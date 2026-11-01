"""Cold-start routing and category diversification on top of the feature ranker.

Addresses two review items:
  1. Users with little or no preceding history are scored by the same model as
     warm users. Here a history-free fallback ranker (current-query matching,
     product metadata, training-set popularity only) scores sessions whose
     history length is at or below a threshold chosen on validation data.
  2. No diversification after relevance scoring. A category-aware MMR re-ranker
     penalises repeating a level-3 category inside the top of the slate; the
     trade-off between NDCG@10 and category coverage is reported for a grid
     of lambda values, with lambda selected on validation data.

Inputs come from stages 2-4 (features, sessions, meta_subset, lgb_model).
Outputs go to RES (results_coldstart_diversity.json, coldstart_segments.csv,
diversity_tradeoff.csv, test_pred_step7.parquet, fig_diversity_tradeoff.png).
"""
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from config import OUT, RES, FIG, SEED, feature_columns, prepare, save_json
from eval_utils import evaluate, per_query, tie_keys
from step4_baselines_lgbm import fit_ranker

# Features that do not depend on the user's preceding history.
HISTORY_FREE = ['query_len', 'slate_size', 'has_meta', 'ov_name', 'ov_cat', 'ov_brand',
                'jacc_name', 'pop_wid', 'pop_cate3', 'pop_brand']
HISTORY_THRESHOLDS = [0, 1, 2, 3, 5, 10, 20]
LAMBDAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
TOP_K = 10


def route_scores(frame, full_col, fallback_col, n_hist, threshold):
    """Fallback score for sessions with n_hist <= threshold, full-model score otherwise.

    Scores of the two rankers live on different scales, but routing is per
    session (the whole slate switches), so no cross-model calibration is needed.
    """
    cold = frame.session_id.map(n_hist).le(threshold).to_numpy()
    return np.where(cold, frame[fallback_col].to_numpy(), frame[full_col].to_numpy()), cold


def mmr_rerank(scores, categories, tie, lam, top_k=TOP_K):
    """Greedy category-aware MMR for one slate.

    Returns a new score array whose ordering is the diversified ranking: the
    first `top_k` chosen positions get descending synthetic scores above all
    others; the remainder keeps the original relevance order. With lam == 0 the
    ranking equals the relevance ranking (ties broken by `tie`).
    """
    s = np.asarray(scores, dtype=float)
    if len(s) == 0:
        return s
    spread = s.max() - s.min()
    rel = (s - s.min()) / spread if spread > 0 else np.zeros_like(s)
    base_order = np.lexsort((tie, -s))
    if lam <= 0:
        return s
    chosen, seen = [], {}
    remaining = list(base_order)
    for _ in range(min(top_k, len(s))):
        best, best_val = None, -np.inf
        for i in remaining:
            penalty = seen.get(categories[i], 0)
            val = (1 - lam) * rel[i] - lam * penalty
            if val > best_val:
                best, best_val = i, val
        chosen.append(best); remaining.remove(best)
        seen[categories[best]] = seen.get(categories[best], 0) + 1
    new = np.empty(len(s)); offset = s.max() + 1.0
    order = chosen + remaining
    new[order] = offset + np.arange(len(s), 0, -1, dtype=float)
    return new


def diversify(frame, score_col, cate_map, lam, seed=0):
    """Apply mmr_rerank per session; returns a score column that encodes the new ranking."""
    out = np.empty(len(frame)); tie = tie_keys(frame, seed)
    cats = frame.cand_wid.map(cate_map).fillna('__unknown__').to_numpy()
    scores = frame[score_col].to_numpy(dtype=float)
    for idx in frame.groupby('session_id', sort=False).indices.values():
        out[idx] = mmr_rerank(scores[idx], cats[idx], tie[idx], lam)
    return out


def diversity_metrics(frame, score_col, cate_map, top_k=TOP_K, seed=0):
    """Mean distinct level-3 categories and mean share of distinct categories in the top-k."""
    tie = tie_keys(frame, seed)
    cats = frame.cand_wid.map(cate_map).fillna('__unknown__').to_numpy()
    scores = frame[score_col].to_numpy(dtype=float)
    distinct, share = [], []
    for idx in frame.groupby('session_id', sort=False).indices.values():
        order = np.lexsort((tie[idx], -scores[idx]))[:top_k]
        top = cats[idx][order]
        distinct.append(len(set(top))); share.append(len(set(top)) / len(top))
    return dict(distinct_categories_top10=float(np.mean(distinct)),
                category_share_top10=float(np.mean(share)))


def main():
    prepare(); start = time.time()
    feat = pd.read_parquet(OUT / 'features.parquet')
    sess = pd.read_parquet(OUT / 'sessions.parquet')
    meta = pd.read_parquet(OUT / 'meta_subset.parquet', columns=['wid', 'cate_id_3'])
    cate_map = dict(zip(meta.wid.astype(str), meta.cate_id_3.astype(str)))
    n_hist = dict(zip(sess.session_id, sess.n_hist))
    columns = feature_columns(feat)
    missing = [c for c in HISTORY_FREE if c not in columns]
    if missing:
        raise ValueError(f'History-free features missing from features.parquet: {missing}')

    # ---- full ranker: reuse the stage-4 booster when present so numbers match stage 4 ----
    model_path = OUT / 'lgb_model.txt'
    if model_path.exists():
        full = lgb.Booster(model_file=str(model_path)); full_source = 'stage4 lgb_model.txt'
    else:
        full = fit_ranker(feat, columns); full_source = 'refit in step 7 (no stage-4 model found)'
    fallback = fit_ranker(feat, HISTORY_FREE)

    val = feat[feat.split == 'val'].copy(); test = feat[feat.split == 'test'].copy()
    for frame in (val, test):
        frame['full_score'] = full.predict(frame[columns])
        frame['fallback_score'] = fallback.predict(frame[HISTORY_FREE])

    # ---- 1. cold-start routing: threshold chosen on validation ----
    routing = []
    for t in [-1] + HISTORY_THRESHOLDS:          # -1 = never route (full model everywhere)
        val['routed'], cold_v = route_scores(val, 'full_score', 'fallback_score', n_hist, t)
        m_v, n_v = evaluate(val, 'routed')
        routing.append(dict(threshold=t, val_NDCG10=m_v['NDCG@10'], val_positive_queries=n_v,
                            val_cold_sessions=int(val.loc[cold_v, 'session_id'].nunique())))
    routing = pd.DataFrame(routing)
    best_t = int(routing.sort_values(['val_NDCG10', 'threshold'], ascending=[False, True]).iloc[0].threshold)
    test['routed_score'], cold_t = route_scores(test, 'full_score', 'fallback_score', n_hist, best_t)

    # per-history-group test comparison: full vs fallback vs routed
    bins = pd.cut(sess.n_hist, [-1, 0, 2, 5, 20, 100, np.inf],
                  labels=['none(0)', 'limited(1-2)', 'very_short(3-5)', 'short(6-20)',
                          'medium(21-100)', 'long(>100)'])
    group_map = dict(zip(sess.session_id, bins.astype(str)))
    test['history_group'] = test.session_id.map(group_map)
    segments = []
    for name, col in [('full', 'full_score'), ('fallback_history_free', 'fallback_score'),
                      ('routed', 'routed_score')]:
        for group, g in test.groupby('history_group', sort=False):
            m, n = evaluate(g, col)
            segments.append(dict(model=name, history_group=group, positive_queries=n,
                                 all_queries=int(g.session_id.nunique()), **m))
        m, n = evaluate(test, col)
        segments.append(dict(model=name, history_group='ALL', positive_queries=n,
                             all_queries=int(test.session_id.nunique()), **m))
    segments = pd.DataFrame(segments)

    # paired bootstrap, routed vs full, on test (fixed models; query resampling only)
    a = per_query(test, 'full_score').set_index('session_id')['NDCG@10'].sort_index()
    b = per_query(test, 'routed_score').set_index('session_id')['NDCG@10'].sort_index()
    delta = (b - a).to_numpy(); rng = np.random.default_rng(SEED)
    boot = [rng.choice(delta, len(delta), replace=True).mean() for _ in range(2000)]
    routed_vs_full = dict(delta=float(delta.mean()), ci95=np.quantile(boot, [.025, .975]).tolist(),
                          positive_queries=int(len(delta)))

    # ---- 2. category diversification: lambda chosen on validation ----
    val['routed'], _ = route_scores(val, 'full_score', 'fallback_score', n_hist, best_t)
    trade = []
    for lam in LAMBDAS:
        val['div'] = diversify(val, 'routed', cate_map, lam)
        test['div'] = diversify(test, 'routed_score', cate_map, lam)
        mv, _ = evaluate(val, 'div'); mt, _ = evaluate(test, 'div')
        trade.append(dict(lam=lam, val_NDCG10=mv['NDCG@10'], test_NDCG10=mt['NDCG@10'],
                          test_MRR200=mt['MRR@200'], test_P1=mt['P@1'],
                          **{f'val_{k}': v for k, v in diversity_metrics(val, 'div', cate_map).items()},
                          **{f'test_{k}': v for k, v in diversity_metrics(test, 'div', cate_map).items()}))
    trade = pd.DataFrame(trade)
    base = trade.loc[trade.lam == 0].iloc[0]
    # Selection rule (validation only): largest lambda whose validation NDCG@10 stays
    # within 0.005 of the undiversified ranking.
    ok = trade[trade.val_NDCG10 >= base.val_NDCG10 - 0.005]
    best_lam = float(ok.lam.max())
    test['final_score'] = diversify(test, 'routed_score', cate_map, best_lam)

    # ---- outputs ----
    segments.to_csv(RES / 'coldstart_segments.csv', index=False)
    trade.to_csv(RES / 'diversity_tradeoff.csv', index=False)
    routing.to_csv(RES / 'coldstart_threshold_selection.csv', index=False)
    test[['session_id', 'cand_wid', 'label', 'full_score', 'fallback_score', 'routed_score',
          'final_score', 'history_group']].to_parquet(OUT / 'test_pred_step7.parquet', index=False)
    fallback.save_model(str(OUT / 'lgb_fallback_history_free.txt'))
    final_m, final_n = evaluate(test, 'final_score')
    full_m, _ = evaluate(test, 'full_score')
    save_json(RES / 'results_coldstart_diversity.json', dict(
        full_model_source=full_source, history_free_features=HISTORY_FREE,
        fallback_best_iteration=fallback.best_iteration,
        routing=dict(selected_threshold=best_t, rule='max validation NDCG@10 (ties -> smaller threshold)',
                     cold_test_sessions=int(test.loc[cold_t, 'session_id'].nunique()),
                     cold_test_share=float(cold_t.mean()), routed_minus_full_test=routed_vs_full),
        diversification=dict(selected_lambda=best_lam, top_k=TOP_K,
                             rule='largest lambda with validation NDCG@10 within 0.005 of lambda=0',
                             category_key='cate_id_3', unknown_metadata_bucket='__unknown__'),
        test_full=full_m, test_final=final_m, test_positive_queries=final_n,
        interpretation='one seed; thresholds and lambda selected on validation only; '
                       'bootstrap reflects query resampling for fixed models'))
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(trade.test_distinct_categories_top10, trade.test_NDCG10, 'o-', color='#2b7bba')
        for _, r in trade.iterrows():
            ax.annotate(f'λ={r.lam:g}', (r.test_distinct_categories_top10, r.test_NDCG10),
                        textcoords='offset points', xytext=(4, 4), fontsize=8)
        ax.set_xlabel('Distinct level-3 categories in top-10 (test)'); ax.set_ylabel('NDCG@10 (test)')
        ax.set_title('Relevance vs category diversity (MMR on routed scores)')
        fig.tight_layout(); fig.savefig(FIG / 'fig_diversity_tradeoff.png', dpi=150); plt.close(fig)
    except Exception as exc:  # figure is optional
        print('figure skipped:', exc)
    print(routing.to_string(index=False)); print()
    print(segments.pivot(index='history_group', columns='model', values='NDCG@10').to_string()); print()
    print(trade[['lam', 'val_NDCG10', 'test_NDCG10', 'test_distinct_categories_top10']].to_string(index=False))
    print(f'\nselected threshold={best_t}, lambda={best_lam}; test NDCG@10 full={full_m["NDCG@10"]:.4f} '
          f'final={final_m["NDCG@10"]:.4f}; step 7 complete ({time.time()-start:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
