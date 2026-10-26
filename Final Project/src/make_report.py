# -*- coding: utf-8 -*-
"""Builds the BT4222 Final Report as a formatted .docx (Arial 11, 1.5 spacing)."""
from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

BASE = Path("/Users/steff/Documents/GitHub/bt4212_finalProj/Final Project")
FIG = BASE/"figures"; REP = BASE/"report"; REP.mkdir(exist_ok=True)

doc = Document()
# base style
st = doc.styles["Normal"]; st.font.name="Arial"; st.font.size=Pt(11)
st.paragraph_format.line_spacing=1.5
for s in ["Heading 1","Heading 2","Heading 3","Title"]:
    try: doc.styles[s].font.name="Arial"
    except KeyError: pass

def h(txt,lvl=1):
    p=doc.add_heading(txt,level=lvl)
    for r in p.runs: r.font.color.rgb=RGBColor(0,0,0)
    return p
def para(txt,bold=False,italic=False,align=None,size=11):
    p=doc.add_paragraph(); r=p.add_run(txt); r.bold=bold; r.italic=italic; r.font.size=Pt(size)
    if align: p.alignment=align
    return p
def bullet(txt,bold_lead=None):
    p=doc.add_paragraph(style="List Bullet")
    if bold_lead:
        r=p.add_run(bold_lead); r.bold=True; p.add_run(txt)
    else: p.add_run(txt)
    return p
def fig(name,caption,width=6.0):
    if (FIG/name).exists():
        doc.add_picture(str(FIG/name),width=Inches(width))
        doc.paragraphs[-1].alignment=WD_ALIGN_PARAGRAPH.CENTER
        c=para(caption,italic=True,size=9.5,align=WD_ALIGN_PARAGRAPH.CENTER)
def table(headers,rows,bold_first_col=False,widths=None):
    t=doc.add_table(rows=1,cols=len(headers)); t.style="Light Grid Accent 1"; t.alignment=WD_TABLE_ALIGNMENT.CENTER
    hdr=t.rows[0].cells
    for i,hd in enumerate(headers):
        hdr[i].text=""; r=hdr[i].paragraphs[0].add_run(hd); r.bold=True; r.font.size=Pt(10)
    for row in rows:
        cells=t.add_row().cells
        for i,v in enumerate(row):
            cells[i].text=""; rr=cells[i].paragraphs[0].add_run(str(v)); rr.font.size=Pt(10)
            if bold_first_col and i==0: rr.bold=True
    return t

# ============ COVER ============
for _ in range(3): doc.add_paragraph()
para("BT4222 — Mining Web Data for Business Insights",bold=True,align=WD_ALIGN_PARAGRAPH.CENTER,size=16)
para("Final Project Report",align=WD_ALIGN_PARAGRAPH.CENTER,size=14)
doc.add_paragraph()
para("Personalised Product Search on JDSearch:",bold=True,align=WD_ALIGN_PARAGRAPH.CENTER,size=15)
para("A Dual-Horizon Re-Ranking System Driven by Log-Retention Dynamics",bold=True,align=WD_ALIGN_PARAGRAPH.CENTER,size=15)
doc.add_paragraph(); doc.add_paragraph()
para("Group No. 13",bold=True,align=WD_ALIGN_PARAGRAPH.CENTER,size=13)
para("National University of Singapore  |  Professor Wang Qiuhong",align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
for nm,mt in [("Chow Ee Xiang","A0300297J"),("Ethan John Ong Jian-Hao","A0308681Y"),
              ("Heng Ann Xin","A0282911E"),("Ku Jia Yi","A0300567J"),
              ("Lara Muy Steffany Mishell","A0350365H")]:
    para(f"{nm}    |    {mt}",align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_page_break()

# ============ ABSTRACT ============
h("Abstract",1)
para("Business objective & RecSys formulation. We build a personalised re-ranking system for "
     "JDSearch (173,831 users, 15.5M query–candidate pairs, 26.7M interactions from JD.com). For each "
     "search query the system re-orders the supplied candidate slate (median 90 products) so that "
     "products the shopper later clicks, carts or purchases rise to the top, using the current query, "
     "product metadata and the user's preceding multi-behaviour history. Ground truth is the ordinal "
     "candidate label (0 none / 1 click / 2 cart / 3 purchase); success is NDCG@10, MRR@200 and P@1.")
para("Prior baseline & critical diagnosis. We reproduce the collaborative-filtering family (NCF, NMF/SVD) "
     "used by past projects and show it is structurally mismatched to JDSearch: test queries are almost "
     "always new, 80.5% of catalogue products never appear in any history, and each user contributes a "
     "single test record, so ID-based CF collapses to near-random (NCF NDCG@10 0.086 vs. random 0.083).")
para("Group contributions & modelling highlights. Tracing the history timeline back in time (per the "
     "instructor's prompt) we uncover a telescoping log-retention window — clicks are retained only ~7 days, "
     "carts ~90 days and orders ≥1 year — which means interaction TYPE is confounded with RECENCY. This "
     "reframes feature engineering: recent clicks/carts encode short-term query intent, whereas old orders "
     "encode long-term brand/category preference. We engineer horizon-aware features and design a "
     "Dual-Horizon Attention Ranker (DHAR) that separates these two roles with candidate-category attention.")
para("Evidence-backed quantitative insights. DHAR reaches NDCG@10 0.336 / MRR@200 0.351 / P@1 0.229, "
     "beating LambdaMART (0.313), SVD (0.274), popularity (0.270) and all weak baselines. Diagnostics show "
     "the model ranks eventual purchases highest (median rank 7) vs. clicks (median 26), validating the "
     "distinct-roles hypothesis, while a warm/cold split (NDCG 0.586 vs 0.180) quantifies the cold-product "
     "ceiling that bounds every model.")
doc.add_page_break()

# ============ SECTION 1 ============
h("Section 1 — Data Understanding & Feature Engineering",1)
para("This section reports what changed since the interim report: a temporal re-reading of user history "
     "that reshaped our feature engineering, followed by the resulting feature set. Dataset overview, slate "
     "sizes, sparsity and the feature dictionary were established in the interim report and are not repeated.")

h("1.1 The decisive finding: a telescoping log-retention window",2)
para("The interim report noted the counter-intuitive fact that historical carts and orders outnumber clicks "
     "(CART 43.2%, CLICK 31.1%, ORD 25.6% of 26.7M events). Following the instructor's guidance we converted "
     "each history event's consecutive intervals (history_time_list, in seconds) into elapsed days before the "
     "final query (a suffix-sum over the gap list) and examined the click/cart/order mix as we trace activity "
     "further back. The pattern is stark and is computed on the full 173,831-user file:")
table(["Elapsed before query","CLICK share","CART share","ORD share","Events"],
      [["≤ 1 day","85.5%","12.8%","1.6%","—"],
       ["1–7 days","84.0%","13.8%","2.2%","7.60M"],
       ["7–30 days","0.00%","85.5%","14.3%","3.86M"],
       ["30–90 days","0.00%","84.2%","15.6%","8.18M"],
       ["90–180 days","0.00%","0.07%","99.9%","1.83M"],
       ["180–365 days","0.00%","0.00%","100.0%","2.96M"]])
fig("fig_temporal_composition.png","Figure 1. Interaction-type composition as history is traced back in time (full data).",5.8)
para("A 1-day-resolution scan pins exact cut-offs: clicks vanish between day 6 (520,760 clicks) and day 7 "
     "(5 clicks); carts collapse between day 85 (100,902) and day 90 (365); orders persist to ~1 year. This "
     "is not user behaviour — it is a data-retention policy: JD keeps ephemeral clicks ~7 days, mid-value "
     "carts ~90 days, and high-value transactions ≥1 year. The millions of 'orders/carts with no clicks' "
     "the instructor highlighted are simply the long tail of history, where only durable signals survive.")
fig("fig_retention_cutoffs.png","Figure 2. Per-type event volume by age (log scale): click ~7d, cart ~90d, order ~1yr retention.",5.8)

h("1.2 Modelling implications (how the finding informs features)",2)
bullet(" Interaction TYPE is confounded with RECENCY. Weighting ORD>CART>CLICK as pure 'intent intensity' "
       "would in fact encode 'old vs recent'. We therefore never treat type as a standalone preference scale.",
       "Confound awareness.")
bullet(" Absence of clicks in old history is a logging artifact, not 'this user never clicks'. Old windows "
       "must not be read as negative click evidence.","No false negatives.")
bullet(" Fields play distinct roles. Recent clicks/carts (≤7–90 d) = short-term intent for the CURRENT query; "
       "old orders = long-term brand/category preference. We build two separate history representations.",
       "Dual horizon.")
bullet(" Content relevance is weak via anonymised term-IDs. Exact query↔name token overlap is ~0.0002 on "
       "average and, alone, ranks no better than random (NDCG 0.087). Category/brand affinity and popularity "
       "carry the personalising signal instead.","Signal reality-check.")

h("1.3 Engineered feature set",2)
para("Features are grouped by the role established above. Popularity priors are computed on TRAIN histories "
     "only (no label leakage); history features use each record's own preceding events.")
table(["Group","Representative features","Rationale"],
      [["Content relevance","ov_name, ov_cat, ov_brand, jacc_name, query_len","Query↔candidate token overlap (weak but non-zero)"],
       ["Popularity prior","pop_wid, pop_brand, pop_cate3 (log, train-only)","Head-vs-long-tail exposure prior"],
       ["Short-term intent","short_cat_hit, short_brand_hit, aff_cat_decay (τ=30d), h_n_recent7","Recent (≤7d) clicks/carts → current intent"],
       ["Long-term preference","long_cat_hit, long_brand_hit, h_n_ord","Order-history category/brand affinity"],
       ["Re-engagement","seen_before, h_min_elapsed","Has user touched this exact item / how recently"],
       ["User activity","h_len, h_n_click, h_n_cart, h_n_distinct_c3, h_queryless","Activity level & diversity"]])
para("Feature means by label confirm the design: long-term category affinity (long_cat_hit) is highest for "
     "purchases (0.12 vs ~0.05 for clicks/carts), while short-term affinity dominates clicks/carts — exactly "
     "the distinct-roles hypothesis showing up in the data before any model is trained.")

# ============ SECTION 2 ============
doc.add_page_break()
h("Section 2 — Models and Performance",1)
h("2.1 Task set-up, splits and metrics",2)
para("Each record is one query with its candidate slate and ordinal labels, so the task is slate re-ranking: "
     "the slate already contains the negatives (97.9% label 0), and catalogue-wide negative sampling is "
     "unnecessary. We sample 20,000 users (seed 42; 1.78M candidate pairs, positive rate 2.05%, matching the "
     "full-data 2.07%) and split by session 70/15/15 into train/val/test. Metrics are computed per query over "
     "sessions containing ≥1 positive (n=2,965 test). Crucially, because the first listed candidate is positive "
     "in 98.8% of slates (interim §2.2.1), we break ranking ties with a fixed-seed random key so candidate "
     "POSITION cannot leak — without this guard a constant scorer spuriously reaches NDCG 0.97.")
table(["Metric","Definition (course Topic 4)"],
      [["NDCG@10","Σ(2^rel−1)/log2(rank+1) over top-10, normalised by ideal DCG; rel = ordinal label"],
       ["MRR@200","Reciprocal rank of the first positive (label>0) within top-200"],
       ["P@1","Top-1 candidate is a positive"],
       ["HR@10","Any positive within top-10"]])

h("2.2 Models",2)
bullet(" ranks the slate by a random key (floor).","Random:")
bullet(" ranks by log popularity prior (non-personalised).","Popularity:")
bullet(" query↔candidate token overlap only.","Content / relevance-only:")
bullet(" 64-factor TruncatedSVD on the user×historical-item implicit matrix; cold candidates have no factor "
       "(NCF/NMF family).","Matrix Factorisation:")
bullet(" GMF+MLP embeddings over (user, item) ids, BCE on positives (course Topic 6 / Basic Recommenders).","NCF (neural):")
bullet(" LightGBM lambdarank over all engineered features, label gains [0,1,3,7] (learning-to-rank).","LambdaMART:")
bullet(" our architecture (below).","DHAR:")
para("DHAR — Dual-Horizon Attention Ranker. Candidate category/brand embeddings form an attention query "
     "that attends separately over (a) the user's RECENT (≤7d) history-category sequence — short-term intent — "
     "and (b) the ORDER-history category sequence — long-term preference. The two context vectors are fused "
     "with candidate embeddings and the engineered features in an MLP head, trained listwise (softmax "
     "cross-entropy with ordinal label gains) per slate on Apple MPS. The architecture is a direct "
     "materialisation of the §1 finding: two retention horizons, two towers, one attention each.")
fig("fig_dhar_architecture.png","Figure 3. DHAR architecture: two retention horizons → two candidate-category attention towers → listwise ordinal ranking.",6.2)

h("2.3 Results",2)
table(["Model","NDCG@10","MRR@200","P@1","HR@10"],
      [["Random","0.083","0.102","0.033","0.227"],
       ["NCF (ID-based CF)","0.086","0.104","0.032","0.230"],
       ["Content / relevance-only","0.087","0.108","0.039","0.232"],
       ["Popularity","0.270","0.283","0.167","0.518"],
       ["Matrix Factorisation (SVD)","0.274","0.289","0.170","0.518"],
       ["LambdaMART (features)","0.313","0.325","0.204","0.579"],
       ["DHAR (ours)","0.336","0.351","0.229","0.603"]],bold_first_col=True)
fig("fig_model_comparison.png","Figure 4. Model comparison on held-out test slates.",6.0)
para("Reading the table. (1) Both CF baselines (NCF, SVD-cold) sit at the random floor on cold candidates — "
     "the structural gap. (2) Popularity is the real workhorse but saturates. (3) Feature-based LambdaMART "
     "adds personalised lift; (4) DHAR's horizon separation adds a further +0.023 NDCG@10 over LambdaMART, "
     "the best of all models on every metric.")
para("Training vs test (overfitting check). Popularity 0.292→0.270; LambdaMART 0.394→0.313 (gap 0.081); "
     "DHAR 0.349→0.336 (gap 0.013). The dual-horizon model not only scores highest on test but also "
     "generalises markedly better than the gradient-boosted ranker, whose larger train–test gap reflects the "
     "popularity/ID features it memorises; DHAR's category-attention towers are more transferable.")

# ============ SECTION 3 ============
doc.add_page_break()
h("Section 3 — Contribution and Justification",1)
h("3.1 Self-Evaluation Table",2)
table(["Contribution aspect","Effort","Effectiveness","Evidence"],
      [["1. Data understanding, rich-signal use & feature engineering","Optional","Optional",
        "Full-data temporal retention analysis (§1.1); horizon-aware features (§1.3)"],
       ["2. Prior baseline benchmarking & gap diagnosis","Optional","Optional",
        "NCF & SVD reproduced; cold-item & new-query failure quantified (§2.3, §3.3)"],
       ["3. Designing/adapting recommender architectures","Optional","Optional",
        "Dual-Horizon Attention Ranker built from the retention finding (§2.2)"],
       ["4. Model diagnostics, error analysis & insights","Optional","Optional",
        "Warm/cold & history segments, rank-by-type, ablation (§3.4)"],
       ["5. Smart use of AI tools & critical collaboration","Optional","Optional",
        "AI Interaction Log (Section 4)"]],bold_first_col=True)
para("('Optional' = the advanced tier of the rubric, justified by the narrative below.)",italic=True,size=10)

h("3.2 Feature engineering & rich-signal utilisation (Aspect 1)",2)
para("Rather than collapsing interactions into binary pairs, we exploited the raw multi-behaviour, timestamped "
     "logs and impression slates. The temporal retention discovery (§1.1) is the project's analytical core: it "
     "explains a dataset anomaly, is computed on all 26.7M events with exact cut-offs, and drives every "
     "downstream feature. We separate short-term intent from long-term preference and compute recency-decayed "
     "category/brand affinities — domain-informed features that a naive type-weighting would get backwards.")

h("3.3 Baseline benchmarking & gap diagnosis (Aspect 2)",2)
para("We reproduced the CF family and diagnosed WHY it fails here rather than merely reporting low numbers: "
     "(i) each user has one test record with an almost-always-new query, so user-ID factors cannot transfer; "
     "(ii) 80.5% of candidate products never appear in any history, so item factors are undefined — on test, "
     "only 14.5% of candidates are 'warm'. NCF therefore equals random (0.086) and SVD only reaches popularity "
     "level. This gap — new-query generalisation under massive item cold-start — is precisely what our "
     "content/affinity features and DHAR target.")

h("3.4 Diagnostics, error analysis & business insight (Aspect 4)",2)
table(["Segment","NDCG@10","Read"],
      [["Warm candidate (seen in train hist)","0.586","Model ranks well when it has a prior"],
       ["Cold candidate (unseen)","0.180","Cold-product ceiling — the binding limitation"],
       ["Cold users (<3 events)","0.323","Popularity/content carry new users"],
       ["Long-history users (>100)","0.311","No degradation with history length"]])
para("Rank-of-positive by interaction type (lower is better): eventual purchases land at median rank 7, carts "
     "at 15, clicks at 26. The model is most confident exactly where business value is highest — because "
     "purchases align with the stable long-term order signal, while clicks are noisier short-term intent. "
     "Ablating the temporal-aware features costs +0.013 NDCG@10, confirming they contribute beyond popularity. "
     "Business takeaway: (a) invest in cold-product content representations (the 0.586→0.180 gap is the main "
     "lever); (b) a two-horizon model should drive purchase-oriented ranking, while a lighter recency model "
     "suffices for click-through objectives.")
fig("fig_feature_importance.png","Figure 5. LambdaMART feature importance: popularity, re-engagement and horizon/affinity features dominate.",4.8)

h("3.5 Reproducibility & pipeline (Aspect: source code)",2)
para("The pipeline is modular and runs end-to-end: step1 temporal analysis (full data), step2 dataset build, "
     "step3 features, step4 baselines+LambdaMART, step5 neural (DHAR, NCF), step6 diagnostics+ablation. "
     "N_USERS is a single knob to scale from the 20k-user laptop run to the full data on Colab/GPU.")

# ============ SECTION 4 ============
doc.add_page_break()
h("Section 4 — AI Interaction Log (Appendix)",1)
table(["Core prompt / objective","AI output summary","Our critical evaluation","Refinement & final action"],
 [["Why do carts/orders outnumber clicks, and what does tracing history back in time reveal?",
   "Suggested converting interval seconds to elapsed days and cross-tabulating type vs age.",
   "Correct direction but could not know JD's retention policy; risk of over-reading medians.",
   "We computed suffix-sum elapsed days on all 26.7M events and found exact 7d/90d/1yr cut-offs."],
  ["How should ordinal labels 0/1/2/3 be handled for ranking?",
   "Proposed cumulative-link ordinal regression and listwise losses with label gains.",
   "Cumulative-link assumes pure intensity ordering — but §1 shows type≈recency confound.",
   "Adopted listwise softmax with ordinal gains [0,1,3,7]; avoided type-as-intensity weighting."],
  ["Our content/relevance baseline scored NDCG 0.97 — is that right?",
   "Flagged possible tie-breaking/position leakage given the 98.8% first-position positive.",
   "Matched the interim's own warning; a stable sort was leaking candidate order.",
   "Added fixed-seed random tie-breaking in evaluation; honest score dropped to 0.087."],
  ["Design an architecture that uses the retention finding.",
   "Suggested a two-tower / dual-horizon encoder with attention over history.",
   "Generic; needed category-level attention and MPS-efficient padded-slate batching.",
   "Implemented DHAR with per-horizon candidate-category attention; vectorised for MPS."]])

# ============ SECTION 5 ============
doc.add_page_break()
h("Section 5 — References",1)
refs=[
 "Liu, J., Dou, Z., Tang, G., & Xu, S. (2023). JDsearch: A Personalized Product Search Dataset with Real Queries and Full Interactions. Proceedings of SIGIR 2023 (pp. 2945–2952). ACM. https://doi.org/10.1145/3539618.3591900",
 "JDsearch Dataset GitHub Repository. https://github.com/rucliujn/JDsearch",
 "He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T.-S. (2017). Neural Collaborative Filtering. WWW 2017.",
 "Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009). BPR: Bayesian Personalized Ranking from Implicit Feedback. UAI 2009.",
 "Burges, C. (2010). From RankNet to LambdaRank to LambdaMART: An Overview. Microsoft Research Technical Report.",
 "Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS 2017.",
 "BT4222 Lecture Topics 4 (Evaluation), 6 (Embedding), 7 (RNN & Transformer); Ordinal Classification Guide, 2026-27 T1.",
]
for r in refs: bullet(r)

out=REP/"BT4222_Group13_Final_Report.docx"
doc.save(str(out))
print("saved", out)
