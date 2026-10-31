"""Build a shareable review/run notebook from canonical source and measured tables."""
import hashlib
import json
from pathlib import Path
import subprocess
from textwrap import dedent

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
REPOSITORY = 'https://github.com/steffanylars/nus-web-mining-business-insights'


def build():
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    cells = []
    hashes = {}

    def cell(kind, source):
        text = dedent(source).strip() + '\n'
        value = dict(cell_type=kind, id=f'ethan-{len(cells):03d}', metadata={},
                     source=text.splitlines(keepends=True))
        if kind == 'code':
            value.update(execution_count=None, outputs=[])
        cells.append(value)

    def markdown(source):
        cell('markdown', source)

    def code(source):
        cell('code', source)

    def snapshot(filename):
        path = BASE / 'src' / filename
        hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        text = path.read_text(encoding='utf-8-sig').rstrip()
        link = f'{REPOSITORY}/blob/{revision}/Final%20Project/src/{filename}'
        markdown(f'### Source to read: {filename}\n\n'
                 f'[Canonical source at the pinned revision]({link}). '
                 'The following is a readable source snapshot, not an executed training cell. '
                 'The run cell below calls this script from the checked-out repository.\n\n'
                 f'```python\n{text}\n```')

    markdown('''
        # Ethan's JDSearch pipeline walkthrough

        This notebook is for teammates reviewing the corrected work on **ethan**.
        It explains the inputs, shows the actual implementation, presents saved
        experiment results, and provides cells to rerun the pipeline.

        **Read first:** the results below come from the previously verified
        20,000-user Windows CPU run. They are not outputs of a new Colab training run.
        The notebook's review cells have been checked locally; a fresh Colab/GPU
        experiment has not been executed for this notebook.

        You can read all explanations, source snapshots and result tables directly
        in GitHub or Colab without running anything. The executable review cells
        do not train models unless you enable the experiment flag.

        1. Understand the task and field roles.
        2. Read the timeline and complete-history construction.
        3. Inspect features and evaluation before the models.
        4. Compare baselines, neural controls and diagnostics.
        5. Discuss what the evidence supports.

        Current outcome labels are targets. They must never enter input features.
        This is reranking of provided slates, not catalogue-wide retrieval.
    ''')
    markdown('''
        ## Setup for review

        In Colab this cell clones the repository into a dedicated folder and checks
        out the pinned source revision. Locally it finds the existing checkout and
        does not change its branch. Pinning separates reproducible source from
        future branch edits. This notebook itself can live on a newer commit.
    ''')
    code(f'''
        import os
        from pathlib import Path
        import subprocess
        import sys
        from IPython.display import Markdown, display

        REPO_URL = {REPOSITORY + '.git'!r}
        SOURCE_REF = {revision!r}
        try:
            import google.colab
            IN_COLAB = True
        except ImportError:
            IN_COLAB = False

        if IN_COLAB:
            REPO_DIR = Path('/content/ethan-jdsearch-review')
            if not (REPO_DIR / '.git').exists():
                subprocess.run(['git', 'clone', REPO_URL, str(REPO_DIR)], check=True)
            subprocess.run(['git', '-C', str(REPO_DIR), 'checkout', SOURCE_REF], check=True)
        else:
            candidates = [Path.cwd(), *Path.cwd().parents, Path.cwd() / 'final-report-review']
            REPO_DIR = next((p for p in candidates
                if (p / '.git').exists() and (p / 'Final Project/src').is_dir()), None)
            if REPO_DIR is None:
                raise RuntimeError('Open this notebook from the repository checkout.')

        PROJECT = REPO_DIR / 'Final Project'
        SAVED = PROJECT / 'artifacts/corrected'
        print('Source revision:', SOURCE_REF)
        print('Repository:', REPO_DIR)
        print('Notebook Python:', sys.version.split()[0])
    ''')
    markdown('''
        ## Saved results for the team to inspect

        These are the checked-in measured results of the fixed 20,000-user sample
        (14,000 train / 3,000 validation / 3,000 test, seed 42). The complete test
        slates contain 266,640 candidates. Ranking averages use 2,965 queries with
        a positive outcome and exclude the other 35 all-negative queries.

        The simple MLP control contains the **same engineered history features**;
        it removes the additional neural history towers. The matched experiment
        does not establish that dual attention improves on those features.
    ''')
    markdown((BASE / 'artifacts/corrected/RESULTS.md').read_text(encoding='utf-8'))
    code('''
        # Review the source-of-truth table without fitting any model.
        display(Markdown((SAVED / 'RESULTS.md').read_text(encoding='utf-8')))
    ''')
    markdown('''
        ## Optional environment and data setup

        The review cells need no raw data. To run experiments, obtain the two raw
        files from the [official JDSearch source](https://github.com/rucliujn/JDsearch)
        and set DATA_DIR. Google Drive can hold the files; mounting it is optional.
        Raw data and large model state are not included in this notebook/repository.

        INSTALL_ENV creates an isolated Python environment for the scripts, keeping
        the notebook kernel's preinstalled packages separate. The requirements were
        validated with Python 3.12 on Windows; Linux wheel/GPU compatibility has not
        been independently verified here. Inspect installation errors and the selected
        runtime rather than assuming a Colab run reproduces the saved CPU results.

        Start with 1,000 users for a trial. Set N_USERS to 20,000 to match the saved
        sample. The full feature build has substantial RAM requirements; a smaller
        sample or a runtime with more RAM may be necessary. Smaller-run scores must
        not be presented as the saved 20,000-user experiment.
    ''')
    code('''
        INSTALL_ENV = False
        MOUNT_DRIVE = False
        RUN_EXPERIMENTS = False
        RUN_REGRESSION_CHECKS = False
        N_USERS = 1000  # Set 20000 to reproduce the saved sample size.
        SEED = 42

        if MOUNT_DRIVE:
            if not IN_COLAB:
                raise RuntimeError('Google Drive mounting is only available in Colab.')
            from google.colab import drive
            drive.mount('/content/drive')

        DATA_DIR = (Path('/content/drive/MyDrive/JDsearch') if IN_COLAB
                    else PROJECT / 'data')
        RUN_DIR = ((Path('/content/jdsearch-runs') if IN_COLAB else REPO_DIR.parent / 'jdsearch-runs')
                   / f'ethan-users{N_USERS}-seed{SEED}')
        EXPERIMENT_PYTHON = sys.executable

        if INSTALL_ENV:
            venv_dir = (Path('/content/jdsearch-venv') if IN_COLAB
                        else REPO_DIR.parent / '.venv-jdsearch-notebook')
            subprocess.run([sys.executable, '-m', 'venv', str(venv_dir)], check=True)
            EXPERIMENT_PYTHON = str(venv_dir / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python'))
            subprocess.run([EXPERIMENT_PYTHON, '-m', 'pip', 'install',
                            '-r', str(PROJECT / 'requirements.txt')], check=True)

        experiment_env = os.environ.copy()
        experiment_env.update(JD_USER_TXT=str(DATA_DIR / 'user_behavior_data.txt'),
                              JD_PRODUCT_TXT=str(DATA_DIR / 'product_meta_data.txt'),
                              JD_RUN_DIR=str(RUN_DIR), JD_SEED=str(SEED))
        print('Raw data directory:', DATA_DIR)
        print('New output directory:', RUN_DIR)
        print('Training enabled:', RUN_EXPERIMENTS)
        # The new run directory is separate from the published corrected evidence.
    ''')
    code('''
        def run_stage(*stages):
            if not RUN_EXPERIMENTS:
                print('Skipped experiment stages:', ', '.join(stages))
                return
            missing = [name for name in ['user_behavior_data.txt', 'product_meta_data.txt']
                       if not (DATA_DIR / name).is_file()]
            if missing:
                raise FileNotFoundError(f'Place the raw files in {DATA_DIR}: {missing}')
            subprocess.run([EXPERIMENT_PYTHON, '-u', str(PROJECT / 'src/run_pipeline.py'),
                            '--users', str(N_USERS), '--stages', *stages],
                           cwd=REPO_DIR, env=experiment_env, check=True)
    ''')
    markdown('''
        ## 1 What the historical fields mean

        Each candidate pair has a current query, product ID and outcome label.
        A user's preceding events have item IDs, action types, optional historical
        queries and consecutive time gaps. Term IDs inside text fields are separated
        by **U+0018**; underscore separates list elements, not terms.

        If n events have gaps [0, g1, ..., gn], the final gap ends at the current
        query. The age of event i is the sum of all subsequent gaps. For [0,12,4,5],
        event ages are [21,9,5] seconds. Using a single gap as an event age is wrong.

        The shared short channel is CLICK <=7 days or CART <=90 days. The order
        channel uses preceding ORD events of any observed age. Sequence sampling
        happens separately within each complete horizon.
    ''')
    snapshot('data_utils.py')
    code('''
        # A small illustrative calculation, not a new dataset/model result.
        from itertools import accumulate
        query = '12\\x1832\\x1856'
        gaps = [0, 12, 4, 5]
        ages_seconds = list(reversed(list(accumulate(reversed(gaps)))))[1:]
        assert query.split('\\x18') == ['12', '32', '56']
        assert ages_seconds == [21, 9, 5]
        print('Query terms:', query.split('\\x18'))
        print('Event ages before query:', ages_seconds, 'seconds')
    ''')
    markdown('''
        ## 2 Temporal analysis and the instructor's feedback

        The full-data analysis finds sharp changes near seven days for clicks and
        90 days for carts. Rare events remain outside those approximate windows.
        This is consistent with differing logging/retention windows; it does not
        verify JD's actual storage policy. Missing older clicks are unobserved,
        not negative preference evidence. Historical input semantics are separate
        from the current outcome gains used by the ranker.

        Stage 1 gives broad age bins; stage 1b gives daily counts. Both scan the
        full raw behavior file even when later modeling uses a smaller sample.
    ''')
    snapshot('step1_preprocess_temporal.py')
    code("run_stage('1', '1b')")
    markdown('''
        ## 3 Build the dataset and preserve complete histories

        Read build_records first: it aligns IDs/types/queries/times, preserves every
        history event, keeps source-row identity and splits whole users. main then
        reads raw product metadata for referenced products, including category names.

        The old shared latest-60-event cap removed most orders before the long-term
        tower saw them. It is removed here. For the saved 20,000-user sample, history
        length increases from 907,554 capped events to 3,091,153 complete events.
        The sample and split membership remain the same.
    ''')
    snapshot('step2_build_dataset.py')
    code("run_stage('2')")
    markdown('''
        ## 4 Engineer the candidate features

        Read the training-only popularity block, then the per-user profiles, then
        the candidate feature loop. Content features compare real term sets. Short
        affinity uses the shared short mask; order affinity uses all preceding orders.
        Training-only popularity is shared across users; personal profiles use each
        user's own available history, including at held-out inference.

        h_min_elapsed is the age of the user's last event. It is not the last time
        this particular candidate was seen. Current labels are carried alongside
        feature rows for training/evaluation and are excluded from predictors.
    ''')
    snapshot('step3_features.py')
    code("run_stage('3')")
    markdown('''
        ## 5 Define a fair evaluation before comparing models

        NDCG@10 uses gains 2^label-1. MRR@200, P@1 and HR@10 treat label >0 as relevant.
        Macro averages use positive queries only. Validation and final evaluation
        use the same policy. Full supplied candidate slates remain intact at evaluation.

        Original candidate order is label-associated. Equal scores must therefore
        not inherit that order. tie_keys uses seeded query/candidate identities so
        a row permutation cannot change the tied ordering.
    ''')
    snapshot('eval_utils.py')
    markdown('''
        ## 6 Baselines and LambdaMART

        Compare random, training-history popularity, content overlap, training-only
        SVD with held-out history fold-in, and the feature-based ranker. Shared SVD
        factors are fit on training histories; a held-out user's preceding history
        is transformed into that fixed space. It is not a joint test-history fit.

        LambdaMART selects its iteration using the validation metric. Test scores
        are for reporting, not choosing settings.
    ''')
    snapshot('step4_baselines_lgbm.py')
    code("run_stage('4')")
    markdown('''
        ## 7 Neural models and matched tower controls

        Read fit_preprocessing and encode_frames, then pack, DHAR, HistoryCF,
        listwise and train_one in that order. Training-only vocabularies and
        normalization are saved. The brand hash is deterministic.

        Numeric input is [batch, slate, features]. Candidate embeddings are
        [batch, slate, 32]. Each history channel holds up to 20 sampled known-category
        positions. Candidate-category attention supplies short/order context vectors.

        The MLP, pooled, short-only, order-only and dual variants share numeric
        features and a matched training procedure. The MLP retains engineered history
        features, isolating the incremental sequence towers. Training slates use up
        to 64 candidates with all positives retained; evaluation uses complete slates.

        HistoryCF is a GMF/MLP adaptation with a pooled preceding-item encoder,
        not a faithful user-ID NCF reproduction. It can represent an unseen user
        without an untrained session-ID embedding.
    ''')
    snapshot('step5_neural.py')
    code("run_stage('5')")
    markdown('''
        ## 8 Diagnostics and interpretation

        Rank each full slate before measuring item subgroups. Removing all warm
        products before assessing cold products would change their competitors.
        Distinguish absence from training history from absence from training
        candidate rows. Limited-history users are not necessarily new customers.

        Compare the dual model with the same-feature MLP and pooled-history
        control. The separate LambdaMART ablation tests a temporal/action feature
        bundle, not the neural towers. Bootstrap intervals are conditional on
        saved models and do not estimate training-seed variation.
    ''')
    snapshot('step6_diagnostics.py')
    code("run_stage('6', 'report')")
    markdown('''
        ## 9 Checks and discussion

        The regression suite targets the original failure modes. The optional
        completed-run verifier additionally requires a full 20,000-user run and
        local checkpoints; it should not be applied to a 1,000-user trial.

        Questions to discuss:

        - Why is a user split appropriate, and how does it differ from the paper's protocol?
        - Which inputs are available before current outcomes, and what is fit only on training data?
        - Why do complete histories matter even though neural sequences remain bounded?
        - What does the MLP comparison tell us about added architecture complexity?
        - Which failure cases and resource measurements should the final report include?

        The inspected test set is not a new untouched holdout. Offline ranking does
        not establish conversion/revenue lift or causal shopper behavior.
    ''')
    code('''
        if RUN_REGRESSION_CHECKS:
            subprocess.run([EXPERIMENT_PYTHON, '-m', 'unittest', 'discover',
                            '-s', str(PROJECT / 'tests'), '-v'],
                           cwd=REPO_DIR, env=experiment_env, check=True)
        else:
            print('Regression tests not rerun; enable RUN_REGRESSION_CHECKS after installing dependencies.')

        if RUN_EXPERIMENTS:
            display(Markdown((RUN_DIR / 'RESULTS.md').read_text(encoding='utf-8')))
        else:
            print('No training was requested; the tables above remain the historical saved results.')
    ''')
    markdown(f'''
        ## Team reference links

        - [Change log]({REPOSITORY}/blob/ethan/Final%20Project/ETHAN_CHANGES.md)
        - [Corrected report]({REPOSITORY}/blob/ethan/Final%20Project/artifacts/corrected/REPORT.md)
        - [A+ reference improvements]({REPOSITORY}/blob/ethan/Final%20Project/APLUS_REFERENCE_IMPROVEMENTS.md)
        - [Reproduction README]({REPOSITORY}/blob/ethan/Final%20Project/README.md)

        This notebook is a review and execution interface for the existing pipeline,
        not an additional model experiment. Source snapshots correspond to commit
        `{revision}`; regenerate the notebook if the canonical source changes.
    ''')
    notebook = dict(nbformat=4, nbformat_minor=5, cells=cells,
        metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                      language_info=dict(name='python'),
                      colab=dict(name='Ethan_JDSearch_Pipeline.ipynb', provenance=[]),
                      jdsearch=dict(source_revision=revision, source_sha256=hashes,
                                    result_provenance='saved verified 20000-user CPU run; not a fresh Colab run')))
    target = BASE / 'notebooks' / 'Ethan_JDSearch_Pipeline.ipynb'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Wrote {target}: {len(cells)} cells')


if __name__ == '__main__':
    build()
