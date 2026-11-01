"""Dataset & Source-Code Listing (v2): the single PDF due 9 Nov with verified repo links and
file-structure descriptions. Numbers and file sizes are read from the repository itself."""
import shutil
import subprocess
from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches

BASE = Path(__file__).resolve().parents[1]
REPO = 'https://github.com/steffanylars/nus-web-mining-business-insights'
BRANCH = 'steff-coldstart-diversity'   # change to the team's final branch before submission
OUT = BASE / 'report'

STRUCTURE = [
    ('README.md', 'Repository overview and pointers.'),
    ('Interim Report/JDSearch_Data_Anlaysis.ipynb', 'Interim EDA notebook (data acquisition, descriptive statistics, feature dictionary).'),
    ('Final Project/README.md', 'Task, protocol, reproduction commands (run_pipeline.py, env vars), stage list.'),
    ('Final Project/ETHAN_CHANGES.md', 'Audit change log: what was corrected in the pipeline and why; corrected vs draft scores.'),
    ('Final Project/STEP7_COLDSTART_DIVERSITY.md', 'Cold-start routing and category diversification: implementation, measured results, disposition of AI suggestions.'),
    ('Final Project/AI_USAGE.md', 'Generative-AI usage disclosure (complements report Section 4).'),
    ('Final Project/requirements.txt', 'Python dependencies (pandas, numpy, pyarrow, scikit-learn, lightgbm, torch, python-docx).'),
    ('Final Project/src/config.py, data_utils.py, eval_utils.py', 'Portable paths/env vars; delimiter + elapsed-time helpers; order-invariant ranking metrics.'),
    ('Final Project/src/step1_preprocess_temporal.py, step1b_retention_cutoffs.py', 'Full-data temporal retention analysis (173,831 users, 26.7M events) and 1-day cut-off scan.'),
    ('Final Project/src/step2_build_dataset.py', 'Seeded 20,000-user sample, complete histories, 70/15/15 user split, raw metadata subset.'),
    ('Final Project/src/step3_features.py', 'Horizon-aware feature engineering (26 features; train-only popularity priors).'),
    ('Final Project/src/step4_baselines_lgbm.py', 'Random, popularity, content, train-only SVD fold-in, LambdaMART.'),
    ('Final Project/src/step5_neural.py, predict_neural.py', 'Listwise neural rankers: MLP control, pooled, short/order single towers, DHAR, history-conditioned CF; saved-checkpoint inference.'),
    ('Final Project/src/step6_diagnostics.py', 'History segments, seen/unseen-item recall on full slates, rank by outcome type, paired bootstrap, matched feature ablation.'),
    ('Final Project/src/step7_coldstart_diversity.py', 'History-free fallback router and category MMR re-ranker, validation-selected; trade-off tables.'),
    ('Final Project/src/run_pipeline.py, make_results_summary.py, make_report_v2.py, make_listing_v2.py', 'One-command runner with run_status.json; report/listing generated from measured JSON so numbers cannot drift.'),
    ('Final Project/tests/test_pipeline.py', '11 regression tests for the audited failure modes.'),
    ('Final Project/notebooks/Ethan_JDSearch_Pipeline.ipynb', 'Colab walkthrough of the pipeline with saved measured results (Open-in-Colab link in README).'),
    ('Final Project/artifacts/corrected/', 'Measured outputs of the corrected run: results_*.json, ablation.json, paired_bootstrap.json, segment_metrics.csv, rank_by_type.csv, positive_subgroup_full_slate.csv, figures/, model/ (LightGBM model, test predictions, sessions).'),
    ('Final Project/artifacts/step7_draft_artifacts/', 'Measured outputs of stage 7 (cold-start segments, threshold selection, diversity trade-off, figure).'),
    ('Final Project/figures/', 'All report and slide figures.'),
    ('Final Project/report/', 'BT4222_Group13_Final_Report_v2.docx/.pdf and this listing. (Files without _v2 are the superseded 26 Oct draft.)'),
    ('Final Project/slides/', 'Final presentation (.pptx).'),
]


def main():
    d = Document(); st = d.styles['Normal']; st.font.name = 'Arial'; st.font.size = Pt(10.5)
    for s in d.sections:
        s.top_margin = s.bottom_margin = s.left_margin = s.right_margin = Inches(0.9)
    d.add_heading('BT4222 Group 13 — Dataset & Source-Code Listing', 1)
    p = d.add_paragraph(); p.add_run('GitHub repository: ').bold = True; p.add_run(f'{REPO}  (branch: {BRANCH})')
    p = d.add_paragraph(); p.add_run('Members: ').bold = True
    p.add_run('Chow Ee Xiang (A0300297J), Ethan John Ong Jian-Hao (A0308681Y), Heng Ann Xin (A0282911E), Ku Jia Yi (A0300567J), Lara Muy Steffany Mishell (A0350365H)')

    d.add_heading('1. Datasets', 2)
    d.add_paragraph('JDSearch (Liu et al., SIGIR 2023) — 173,831 users, 12.8M products, 26.7M interactions. '
                    'Source and download (JD Cloud): https://github.com/rucliujn/JDsearch · paper https://doi.org/10.1145/3539618.3591900. '
                    'Raw files are not redistributed (licence and size): user_behavior_data.txt (957 MB) and product_meta_data.txt (4.8 GB). '
                    'Set JD_USER_TXT / JD_PRODUCT_TXT or place them in Final Project/data/.')
    d.add_paragraph('Cleaned / intermediary artifacts. Regenerated deterministically by stages 2–3 (seed 42; split hash recorded in '
                    'artifacts/corrected/dataset_manifest.json): candidates.parquet, history.parquet, sessions.parquet, meta_subset.parquet, '
                    'features.parquet. These are git-ignored for size (~360 MB); a copy is shared at the Drive link below. '
                    'Lightweight measured outputs (predictions, metrics, model files) are committed under artifacts/corrected and '
                    'artifacts/step7_draft_artifacts.')
    p = d.add_paragraph(); p.add_run('Intermediate-artifact share link: ').bold = True
    p.add_run('<<INSERT Google Drive / OneDrive link before submission>>')

    d.add_heading('2. Repository structure', 2)
    t = d.add_table(rows=1, cols=2); t.style = 'Light Grid Accent 1'
    t.rows[0].cells[0].text = 'Path'; t.rows[0].cells[1].text = 'Contents'
    for path, desc in STRUCTURE:
        c = t.add_row().cells; c[0].text = path; c[1].text = desc
    for row in t.rows:
        row.cells[0].width = Inches(2.8); row.cells[1].width = Inches(4.0)
        for cell in row.cells:
            for para in cell.paragraphs:
                for r in para.runs: r.font.size = Pt(9)

    d.add_heading('3. How to reproduce', 2)
    d.add_paragraph('pip install -r "Final Project/requirements.txt"')
    d.add_paragraph('export JD_USER_TXT=/path/user_behavior_data.txt; export JD_PRODUCT_TXT=/path/product_meta_data.txt')
    d.add_paragraph('python "Final Project/src/run_pipeline.py" --users 20000        # stages 1,1b,2,3,4,5,6,7,report')
    d.add_paragraph('python -m unittest "Final Project/tests/test_pipeline.py"        # 11 regression tests')
    d.add_paragraph('Device is chosen CUDA → MPS → CPU. The 20k-user run completes on CPU; raise --users for the full data on a GPU.')

    OUT.mkdir(exist_ok=True); path = OUT / 'BT4222_Group13_Dataset_and_SourceCode_Listing_v2.docx'; d.save(path)
    if shutil.which('soffice'):
        subprocess.run(['soffice', '--headless', '--convert-to', 'pdf', '--outdir', str(OUT), str(path)],
                       check=False, capture_output=True, timeout=180)
    print('wrote', path)


if __name__ == '__main__':
    main()
