# -*- coding: utf-8 -*-
"""Dataset & Source-Code Listing (single PDF for Canvas)."""
from pathlib import Path
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
REP=Path("/Users/steff/Documents/GitHub/bt4212_finalProj/Final Project/report")
doc=Document(); s=doc.styles["Normal"]; s.font.name="Arial"; s.font.size=Pt(11); s.paragraph_format.line_spacing=1.3
def H(t,l=1):
    p=doc.add_heading(t,l)
    for r in p.runs: r.font.color.rgb=RGBColor(0,0,0)
def P(t,b=False):
    p=doc.add_paragraph(); r=p.add_run(t); r.bold=b
def B(lead,t):
    p=doc.add_paragraph(style="List Bullet"); r=p.add_run(lead); r.bold=True; p.add_run(t)

P("BT4222 Group 13 — Dataset & Source-Code Listing",True); doc.paragraphs[-1].alignment=WD_ALIGN_PARAGRAPH.CENTER
P("GitHub repository: https://github.com/steffanylars/BT4222_finalProj"); doc.paragraphs[-1].alignment=WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph()

H("1. Dataset",1)
B("JDSearch (primary). ","Liu et al., SIGIR 2023. 173,831 users, 12.8M products, 26.7M "
  "interactions. Source & download (JD Cloud): https://github.com/rucliujn/JDsearch · "
  "paper https://doi.org/10.1145/3539618.3591900")
B("Files. ","user_behavior_data.txt (919 MB; query, candidate_wid_list, candidate_label_list, "
  "history_qry/wid/type/time lists) and product_meta_data.txt (4.5 GB; wid, name, brand, 4 category "
  "levels, shop_id). Pre-processed to Parquet chunks under 'Interim Report/data/JDsearch/product_chunks/'.")
B("Intermediate artifacts. ","Final Project/artifacts/model/ (candidates, history, sessions, "
  "features, meta_subset .parquet) — git-ignored due to size; regenerate with src/step2 & step3.")

H("2. Repository structure",1)
B("Interim Report/ ","— interim deliverable, EDA notebooks, product_chunks, figures.")
B("Final Project/src/ ","— modular pipeline step1–step6 (.py), make_report/make_slides/make_listing.")
B("Final Project/report/ ","— BT4222_Group13_Final_Report.pdf (+ .docx) and this listing.")
B("Final Project/slides/ ","— BT4222_Group13_Final_Presentation.pptx.")
B("Final Project/figures/ ","— all figures (temporal, retention, model comparison, importance, segments).")
B("Final Project/artifacts/ ","— result tables (JSON/CSV: temporal_summary, retention_cutoffs, "
  "results_*, segment_metrics, rank_by_type, ablation, lgb_feature_importance) + trained models "
  "(lgb_model.txt, dhar.pt).")
B("Final Project/README.md & RESULTS.md ","— run order, scaling knob, and preserved step outputs.")

H("3. Source code (run in order)",1)
for n,d in [("step1_preprocess_temporal.py","full-data temporal retention analysis (clicks 7d / carts 90d / orders 1yr)"),
            ("step1b_retention_cutoffs.py","exact 1-day-resolution cut-offs"),
            ("step2_build_dataset.py [N_USERS]","build sampled modelling dataset + metadata join"),
            ("step3_features.py","horizon-aware feature engineering"),
            ("step4_baselines_lgbm.py","Random/Popularity/Content/MF + LambdaMART, position-leak-safe eval"),
            ("step5_neural.py","Dual-Horizon Attention Ranker (ours) + NCF baseline (PyTorch/MPS)"),
            ("step6_diagnostics.py","segment analysis, rank-by-type, temporal-feature ablation")]:
    B(n+" ","— "+d)

out=REP/"BT4222_Group13_Dataset_and_SourceCode_Listing.docx"; doc.save(str(out)); print("saved",out)
