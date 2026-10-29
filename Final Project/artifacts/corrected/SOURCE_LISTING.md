# Dataset and source code inventory

Raw data: JDSearch user_behavior_data.txt and product_meta_data.txt.
Obtain from https://github.com/rucliujn/JDsearch. Raw files are not redistributed.
Set JD_USER_TXT and JD_PRODUCT_TXT; raw metadata is streamed directly.
Corrected outputs: artifacts/corrected. See README.md for reproduction.

| Source | Bytes |
|---|---:|
| src/config.py | 1,198 |
| src/data_utils.py | 1,317 |
| src/eval_utils.py | 3,715 |
| src/make_listing.py | 815 |
| src/make_report.py | 219 |
| src/make_results_summary.py | 12,795 |
| src/make_slides.py | 3,456 |
| src/predict_neural.py | 1,597 |
| src/run_pipeline.py | 2,037 |
| src/step1_preprocess_temporal.py | 8,358 |
| src/step1b_retention_cutoffs.py | 2,123 |
| src/step2_build_dataset.py | 4,565 |
| src/step3_features.py | 8,086 |
| src/step4_baselines_lgbm.py | 5,273 |
| src/step5_neural.py | 13,316 |
| src/step6_diagnostics.py | 6,250 |
