# BT4222 Group 13 JDSearch project

Personalised reranking of provided JDSearch candidate slates using query terms,
product metadata and preceding user history.

On branch ethan, start with [Ethan's change log](Final%20Project/ETHAN_CHANGES.md).
For the team's Colab notebook exchange, use the
[review notebook](Final%20Project/notebooks/Ethan_JDSearch_Pipeline.ipynb)
or [open it in Colab](https://colab.research.google.com/github/steffanylars/nus-web-mining-business-insights/blob/ethan/Final%20Project/notebooks/Ethan_JDSearch_Pipeline.ipynb).
The [pipeline README](Final%20Project/README.md) contains reproduction instructions.
Measured corrected results and the report narrative are in
[artifacts/corrected](Final%20Project/artifacts/corrected).

The corrected run preserves the original 20,000-user split and repairs term
tokenization, history retention, CF/SVD protocols, evaluation and ablations.
The temporal pattern supports horizon-aware features; the matched experiment
does not establish a benefit for dual attention beyond the same-feature MLP.

Submission documents with the corrected numbers, generated from the measured artifacts:
[Final Report v2](Final%20Project/report/BT4222_Group13_Final_Report_v2.pdf) and
[Dataset & Source-Code Listing v2](Final%20Project/report/BT4222_Group13_Dataset_and_SourceCode_Listing_v2.pdf)
(`src/make_report_v2.py`, `src/make_listing_v2.py`). Stage 7 (cold-start routing and
category diversification, both measured and declined) is documented in
[STEP7_COLDSTART_DIVERSITY.md](Final%20Project/STEP7_COLDSTART_DIVERSITY.md).

Files in report/ and slides/ without `_v2`, and root artifacts/ inside Final Project, are the
26 Oct draft, retained for comparison. They are not the corrected submission. Still open for
the team: slides regenerated from the v2 numbers, a fresh Colab run with saved outputs, the
intermediate-artifact share link in the listing, and the final branch name in the listing.

Raw data are not redistributed. Obtain them from the
[official JDSearch repository](https://github.com/rucliujn/JDsearch).
