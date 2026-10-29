# BT4222 Group 13 JDSearch project

Personalised reranking of provided JDSearch candidate slates using query terms,
product metadata and preceding user history.

On branch ethan, start with [Ethan's change log](Final%20Project/ETHAN_CHANGES.md).
The [pipeline README](Final%20Project/README.md) contains reproduction instructions.
Measured corrected results and the report narrative are in
[artifacts/corrected](Final%20Project/artifacts/corrected).

The corrected run preserves the original 20,000-user split and repairs term
tokenization, history retention, CF/SVD protocols, evaluation and ablations.
The temporal pattern supports horizon-aware features; the matched experiment
does not establish a benefit for dual attention beyond the same-feature MLP.

Original report/, slides/ and root artifacts/ inside Final Project are historical
draft snapshots, retained for comparison. They are not the corrected submission.
The team still needs to agree the report narrative, contribution statements
and formatting against the course guideline.

Raw data are not redistributed. Obtain them from the
[official JDSearch repository](https://github.com/rucliujn/JDsearch).
