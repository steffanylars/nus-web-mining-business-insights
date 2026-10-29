# AI assistance record for the ethan branch

This records the Codex-assisted audit and correction work performed on
29 October 2026. It does not certify the team's prior AI interactions.
The previous draft log is preserved in Git at commit
3d6b62b90fbdee102331f2aaf099be0ec52cb5e8.

## User requests and source of the temporal question

Ethan asked Codex to inspect the final-report pipeline against the instructor's
interim feedback, then implement the corrections and push them to a separate
branch named ethan with a teammate-facing change log. The instruction to trace
historical action distributions backward in time came from the instructor's
feedback supplied by Ethan, not from AI.

## Work performed with Codex

- Read the source, report narrative, saved results and official JDSearch field definitions.
- Independently reconstructed raw history ages and checked the original saved predictions.
- Identified the actual-delimiter bug, missing category-name input, early global history cap,
  unsupported held-out user-ID baseline and inconsistent diagnostic populations.
- Proposed and implemented code corrections, regression tests, training-only SVD fold-in,
  a history-conditioned CF adaptation, tower ablations and deterministic evaluation.
- Ran the corrected 20,000-user pipeline on local raw data and generated measured tables.
- Drafted ETHAN_CHANGES.md, reproduction instructions and the corrected Markdown report.
- Added saved-checkpoint inference and metric/reload verification.

## Findings that changed the draft interpretation

The corrected token overlap is nonzero; anonymization cannot explain the original
broken content feature. Complete history features improve the tree result, but
the matched neural comparison does not show a dual-attention advantage over the
same-feature MLP. The report therefore avoids claiming that the architecture's
superiority or a specific JD storage policy has been established.

## Human review status and limitations

The code, experiment choices and generated narrative are proposals for the team's
review. This conversation does not establish that every team member reviewed them.
No claim is made that the new report text was independently authored or verified
by the whole team. No source attribution, team contribution assignment or external
rubric compliance statement is invented.

The history-conditioned CF baseline is an adaptation, and the corrected run uses
one seed and a previously inspected test partition. Recorded numeric checks verify
the saved run; they do not prove causal business effects or generalization to an
untouched population. The team should describe its own subsequent checks and
decisions before submission.
