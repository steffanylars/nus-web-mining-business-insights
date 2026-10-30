# Improvements to consider from the A+ project reference

Discussion note for the team, 30 October 2026. These are proposed next steps, not
additional experiments or report changes that have already been completed.

## Purpose and source

Ethan supplied a previous semester's Group 14 submission, reported to have received
an A+, in Project.zip. The reference project concerns contextual hate-speech
classification, while ours concerns personalised product search. We should borrow
its clarity of explanation and analytical depth, adapting the methods to ranking.
Its grade does not establish which individual choices earned marks, and its report
does not replace this semester's guideline or our instructor's feedback.

Page references below refer to Group 14 Final Report.pdf. The reference archive is
not redistributed in this branch. These notes are based on the report and selected
supporting materials, not an independent reproduction of their entire pipeline.

## A useful lesson for our results

Their proposed hybrid model did not outperform its baseline. Table 4 on page 19
reports F1 of 90.23% for baseline RoBERTa and 89.50% for RoBERTa with metadata.
They acknowledge the result and discuss possible explanations.

Our corrected DHAR similarly does not outperform the same-feature MLP in this run:
NDCG@10 is 0.3713 versus 0.3725. The paired query bootstrap interval for the difference
includes zero. We should explain this result rather than force a claim that a more
complex architecture must be better. Possible explanations remain hypotheses until
tested. See [measured results](artifacts/corrected/RESULTS.md).

## Improvements mapped to our project

| Priority | Reference strength | Equivalent for JDSearch | Current status and next action |
|---|---|---|---|
| High | Overall workflow diagram, p. 4 | Show raw query/candidate/history inputs, suffix-sum ages, user split, training-only priors, features, models and ranked output | Pipeline is implemented. Add a readable diagram to the agreed final report and slides; distinguish inputs from outcome labels. |
| High | Feature groups linked to their purpose, p. 5 | Group features into query relevance, popularity, recent intent, order preference, re-engagement and activity | Group definitions exist in code and narrative. Add a compact dictionary with calculation, time window, missing-value treatment and training/inference availability. |
| High | Component-by-component architecture explanation, pp. 8–12 | Explain candidate embeddings, short/order sequences, attention, numeric features and the ranking loss | Code and matched ablations exist. Add a diagram with tensor dimensions and an explanation of what each control removes. |
| High | Direct baseline comparison and discussion of the unsuccessful hybrid, pp. 19–20 | Present all corrected models and the MLP/pooled/single-horizon controls honestly | Results are measured. Use them to revise the report's contribution claims; do not choose the primary model by test score alone. |
| High | Error analysis beyond one headline metric, pp. 21–25 | Examine failures by history length, item exposure, query/slate characteristics and outcome type | Full-slate history/item diagnostics are implemented. Add subgroup counts, selected charts and representative ranking examples. Query/slate slices would be additional analysis. |
| Medium | Explicit model settings and validation comparisons, pp. 17–18 | State sample size, split, gains, history sampling, optimizer, early stopping and model-selection rule | Settings and logs are saved. Summarise them in one table. Consider a small validation-only sensitivity study if time permits. |
| Medium | Business discussion grounded in diagnostics, pp. 24–26 | Explain what weak cold-item ranking or limited-history performance means for search | Link each recommendation to a measured result. Distinguish proposed product actions from proven conversion or revenue improvements. |
| Medium | Traceability between workflow and source notebooks | Map report claims to scripts, saved artifacts and reproduction commands | README, source listing and verification are present. Add a short claim-to-evidence index for the final report. |

## Concrete additions worth prioritising

- [ ] One overall pipeline diagram, including the split and the boundary between
  preceding history inputs and current outcome labels.
- [ ] One feature dictionary with precise definitions. In particular, describe
  h_min_elapsed as the user's latest event age, not candidate-specific recency.
- [ ] One architecture diagram showing the short and order channels and their
  independent sequence sampling before fusion with candidate/numeric features.
- [ ] One compact model and ablation table, sourced from corrected JSON outputs.
- [ ] A small, transparently selected set of ranking examples: successful ranking,
  a missed positive item absent from training histories, and a case where added
  towers do not improve the ranking. Include the selection rule to avoid cherry-picking.
- [ ] Subgroup charts that retain the original candidate slate and show sample
  sizes, with descriptive interpretation rather than causal explanations.
- [ ] A clear contribution statement separating the temporal finding, pipeline
  corrections, engineered features, architecture experiment and diagnostics.
- [ ] Final layout review after the team agrees the narrative and checks the
  current deliverable guideline.

For ranking examples, product/query terms are anonymised. We can discuss shared
term IDs, category matches, history ages, model scores and observed outcomes; we
cannot invent product meanings or shopper intent from the anonymised strings.

## Optional experiments after the presentation essentials

If resources permit, test sensitivity to history sequence length, window choices
or training seed using a predeclared validation procedure. A fresh final holdout
would strengthen claims because the existing test partition has already been
inspected. These are proposals, not completed checks or requirements inferred
from the reference's grade.

Measure training/inference time and model size if arguing that a simpler model
is preferable. Our current accuracy comparison alone does not establish a resource
advantage. Additional experiments should answer a specific question, rather than
increase model count without improving the explanation.

## What not to transfer directly

- Classification accuracy, F1, confusion matrices and ROC curves do not replace
  query-level ranking metrics. A ranker's score is not automatically a calibrated
  interaction probability.
- More datasets or a pretrained language model are not prerequisites for a strong
  project. Any addition needs a reason grounded in our task and available fields.
- Do not copy the previous report's wording, unverified explanations, claimed
  calibration or self-assessed contribution levels.
- Do not infer that novelty guarantees improvement, or that a negative architecture
  result makes the whole project unsuccessful.
- Do not treat a reported A+ as evidence that every implementation detail or claim
  in that submission is correct.

## Suggested report narrative

Start with the search task and what each data field represents. Explain the
temporal pattern identified through the instructor's feedback, then show how it
changes history construction and features. Present fair baseline comparisons,
test the incremental value of the proposed architecture, and finish with failure
analysis and bounded business implications. The contribution should be supported
by the evidence even where the more complex model does not win.

Related files: [change log](ETHAN_CHANGES.md), [corrected report draft](artifacts/corrected/REPORT.md),
[results](artifacts/corrected/RESULTS.md) and [reproduction instructions](README.md).
