# AI Usage Log — BT4222 Final Project (Group 13)

This document discloses how generative-AI tools were used in this project, in line with the
BT4222 Generative-AI policy (AI as a thought partner, debugger and learning multiplier — **not** a
substitute for our own analytical thinking). It complements Section 4 (AI Interaction Log) of the
final report. Every AI suggestion below was critically evaluated, verified against the data, and
revised by us before adoption.

## Tools
- **Claude (Opus 4.8)** — conceptual sparring, code scaffolding, debugging, and review.

## How AI was used (by area)

### 1. Data understanding — the temporal retention finding
- **Our question / objective:** The instructor pointed out that carts/orders outnumber clicks and
  asked us to trace history back in time (seconds → elapsed days) and watch the click/cart/order mix.
- **AI contribution:** Helped confirm the `history_time_list` semantics (n+1 values; elapsed-before-query
  = suffix sum `sum(time[i+1:])`) and scaffolded the aggregation code.
- **Our critical evaluation:** AI could not know JD's retention policy and initially risked over-reading
  medians. We insisted on computing on the **full 173,831-user file** and at **1-day resolution**.
- **Final action (ours):** We ran it on all 26.7M events and found exact cut-offs (clicks ~7d, carts ~90d,
  orders ≥1yr) and interpreted them as a **log-retention policy**, not user behaviour.

### 2. Feature engineering & problem formulation
- **AI contribution:** Suggested separating short-term intent from long-term preference and recency decay.
- **Our evaluation/refinement:** We rejected naive type-intensity weighting (ORD>CART>CLICK) precisely
  because our finding shows type is confounded with recency; we designed horizon-split features instead.

### 3. Modelling
- **AI contribution:** Provided baseline templates (NCF/GMF+MLP, SVD, LambdaMART) and an initial
  dual-horizon attention sketch; helped make the training loop MPS-efficient (padded-slate batching).
- **Our evaluation/refinement:** The first neural loop was too slow and opaque — we rewrote it as a
  vectorised, padded-slate listwise trainer. We chose listwise softmax with ordinal gains [0,1,3,7].

### 4. Evaluation integrity (critical catch)
- **AI contribution:** When our content-only baseline scored an impossible NDCG 0.965, AI flagged likely
  **position/tie-break leakage** (the first candidate is positive in 98.8% of slates).
- **Our action:** We added fixed-seed random tie-breaking in `eval_utils.py`; the honest score dropped to
  0.087, and we applied the guard uniformly to all models.

### 5. Verification & self-grading (feedback round)
- We used AI reviewers to adversarially audit the project (rubric compliance, leakage, instructor-feedback
  coverage, reproducibility). Their findings led us to fix a **transductive normalisation leak** (now
  TRAIN-only statistics), correct a mislabelled split comment, add **train-vs-test** metrics, add an
  architecture diagram, and commit the repo with `.gitignore` for large data.

## What AI did NOT do
- It did not define our business problem, choose our dataset, or interpret the results for us.
- All numbers in the report were produced by our own code runs and cross-checked against saved artifacts.
- No AI-generated text was copied into the report without our review and editing.
