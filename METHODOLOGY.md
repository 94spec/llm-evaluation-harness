# Evaluation methodology

## 1. Define the evaluation unit before sampling

“Accuracy” is not a universal metric. Define the unit being compared:

- a criterion state in a QA card;
- a weighted deal score;
- a categorical summary slot;
- an annotated content unit in free text;
- a semantic label;
- a violation flag.

Reports, dashboards, and regression gates use the same unit and denominator.

## 2. Golden panels

Maintain four independently versioned golden panels, one per task family. Recommended size:
500 deals per family. Overlap is permitted, but task-specific eligibility rules remain explicit.

For each panel:

1. sample from a fixed eligibility window before seeing candidate-model outputs;
2. stratify by outcome, duration, call-chain length, transcript quality, and relevant edge cases;
3. oversample rare critical violations for diagnostic power;
4. store natural-prevalence weights if oversampling is used;
5. freeze a development split and a never-tuned-on test split;
6. version dataset, rubric, prompt, schema, and evaluator together.

Do not repeatedly tune against the frozen 500-deal test panel. A practical split is a rotating
development set plus a sequestered release set scored only at a release decision.

## 3. Annotation

### Annotators

- Two trained reviewers independently label every decision unit.
- A senior reviewer adjudicates disagreements without seeing the model answer.
- Annotators use the same rubric version as production.
- At least 10% of already-adjudicated records are relabeled blind to detect drift.

### Task-specific instructions

**QA scorecard:** label each criterion as `pass`, `fail`, `not_applicable`, or
`insufficient_data`. Record criterion weights separately from model output.

**Deal summary:** annotate structured categorical slots and atomic content units. Evaluate facts,
not writing style. Each fact should be independently verifiable from the source conversation.

**Semantic analytics:** use controlled taxonomies. New labels are proposed through a change
process rather than invented during annotation.

**Violations:** define an observable evidence threshold, severity, and treatment of uncertainty.
Critical violations receive a second mandatory review.

### Reliability

Before adjudication, report:

- raw agreement for interpretability;
- Cohen’s kappa when exactly two raters score the same nominal units;
- Krippendorff’s alpha (nominal) when more than two raters or missing ratings occur.

Agreement is a property of the annotation process, not model performance. Do not mix
human-vs-human reliability with model-vs-adjudicated-gold agreement.

## 4. Model evaluation

Validate output schema before semantic scoring. Invalid output remains in the denominator and is
not treated as a skipped success.

### QA scorecard

- criterion-state accuracy;
- macro-F1 across criterion states;
- Cohen’s kappa;
- model-reported total-score MAE;
- recomputed weighted-score MAE;
- model score vs recomputed score consistency.

The canonical score is recomputed by the evaluator from criterion labels and rubric weights.

### Deal summary

- schema validity and full normalized exact match;
- categorical outcome accuracy and kappa;
- exact match for controlled slots;
- content-unit precision, recall, and F1;
- omission and unsupported-fact counts;
- next-step presence and exact match.

Embedding similarity or an LLM judge may be added as secondary diagnostics, never as the only
truth signal.

### Semantic analytics

- micro/macro-F1 across controlled labels;
- exact-set match and mean Jaccard;
- per-dimension F1;
- readiness accuracy and kappa.

### Violations

- precision, recall, and F1 for positive flags;
- critical-positive recall;
- false-negative rate;
- exact flag-set match;
- state accuracy and kappa including `insufficient_data`.

A high overall accuracy can coexist with unacceptable rare-event recall. Critical recall is
therefore a separate gate.

## 5. Uncertainty

Report 95% percentile-bootstrap confidence intervals with at least 1,000 resamples. Resample
whole deals, not individual criterion decisions, because decisions within a deal are correlated.
If a deal contains several calls or task outputs, keep that cluster intact.

For model comparisons, prefer paired bootstrap differences using the same deal IDs. The current
harness reports per-run intervals; paired-difference support is required before treating a
small model-to-model delta as statistically meaningful.

## 6. Error analysis

Every mismatch is mapped to a stable code from [ERROR_TAXONOMY.md](ERROR_TAXONOMY.md). Review:

- counts and rates by error code;
- errors by stratum;
- top recurring criteria/labels;
- critical misses;
- new errors introduced relative to baseline.

Quality work is incomplete until examples are inspected and assigned an owner or accepted-risk
decision.

## 7. Release protocol

1. Pin dataset, rubric, prompt, schema, provider/model, inference parameters, and code revision.
2. Run the candidate once and retain raw outputs and telemetry.
3. Evaluate without changing gates.
4. Compare with the production baseline on identical record IDs.
5. Block release on any hard gate.
6. Review confidence intervals and error deltas.
7. Obtain business and quality-owner approval for accepted regressions.
8. Roll out gradually and monitor live drift separately from offline test quality.

Suggested gate classes:

- schema validity and missing-output rate;
- critical-violation recall;
- criterion macro-F1 / summary content recall / analytics micro-F1;
- weighted score MAE;
- p95 latency;
- estimated cost per 1,000 deals.

`configs/regression_gates.demo.json` contains an illustrative policy. Each deployment supplies
versioned thresholds approved for its task, risk profile, and operating constraints.

## 8. Monitoring after release

Offline agreement does not prove permanent correctness. Track:

- input and transcript distribution drift;
- label prevalence drift;
- schema failures and empty outputs;
- cost and latency;
- human appeal / override rate;
- weekly blind audit sample;
- a canary golden subset after provider or model changes.

Recalibrate after rubric changes. Never compare scores from different rubric versions as though
they were the same measurement.

