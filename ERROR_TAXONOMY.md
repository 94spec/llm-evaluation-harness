# Error taxonomy

Codes are stable analytical dimensions. Descriptions may evolve; identifiers should not be
renamed after reports depend on them.

| Code | Family | Meaning | Default severity |
|---|---|---|---|
| `MISSING_PREDICTION` | all | No prediction exists for a golden record | critical |
| `SCHEMA_INVALID` | all | Output violates the versioned task schema | high |
| `UNEXPECTED_PREDICTION` | all | Prediction has no matching golden record | medium |
| `CRITERION_FALSE_NEGATIVE` | QA | Expected `pass`, model did not pass | medium |
| `CRITERION_FALSE_POSITIVE` | QA | Model passed an unmet criterion | high |
| `CRITERION_STATE_MISMATCH` | QA | Other criterion-state mismatch | medium |
| `MODEL_SCORE_DRIFT` | QA | Model total differs materially from rubric-recomputed score | high |
| `OUTCOME_MISMATCH` | summary | Consultation outcome is wrong | high |
| `PRIMARY_NEED_MISMATCH` | summary | Controlled primary-need slot is wrong | medium |
| `SUMMARY_FACT_OMISSION` | summary | Gold content unit is absent | medium |
| `SUMMARY_UNSUPPORTED_FACT` | summary | Model adds a content unit absent from gold | high |
| `NEXT_STEP_MISMATCH` | summary | Presence or normalized value of next step is wrong | medium |
| `SEMANTIC_LABEL_MISSED` | analytics | Expected controlled label is absent | medium |
| `SEMANTIC_LABEL_SPURIOUS` | analytics | Unsupported controlled label is added | medium |
| `READINESS_MISMATCH` | analytics | Readiness state is wrong | medium |
| `CRITICAL_FLAG_MISSED` | violations | Positive critical violation is not detected | critical |
| `VIOLATION_FALSE_NEGATIVE` | violations | Other positive violation is missed | high |
| `VIOLATION_FALSE_POSITIVE` | violations | Violation is raised without gold support | high |
| `VIOLATION_STATE_MISMATCH` | violations | Other flag-state mismatch | medium |

Recommended triage dimensions stored beside the code:

- record and deal identifiers;
- criterion, field, or label;
- expected and predicted values;
- task family and schema version;
- dataset stratum;
- provider/model, prompt version, and run ID.

Evaluation fixtures should be synthetic or de-identified before they enter the repository.

