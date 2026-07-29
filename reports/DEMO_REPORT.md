# Synthetic smoke-test evaluation

> **Synthetic fixture run generated from bundled test data**

This report validates evaluator behavior against known, deliberately injected errors. It uses
four handwritten synthetic records—one per task family. The smoke-test sample exercises the
reporting path without external data or model calls.

## Run identity

| Field | Value |
|---|---|
| Dataset | `synthetic-smoke-v1` |
| Run | `smoke-candidate-demo` |
| Provider | `synthetic-provider` |
| Model | `synthetic-model-b` |
| API calls | `0` |
| Pricing | synthetic snapshot |

## Coverage and operations

| Metric | Fixture value |
|---|---:|
| Golden records | 4 |
| Task families | 4 |
| Schema validity | 100% |
| Latency p50 | 600 ms |
| Latency p95 | 669.5 ms |
| Input tokens | 9,700 |
| Output tokens | 1,110 |
| Estimated cost / 1,000 records | $0.707 using synthetic prices |

## Selected quality checks

| Task family | Metric | Fixture value |
|---|---|---:|
| QA scorecard | criterion accuracy | 0.6667 |
| QA scorecard | weighted-score MAE | 42.8571 |
| Deal summary | content-unit precision / recall | 0.5000 / 0.5000 |
| Semantic analytics | label micro-F1 | 0.8571 |
| Violations | critical recall | 0.0000 |

The poor values are expected: the candidate fixture deliberately marks a failed criterion as
passed, omits and invents summary facts, misses a semantic label, changes readiness, and misses a
critical violation.

## Detected error ledger

| Error code | Count |
|---|---:|
| `CRITERION_FALSE_POSITIVE` | 1 |
| `CRITICAL_FLAG_MISSED` | 1 |
| `NEXT_STEP_MISMATCH` | 1 |
| `READINESS_MISMATCH` | 1 |
| `SEMANTIC_LABEL_MISSED` | 1 |
| `SUMMARY_FACT_OMISSION` | 1 |
| `SUMMARY_UNSUPPORTED_FACT` | 1 |

The regression gate correctly **fails** because critical recall and several
baseline-relative quality rules regress.

## Reproduce

```bash
PYTHONPATH=src python -m conv_eval evaluate \
  --gold fixtures/golden.demo.jsonl \
  --predictions fixtures/predictions.candidate.demo.jsonl \
  --manifest fixtures/run.candidate.demo.json \
  --json-out reports/generated/candidate.json \
  --markdown-out reports/generated/candidate.md \
  --bootstrap 100
```

