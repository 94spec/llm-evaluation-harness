"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .compare import compare_reports
from .evaluate import evaluate_run
from .gates import evaluate_gates
from .io import DataError, load_json, load_jsonl, write_json
from .report import render_markdown
from .synthetic import write_bundle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ci-eval",
        description="Evaluate structured conversation-intelligence LLM outputs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate deterministic synthetic panels")
    generate.add_argument("--out", required=True, help="output directory")
    generate.add_argument("--deals-per-family", type=int, default=500)
    generate.add_argument("--seed", type=int, default=20260728)

    evaluate = subparsers.add_parser("evaluate", help="evaluate one immutable prediction run")
    evaluate.add_argument("--gold", required=True)
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--json-out", required=True)
    evaluate.add_argument("--markdown-out")
    evaluate.add_argument("--bootstrap", type=int, default=1000)
    evaluate.add_argument("--seed", type=int, default=20260728)
    evaluate.add_argument("--max-error-examples", type=int, default=100)

    compare = subparsers.add_parser("compare", help="compare reports on the same dataset")
    compare.add_argument("--reports", nargs="+", required=True)
    compare.add_argument("--out", required=True)

    gate = subparsers.add_parser("gate", help="enforce absolute and baseline-relative rules")
    gate.add_argument("--candidate", required=True)
    gate.add_argument("--baseline")
    gate.add_argument("--config", required=True)
    gate.add_argument("--out", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "generate":
            write_bundle(
                args.out,
                deals_per_family=args.deals_per_family,
                seed=args.seed,
            )
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "out": str(Path(args.out).resolve()),
                        "deals_per_family": args.deals_per_family,
                        "synthetic": True,
                    },
                    ensure_ascii=False,
                )
            )
            return 0

        if args.command == "evaluate":
            report = evaluate_run(
                load_jsonl(args.gold),
                load_jsonl(args.predictions),
                load_json(args.manifest),
                bootstrap_resamples=args.bootstrap,
                seed=args.seed,
                max_error_examples=args.max_error_examples,
            )
            write_json(args.json_out, report)
            if args.markdown_out:
                target = Path(args.markdown_out)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(render_markdown(report), encoding="utf-8")
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "report": str(Path(args.json_out).resolve()),
                        "schema_validity": report["overall"]["schema_validity"],
                        "synthetic": report["dataset"]["synthetic"],
                    },
                    ensure_ascii=False,
                )
            )
            return 0

        if args.command == "compare":
            comparison = compare_reports([load_json(path) for path in args.reports])
            write_json(args.out, comparison)
            print(json.dumps({"status": "ok", "comparable": comparison["comparable"]}))
            return 0 if comparison["comparable"] else 2

        if args.command == "gate":
            candidate = load_json(args.candidate)
            baseline = load_json(args.baseline) if args.baseline else None
            result = evaluate_gates(candidate, baseline, load_json(args.config))
            write_json(args.out, result)
            print(
                json.dumps(
                    {
                        "status": "passed" if result["passed"] else "failed",
                        "blocking_failures": result["blocking_failures"],
                        "warnings": result["warnings"],
                    }
                )
            )
            return 0 if result["passed"] else 2
    except (DataError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 1

