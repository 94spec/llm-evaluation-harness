"""Deterministic synthetic panels and model-like outputs; no API calls."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

from .evaluate import recompute_weighted_score
from .io import write_json, write_jsonl
from .schema import CRITERION_STATES, FLAG_STATES

_QA_CRITERIA = [f"criterion_{index:02d}" for index in range(1, 13)]
_QA_WEIGHTS = {
    criterion: weight
    for criterion, weight in zip(
        _QA_CRITERIA,
        (8, 8, 8, 8, 9, 9, 8, 8, 9, 9, 8, 8),
    )
}
_NEEDS = [
    "career_change",
    "income_growth",
    "career_growth",
    "move_to_it",
    "skill_upgrade",
]
_OBJECTIONS = ["price", "time", "trust", "needs_comparison", "not_ready"]
_TARIFFS = ["base", "advanced", "individual"]
_COMPETITORS = ["alternative_course", "self_study", "university"]
_FLAGS = [
    "pressure",
    "disrespect",
    "policy_violation",
    "missing_consent",
    "unsupported_guarantee",
]
_CRITICAL_FLAGS = ["policy_violation", "unsupported_guarantee"]


def _sample_list(rng: random.Random, population: list[str], maximum: int) -> list[str]:
    count = rng.randint(0, min(maximum, len(population)))
    return sorted(rng.sample(population, count))


def _telemetry(rng: random.Random) -> dict[str, int | float]:
    return {
        "latency_ms": round(rng.uniform(350, 1800), 3),
        "input_tokens": rng.randint(1400, 5200),
        "output_tokens": rng.randint(100, 850),
    }


def _qa_reference(rng: random.Random) -> dict[str, Any]:
    criteria = {
        criterion: rng.choices(
            ["pass", "fail", "not_applicable", "insufficient_data"],
            weights=[62, 25, 8, 5],
            k=1,
        )[0]
        for criterion in _QA_CRITERIA
    }
    score = recompute_weighted_score(criteria, _QA_WEIGHTS)
    return {
        "criteria": criteria,
        "weights": _QA_WEIGHTS,
        "total_score": round(score, 4),
    }


def _summary_reference(rng: random.Random, index: int) -> dict[str, Any]:
    outcome = rng.choice(["won", "lost", "follow_up", "undecided"])
    need = rng.choice(_NEEDS)
    objections = _sample_list(rng, _OBJECTIONS, 2)
    next_step = (
        None
        if outcome in {"won", "lost"} and rng.random() < 0.6
        else f"synthetic_next_step_{index % 7}"
    )
    facts = [
        f"need:{need}",
        f"outcome:{outcome}",
        *[f"objection:{item}" for item in objections],
    ]
    return {
        "outcome": outcome,
        "primary_need": need,
        "objections": objections,
        "next_step": next_step,
        "facts": facts,
    }


def _semantic_reference(rng: random.Random) -> dict[str, Any]:
    return {
        "needs": _sample_list(rng, _NEEDS, 2),
        "objections": _sample_list(rng, _OBJECTIONS, 2),
        "readiness": rng.choice(["cold", "warm", "hot", "insufficient_data"]),
        "tariff_interest": _sample_list(rng, _TARIFFS, 2),
        "competitor_mentions": _sample_list(rng, _COMPETITORS, 1),
    }


def _violation_reference(rng: random.Random) -> dict[str, Any]:
    flags = {
        flag: rng.choices(
            ["yes", "no", "insufficient_data"],
            weights=[12 if flag in _CRITICAL_FLAGS else 18, 76, 12],
            k=1,
        )[0]
        for flag in _FLAGS
    }
    return {"flags": flags, "critical_flags": list(_CRITICAL_FLAGS)}


def _perturb_qa(
    reference: dict[str, Any],
    rng: random.Random,
    error_rate: float,
) -> dict[str, Any]:
    criteria = copy.deepcopy(reference["criteria"])
    for criterion, state in list(criteria.items()):
        if rng.random() < error_rate:
            criteria[criterion] = rng.choice(sorted(CRITERION_STATES - {state}))
    recomputed = recompute_weighted_score(criteria, reference["weights"])
    reported = max(0.0, min(100.0, recomputed + rng.gauss(0, error_rate * 8)))
    return {"criteria": criteria, "total_score": round(reported, 4)}


def _perturb_summary(
    reference: dict[str, Any],
    rng: random.Random,
    error_rate: float,
) -> dict[str, Any]:
    output = copy.deepcopy(reference)
    if rng.random() < error_rate:
        output["outcome"] = rng.choice(
            [item for item in ("won", "lost", "follow_up", "undecided") if item != output["outcome"]]
        )
    if rng.random() < error_rate:
        output["primary_need"] = rng.choice(
            [item for item in _NEEDS if item != output["primary_need"]]
        )
    if output["objections"] and rng.random() < error_rate:
        output["objections"].pop()
    elif rng.random() < error_rate:
        additions = [item for item in _OBJECTIONS if item not in output["objections"]]
        if additions:
            output["objections"].append(rng.choice(additions))
            output["objections"].sort()
    if output["facts"] and rng.random() < error_rate:
        output["facts"].pop()
    if rng.random() < error_rate:
        output["facts"].append(f"unsupported_synthetic_fact_{rng.randint(1, 9)}")
    if rng.random() < error_rate:
        output["next_step"] = (
            None if output["next_step"] is not None else "synthetic_follow_up"
        )
    return output


def _perturb_semantic(
    reference: dict[str, Any],
    rng: random.Random,
    error_rate: float,
) -> dict[str, Any]:
    output = copy.deepcopy(reference)
    for dimension, population in (
        ("needs", _NEEDS),
        ("objections", _OBJECTIONS),
        ("tariff_interest", _TARIFFS),
        ("competitor_mentions", _COMPETITORS),
    ):
        if output[dimension] and rng.random() < error_rate:
            output[dimension].pop()
        elif rng.random() < error_rate:
            additions = [item for item in population if item not in output[dimension]]
            if additions:
                output[dimension].append(rng.choice(additions))
                output[dimension].sort()
    if rng.random() < error_rate:
        output["readiness"] = rng.choice(
            [
                item
                for item in ("cold", "warm", "hot", "insufficient_data")
                if item != output["readiness"]
            ]
        )
    return output


def _perturb_violations(
    reference: dict[str, Any],
    rng: random.Random,
    error_rate: float,
) -> dict[str, Any]:
    flags = copy.deepcopy(reference["flags"])
    for flag, state in list(flags.items()):
        if rng.random() < error_rate:
            flags[flag] = rng.choice(sorted(FLAG_STATES - {state}))
    return {"flags": flags}


def _annotator_copy(
    family: str,
    reference: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    output = copy.deepcopy(reference)
    if rng.random() >= 0.05:
        return output
    if family == "qa_scorecard":
        criterion = rng.choice(_QA_CRITERIA)
        current = output["criteria"][criterion]
        output["criteria"][criterion] = rng.choice(sorted(CRITERION_STATES - {current}))
        output["total_score"] = round(
            recompute_weighted_score(output["criteria"], output["weights"]),
            4,
        )
    elif family == "deal_summary":
        output["outcome"] = rng.choice(
            [
                item
                for item in ("won", "lost", "follow_up", "undecided")
                if item != output["outcome"]
            ]
        )
    elif family == "semantic_analytics":
        output["readiness"] = rng.choice(
            [
                item
                for item in ("cold", "warm", "hot", "insufficient_data")
                if item != output["readiness"]
            ]
        )
    else:
        flag = rng.choice(_FLAGS)
        current = output["flags"][flag]
        output["flags"][flag] = rng.choice(sorted(FLAG_STATES - {current}))
    return output


def _prediction(
    gold: dict[str, Any],
    rng: random.Random,
    error_rate: float,
) -> dict[str, Any]:
    family = gold["task_family"]
    reference = gold["reference"]
    if family == "qa_scorecard":
        output = _perturb_qa(reference, rng, error_rate)
    elif family == "deal_summary":
        output = _perturb_summary(reference, rng, error_rate)
    elif family == "semantic_analytics":
        output = _perturb_semantic(reference, rng, error_rate)
    else:
        output = _perturb_violations(reference, rng, error_rate)
    return {
        "record_id": gold["record_id"],
        "task_family": family,
        "output": output,
        "telemetry": _telemetry(rng),
    }


def _manifest(
    *,
    run_id: str,
    model: str,
    error_profile: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "dataset_id": "synthetic-four-panels-v1",
        "provider": "synthetic-provider",
        "model": model,
        "prompt_version": "synthetic-prompt-v1",
        "output_schema_version": "1.0",
        "code_revision": "synthetic-fixture-v1",
        "created_at": "2026-07-28T00:00:00Z",
        "generation": {
            "temperature": 0,
            "seed": seed,
            "error_profile": error_profile,
            "api_calls": 0,
        },
        "pricing": {
            "currency": "USD",
            "input_per_1m_tokens": 0.25,
            "output_per_1m_tokens": 1.0,
            "effective_at": "2026-07-28",
            "source": "FICTIONAL DEMO PRICE — NOT A VENDOR QUOTE",
        },
        "is_demo": True,
    }


def generate_bundle(
    *,
    deals_per_family: int,
    seed: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    if deals_per_family <= 0:
        raise ValueError("deals_per_family must be positive")
    rng = random.Random(seed)
    reference_factories = {
        "qa_scorecard": lambda index: _qa_reference(rng),
        "deal_summary": lambda index: _summary_reference(rng, index),
        "semantic_analytics": lambda index: _semantic_reference(rng),
        "violation_flags": lambda index: _violation_reference(rng),
    }
    gold: list[dict[str, Any]] = []
    for family, factory in reference_factories.items():
        for index in range(1, deals_per_family + 1):
            reference = factory(index)
            record = {
                "record_id": f"{family}-{index:04d}",
                "deal_id": f"synthetic-deal-{family}-{index:04d}",
                "task_family": family,
                "schema_version": "1.0",
                "reference": reference,
                "metadata": {
                    "synthetic": True,
                    "stratum": rng.choice(
                        ["single_call", "multi_call", "noisy_transcript", "edge_case"]
                    ),
                    "generator_seed": seed,
                },
                "annotations": [
                    {"annotator_id": "synthetic-rater-a", "output": copy.deepcopy(reference)},
                    {
                        "annotator_id": "synthetic-rater-b",
                        "output": _annotator_copy(family, reference, rng),
                    },
                ],
            }
            gold.append(record)

    baseline_rng = random.Random(seed + 1)
    candidate_rng = random.Random(seed + 2)
    baseline = [_prediction(record, baseline_rng, 0.12) for record in gold]
    candidate = [_prediction(record, candidate_rng, 0.06) for record in gold]
    baseline_manifest = _manifest(
        run_id="synthetic-baseline",
        model="synthetic-model-a",
        error_profile="deterministic-demo-12pct",
        seed=seed + 1,
    )
    candidate_manifest = _manifest(
        run_id="synthetic-candidate",
        model="synthetic-model-b",
        error_profile="deterministic-demo-6pct",
        seed=seed + 2,
    )
    return gold, baseline, candidate, baseline_manifest, candidate_manifest


def write_bundle(out_dir: str | Path, *, deals_per_family: int, seed: int) -> None:
    target = Path(out_dir)
    gold, baseline, candidate, baseline_manifest, candidate_manifest = generate_bundle(
        deals_per_family=deals_per_family,
        seed=seed,
    )
    write_jsonl(target / "golden.jsonl", gold)
    write_jsonl(target / "predictions.baseline.jsonl", baseline)
    write_jsonl(target / "predictions.candidate.jsonl", candidate)
    write_json(target / "run.baseline.json", baseline_manifest)
    write_json(target / "run.candidate.json", candidate_manifest)
