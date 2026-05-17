from __future__ import annotations

import argparse
import math
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any

from discover_pseudo_labels_llm_structured import (
    RAW_PATH,
    ROOT,
    build_prompt_record,
    coerce_extraction,
    llm_extract_one,
    load_env_file,
    load_joined,
    summarize_named_entities,
)
from openai import OpenAI


def simulate_total_seconds(
    n_rows: int,
    seconds_per_call: float,
    retry_error_rate: float,
    max_retries: int,
    n_trials: int,
    seed: int,
) -> list[float]:
    rng = random.Random(seed)
    totals: list[float] = []
    for _ in range(n_trials):
        total = 0.0
        for _row in range(n_rows):
            for attempt in range(max_retries):
                total += seconds_per_call
                if rng.random() >= retry_error_rate:
                    break
                if attempt < max_retries - 1:
                    total += 1.25 * (attempt + 1)  # backoff in production script
        totals.append(total)
    return totals


def pretty_minutes(seconds: float) -> str:
    return f"{seconds / 60.0:.1f} min ({seconds / 3600.0:.2f} hr)"


def run_live_smoke(sample_size: int, model: str) -> tuple[list[float], list[dict[str, Any]]]:
    load_env_file(ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing; cannot run live smoke test")

    client = OpenAI(api_key=api_key)
    df = load_joined(RAW_PATH).head(sample_size)

    times: list[float] = []
    examples: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rec = build_prompt_record(row)
        t0 = time.perf_counter()
        extracted = llm_extract_one(client=client, model=model, rec=rec)
        times.append(time.perf_counter() - t0)

        normalized = coerce_extraction(extracted)
        examples.append(
            {
                "encounter_id": row["encounter_id"],
                "entities": summarize_named_entities(normalized),
            }
        )
    return times, examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test LLM extraction runtime")
    parser.add_argument("--seconds-per-call", type=float, default=3.0)
    parser.add_argument("--retry-error-rate", type=float, default=0.03)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--simulate-only", action="store_true")
    parser.add_argument("--model", type=str, default=os.getenv("OPENAI_MODEL", "gpt-5.5"))
    parser.add_argument(
        "--show-examples",
        type=int,
        default=3,
        help="Number of live extracted-entity examples to print (default: 3).",
    )
    args = parser.parse_args()

    n_rows = len(load_joined(RAW_PATH))

    if not args.simulate_only:
        live, examples = run_live_smoke(sample_size=args.sample_size, model=args.model)
        per_call = statistics.mean(live)
        retry_rate = 0.0
        print(f"Live smoke sample size: {len(live)}")
        print(f"Observed mean extraction call time: {per_call:.2f}s")
        if len(live) >= 2:
            print(f"Observed p95 extraction call time: {statistics.quantiles(live, n=20)[18]:.2f}s")

        n_show = max(0, min(int(args.show_examples), len(examples)))
        if n_show > 0:
            print("\nSample extracted named entities")
            for ex in examples[:n_show]:
                ent = ex["entities"]
                print(f"- encounter_id={ex['encounter_id']}")
                print(f"  symptoms={ent['symptoms']}")
                print(f"  time_prior_to_arrival_minutes={ent['time_prior_to_arrival_minutes']}")
                print(
                    "  context="
                    f"arrival_mode={ent['arrival_mode']}, "
                    f"origin={ent['patient_origin']}"
                )
    else:
        per_call = args.seconds_per_call
        retry_rate = args.retry_error_rate
        print("Simulation mode (no API calls)")
        print(f"Assumed mean extraction call time: {per_call:.2f}s")
        print(f"Assumed transient error rate per attempt: {retry_rate:.1%}")

    totals = simulate_total_seconds(
        n_rows=n_rows,
        seconds_per_call=per_call,
        retry_error_rate=retry_rate,
        max_retries=args.max_retries,
        n_trials=args.trials,
        seed=args.seed,
    )

    mean_total = statistics.mean(totals)
    p50 = statistics.median(totals)
    p90 = statistics.quantiles(totals, n=10)[8]
    p99 = statistics.quantiles(totals, n=100)[98]

    print("\nEstimated extraction runtime for full run")
    print(f"Rows: {n_rows}")
    print(f"Mean: {pretty_minutes(mean_total)}")
    print(f"P50:  {pretty_minutes(p50)}")
    print(f"P90:  {pretty_minutes(p90)}")
    print(f"P99:  {pretty_minutes(p99)}")

    timeout_hint = math.ceil(p90 * 1.2)
    print(f"\nRecommended job timeout budget: {pretty_minutes(timeout_hint)}")


if __name__ == "__main__":
    main()
