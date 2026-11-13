#!/usr/bin/env python3
"""Scheduling simulator for recorded agent programs (CPU only; no model is run).

Replays complete programs from a schema-version-2 ingestion run on a simulated
LLM server with a fixed number of slots and compares scheduling policies on
identical programs, arrivals, and sampled external waits.

Every time in the output is a SIMULATION ESTIMATE. The dataset has no timing
fields, so this script assumes:

* LLM service per call = context_tokens / prefill_tps + output_tokens / decode_tps.
  Output tokens are the recorded reasoning + content tokens plus an estimate
  for the serialized tool-call payload (characters / 4), which the recorded
  counts omit.
* Each slot serves one call at full speed; batching slowdown is not modelled.
* A tool batch waits for the slowest of its results; each result's latency is
  lognormal. A follow-up user message waits a constant think time.
* External waits release the slot. Preemption is free (no KV-cache cost).

Policies:

* fcfs    Call-level first-come, first-served, the usual serving default.
* plas    Agentix-style program-level attained service: calls are ranked by
          the LLM service their whole program has received, discretized into
          queue levels with geometric thresholds, FCFS within a level, and
          preemption when a running call crosses into a lower level. This is
          our reading of the paper's rule, not the Agentix runtime, and it has
          no anti-starvation promotion.
* oracle  Shortest remaining program service with perfect information. A
          reference for the same slot model, not a global optimum.
"""

import argparse
import bisect
import heapq
import json
import math
import random
import statistics
import sys
from pathlib import Path

POLICIES = ("fcfs", "plas", "oracle")
EPS = 1e-9
# Two-sided 95% Student t critical values by degrees of freedom.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
       8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086, 30: 2.042}


def plas_thresholds(base_s, ratio, levels):
    return [base_s * ratio ** i for i in range(levels - 1)]


def simulate(programs, policy, slots, thresholds=()):
    """Run one policy and return each program's finish time.

    `programs` is a list of dicts with `arrival` (seconds) and `calls`, a list
    of (wait_before_s, service_s) pairs in execution order. The wait happens
    before the call becomes ready and does not hold a slot.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    count = len(programs)
    index = [0] * count          # current call
    remaining = [0.0] * count    # service left in the current call
    attained = [0.0] * count     # LLM service the program has received
    future = [sum(s for _, s in p["calls"]) for p in programs]  # service not yet started
    ready_at = [0.0] * count
    finish = [None] * count
    pending = []                 # (time the next call becomes ready, program)
    for pid, program in enumerate(programs):
        if not program["calls"]:
            raise ValueError("a program needs at least one call")
        heapq.heappush(pending, (program["arrival"] + program["calls"][0][0], pid))
    active, running, now = set(), [], 0.0

    def level(pid):
        return bisect.bisect_right(thresholds, attained[pid] + EPS)

    def key(pid):
        if policy == "fcfs":
            return (ready_at[pid], pid)
        if policy == "plas":
            return (level(pid), ready_at[pid], pid)
        return (remaining[pid] + future[pid], ready_at[pid], pid)

    while pending or active:
        step = math.inf
        if pending:
            step = pending[0][0] - now
        for pid in running:
            step = min(step, remaining[pid])
            if policy == "plas" and level(pid) < len(thresholds):
                step = min(step, thresholds[level(pid)] - attained[pid])
        step = max(step, 0.0)
        now += step
        for pid in running:
            remaining[pid] -= step
            attained[pid] += step
        for pid in [p for p in running if remaining[p] <= EPS]:
            active.discard(pid)
            index[pid] += 1
            calls = programs[pid]["calls"]
            if index[pid] == len(calls):
                finish[pid] = now
            else:
                heapq.heappush(pending, (now + calls[index[pid]][0], pid))
        while pending and pending[0][0] <= now + EPS:
            _, pid = heapq.heappop(pending)
            service = programs[pid]["calls"][index[pid]][1]
            remaining[pid] = service
            future[pid] -= service
            ready_at[pid] = now
            active.add(pid)
        running = heapq.nsmallest(slots, active, key=key)
    return finish


def load_programs(processed, *, split=None, subsets=None):
    """Read complete programs as {program_id, subset, calls: [token facts]}."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    processed = Path(processed)
    if not (processed / "_SUCCESS").exists():
        raise SystemExit(f"{processed} is not a completed ingestion run")
    calls = pq.read_table(processed / "calls.parquet", columns=[
        "program_id", "subset", "split", "prefix_end", "interrupt_message_indices",
        "recorded_context_tokens", "recorded_output_tokens", "program_complete"])
    mask = calls["program_complete"]
    if split:
        mask = pc.and_(mask, pc.equal(calls["split"], split))
    if subsets:
        mask = pc.and_(mask, pc.is_in(calls["subset"], value_set=__import__("pyarrow").array(subsets)))
    calls = calls.filter(mask).to_pylist()
    wanted = {row["program_id"] for row in calls}
    events = pq.read_table(processed / "events.parquet", columns=[
        "trajectory_id", "message_index", "role", "tool_calls_json"]).to_pylist()
    roles, payload_chars = {}, {}
    for event in events:
        if event["trajectory_id"] in wanted:
            where = (event["trajectory_id"], event["message_index"])
            roles[where] = event["role"]
            if event["role"] == "assistant" and event["tool_calls_json"]:
                payload_chars[where] = sum(
                    len(call.get("name") or "") + len(call.get("arguments") or "")
                    for call in json.loads(event["tool_calls_json"]))
    programs = {}
    skipped = set()
    for row in calls:  # ingestion order is execution order within a program
        pid = row["program_id"]
        if row["recorded_context_tokens"] is None or row["recorded_output_tokens"] is None:
            skipped.add(pid)
            continue
        interrupts = [roles[(pid, i)] for i in row["interrupt_message_indices"]]
        program = programs.setdefault(pid, {"program_id": pid, "subset": row["subset"], "calls": []})
        program["calls"].append({
            "context_tokens": row["recorded_context_tokens"],
            "output_tokens": row["recorded_output_tokens"]
                             + payload_chars.get((pid, row["prefix_end"]), 0) // 4,
            "tool_results": interrupts.count("tool"),
            "user_followup": bool(program["calls"]) and "user" in interrupts,
        })
    return [p for pid, p in programs.items() if pid not in skipped], len(skipped)


def apply_timing(program, *, prefill_tps, decode_tps, tool_median_s, tool_sigma,
                 user_think_s, seed):
    """Turn token facts into (wait_before_s, service_s); same draws for every policy."""
    rng = random.Random(f"{seed}:{program['program_id']}")
    timed = []
    for call in program["calls"]:
        wait = max((rng.lognormvariate(math.log(tool_median_s), tool_sigma)
                    for _ in range(call["tool_results"])), default=0.0)
        if call["user_followup"]:
            wait += user_think_s
        service = call["context_tokens"] / prefill_tps + call["output_tokens"] / decode_tps
        timed.append((wait, service))
    return timed


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)]


def mean_ci(values):
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, None
    df = len(values) - 1
    t = T95.get(df) or T95[max(k for k in T95 if k <= df)]
    return mean, t * statistics.stdev(values) / math.sqrt(len(values))


def run_experiment(pool, args):
    thresholds = plas_thresholds(args.plas_base_s, args.plas_ratio, args.plas_levels)
    results = []
    for load in args.loads:
        per_trial = {policy: [] for policy in args.policies}
        for trial in range(args.trials):
            rng = random.Random(f"{args.seed}:{load}:{trial}")
            chosen = rng.sample(pool, min(args.programs, len(pool)))
            mean_service = statistics.fmean(sum(s for _, s in p["calls"]) for p in chosen)
            clock, programs = 0.0, []
            for program in chosen:
                if load != "batch":
                    clock += rng.expovariate(float(load) * args.slots / mean_service)
                programs.append({"arrival": clock, "calls": program["calls"]})
            alone = [sum(w + s for w, s in p["calls"]) for p in programs]
            for policy in args.policies:
                finish = simulate(programs, policy, args.slots, thresholds)
                response = [f - p["arrival"] for f, p in zip(finish, programs)]
                per_trial[policy].append({
                    "mean_response_s": statistics.fmean(response),
                    "p95_response_s": percentile(response, 0.95),
                    "p95_slowdown": percentile([r / a for r, a in zip(response, alone)], 0.95),
                })
        for policy in args.policies:
            row = {"load": load, "policy": policy, "trials": args.trials}
            for metric in ("mean_response_s", "p95_response_s", "p95_slowdown"):
                row[metric] = statistics.fmean(t[metric] for t in per_trial[policy])
            if policy != "fcfs" and "fcfs" in per_trial:
                diffs = [t["mean_response_s"] - b["mean_response_s"]
                         for t, b in zip(per_trial[policy], per_trial["fcfs"])]
                row["mean_response_vs_fcfs_s"], row["vs_fcfs_ci95_s"] = mean_ci(diffs)
            results.append(row)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("processed", help="schema-version-2 ingestion output directory")
    parser.add_argument("--output", help="write results and settings to this new JSON file")
    parser.add_argument("--policies", nargs="+", default=list(POLICIES), choices=POLICIES)
    parser.add_argument("--loads", nargs="+", default=["batch", "0.5", "0.8", "0.95"],
                        help="'batch' (all arrive at 0) or an offered LLM load in (0, 1)")
    parser.add_argument("--programs", type=int, default=500, help="programs per trial")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--slots", type=int, default=8, help="concurrent calls on the server")
    parser.add_argument("--split", choices=["train", "validation", "test"])
    parser.add_argument("--subsets", nargs="+")
    parser.add_argument("--prefill-tps", type=float, default=5000.0)
    parser.add_argument("--decode-tps", type=float, default=50.0)
    parser.add_argument("--tool-median-s", type=float, default=1.0)
    parser.add_argument("--tool-sigma", type=float, default=1.0)
    parser.add_argument("--user-think-s", type=float, default=0.0)
    parser.add_argument("--plas-base-s", type=float, default=2.0,
                        help="attained service at which a program leaves the top queue")
    parser.add_argument("--plas-ratio", type=float, default=2.0)
    parser.add_argument("--plas-levels", type=int, default=8)
    parser.add_argument("--seed", type=int, default=585)
    args = parser.parse_args()
    if args.output and Path(args.output).exists():
        raise SystemExit(f"{args.output} already exists")

    pool, skipped = load_programs(args.processed, split=args.split, subsets=args.subsets)
    if not pool:
        raise SystemExit("no complete programs matched")
    for program in pool:
        program["calls"] = apply_timing(
            program, prefill_tps=args.prefill_tps, decode_tps=args.decode_tps,
            tool_median_s=args.tool_median_s, tool_sigma=args.tool_sigma,
            user_think_s=args.user_think_s, seed=args.seed)
    results = run_experiment(pool, args)

    print(f"{len(pool)} complete programs available ({skipped} skipped for missing token counts); "
          f"{min(args.programs, len(pool))} per trial, {args.trials} trials, {args.slots} slots")
    print("Simulation estimates under assumed timing; see --help.\n")
    print(f"{'load':>6} {'policy':>7} {'mean resp s':>12} {'p95 resp s':>11} "
          f"{'p95 slowdown':>13} {'vs fcfs s (95% CI)':>22}")
    for row in results:
        versus = ""
        if "mean_response_vs_fcfs_s" in row:
            versus = f"{row['mean_response_vs_fcfs_s']:+.1f}"
            if row["vs_fcfs_ci95_s"] is not None:
                versus += f" ± {row['vs_fcfs_ci95_s']:.1f}"
        print(f"{row['load']:>6} {row['policy']:>7} {row['mean_response_s']:>12.1f} "
              f"{row['p95_response_s']:>11.1f} {row['p95_slowdown']:>13.1f} {versus:>22}")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(
            {"settings": vars(args), "programs_available": len(pool), "results": results}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
