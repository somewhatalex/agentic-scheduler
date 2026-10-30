"""Sanity-check a downloaded agentic trajectory dataset.

Finds the .parquet / .jsonl files under a path and, for each file, prints the
columns and counts assistant turns and tool messages per trajectory for the
first N rows.

Usage: python scripts/check_data.py /mydata/raw/nemotron [--rows 1000] [--show 10]
"""

import argparse
import json
import statistics
from pathlib import Path

import pyarrow.parquet as pq


def find_files(root):
    root = Path(root)
    files = sorted(root.rglob("*.parquet")) + sorted(root.rglob("*.jsonl"))
    # Skip the HF cache metadata that hf download leaves next to the data.
    return [f for f in files if ".cache" not in f.parts]


def iter_rows(path):
    if path.suffix == ".parquet":
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1000):
            yield from batch.to_pylist()
    else:
        with open(path) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def columns(path):
    if path.suffix == ".parquet":
        return pq.read_schema(path).names
    with open(path) as f:
        return list(json.loads(f.readline()).keys())


def count_roles(messages):
    # Some exports store messages as a JSON string rather than a list.
    if isinstance(messages, str):
        messages = json.loads(messages)
    assistant = sum(1 for m in messages if m.get("role") == "assistant")
    tool = sum(1 for m in messages if m.get("role") == "tool")
    return len(messages), assistant, tool


def summarize(name, values):
    print(
        f"  {name:<10} min={min(values)} median={statistics.median(values)} "
        f"mean={statistics.mean(values):.1f} max={max(values)}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", help="directory containing the dataset files")
    parser.add_argument("--rows", type=int, default=1000, help="rows to scan per file")
    parser.add_argument("--show", type=int, default=10, help="per-trajectory rows to print per file")
    args = parser.parse_args()

    files = find_files(args.path)
    if not files:
        raise SystemExit(f"no .parquet or .jsonl files under {args.path}")
    print(f"found {len(files)} data files")

    for path in files:
        print(f"\n=== {path}")
        print(f"columns: {columns(path)}")

        counts = []
        for row in iter_rows(path):
            if "messages" not in row:
                raise SystemExit(f"{path} has no 'messages' column: {list(row)}")
            counts.append(count_roles(row["messages"]))
            if len(counts) >= args.rows:
                break
        if not counts:
            print("no rows")
            continue

        print(f"per-trajectory counts (first {min(args.show, len(counts))} of {len(counts)}):")
        print(f"  {'idx':>5}  {'messages':>8}  {'assistant':>9}  {'tool':>5}")
        for i, (total, assistant, tool) in enumerate(counts[: args.show]):
            print(f"  {i:>5}  {total:>8}  {assistant:>9}  {tool:>5}")

        print(f"summary over {len(counts)} trajectories:")
        summarize("messages", [c[0] for c in counts])
        summarize("assistant", [c[1] for c in counts])
        summarize("tool", [c[2] for c in counts])
        no_tools = sum(1 for c in counts if c[2] == 0)
        print(f"  trajectories with no tool messages: {no_tools}")


if __name__ == "__main__":
    main()
