"""Check output row counts against the audit and display a few prediction examples."""

import argparse
import itertools
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def inspect(path, show=3):
    path = Path(path)
    if not (path / "_SUCCESS").is_file():
        raise ValueError(f"{path} has no _SUCCESS marker; ingestion did not complete with examples")
    report = json.loads((path / "audit.json").read_text())
    if report["status"] != "complete":
        raise ValueError(f"Unexpected ingestion status: {report['status']}")
    tables = ["trajectories", "events", "examples"]
    if report["schema_version"] >= 2:
        tables.append("calls")
    for table in tables:
        parquet = pq.ParquetFile(path / f"{table}.parquet")
        actual, expected = parquet.metadata.num_rows, report["counts"][table]
        if actual != expected:
            raise ValueError(f"{table}: audit says {expected} rows, Parquet contains {actual}")
        print(f"{table}: {actual} rows")
    print("Audit counts:", json.dumps(report["counts"], sort_keys=True))
    print("Excluded episodes:", json.dumps(report["episode_exclusions"], sort_keys=True))
    print("Token metadata:", json.dumps(report["token_count_statuses"], sort_keys=True))
    print("Split counts:", json.dumps(report.get("split_counts", {}), sort_keys=True))
    if "agentix" in report:
        print("Agentix scope:", json.dumps(report["agentix"], sort_keys=True))
    print("Feature columns:", ", ".join(report["feature_columns"]))
    print(f"First {show} examples:")
    batches = pq.ParquetFile(path / "examples.parquet").iter_batches(batch_size=max(show, 1))
    rows = (row for batch in batches for row in batch.to_pylist())
    for row in itertools.islice(rows, show):
        print(json.dumps(row, sort_keys=True))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--show", type=int, default=3)
    args = parser.parse_args()
    if args.show < 0:
        parser.error("--show must be nonnegative")
    try:
        inspect(args.path, args.show)
    except (OSError, ValueError, pa.ArrowException) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
