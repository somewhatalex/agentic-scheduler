"""Write tiny, synthetic Nemotron-shaped JSONL files for an offline smoke test."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new directory for demo JSONL files")
    args = parser.parse_args()
    fixture = Path(__file__).resolve().parents[1] / "tests/fixtures/nemotron.json"
    rows_by_subset = json.loads(fixture.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    for subset, rows in rows_by_subset.items():
        with (args.output / f"{subset}.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
    print(f"Wrote {sum(map(len, rows_by_subset.values()))} synthetic trajectories to {args.output}")


if __name__ == "__main__":
    main()
