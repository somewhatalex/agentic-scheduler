"""Export complete serial programs as recorded OpenAI-shaped call replays.

This is an agentic-scheduler interchange format based on the NSDI 2026 Agentix
execution model, not an upstream Agentix file-loader contract. No requests are
sent. A runtime adapter must supply session APIs, timings, and tool/user delays.
"""

import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


REPLAY_FORMAT = "agentic-scheduler.replay.v1"


def chat_message(event):
    message = {"role": event["role"], "content": event["content"]}
    if event["role"] == "assistant":
        calls = json.loads(event["tool_calls_json"])
        if calls:
            message["tool_calls"] = [
                {"id": call["id"], "type": "function",
                 "function": {"name": call["name"], "arguments": call["arguments"]}}
                for call in calls
            ]
    elif event["role"] == "tool":
        message["tool_call_id"] = event["tool_call_id"]
    return message


def request_payload(events, tools, prefix_end, model):
    if not 0 <= prefix_end < len(events) or events[prefix_end]["role"] != "assistant":
        raise ValueError("prefix_end must identify the next assistant message")
    request = {"model": model, "messages": [chat_message(e) for e in events[:prefix_end]]}
    if tools:
        normalized_tools = []
        for tool in tools:
            tool = json.loads(tool) if isinstance(tool, str) else copy.deepcopy(tool)
            if not isinstance(tool, dict) or tool.get("type") != "function":
                raise ValueError("Replay supports function tool definitions only")
            function = tool.get("function")
            if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                raise ValueError("Invalid replay tool definition")
            parameters = function.get("parameters", {"type": "object"})
            if isinstance(parameters, str):
                parameters = json.loads(parameters)
            if not isinstance(parameters, dict):
                raise ValueError("Tool parameters must be JSON schema objects")
            function["parameters"] = parameters
            normalized_tools.append(tool)
        request["tools"] = normalized_tools
    return request


def parquet_rows(path):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1000):
        yield from batch.to_pylist()


def export(processed, output, *, model, limit_programs=100, split=None):
    processed, output = Path(processed), Path(output)
    if not model.strip() or (limit_programs is not None and limit_programs <= 0):
        raise ValueError("Supply a model name and a positive program limit")
    if not (processed / "_SUCCESS").is_file():
        raise ValueError("Processed input has no _SUCCESS marker")
    audit_bytes = (processed / "audit.json").read_bytes()
    audit = json.loads(audit_bytes)
    if audit["schema_version"] != 2 or audit["status"] != "complete":
        raise ValueError("Replay export requires a completed schema-version-2 ingestion run")
    for name in ("trajectories", "events", "calls"):
        if pq.ParquetFile(processed / f"{name}.parquet").metadata.num_rows != audit["counts"][name]:
            raise ValueError(f"{name} row count disagrees with ingestion audit")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "format": REPLAY_FORMAT, "status": "running", "model": model,
        "processed_path": str(processed.resolve()),
        "ingestion_audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
        "limit_programs": limit_programs, "split": split, "programs": 0, "calls": 0,
        "skipped_incomplete_programs": 0, "native_agentix_runtime_verified": False,
    }
    events_iter = parquet_rows(processed / "events.parquet")
    calls_iter = parquet_rows(processed / "calls.parquet")
    try:
        with (output / "replay.jsonl").open("w", encoding="utf-8") as stream:
            for parent in parquet_rows(processed / "trajectories.parquet"):
                if limit_programs is not None and report["programs"] >= limit_programs:
                    break
                events = list(itertools.islice(events_iter, parent["num_messages"]))
                calls = list(itertools.islice(calls_iter, sum(e["role"] == "assistant" for e in events)))
                if len(events) != parent["num_messages"] or any(
                    row["trajectory_id"] != parent["trajectory_id"] for row in events + calls
                ) or len(calls) != sum(e["role"] == "assistant" for e in events):
                    raise ValueError("Processed tables are not in matching trajectory order")
                if not parent["program_complete"]:
                    report["skipped_incomplete_programs"] += 1
                    continue
                if split is not None and parent["split"] != split:
                    continue
                tools = json.loads(parent["tools_json"])
                for index, call in enumerate(calls):
                    event = events[call["prefix_end"]]
                    record = {
                        "format": REPLAY_FORMAT, "program_id": call["program_id"],
                        "thread_id": call["thread_id"], "call_id": call["call_id"],
                        "parent_call_ids": call["parent_call_ids"],
                        "subset": parent["subset"], "split": parent["split"],
                        "session_start": index == 0, "session_end": index == len(calls) - 1,
                        "request": request_payload(events, tools, call["prefix_end"], model),
                        "expected_response": chat_message(event),
                        "recorded_reasoning": event["reasoning_content"],
                        "interrupts_before": [chat_message(events[i]) for i in call["interrupt_message_indices"]],
                        "recorded_context_tokens": call["recorded_context_tokens"],
                        "recorded_output_tokens": call["recorded_output_tokens"],
                        "service_time_s": call["service_time_s"],
                        "external_wait_s": call["external_wait_s"],
                    }
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    report["calls"] += 1
                report["programs"] += 1
        report["status"] = "complete" if report["calls"] else "no_calls"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (output / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["status"] == "complete":
        (output / "_SUCCESS").write_text(REPLAY_FORMAT + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed", type=Path)
    parser.add_argument("output", type=Path, help="new output directory")
    parser.add_argument("--model", required=True, help="target serving model ID; no model is called")
    parser.add_argument("--limit-programs", type=int, default=100)
    parser.add_argument("--split", choices=("train", "validation", "test"))
    args = parser.parse_args()
    try:
        report = export(args.processed, args.output, model=args.model,
                        limit_programs=args.limit_programs, split=args.split)
    except (OSError, ValueError, pa.ArrowException) as error:
        parser.exit(1, f"error: {error}\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "complete":
        parser.exit(2, "No replay calls were produced; inspect manifest.json.\n")


if __name__ == "__main__":
    main()
