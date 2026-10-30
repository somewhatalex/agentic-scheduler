"""Stream Nemotron JSONL into normalized events and LLM-ready prediction examples.

Each user message starts a response episode. Incomplete episodes retain their
events but receive no labels. Token quantities use the supplied per-message
reasoning + content counts; they are not serving-token counts or timings.
"""

import argparse
from collections import Counter
from contextlib import ExitStack, closing
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sqlite3
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA_VERSION = 2
FEATURE_TYPES = {
    "step_index": pa.int64(),
    "program_step_index": pa.int64(),
    "context_message_count": pa.int64(),
    "episode_tool_calls": pa.int64(),
    "prefix_tool_calls": pa.int64(),
    "prefix_recorded_tokens": pa.int64(),
    "prefix_assistant_recorded_tokens": pa.int64(),
    "last_assistant_recorded_tokens": pa.int64(),
    "last_tool_name": pa.string(),
    "last_tool_output_chars": pa.int64(),
    "last_call_repeated": pa.bool_(),
}
FEATURE_COLUMNS = tuple(FEATURE_TYPES)
TARGET_TYPES = {
    "y_remaining_tool_calls": pa.int64(),
    "y_remaining_assistant_turns": pa.int64(),
    "y_remaining_recorded_assistant_tokens": pa.int64(),
    "y_program_remaining_tool_calls": pa.int64(),
    "y_program_remaining_assistant_turns": pa.int64(),
    "y_program_remaining_recorded_assistant_tokens": pa.int64(),
}
COMMON_TYPES = {
    "trajectory_id": pa.string(), "subset": pa.string(),
    "split_group_id": pa.string(), "split": pa.string(),
}
SCHEMAS = {
    "trajectories": pa.schema({
        **COMMON_TYPES, "source_id": pa.string(), "source": pa.string(),
        "domain": pa.string(), "input_file": pa.string(), "input_line": pa.int64(),
        "tools_json": pa.string(), "num_messages": pa.int64(),
        "num_tool_calls": pa.int64(), "complete_episodes": pa.int64(),
        "excluded_episodes": pa.int64(), "token_count_status": pa.string(),
        "program_complete": pa.bool_(),
    }),
    "events": pa.schema({
        "trajectory_id": pa.string(), "message_index": pa.int64(),
        "episode_index": pa.int64(), "role": pa.string(), "content": pa.string(),
        "reasoning_content": pa.string(), "tool_calls_json": pa.string(),
        "tool_call_id": pa.string(), "tool_name": pa.string(),
        "assistant_message_index": pa.int64(), "content_tokens": pa.int64(),
        "reasoning_tokens": pa.int64(),
    }),
    "examples": pa.schema({
        **COMMON_TYPES, "episode_index": pa.int64(),
        "call_id": pa.string(),
        "request_message_index": pa.int64(), "prefix_end": pa.int64(),
        **FEATURE_TYPES, **TARGET_TYPES,
    }),
    "calls": pa.schema({
        **COMMON_TYPES, "program_id": pa.string(), "thread_id": pa.int64(),
        "call_id": pa.string(), "parent_call_ids": pa.list_(pa.string()),
        "program_step_index": pa.int64(), "episode_index": pa.int64(),
        "request_message_index": pa.int64(), "prefix_end": pa.int64(),
        "interrupt_message_indices": pa.list_(pa.int64()),
        "recorded_context_tokens": pa.int64(), "recorded_output_tokens": pa.int64(),
        "recorded_attained_assistant_tokens": pa.int64(),
        "service_time_s": pa.float64(), "external_wait_s": pa.float64(),
        "episode_complete": pa.bool_(), "program_complete": pa.bool_(),
    }),
}


class RowError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def json_text(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(json_text(value).encode("utf-8")).hexdigest()


def decode(value, expected, code):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, RecursionError) as error:
            raise RowError(code) from error
    if not isinstance(value, expected):
        raise RowError(code)
    return value


def optional_text(value):
    return value if isinstance(value, str) else None


def token_counts(metadata, length):
    counts = metadata.get("turn_token_count")
    if counts is None:
        return "missing", None
    if not isinstance(counts, list) or len(counts) != length:
        return "misaligned", None
    for count in counts:
        if not isinstance(count, dict) or any(
            type(count.get(key)) is not int or not 0 <= count[key] <= 10**12
            for key in ("reasoning", "content")
        ):
            return "invalid", None
    return "valid", counts


def normalize_messages(raw_messages):
    messages = decode(raw_messages, list, "invalid_messages")
    events = []
    for message in messages:
        message = decode(message, dict, "invalid_message")
        role = message.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            raise RowError("unsupported_role")
        content = message.get("content")
        reasoning = message.get("reasoning_content")
        content = "" if content is None else content
        reasoning = "" if reasoning is None else reasoning
        if not isinstance(content, str) or not isinstance(reasoning, str):
            raise RowError("unsupported_content")
        if message.get("function_call") is not None:
            raise RowError("unsupported_legacy_function_call")
        raw_calls = message.get("tool_calls")
        calls = decode([] if raw_calls is None else raw_calls, list, "invalid_tool_calls")
        normalized_calls = []
        for call in calls:
            call = decode(call, dict, "invalid_tool_call")
            function = decode(call.get("function"), dict, "invalid_tool_function")
            if not isinstance(call.get("id"), str) or not call["id"]:
                raise RowError("missing_call_id")
            if not isinstance(function.get("name"), str) or not function["name"]:
                raise RowError("missing_tool_name")
            arguments = function.get("arguments", "{}")
            if not isinstance(arguments, str):
                arguments = json_text(arguments)
            normalized_calls.append({
                "id": call["id"], "name": function["name"], "arguments": arguments,
            })
        if calls and role != "assistant":
            raise RowError("calls_on_non_assistant")
        if role == "tool" and not isinstance(message.get("tool_call_id"), str):
            raise RowError("missing_result_call_id")
        events.append({
            "role": role, "content": content, "reasoning_content": reasoning,
            "tool_call_id": optional_text(message.get("tool_call_id")),
            "tool_name": optional_text(message.get("name")),
            "tool_calls_json": json_text(normalized_calls), "_calls": normalized_calls,
        })
    return events


@dataclass
class PreparedTrajectory:
    trajectory: dict
    events: list
    examples: list
    exclusions: Counter
    calls: list


def make_calls(events, common, user_indices, complete_episodes, counts):
    """Map the observed serial LLM trace to programs, calls, and interrupts.

    Program/call IDs and parent edges implement the paper's execution model.
    They are our interchange fields, not undocumented Agentix wire parameters.
    Tool batches do not establish concurrent LLM threads or tool durations.
    """
    calls, interrupts = [], []
    context_tokens = attained_tokens = 0 if counts is not None else None
    program_complete = len(complete_episodes) == len(user_indices)
    for i, event in enumerate(events):
        if event["role"] in ("user", "tool"):
            interrupts.append(i)
        if event["role"] == "assistant":
            calls.append({
                **common, "program_id": common["trajectory_id"], "thread_id": 0,
                "call_id": f"{common['trajectory_id']}:{i}",
                "parent_call_ids": [calls[-1]["call_id"]] if calls else [],
                "program_step_index": len(calls), "episode_index": event["episode_index"],
                "request_message_index": user_indices[event["episode_index"]], "prefix_end": i,
                "interrupt_message_indices": interrupts,
                "recorded_context_tokens": context_tokens,
                "recorded_output_tokens": event["content_tokens"] + event["reasoning_tokens"] if counts is not None else None,
                "recorded_attained_assistant_tokens": attained_tokens,
                "service_time_s": None, "external_wait_s": None,
                "episode_complete": event["episode_index"] in complete_episodes,
                "program_complete": program_complete,
            })
            interrupts = []
        if counts is not None:
            tokens = event["content_tokens"] + event["reasoning_tokens"]
            context_tokens += tokens
            if event["role"] == "assistant":
                attained_tokens += tokens
    return calls


def prepare_trajectory(row, subset, seed=42):
    row = decode(row, dict, "invalid_row")
    raw_metadata, raw_tools = row.get("metadata"), row.get("tools")
    metadata = decode({} if raw_metadata is None else raw_metadata, dict, "invalid_metadata")
    tools = decode([] if raw_tools is None else raw_tools, list, "invalid_tools")
    events = normalize_messages(row.get("messages"))
    user_indices = [i for i, event in enumerate(events) if event["role"] == "user"]
    if not user_indices:
        raise RowError("missing_user_request")
    if any(e["role"] != "system" for e in events[:user_indices[0]]):
        raise RowError("execution_before_first_user")
    identity = [{k: v for k, v in e.items() if not k.startswith("_")} for e in events]
    trajectory_id = fingerprint([identity, tools])
    # Group alternative completions of the same initial context together.
    split_group_id = fingerprint([identity[:user_indices[0] + 1], tools])
    bucket = int(fingerprint([seed, split_group_id])[:16], 16) % 100
    split = "train" if bucket < 80 else "validation" if bucket < 90 else "test"
    common = dict(trajectory_id=trajectory_id, subset=subset, split_group_id=split_group_id, split=split)
    status, counts = token_counts(metadata, len(events))
    for i, event in enumerate(events):
        event.update(
            trajectory_id=trajectory_id, message_index=i, episode_index=None,
            assistant_message_index=i if event["role"] == "assistant" else None,
            content_tokens=counts[i]["content"] if counts is not None else None,
            reasoning_tokens=counts[i]["reasoning"] if counts is not None else None,
        )

    prefix_tokens = 0 if counts is not None else None
    assistant_tokens = 0 if counts is not None else None
    last_assistant_tokens = 0 if counts is not None else None
    prefix_calls = 0
    last_tool_name, last_tool_chars, repeated = None, 0, False
    seen_calls = set()
    examples, exclusions = [], Counter()
    complete_episodes = excluded_episodes = 0
    complete_episode_indices = set()
    episode_index = -1
    pending, candidates, errors = {}, [], set()
    episode_calls = step = program_step = 0

    def finish_episode():
        nonlocal complete_episodes, excluded_episodes
        if episode_index < 0:
            return
        if pending:
            errors.add("unresolved_tool_calls")
        if not candidates:
            errors.add("no_ready_assistant_turn")
        if events[episode_end - 1]["role"] != "assistant":
            errors.add("no_final_assistant_response")
        if errors:
            excluded_episodes += 1
            exclusions.update(errors)
            return
        complete_episodes += 1
        complete_episode_indices.add(episode_index)
        # A reverse scan avoids recounting the suffix at every prediction point.
        remaining_calls = remaining_turns = remaining_tokens = 0
        candidate_by_index = {candidate["prefix_end"]: candidate for candidate in candidates}
        for j in range(episode_end - 1, request_index, -1):
            event = events[j]
            if event["role"] == "assistant":
                remaining_calls += len(event["_calls"])
                remaining_turns += 1
                if counts is not None:
                    remaining_tokens += event["content_tokens"] + event["reasoning_tokens"]
            if j in candidate_by_index:
                candidate_by_index[j].update(
                    y_remaining_tool_calls=remaining_calls,
                    y_remaining_assistant_turns=remaining_turns,
                    y_remaining_recorded_assistant_tokens=remaining_tokens if counts is not None else None,
                )
        examples.extend(candidates)

    request_index = episode_end = 0
    for i, event in enumerate(events):
        role = event["role"]
        if role == "user":
            episode_end = i
            finish_episode()
            episode_index += 1
            request_index = i
            pending, candidates, errors = {}, [], set()
            episode_calls = step = 0
        event["episode_index"] = episode_index if episode_index >= 0 else None
        if role == "assistant":
            if pending:
                errors.add("assistant_before_tool_results")
            else:
                candidates.append({
                    **common, "episode_index": episode_index,
                    "call_id": f"{trajectory_id}:{i}",
                    "request_message_index": request_index, "prefix_end": i,
                    "step_index": step, "program_step_index": program_step, "context_message_count": i,
                    "episode_tool_calls": episode_calls, "prefix_tool_calls": prefix_calls,
                    "prefix_recorded_tokens": prefix_tokens,
                    "prefix_assistant_recorded_tokens": assistant_tokens,
                    "last_assistant_recorded_tokens": last_assistant_tokens,
                    "last_tool_name": last_tool_name, "last_tool_output_chars": last_tool_chars,
                    "last_call_repeated": repeated,
                })
            repeated = False
            for call in event["_calls"]:
                if call["id"] in pending:
                    errors.add("duplicate_pending_call_id")
                pending[call["id"]] = (call["name"], i)
                try:
                    arguments = json.loads(call["arguments"])
                except ValueError:
                    arguments = call["arguments"]
                signature = fingerprint([call["name"], arguments])
                repeated |= signature in seen_calls
                seen_calls.add(signature)
            prefix_calls += len(event["_calls"])
            episode_calls += len(event["_calls"])
            step += 1
            program_step += 1
        elif role == "tool":
            match = pending.pop(event["tool_call_id"], None)
            if match is None:
                errors.add("unmatched_tool_result")
            else:
                event["tool_name"], event["assistant_message_index"] = match
            last_tool_name, last_tool_chars = event["tool_name"], len(event["content"])
        if counts is not None:
            tokens = event["content_tokens"] + event["reasoning_tokens"]
            prefix_tokens += tokens
            if role == "assistant":
                assistant_tokens += tokens
                last_assistant_tokens = tokens
    episode_end = len(events)
    finish_episode()
    calls = make_calls(events, common, user_indices, complete_episode_indices, counts)
    program_complete = excluded_episodes == 0
    examples_by_index = {example["prefix_end"]: example for example in examples}
    remaining_calls = remaining_turns = remaining_tokens = 0
    for event in reversed(events):
        if event["role"] == "assistant":
            remaining_calls += len(event["_calls"])
            remaining_turns += 1
            if counts is not None:
                remaining_tokens += event["content_tokens"] + event["reasoning_tokens"]
        example = examples_by_index.get(event["message_index"])
        if example is not None:
            example.update(
                y_program_remaining_tool_calls=remaining_calls if program_complete else None,
                y_program_remaining_assistant_turns=remaining_turns if program_complete else None,
                y_program_remaining_recorded_assistant_tokens=remaining_tokens if program_complete and counts is not None else None,
            )
    trajectory = {
        **common, "source_id": optional_text(metadata.get("uuid") or row.get("uuid") or metadata.get("id")),
        "source": optional_text(metadata.get("source")), "domain": optional_text(row.get("domain")),
        "tools_json": json_text(tools), "num_messages": len(events), "num_tool_calls": prefix_calls,
        "complete_episodes": complete_episodes, "excluded_episodes": excluded_episodes,
        "token_count_status": status,
        "program_complete": program_complete,
    }
    for event in events:
        del event["_calls"]
    return PreparedTrajectory(trajectory, events, examples, exclusions, calls)


class ParquetSink:
    def __init__(self, path, schema, batch_size):
        self.schema, self.batch_size, self.buffer = schema, batch_size, []
        self.writer = pq.ParquetWriter(path, schema, compression="zstd")

    def append(self, row):
        self.buffer.append(row)
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self):
        if self.buffer:
            self.writer.write_table(pa.Table.from_pylist(self.buffer, schema=self.schema))
            self.buffer.clear()

    def close(self):
        try:
            self.flush()
        finally:
            self.writer.close()


class JsonlReader:
    """Bound memory even for a corrupted file containing an enormous JSONL line."""

    def __init__(self, path, max_line_bytes):
        self.path, self.max_line_bytes = path, max_line_bytes
        self.sha256 = hashlib.sha256()
        self.bytes_scanned = 0
        self.scan_complete = False

    def records(self, limit):
        with self.path.open("rb") as stream:
            seen = line_number = 0
            while limit is None or seen < limit:
                line = stream.readline(self.max_line_bytes + 1)
                if not line:
                    self.scan_complete = True
                    break
                line_number += 1
                self.sha256.update(line)
                self.bytes_scanned += len(line)
                oversized = len(line) > self.max_line_bytes
                if oversized:
                    while not line.endswith(b"\n"):
                        line = stream.readline(self.max_line_bytes + 1)
                        self.sha256.update(line)
                        self.bytes_scanned += len(line)
                        if not line:
                            break
                elif not line.strip():
                    continue
                seen += 1
                yield line_number, None if oversized else line


def run(input_path, output_path, *, limit=None, batch_size=1000,
        max_line_bytes=16 * 1024 * 1024, seed=42, strict=False,
        dataset_revision=None, progress_every=1000):
    if any(value is not None and value <= 0 for value in (limit, batch_size, max_line_bytes, progress_every)):
        raise ValueError("limits and batch sizes must be positive")
    input_path, output_path = Path(input_path), Path(output_path)
    files = [input_path] if input_path.is_file() else sorted(
        p for p in input_path.rglob("*.jsonl") if ".cache" not in p.parts
    )
    if not files or any(path.suffix != ".jsonl" for path in files):
        raise ValueError(f"No JSONL input files found at {input_path}")
    output_path.mkdir(parents=True, exist_ok=False)
    counts = Counter(rows_seen=0, rejected=0, duplicates=0, trajectories=0, events=0, examples=0, calls=0,
                     complete_episodes=0, excluded_episodes=0, complete_programs=0, incomplete_programs=0)
    by_subset, reasons, exclusions, token_statuses = {}, Counter(), Counter(), Counter()
    source_counts, domain_counts, lengths, split_counts = Counter(), Counter(), Counter(), {}
    report = {
        "schema_version": SCHEMA_VERSION, "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(), "pyarrow_version": pa.__version__,
        "dataset": {"repo": "nvidia/Nemotron-SFT-Agentic-v2", "revision": dataset_revision,
                    "revision_verified": False},
        "config": {"limit_per_file": limit, "batch_size": batch_size,
                   "max_line_bytes": max_line_bytes, "seed": seed, "strict": strict},
        "feature_columns": list(FEATURE_COLUMNS), "target_columns": list(TARGET_TYPES),
        "agentix": {"target": "NSDI 2026 Agentix", "compatibility": "paper_execution_model",
                    "native_runtime_verified": False, "program_unit": "parent_conversation",
                    "llm_threads_per_program": 1, "timings": "unavailable"},
        "counts": counts, "by_subset": by_subset, "rejection_reasons": reasons,
        "episode_exclusions": exclusions, "token_count_statuses": token_statuses,
        "source_counts": source_counts, "domain_counts": domain_counts,
        "split_counts": split_counts,
        "message_length_log2_histogram": lengths, "files": [], "rejection_examples": [],
    }
    try:
        with ExitStack() as stack:
            scratch = stack.enter_context(tempfile.TemporaryDirectory(prefix=".dedup-", dir=output_path))
            db = stack.enter_context(closing(sqlite3.connect(str(Path(scratch) / "seen.sqlite"))))
            db.execute("CREATE TABLE seen (id TEXT PRIMARY KEY)")
            sinks = {}
            for name, schema in SCHEMAS.items():
                sinks[name] = ParquetSink(output_path / f"{name}.parquet", schema, batch_size)
                stack.callback(sinks[name].close)
            for path in files:
                subset = path.stem
                subset_counts = by_subset.setdefault(subset, Counter())
                reader = JsonlReader(path, max_line_bytes)
                initial_stat = path.stat()
                file_report = {"path": str(path.resolve()), "subset": subset,
                               "size_bytes": initial_stat.st_size,
                               "mtime_ns": initial_stat.st_mtime_ns, "rows_seen": 0}
                report["files"].append(file_report)
                for line_number, line in reader.records(limit):
                    counts["rows_seen"] += 1
                    subset_counts["rows_seen"] += 1
                    file_report["rows_seen"] += 1
                    try:
                        if line is None:
                            raise RowError("line_too_large")
                        try:
                            row = json.loads(line)
                        except (ValueError, UnicodeDecodeError, RecursionError) as error:
                            raise RowError("invalid_json") from error
                        prepared = prepare_trajectory(row, subset, seed)
                    except (RowError, ValueError, RecursionError) as error:
                        code = error.code if isinstance(error, RowError) else "invalid_value"
                        counts["rejected"] += 1
                        subset_counts["rejected"] += 1
                        reasons[code] += 1
                        if len(report["rejection_examples"]) < 20:
                            report["rejection_examples"].append(
                                {"file": str(path), "line": line_number, "reason": code})
                        if strict:
                            raise ValueError(f"{path}:{line_number}: {code}") from error
                        continue
                    trajectory = prepared.trajectory
                    inserted = db.execute("INSERT OR IGNORE INTO seen VALUES (?)",
                                          (trajectory["trajectory_id"],)).rowcount
                    if not inserted:
                        counts["duplicates"] += 1
                        subset_counts["duplicates"] += 1
                        continue
                    trajectory.update(input_file=str(path.resolve()), input_line=line_number)
                    sinks["trajectories"].append(trajectory)
                    for event in prepared.events:
                        sinks["events"].append(event)
                    for example in prepared.examples:
                        sinks["examples"].append(example)
                    for call in prepared.calls:
                        sinks["calls"].append(call)
                    increments = dict(trajectories=1, events=len(prepared.events), examples=len(prepared.examples), calls=len(prepared.calls),
                                      complete_episodes=trajectory["complete_episodes"],
                                      excluded_episodes=trajectory["excluded_episodes"],
                                      complete_programs=int(trajectory["program_complete"]),
                                      incomplete_programs=int(not trajectory["program_complete"]))
                    counts.update(increments)
                    subset_counts.update(increments)
                    split_counts.setdefault(trajectory["split"], Counter()).update(increments)
                    exclusions.update(prepared.exclusions)
                    token_statuses[trajectory["token_count_status"]] += 1
                    source_counts[trajectory["source"] or "<missing>"] += 1
                    domain_counts[trajectory["domain"] or "<missing>"] += 1
                    lengths[str(int(math.log2(trajectory["num_messages"])))] += 1
                    if counts["trajectories"] % progress_every == 0:
                        db.commit()
                        print(f"Inspected {counts['rows_seen']} rows; wrote {counts['examples']} examples", file=sys.stderr)
                file_report.update(bytes_scanned=reader.bytes_scanned,
                                   scanned_sha256=reader.sha256.hexdigest(),
                                   scan_complete=reader.scan_complete)
        report["status"] = "complete" if counts["examples"] else "no_examples"
        for file_report, path in zip(report["files"], files):
            final_stat = path.stat()
            if (final_stat.st_size, final_stat.st_mtime_ns) != (file_report["size_bytes"], file_report["mtime_ns"]):
                raise ValueError(f"Input changed during ingestion: {path}")
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        (output_path / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["status"] == "complete":
        (output_path / "_SUCCESS").write_text(f"schema_version={SCHEMA_VERSION}\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="raw JSONL file or directory (searched recursively)")
    parser.add_argument("output", type=Path, help="new output directory; existing paths are refused")
    parser.add_argument("--limit", type=int, help="maximum input records per file; omit for a full scan")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--max-line-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict", action="store_true", help="stop at the first malformed record")
    parser.add_argument("--dataset-revision", help="provenance annotation for your downloaded snapshot")
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()
    try:
        report = run(args.input, args.output, limit=args.limit, batch_size=args.batch_size,
                     max_line_bytes=args.max_line_bytes, seed=args.seed, strict=args.strict,
                     dataset_revision=args.dataset_revision, progress_every=args.progress_every)
    except (OSError, ValueError, pa.ArrowException) as error:
        parser.exit(1, f"error: {error}\n")
    print(json.dumps({"output": str(args.output), "status": report["status"], **report["counts"]}, indent=2))
    if report["status"] != "complete":
        parser.exit(2, "No labeled examples were produced; inspect audit.json.\n")


if __name__ == "__main__":
    main()
