# Nemotron ingestion contract (schema version 2)

This prototype converts recorded conversations into scheduling prediction
examples. It runs the same way locally and on CloudLab. It requires no GPU,
model weights, tokenizer download, or live tools.

## Scheduling unit and boundaries

The whole parent conversation is a **program**, preserving the unit used by
Agentix. Each user message starts a **response episode**, ending immediately before the
next user message or at the end of the record. Earlier conversation messages
remain context. This avoids treating a future user request as work the agent
must finish for the current request. A consecutive user message ends an episode
without a response; that episode is excluded.

An example represents the prefix immediately **before a recorded assistant
message**, provided all previously requested tool results in the episode have
arrived. `prefix_end` is an exclusive, zero-based message cutoff. Reconstruct
the observed context from events with the same `trajectory_id` and
`message_index < prefix_end`. `request_message_index` identifies the current
user request. Text is stored once in events, not copied into every example.

Tool requests are counted individually within an assistant's call list. Results
are matched by `tool_call_id`, including out-of-order responses to a batch.
Resolved call IDs can be reused by a later invocation. Events preserve each
batch in `tool_calls_json`; a matched tool result's `assistant_message_index`
identifies the requesting assistant message. Calls in a batch are not assumed
to have run in parallel: the dataset does not establish their timing.
The normalized call list contains objects with `id`, `name`, and `arguments`
(argument text), rather than the original nested OpenAI call envelope.

An episode receives **no examples** when calls are unresolved, a result has no
matching pending call, an assistant message appears before pending calls finish,
a pending ID is reused, or no final assistant response is recorded. Its events
are still saved, and exclusion reasons appear in the audit. Earlier complete
episodes in that conversation remain usable. This is deliberately conservative:
a valid SFT target ending at a tool call is not a complete execution trace.

These are observed transcript boundaries, not proof of real execution, successful
answers, or actual request completion. External user think time is not modeled.

## Features and targets

Use **only `feature_columns` in `audit.json`** as the initial tabular model input.
The example table also contains identifiers, split annotations, and labels.
Do not feed the entire table into a model without selecting its columns.

| Feature | Meaning at the prefix cutoff |
| --- | --- |
| `step_index` | Assistant messages already completed in this episode |
| `program_step_index` | Assistant messages already completed in the whole program; does not reset at a user turn |
| `context_message_count` | Messages observed across the parent conversation |
| `episode_tool_calls` | Calls already issued in this episode |
| `prefix_tool_calls` | Calls already issued across the conversation |
| `prefix_recorded_tokens` | Recorded reasoning + content tokens in all observed messages |
| `prefix_assistant_recorded_tokens` | Same count, restricted to observed assistant messages |
| `last_assistant_recorded_tokens` | Recorded tokens in the most recent assistant message; initially zero |
| `last_tool_name` | Tool whose result arrived most recently; initially null |
| `last_tool_output_chars` | Length of that result in Unicode characters; initially zero |
| `last_call_repeated` | Whether the last assistant message repeated a previously observed tool name + arguments |

The last-tool and repetition features can include earlier episodes as observed
history. Equivalent JSON arguments are compared independent of key order and
whitespace. Arbitrary argument text is preserved when it is not valid JSON.

The original targets count work **from the next assistant message through the
end of the current response episode**:

- `y_remaining_tool_calls`: number of individual tool requests yet to be issued.
- `y_remaining_assistant_turns`: assistant messages still to generate, including
  the final response.
- `y_remaining_recorded_assistant_tokens`: sum of supplied assistant reasoning
  and content token counts over that suffix.

The corresponding `y_program_remaining_tool_calls`,
`y_program_remaining_assistant_turns`, and
`y_program_remaining_recorded_assistant_tokens` targets continue through the
entire recorded parent program, including later user turns. These are offline
labels; future requests never enter the observed prefix. Whole-program labels
are null if any episode is incomplete. Complete earlier episodes can still have
valid response-episode labels in an incomplete program.

Choose one task unit consistently when training and comparing schedulers.
Use program targets for a whole-program Agentix/PLAS comparison. External user
think time remains unknown and requires an explicit evaluation assumption.

`metadata.turn_token_count` is aligned to **all messages**, not just assistant
turns. Missing, misaligned, invalid, or negative counts result in null token
features and labels for that trajectory. Call and assistant-turn labels remain
available. Inspect `token_count_status` on the parent and coverage in the audit.

The provided counts are not a complete serving-token accounting. In particular,
assistant messages with structured tool calls often report zero content tokens;
this prototype does not tokenize their serialized call payload. It does not
estimate prefill cost, KV-cache behavior, tool latency, or wall-clock time.
Keep the eventual serving tokenizer and timing model separate from these labels.

## Program and call interchange

`calls.parquet` has one row per recorded assistant invocation, including calls
from incomplete programs. `program_id` equals the parent `trajectory_id`, and
`call_id` adds the assistant message index. `examples.call_id` joins to this table.
The first call has no parents; subsequent calls depend on the preceding LLM call
across user and tool interrupts. All calls in the observed serial conversation
use `thread_id = 0`; a batch of external tools does not create parallel LLM threads.

`interrupt_message_indices` references user/tool events observed since the prior
LLM call. `recorded_context_tokens`, `recorded_output_tokens`, and
`recorded_attained_assistant_tokens` preserve supplied token quantities.
`service_time_s` and `external_wait_s` are null because no timings were observed.
`episode_complete` and `program_complete` identify supported replay scope.

The optional replay exporter consumes tables in ingestion order and exports only
complete programs. It rebuilds nested function-call messages and JSON schema
objects for tool definitions, while keeping expected responses and recorded
reasoning separate from request payloads. See [agentix.md](agentix.md).

These fields implement our mapping of the paper's execution model; they are not
a verified native Agentix wire or file format. Schema version 1 had no call table
or whole-program targets; regenerate it to use version-2 export. The inspection
script still accepts version-1 runs.

Future metadata such as full-trajectory totals, `num_tool_calls`, `ground_truth`,
finish reasons, and final answers is never copied into features. Text for all
events is available for audit, so downstream text models must respect the cutoff.

## Identity and splits

`trajectory_id` hashes normalized messages and tool schemas. Exact duplicates
are discarded using an on-disk SQLite index so deduplication memory does not grow
with corpus size. First occurrence wins; source coverage counts describe retained
records. Metadata UUIDs are retained separately as `source_id`. The audit records
retained counts by subset, source, domain, and split.

`split_group_id` hashes the initial system/user context and tool schemas. An
80/10/10 train/validation/test split is assigned by a seeded hash of that group,
before examples are written. All episodes and alternative completions sharing
that initial context have the same split. Assignment does not depend on file
ordering or the number of prefixes. Near-duplicate prompts are not detected.

This prototype does not yet implement the proposal's whole-domain or whole-subset
holdout evaluation. Source, subset, and domain are retained on parent records for
that next stage; those holdouts must preserve split groups when constructed.

## Input quality, memory, and provenance

Pass a JSONL file or a directory containing JSONL files. Files are scanned
recursively, excluding Hugging Face `.cache` metadata. Each file stem becomes its
subset name. The three original Nemotron filenames therefore retain their subset
names. Messages, message elements, metadata, and tools may be JSON-encoded strings
at the levels handled by the normalizer. Message content must be text or null;
non-null legacy `function_call` messages are explicitly rejected.

Malformed rows are skipped and counted, with the first 20 locations/reasons in
`audit.json`. `--strict` stops on the first malformed/unsupported row; an incomplete
episode is an audited exclusion rather than a malformed row.

Memory is bounded by a single input record, its normalized events/examples,
and Parquet buffers controlled by `--batch-size` (default 1000 rows per table).
Lines larger than `--max-line-bytes` (default 16 MiB) are drained in bounded chunks
and rejected. This also limits exceptional valid records; inspect the audit
before choosing a different cap. The temporary deduplication index uses disk
space under the output directory and is removed on normal exit.

`--limit N` reads at most N nonblank records per file, not a random sample.
An oversized or malformed row still consumes one record of that limit. Parquet
schemas are explicit; subset differences do not rely on inferred Arrow types.

The audit records Python/PyArrow versions, schema version, seed, source file sizes,
bytes scanned, and SHA-256 fingerprints of scanned bytes. A limited run records
`scan_complete: false` when it stops at its limit; this fingerprint is not a hash
of the unread suffix. `--dataset-revision SHA` records an optional user-supplied
provenance annotation; it does not verify that the local files came from that
revision. Raw input should remain immutable for the duration of a run.

Output paths must be new. `_SUCCESS` is written only after all Parquet writers
close and the complete audit is saved, with at least one example produced. A
failed run saves an audit where possible and exits with code 1; a completed audit
with no usable examples exits with code 2. Neither creates `_SUCCESS`. There is
no resume support yet: choose a new directory for another run.
