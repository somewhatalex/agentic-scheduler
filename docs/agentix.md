# Agentix compatibility assessment

Checked against the final [NSDI '26 paper](https://www.usenix.org/system/files/nsdi26-luo.pdf)
and its [USENIX publication page](https://www.usenix.org/conference/nsdi26/presentation/luo)
on October 29, 2026. This targets **Agentix by Luo et al.**, previously titled
Autellix, as cited in the proposal. Other projects sharing the name are outside
this compatibility assessment.

## Finding

The original three-table output supported training per-response predictors, but
did not expose explicit LLM-call dependencies or whole-program targets. Schema
version 2 adds those fields and a recorded-request exporter.

The format now supports the paper's **serial program execution model and a PLAS
comparison after adding a timing model**. It is not verified as directly loadable
by an Agentix runtime. The reviewed primary sources describe a stateful frontend
and its session lifecycle, but provide no native dataset-loader schema against
which to test Parquet or JSONL. No pinned upstream runtime was available in this
project, so native compatibility is explicitly marked unverified in both manifests.

## Mapping to the paper

Agentix treats programs as DAGs of LLM calls and external interrupts (§§2, 4).
PLAS tracks completed LLM execution time across a program; ATLAS extends this to
parallel LLM paths (§4.2). Its frontend starts and ends persistent sessions (§5).
These facts inform the following project-owned mapping, rather than establishing
an upstream file format. [Agentix paper](https://www.usenix.org/system/files/nsdi26-luo.pdf)

| Paper concept | Our representation |
| --- | --- |
| Program / persistent session | Entire parent conversation; `calls.program_id = trajectory_id` |
| LLM call | One recorded assistant message; stable `call_id` and exclusive `prefix_end` |
| Call dependency | `parent_call_ids` retains the preceding LLM call across user turns |
| External interrupt | Referenced user/tool events in `interrupt_message_indices` |
| Single observed LLM path | `thread_id = 0`; tool batches remain external operations |
| Program progress | `program_step_index` and recorded attained assistant tokens |
| Remaining program work | `y_program_remaining_*` targets at the same prediction points |
| Runtime service / external waiting | Nullable `service_time_s` and `external_wait_s` |
| Session lifecycle | Exported `session_start` / `session_end` flags for a runtime adapter |

`step_index` and the original `y_remaining_*` labels remain scoped to one
user-response episode. Whole-program targets include later recorded user
requests; those future messages are labels/history for replay, never predictor
inputs before their cutoff. Incomplete programs keep events and calls for audit,
but receive no whole-program labels and are excluded from replay exports. This
prevents silently presenting an unfinished conversation as a completed program.

The dataset establishes a serial LLM-call order. Parallel external tools do not
establish parallel LLM branches, so it cannot validate ATLAS's multithreaded
benefit without a workload that records actual branching LLM calls.

## Export contract

```bash
python scripts/export_replay.py data/processed/demo-v2 data/replay/demo-v2 --model replay-placeholder
```

The output directory must be new. The exporter reads schema-version-2 tables in
their original ingestion order; do not independently reorder them. It streams
one parent program at a time and defaults to at most 100 complete programs.
`--split train|validation|test` selects programs from that split, and
`--limit-programs N` changes the cap without cutting a program in half.

`replay.jsonl` uses the project format `agentic-scheduler.replay.v1`, one row per
LLM call. Rows contain:

- Program/thread/call IDs, parent-call references, source subset, and split.
- Session boundary flags, applied before the first call and after the last call.
- `request`, containing `model`, observed-prefix `messages`, and tool schemas.
- `expected_response` and `recorded_reasoning`, stored separately from `request`.
- User/tool `interrupts_before` and nullable runtime timing fields.
- Recorded context and output token counts, retaining their ingestion convention.

The exporter restores nested `{id, type, function: {name, arguments}}` call
envelopes and parses JSON-encoded tool parameter schemas. It omits
`reasoning_content` from request messages because it is not part of the common
chat-message envelope; provider-specific reasoning replay requires an explicit
adapter. The `--model` value is an annotation for the intended serving model.
No model, session API, tool, or external endpoint is invoked.

This is a **recorded transcript replay**: later request prefixes use recorded
assistant responses and tool results, regardless of what a live model would
generate. It is not a live-agent correctness test. Full prefixes are repeated in
this optional JSONL export, so it can be substantially larger than the canonical
Parquet representation. Keep initial exports small.

`manifest.json` records the model annotation, split, limit, counts, and a hash of
the input ingestion audit. `_SUCCESS` appears only after a completed export with
calls. Empty exports exit 2; failures exit 1 and save a failure manifest where
possible. Output files are never overwritten.

## Runtime work still required

To run an actual Agentix baseline, pin a concrete runtime revision and implement
its session/client adapter. That adapter must translate our IDs to the runtime's
session handles, submit only `request`, apply chosen tool/user interrupt delays
outside the LLM engine, and close sessions after the final call completes.
The manifest fields are not assumed valid keyword arguments for an upstream API.

Profile or simulate LLM **execution service** separately from queue waiting and
external delays. Use that service to update PLAS; summing client wall-clock
latency or tool delays changes the scheduling policy. Supplied token counts are
only a proxy and omit some serving costs, including serialized call payloads,
chat-template/tool-schema overhead, and caching behavior.

For comparisons with the proposed predictor, give both schedulers the same
arrivals, program boundaries, service costs, interrupt model, and available
history. Construct timing-dependent labels in that separate evaluation layer;
this ingester supplies no measured timing ground truth.

## Validation performed

The local tests cover dependencies across user interrupts, complete-program
filtering, single LLM threads despite tool batches, prefix-only request payloads,
session boundaries, and both label scopes. A leading sample from the three
Nemotron files at revision `7c804833427f633ccd53b582dbf02525fd680f78` processed
208 conversations into 2,452 events, 1,120 recorded LLM calls, and 1,093 training
examples. Replay export retained 191 complete programs and 1,093 calls, excluding
17 incomplete programs. This sample check does not establish whole-corpus
coverage, real service-time accuracy, or native Agentix runtime compatibility.
