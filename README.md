# agentic-scheduler

Step-Level Remaining-Work Prediction for Scheduling LLM Agents (CSE 585).

## Data setup on CloudLab

Raw and processed data live on a persistent dataset (`agentic-sched-data`: ext4,
100 GB, project Michigan-BigData, **Utah** cluster) that is mounted at `/mydata`.
Experiments expire after a few hours and their local disks (including your home
directory) are wiped, so keep everything that matters under `/mydata`:

| Path | Contents |
| --- | --- |
| `/mydata/venv` | Python virtualenv |
| `/mydata/hf` | `HF_HOME` (Hugging Face cache) |
| `/mydata/raw/nemotron` | raw `nvidia/Nemotron-SFT-Agentic-v2` download |
| `/mydata/processed` | formatted Parquet output |

### Status (Oct 29, 2026)

- [x] Profile `agentic-sched-data` created from this repo in the CloudLab portal.
- [x] Nemotron-SFT-Agentic-v2 downloaded to `/mydata/raw/nemotron/data/`
      (21 GB used, ~66 GB free). File sizes match Hugging Face.
- [x] `check_data.py` parsed 1000 trajectories from each file.
- [x] Persistence verified: data written in one experiment was read back
      unchanged (same MD5) from a different node in a second, read-only
      experiment, without re-downloading.
- [ ] The dataset is **short-term and expires Nov 4, 2026**. Long-term storage
      needs approval; if that means a new dataset, update the URN in `profile.py`
      and rerun `setup.sh` (the download takes ~2 minutes).
- [ ] Format the raw JSONL into Parquet in `/mydata/processed`.
- [x] Local ingestion prototype, program/call dependencies, and recorded replay
      export tested on a small Nemotron sample. Full CloudLab ingestion remains
      to be run.
- [ ] SWE-smith.

### Data format

Each file is JSONL: one line is one recorded agent conversation in OpenAI chat
format with tool calls.

| File | Size | Contents | Typical length (median) |
| --- | --- | --- | --- |
| `search.jsonl` | 0.6 GB | multi-hop web research, 1 tool (`web-search`) | 25 messages, 11 tool calls |
| `interactive_agent.jsonl` | 6.3 GB | customer-service agent following a policy, many tools | 11 messages, 2 tool calls |
| `tool_calling.jsonl` | 14.9 GB | general tool use (e.g. Wolfram) | 7 messages, 2 tool calls |

All files share `messages`, `tools`, and `metadata`; the other columns differ
per file (for example, only `interactive_agent` has `model` and `domain`).
Each message has a `role` (`system`, `user`, `assistant`, `tool`) and
`content`; assistant messages may also have `reasoning_content` and
`tool_calls`, and tool messages have a `tool_call_id` linking them to the call.
Complete response episodes end with an assistant response and no pending tool
calls; some records instead stop at an unresolved tool call. There are no timing
fields. When present, `metadata.turn_token_count` supplies per-message reasoning
and content token counts, aligned to all message roles.

To look at a record on a node:

```bash
head -1 /mydata/raw/nemotron/data/search.jsonl | /mydata/venv/bin/python -m json.tool | less
```

### 1. Create the profile (once)

The dataset URN is already the default in `profile.py`. It can also be overridden
when you instantiate.

In the CloudLab portal, go to **Experiments > Create Experiment Profile**, pick
project Michigan-BigData, and choose the git repository option with
`https://github.com/somewhatalex/agentic-scheduler`. The portal reads `profile.py`
from the repo root (it must stay there) and picks up later pushes to the repo.

### 2. Start an experiment

Instantiate the profile and select the **Utah** cluster. The remote blockstore
only attaches to nodes on the same cluster as the dataset.

`/mydata` is mounted read-only by default, which is what teammates should use.
Only the dataset owner can write to it, and only one experiment can mount it
read-write at a time. Uncheck "Mount /mydata read-only" only when you are
downloading or processing data. When the node is ready,
the List View tab shows its SSH command, for example:

```bash
ssh <cloudlab-user>@<node>.utah.cloudlab.us
```

Tips:

- SSH logs you into your home directory; the data is in `/mydata`.
- `/mydata` can take about a minute to mount after SSH starts working. If it is
  missing, wait and check again with `findmnt /mydata`.
- The default hardware is often an m400 (ARM). For heavier processing, set the
  hardware type to an x86 node such as m510 or c6525.
- The image has no `tmux`. Install it (`sudo apt-get install -y tmux`) before
  long-running jobs so they survive a dropped SSH connection.

### 3. Download and check the data

```bash
git clone https://github.com/somewhatalex/agentic-scheduler
cd agentic-scheduler
bash scripts/setup.sh                 # venv + hf download; skips steps already done
source /mydata/venv/bin/activate
export HF_HOME=/mydata/hf
python scripts/check_data.py /mydata/raw/nemotron
```

The raw download is about 22 GB of JSONL (`tool_calling`, `interactive_agent`,
`search`) and needs no Hugging Face token. `setup.sh` needs a read-write mount.

For each file, `check_data.py` prints the columns and the number of assistant
turns and tool messages per trajectory for the first 1000 rows (`--rows`, `--show`
change how many).

### 4. Copy processed output to your laptop

Run this from your laptop:

```bash
rsync -avz --progress <cloudlab-user>@<node>.utah.cloudlab.us:/mydata/processed/ ./data/processed/
```

`data/` is gitignored. Do not commit data files.

## Ingestion prototype

The ingester runs on a laptop or a CloudLab CPU node. It reads raw JSONL directly;
it does not call models, execute tools, or need a Hugging Face token. Python 3.10+
and the PyArrow dependency in `requirements.txt` are sufficient.

### Test locally, without downloading Nemotron

Use `venv` as the local virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/make_demo_data.py data/raw/demo
python scripts/prepare_data.py data/raw/demo data/processed/demo-v2 --strict
python scripts/inspect_processed.py data/processed/demo-v2
```

The synthetic fixture covers parallel tool calls, out-of-order tool results,
multiple user requests, missing token metadata, and an unresolved tool call.
Expected output: **4 trajectories, 15 events, 6 LLM calls, 5 examples,
1 excluded episode**. Three parent programs are complete; one is incomplete.
The inspection command checks Parquet row counts against the audit and displays
examples. The tests separately check label correctness, prefix-only features,
split grouping, malformed input, and incomplete episodes.

Both demo generation and ingestion require a **new output directory**. For a
repeat run, choose another name, such as `demo-v3`; existing results are refused.

### Test a real subset on CloudLab

On a Utah experiment with `/mydata` mounted read-write, after the data setup above:

```bash
source /mydata/venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/prepare_data.py /mydata/raw/nemotron /mydata/processed/smoke-v2 --limit 1000
python scripts/inspect_processed.py /mydata/processed/smoke-v2
```

If you switch between ARM and x86 nodes, create a fresh node-local `venv` and
install `requirements.txt` for that architecture. Compiled packages in a
persistent virtualenv are architecture-specific; the raw and Parquet data are
portable.

`--limit` caps input records **per file**, including rejected and duplicate rows.
It does not limit downloads or randomly sample the corpus. Check `audit.json` for
rejection reasons, usable examples in each subset, excluded episodes, and token
metadata coverage before processing everything:

```bash
python scripts/prepare_data.py /mydata/raw/nemotron /mydata/processed/nemotron-v2
python scripts/inspect_processed.py /mydata/processed/nemotron-v2
```

A read-only experiment can run the smoke test with `/tmp/smoke-v2` as its output;
copy that output off the node before the experiment ends. Input data is never
modified. Teammates can also copy a small sample to a laptop: on CloudLab run
`head -n 100 /mydata/raw/nemotron/data/tool_calling.jsonl > /tmp/tool_calling.jsonl`,
then use `scp` from the laptop to copy that file into `data/raw/sample/` and pass
that directory to `prepare_data.py`. Malformed sample rows will be audited.

### Output and interpretation

| File | Contents |
| --- | --- |
| `trajectories.parquet` | Parent identity, source/domain, tool schemas, split, and quality counts |
| `events.parquet` | One row per message, text/reasoning, calls, matched results, and recorded token counts |
| `examples.parquet` | Features and remaining-work targets at LLM-ready boundaries |
| `calls.parquet` | Program/thread/call IDs, parent calls, interrupt references, and nullable timing fields |
| `audit.json` | Configuration, versions, input fingerprints, counts, and rejection/exclusion reasons |
| `_SUCCESS` | Created only after all outputs close successfully and at least one example exists |

Detailed semantics, feature columns, and limitations are in
[docs/ingestion.md](docs/ingestion.md). A failed or interrupted run must not be
used for training. Runs producing no examples exit with code 2 and no `_SUCCESS`.

### Agentix compatibility and recorded replay

The target is Luo et al.'s [NSDI '26 Agentix serving system](https://www.usenix.org/conference/nsdi26/presentation/luo).
The output now preserves its **program-level execution model**: a whole parent
conversation is one program, assistant messages are LLM calls, and user/tool
messages are external interrupts. Program history continues across user turns.
Both per-response and whole-program remaining-work targets are available.

This is **execution-model compatibility**, with an adapter-friendly export;
direct loading into an Agentix runtime has not been verified. The reviewed paper
describes session APIs but does not specify a native Parquet/JSONL loader. Our
program/thread/session fields are interchange metadata, not claimed upstream API
arguments. Measured LLM service times and external delays are still missing, and
these serial conversations do not supply parallel LLM branches for ATLAS.

After a schema-version-2 ingestion run, export a small recorded workload:

```bash
python scripts/export_replay.py data/processed/demo-v2 data/replay/demo-v2 --model replay-placeholder
```

The demo exports **3 complete programs and 5 calls**. No model or tool is invoked.
`replay.jsonl` contains OpenAI-shaped request payloads, separately stored expected
responses, call dependencies, and session-boundary flags. `manifest.json` records
export scope; `_SUCCESS` marks a completed export with calls. The default cap is
100 complete programs; use `--limit-programs N` and optionally `--split test`.
Full request prefixes are repeated in this optional export, so keep its size
small. Existing schema-version-1 output must be regenerated for replay export;
the inspection script continues to support version 1.

For a PLAS comparison matching Agentix's program unit, train using the
`y_program_remaining_*` targets and accumulate **LLM service time** across that
same program. Recorded token counts are a proxy, not attained service time.
See [docs/agentix.md](docs/agentix.md) for the compatibility assessment, field
mapping, and the runtime work still required.
