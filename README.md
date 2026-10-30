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
- [ ] SWE-smith.

### Data format

Each file is JSONL: one line is one complete agent trajectory in OpenAI chat
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
A trajectory ends with an assistant message that has no tool call. There are
no timings or token counts in the data.

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
