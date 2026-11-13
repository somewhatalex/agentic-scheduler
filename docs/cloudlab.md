# CloudLab setup

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


## 1. Create the profile (once)

The dataset URN is already the default in `profile.py`. It can also be overridden
when you instantiate.

In the CloudLab portal, go to **Experiments > Create Experiment Profile**, pick
project Michigan-BigData, and choose the git repository option with
`https://github.com/somewhatalex/agentic-scheduler`. The portal reads `profile.py`
from the repo root (it must stay there) and picks up later pushes to the repo.

## 2. Start an experiment

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

## 3. Download and check the data

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

## 4. Copy processed output to your laptop

Run this from your laptop:

```bash
rsync -avz --progress <cloudlab-user>@<node>.utah.cloudlab.us:/mydata/processed/ ./data/processed/
```

`data/` is gitignored. Do not commit data files.

## Run ingestion on a real subset

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
