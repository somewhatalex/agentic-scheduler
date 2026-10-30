# agentic-scheduler

Step-Level Remaining-Work Prediction for Scheduling LLM Agents (CSE 585).

## Data setup on CloudLab

Raw and processed data live on a long-term persistent dataset (ext4, project
Michigan-BigData, **Utah** cluster) that is mounted at `/mydata`. Experiments
expire after a few hours and their local disks are wiped, so keep everything that
matters under `/mydata`:

| Path | Contents |
| --- | --- |
| `/mydata/venv` | Python virtualenv |
| `/mydata/hf` | `HF_HOME` (Hugging Face cache) |
| `/mydata/raw/nemotron` | raw `nvidia/Nemotron-SFT-Agentic-v2` download |
| `/mydata/processed` | formatted Parquet output |

### 1. Create the profile (once)

1. Copy the dataset's URN from its info page (Storage > My Datasets > the dataset)
   into the `dataset_urn` default in `cloudlab/profile.py`. You can also override it
   when you instantiate.
2. In the CloudLab portal, go to **Experiments > Create Experiment Profile**, pick
   project Michigan-BigData, and paste or upload `cloudlab/profile.py` as the source.
   - The portal can also create a profile from a git repository, but it looks for
     `profile.py` at the repo root, so the paste option is simpler here.

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
