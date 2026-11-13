# agentic-scheduler

Step-Level Remaining-Work Prediction for Scheduling LLM Agents (CSE 585).

## Status (Nov 12, 2026)

- [x] Ingestion of Nemotron-SFT-Agentic-v2 into Parquet, tested on a local slice.
- [x] FCFS and Agentix-style baseline in a scheduling simulator, on that slice
      with assumed token speeds.
- [ ] Recreate the CloudLab dataset: `agentic-sched-data` had a Nov 4 expiry and
      no longer appears in the owner's dataset list.
- [ ] Full-corpus ingestion, then rerun the baseline on it.
- [ ] Measure token speeds on a GPU and rerun the baseline with them.
- [ ] Domain and subset holdouts, SWE-smith, predictors, and the proposed scheduler.

## Quick start

No GPU, model, or Hugging Face token is needed. Python 3.10+.

```bash
python3 -m venv venv && source venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/make_demo_data.py data/raw/demo
python scripts/prepare_data.py data/raw/demo data/processed/demo-v2 --strict
python scripts/inspect_processed.py data/processed/demo-v2
```

Output directories must be new; existing results are never overwritten.
`data/` is gitignored. Do not commit data files.

## Data

Nemotron is three JSONL files of recorded agent conversations in OpenAI chat
format with tool calls. It has no timing fields.

| File | Size | Contents | Typical length (median) |
| --- | --- | --- | --- |
| `search.jsonl` | 0.6 GB | multi-hop web research, 1 tool (`web-search`) | 25 messages, 11 tool calls |
| `interactive_agent.jsonl` | 6.3 GB | customer-service agent following a policy, many tools | 11 messages, 2 tool calls |
| `tool_calling.jsonl` | 14.9 GB | general tool use (e.g. Wolfram) | 7 messages, 2 tool calls |

`scripts/prepare_data.py` turns them into Parquet tables of trajectories,
events, prediction examples with remaining-work targets, and LLM calls, plus an
`audit.json`. A run is usable only if it wrote `_SUCCESS`.

## Baseline simulation

`scripts/simulate.py` replays recorded programs on a simulated server and
compares scheduling policies. Times come from token counts and assumed rates,
so results are **simulation estimates**.

- **FCFS:** each LLM call is served in the order it became ready.
- **Agentix-style:** each call is ranked by the LLM service its whole program
  has already received, so long-running programs give way to short ones. This is
  our reading of the rule in Luo et al. (NSDI '26), not the Agentix runtime.

```bash
python scripts/simulate.py data/processed/slice-v2 --output data/results/slice-v2-baseline.json
```

Mean response time in seconds on the first slice (5 trials of 500 programs, 8
server slots):

| Load | FCFS | Agentix-style |
| --- | --- | --- |
| batch | 2429.7 | 2235.9 |
| 0.5 | 80.8 | 80.9 |
| 0.8 | 89.4 | 89.2 |
| 0.95 | 157.9 | 157.2 |

The Agentix-style policy is about 8% better when all programs arrive at once and
makes no measurable difference under streaming arrivals.

## More detail

- [docs/baseline.md](docs/baseline.md): policy definitions, full results, assumptions, and measuring token speeds with `scripts/profile_llm.py`.
- [docs/cloudlab.md](docs/cloudlab.md): CloudLab profile, experiments, downloading the data, and running ingestion there.
- [docs/ingestion.md](docs/ingestion.md): table semantics, features, targets, and splits.
- [docs/agentix.md](docs/agentix.md): Agentix compatibility and the recorded replay export.
- [docs/proposal.md](docs/proposal.md): the project proposal.
