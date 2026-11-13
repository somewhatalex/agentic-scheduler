# agentic-scheduler

Step-Level Remaining-Work Prediction for Scheduling LLM Agents (CSE 585).

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
- **SSJF (perfect length):** the waiting call with the shortest output runs
  first, per call and without preemption, after Qiu et al. (AIOps '24).
- **SSJF (noisy):** the same, with random error added to each length.

Both SSJF versions rank calls by the **recorded output length, not a trained
predictor**, so the perfect-length version is an upper bound on real SSJF.

```bash
python scripts/simulate.py data/processed/slice-v2 --output data/results/slice-v2-baseline-ssjf.json
```

Mean response time in seconds on the first slice (5 trials of 500 programs, 8
server slots):

| Load | FCFS | Agentix-style | SSJF (perfect length) | SSJF (noisy) |
| --- | --- | --- | --- | --- |
| batch | 2429.7 | 2235.9 | 1729.3 | 1912.8 |
| 0.5 | 80.8 | 80.9 | 80.6 | 80.7 |
| 0.8 | 89.4 | 89.2 | 86.8 | 87.5 |
| 0.95 | 157.9 | 157.2 | 127.3 | 143.7 |

When all programs arrive at once, the Agentix-style policy is about 8% better
than FCFS, noisy SSJF about 21%, and perfect-length SSJF about 29%. Under
streaming arrivals the Agentix-style policy makes no measurable difference,
and SSJF helps only at loads 0.8 and 0.95.

## Next: GPU session (Nov 13)

One `c240g5` node (NVIDIA P100, 12 GB) at CloudLab Wisconsin; details and the
full task list are in [docs/gpu-session.md](docs/gpu-session.md).

- Measure prefill and decode speed with vLLM and `scripts/profile_llm.py`.
- Rerun the baseline with the measured speeds.
- Start a BERT-base output-length predictor on Nemotron for a real SSJF baseline.
- Copy results off the node before it expires.

## More detail

- [docs/baseline.md](docs/baseline.md): policy definitions, full results, assumptions, and measuring token speeds with `scripts/profile_llm.py`.
- [docs/gpu-session.md](docs/gpu-session.md): the GPU node, its reservation, and the tasks for the Nov 13 session.
- [docs/cloudlab.md](docs/cloudlab.md): CloudLab profile, experiments, downloading the data, and running ingestion there.
- [docs/ingestion.md](docs/ingestion.md): table semantics, features, targets, and splits.
- [docs/agentix.md](docs/agentix.md): Agentix compatibility and the recorded replay export.
- [docs/proposal.md](docs/proposal.md): the project proposal.
