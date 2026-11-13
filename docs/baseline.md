# Baseline scheduling simulation (Nov 12, 2026)

This is the first scheduling baseline: FCFS against an Agentix-style policy and
two SSJF-style stand-ins on recorded Nemotron programs. It runs on a CPU. No model is served, so **every
time below is a simulation estimate under assumed timing**, not a measurement.

## What the simulator does

`scripts/simulate.py` replays complete programs from a schema-version-2
ingestion run on a simulated LLM server with a fixed number of slots. Each
program is one recorded conversation; each assistant message is one LLM call;
tool results and follow-up user messages are external waits that release the
slot. Every policy sees the same programs, the same arrival times, and the same
sampled waits, so differences between policies are paired.

Nemotron has no timing fields, so the simulator derives times from token counts:

- LLM service per call = context tokens / prefill rate + output tokens / decode rate.
- Output tokens are the recorded reasoning + content tokens plus characters / 4
  of the serialized tool call, which the recorded counts omit.
- A tool batch waits for the slowest of its results; each result's latency is
  drawn from a lognormal distribution.

It reports, per policy and load, the mean and p95 response time (arrival to
completion of the whole program) and the p95 slowdown (response time divided by
the program's time running alone), averaged over trials, plus the paired
difference in mean response time from FCFS with a 95% confidence interval.

## Policies

### FCFS

Call-level first-come, first-served: whenever a slot frees, the LLM call that
has been ready longest gets it. Calls are never preempted. This is the usual
default of LLM serving systems and ignores which program a call belongs to.

### Agentix-style (PLAS)

Agentix (Luo et al., *Agentix: An Efficient Serving Engine for LLM Agents as
General Programs*, NSDI '26; previously titled Autellix) schedules whole agent
programs, not individual calls. Its rule for single-threaded programs is
**program-level attained service (PLAS)**: each LLM call is ranked by the total
LLM service its whole program has already received. A program that has used
little service so far goes first; a long-running program sinks. The rule needs
no prediction of how much work is left: short programs finish early because
they never accumulate much service.

Our version (`--policies plas`) discretizes attained service into queue levels:

- Thresholds are 2 s, doubling, with 8 levels (2, 4, 8, ... 128 s).
- A lower level always runs before a higher one; within a level, calls run FCFS.
- A running call that crosses a threshold drops a level and can be preempted by
  a waiting call from a lower level.
- Attained service carries over between a program's calls, across tool waits
  and user turns.

This is **our reading of the paper's rule, not the Agentix runtime**. We chose
the thresholds ourselves, there is no anti-starvation rule, and preemption costs
nothing. Nemotron conversations are serial, so the paper's multi-threaded
variant (ATLAS) does not apply; see [agentix.md](agentix.md).

### SSJF-style

SSJF (Qiu et al., *Efficient Interactive LLM Serving with Proxy Model-based
Sequence Length Prediction*, AIOps '24, arXiv:2404.08509) predicts each
request's output length from its prompt with a small BERT model and runs the
request with the shortest predicted output first. It works per LLM call, does
not preempt, and ignores which program a call belongs to and what that program
has already done.

We have no trained predictor yet, so both versions here rank calls by the
**recorded output length**, not a prediction:

- **Perfect length** (`--policies ssjf`): the recorded output length as is. A
  predictor that is never wrong, so this is an upper bound on real SSJF.
- **Noisy** (`--policies ssjf-noisy`): the recorded length times a lognormal
  error with sigma 1.0 (`--ssjf-noise-sigma`). The error level is our
  assumption, not the paper's measured accuracy.

The paper's [repository](https://github.com/james-qiuhaoran/llm-serving-with-proxy-models)
has the training code, but we found no released weights, and its predictor was
trained on chatbot requests. A real SSJF baseline needs BERT-base retrained on
Nemotron calls; see [gpu-session.md](gpu-session.md).

### Perfect information (reference only)

`oracle` runs the program with the least remaining LLM service first, using the
recorded future. No real scheduler can do this. It shows how much room a
remaining-work predictor has under the same slot model; it is not a global
optimum, because tool waits are ignored in the ranking.

## Results

5 trials of 500 programs each, 8 slots. Times are simulated seconds. "Load" is
the offered LLM load with Poisson arrivals; "batch" means all programs arrive
at time 0.

| Load | Policy | Mean response (s) | p95 response (s) | p95 slowdown | Mean vs FCFS (s, 95% CI) |
| --- | --- | --- | --- | --- | --- |
| batch | FCFS | 2429.7 | 4249.7 | 105.2 | |
| batch | Agentix-style | 2235.9 | 4170.5 | 63.6 | -193.8 ± 59.8 |
| batch | SSJF (perfect length) | 1729.3 | 3889.6 | 75.5 | -700.4 ± 140.0 |
| batch | SSJF (noisy) | 1912.8 | 4016.8 | 111.3 | -516.9 ± 84.4 |
| batch | *Perfect information* | 1172.1 | 3623.5 | 17.7 | -1257.6 ± 106.4 |
| 0.5 | FCFS | 80.8 | 223.2 | 1.1 | |
| 0.5 | Agentix-style | 80.9 | 232.0 | 1.0 | +0.1 ± 0.2 |
| 0.5 | SSJF (perfect length) | 80.6 | 224.9 | 1.1 | -0.1 ± 0.2 |
| 0.5 | SSJF (noisy) | 80.7 | 225.0 | 1.1 | -0.1 ± 0.1 |
| 0.5 | *Perfect information* | 80.3 | 226.2 | 1.0 | -0.5 ± 0.6 |
| 0.8 | FCFS | 89.4 | 245.1 | 2.0 | |
| 0.8 | Agentix-style | 89.2 | 281.2 | 1.5 | -0.2 ± 1.3 |
| 0.8 | SSJF (perfect length) | 86.8 | 242.3 | 1.6 | -2.6 ± 2.2 |
| 0.8 | SSJF (noisy) | 87.5 | 238.6 | 1.6 | -2.0 ± 1.9 |
| 0.8 | *Perfect information* | 83.2 | 241.0 | 1.2 | -6.2 ± 3.9 |
| 0.95 | FCFS | 157.9 | 419.1 | 5.0 | |
| 0.95 | Agentix-style | 157.2 | 700.3 | 3.5 | -0.7 ± 9.3 |
| 0.95 | SSJF (perfect length) | 127.3 | 398.3 | 2.7 | -30.6 ± 27.1 |
| 0.95 | SSJF (noisy) | 143.7 | 451.0 | 4.2 | -14.2 ± 7.9 |
| 0.95 | *Perfect information* | 105.1 | 334.5 | 1.7 | -52.8 ± 42.8 |

### Findings

- With everything arriving at once, the Agentix-style policy cuts mean response
  time by about 8% against FCFS (2429.7 s to 2235.9 s). Under streaming arrivals
  at loads 0.5 to 0.95, the difference in mean response time is within the
  confidence interval.
- It lowers p95 slowdown at every load, so short programs wait less, but long
  programs wait more: p95 response time rises from 419.1 s to 700.3 s at load 0.95.
- Perfect-length SSJF cuts mean response time by about 29% in batch (to
  1729.3 s) and noisy SSJF by about 21% (to 1912.8 s), both more than the
  Agentix-style policy. Under streaming arrivals SSJF's gain is outside the
  confidence interval only at loads 0.8 and 0.95, and it is small at 0.8.
- Noise costs SSJF about a quarter of its batch gain and about half of its gain
  at load 0.95, and noisy SSJF's p95 slowdown in batch (111.3) is worse than
  FCFS (105.2). These SSJF numbers use recorded lengths, so a trained predictor
  can only do as well as or worse than the perfect-length rows.
- Perfect information cuts mean response time by about half in batch and a third
  at load 0.95. That gap is the room a remaining-work predictor has.

## Assumptions behind the numbers

- **Data:** the first 1,500 rows of each of the three Nemotron files
  (`search`, `interactive_agent`, `tool_calling`), giving 4,209 complete
  programs. This is not a random sample, and the three subsets are mixed
  equally, not in the corpus's proportions.
- **Token speeds:** assumed 5,000 tokens/s prefill and 50 tokens/s decode. These
  are placeholders until measured on a GPU.
- **Tool latency:** lognormal with a 1 s median and sigma 1, per tool result.
- **User think time:** 0 s. A follow-up user message arrives immediately.
- **Server:** 8 slots, each serving one call at full speed. Batching slowdown is
  not modelled.
- **Preemption:** free. No KV-cache eviction or recomputation cost.
- **Sampling:** 5 trials of 500 programs drawn from the pool, seed 585.
- **Policy:** thresholds of 2 s, doubling, 8 levels; no anti-starvation rule.
- **SSJF:** ranks by recorded output length; the noisy version multiplies it by
  lognormal error with sigma 1.0, drawn once per call and shared across loads.

## Reproduce

```bash
python3 -m venv venv && source venv/bin/activate
python -m pip install -r requirements.txt

# First 1,500 rows of each file (about 220 MB); no Hugging Face token needed.
mkdir -p data/raw/slice
for f in search interactive_agent tool_calling; do
  curl -sL "https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2/resolve/main/data/$f.jsonl" \
    | head -n 1500 > data/raw/slice/$f.jsonl
done

python scripts/prepare_data.py data/raw/slice data/processed/slice-v2
python scripts/simulate.py data/processed/slice-v2 --output data/results/slice-v2-baseline-ssjf.json
```

Every assumption above is a flag; see `python scripts/simulate.py --help`.
Output paths must be new. `data/` is gitignored.

## Replacing the assumed token speeds

`scripts/profile_llm.py` measures the two rates from a running
OpenAI-compatible server such as vLLM. On a GPU node:

```bash
vllm serve <model> --port 8000
python scripts/profile_llm.py --model <model> --concurrency 8
```

It sends streaming completions with synthetic prompts of several lengths and
prints `--prefill-tps` and `--decode-tps` values to pass to `simulate.py`. Set
`--concurrency` to the simulator's `--slots` so the per-request rates include
batching slowdown at that occupancy. The script has been tested only against a
local mock server, not yet against a real one.
