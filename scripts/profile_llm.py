#!/usr/bin/env python3
"""Measure prefill and decode speed of an OpenAI-compatible server (e.g. vLLM).

Run on the GPU node after starting the server, for example:

    vllm serve <model> --port 8000
    python scripts/profile_llm.py --model <model> --concurrency 8

It sends streaming completions with synthetic prompts of several lengths at a
fixed concurrency and reports the two rates that scripts/simulate.py takes as
--prefill-tps and --decode-tps:

* prefill tokens/s: least-squares slope of time-to-first-token against prompt
  tokens (the intercept absorbs fixed per-request overhead).
* decode tokens/s per request: output tokens after the first, divided by the
  time from the first token to the last.

Set --concurrency to the simulator's --slots so the per-request rates include
batching slowdown at that occupancy. Standard library only.
"""

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

WORDS = "the scheduler measures remaining work after every agent step and tool call ".split()


def one_request(base_url, model, prompt_words, max_tokens):
    prompt = " ".join(WORDS[i % len(WORDS)] for i in range(prompt_words))
    body = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "min_tokens": max_tokens, "ignore_eos": True, "temperature": 0.0,
        "stream": True, "stream_options": {"include_usage": True},
    }).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    first = last = None
    usage = None
    with urllib.request.urlopen(request, timeout=600) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            chunk = json.loads(line[5:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            if any(choice.get("text") for choice in chunk.get("choices", [])):
                last = time.perf_counter()
                first = first or last
    if first is None or usage is None:
        raise RuntimeError("the server returned no tokens or no usage; is it vLLM-compatible?")
    return {"prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "ttft_s": first - start, "decode_s": last - first}


def slope(xs, ys):
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    return (sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
            / sum((x - mean_x) ** 2 for x in xs))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--prompt-words", type=int, nargs="+",
                        default=[250, 500, 1000, 2000, 4000, 8000])
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=3, help="batches per prompt length")
    parser.add_argument("--output", help="write raw measurements to this JSON file")
    args = parser.parse_args()

    one_request(args.base_url, args.model, 50, 8)  # warm up
    rows = []
    with ThreadPoolExecutor(args.concurrency) as pool:
        for words in args.prompt_words:
            for _ in range(args.rounds):
                rows += pool.map(lambda _: one_request(args.base_url, args.model, words,
                                                       args.max_tokens), range(args.concurrency))
    prefill_tps = 1.0 / slope([r["prompt_tokens"] for r in rows], [r["ttft_s"] for r in rows])
    decode_tps = statistics.median(
        (r["completion_tokens"] - 1) / r["decode_s"] for r in rows if r["decode_s"] > 0)
    print(f"{len(rows)} requests at concurrency {args.concurrency}, "
          f"prompts of {min(r['prompt_tokens'] for r in rows)}-"
          f"{max(r['prompt_tokens'] for r in rows)} tokens")
    print(f"--prefill-tps {prefill_tps:.0f} --decode-tps {decode_tps:.1f}")
    if args.output:
        with open(args.output, "x") as handle:
            json.dump({"settings": vars(args), "prefill_tps": prefill_tps,
                       "decode_tps": decode_tps, "requests": rows}, handle, indent=2)


if __name__ == "__main__":
    main()
