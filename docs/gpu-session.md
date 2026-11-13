# GPU session plan (Nov 13, 2026)

The baseline in [baseline.md](baseline.md) uses assumed token speeds and no
trained predictor. This session replaces the assumed speeds with measured ones
and starts the real SSJF predictor.

## GPU node

- **Node:** one `c240g5` at CloudLab Wisconsin.
- **Reservation:** prepared for Nov 13, 7:00 PM to Nov 14, 7:00 PM, in the
  reserving user's local time as shown on the CloudLab form. **Confirm it was
  submitted and approved**; we have not verified either. The form says most
  reservations need administrator approval.
- **Start deadline:** an experiment must be started on the node within three
  hours of the start (by 10:00 PM on Nov 13) or the reservation is cancelled.
  Only the reserving user can start it. The end time is a hard deadline.
- **Profile:** use the project profile `c240g5-ubuntu24-nvidia-drivers`. Do
  **not** use `agentic-sched-data`: it always attaches the Utah dataset and
  cannot run at Wisconsin.

Hardware, from the [CloudLab hardware page](https://docs.cloudlab.us/hardware.html)
(read through a search result, not the live page; check `nvidia-smi` on the node):

| Part | c240g5 |
| --- | --- |
| GPU | One NVIDIA P100, 12 GB |
| CPU | Two Intel Xeon Silver 4114, 10 cores each, 2.20 GHz |
| Memory | 192 GB |

What this means for the session:

- **Model choice:** the served model, its KV cache, and the batch must fit in
  12 GB, so use a small model (a few billion parameters at most, fewer at full
  precision). Measured speeds will describe that model on this GPU, not a
  production server.
- **Open risk:** the P100 is an old GPU generation. Check on the node that the
  current vLLM and PyTorch builds still support it before anything else; if they
  do not, fall back to an older pinned release or another OpenAI-compatible
  server.

## Tasks, in order

1. Confirm the reservation, then start the experiment inside the three-hour window.
2. Run `nvidia-smi` and confirm the GPU and driver.
3. Install vLLM and serve a small model on port 8000.
4. Run `scripts/profile_llm.py` at several concurrency levels and save each output:

   ```bash
   for c in 1 4 8 16; do
     python scripts/profile_llm.py --model <model> --concurrency $c --output profile-c$c.json
   done
   ```

   The script has only been tested against a local mock server so far.
5. Rerun `scripts/simulate.py` with the measured `--prefill-tps` and
   `--decode-tps`, and `--slots` equal to the concurrency they were measured at.
6. Fine-tune a BERT-base output-length predictor on Nemotron calls, for a real
   SSJF baseline in place of the recorded-length stand-ins. The training script
   is still to be written. It needs a data slice on the node, which can be
   fetched from Hugging Face without a token (see the commands in
   [baseline.md](baseline.md)).
7. Copy every result off the node before it expires. There is no persistent
   dataset at Wisconsin, so anything left on the node is lost.

## Separate from the GPU work

The Utah dataset `agentic-sched-data` no longer appears in the owner's dataset
list. It must be recreated before full-corpus ingestion; see
[cloudlab.md](cloudlab.md).
