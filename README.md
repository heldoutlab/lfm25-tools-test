# Is LFM2.5-24B-A2B competitive with Qwen3.6-35B-A3B at tool calling?

Pre-registered test by [Held Out Lab](https://heldoutlab.substack.com): independent tests of AI model claims, with the rules fixed before the run.

**Status: pre-registered, not yet run.** The claim, the pass/fail rule, the author's prediction, the settings and the sha256 of every task and harness file are in [`PREREG.md`](PREREG.md), committed and pushed before any test model was run. Results, raw model outputs and the write-up will be added here when the runs finish.

- **Claim (the vendor's wording):** LFM2.5-24B-A2B is competitive with Qwen3.6-35B-A3B on tool use.
- **Models:** both at Q4_K_M, on stock llama.cpp, on a 32 GB M1 Pro MacBook.
- **Tasks:** [`items_public/tool_items.py`](items_public/tool_items.py), 75 fresh multi-step tool-calling tasks in five mock environments, each graded by code on the final state, plus 6 calibration tasks. A separate private set of 40 tasks is not published; only its hash is.
- **Harness:** [`harness/`](harness/).

To check the frozen hashes against the files:

```bash
grep -E '^    [0-9a-f]{64}  ' PREREG.md | grep -v items_private | sed 's/^    //' | shasum -a 256 -c
```

Questions or corrections: hello@heldoutlab.com. [Subscribe](https://heldoutlab.substack.com/subscribe) for the results.

Code is MIT-licensed (see `LICENSE`); task texts and results are CC BY 4.0. Tasks, graders and harness were written with help from Claude (Anthropic).
