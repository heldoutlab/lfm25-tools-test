# Is LFM2.5-24B-A2B competitive with Qwen3.6-35B-A3B at tool calling?

Pre-registered test by [Held Out Lab](https://heldoutlab.substack.com): independent tests of AI model claims, with the rules fixed before the run.

**Status: done.** Verdict under the pre-registered rule: **not competitive.** Qwen3.6-35B-A3B solved 68 of 75 tasks (90.7%), LFM2.5-24B-A2B 37 (49.3%); gap −41.3 points, 95% interval [−52, −31]. Harder supplement: 64% vs 20%. Write-up: [heldoutlab.substack.com](https://heldoutlab.substack.com)

Everything is here: the pre-registration and amendment ([`PREREG.md`](PREREG.md)), all public tasks, the harness, every raw model output (`results/`) and the analysis (`results/report.json`, `results/report.txt`). The private set's results are summarised in the write-up; its tasks stay private.

### Re-run it

With llama.cpp (`brew install llama.cpp`) and Python 3.12:

```bash
python3.12 fetch.py
mv ~/models/lfm25-test ~/models/lfm25-tools-test
cd harness
python3.12 run.py --arm A --run main --server-bin "$(which llama-server)"
python3.12 run.py --arm B --run main --server-bin "$(which llama-server)"
python3.12 analyze.py
```

`fetch.py` downloads both models and checks their sha256; the `mv` fixes the folder-name mismatch recorded in `PREREG.md`. The harness is pinned to llama.cpp build b29c606e2; `--server-bin` runs it on your build and records which one. Results from other hardware, other builds or other quantizations (LFM2.5 at Q8_0 or BF16 especially) are welcome as issues.

- **Claim (the vendor's wording):** LFM2.5-24B-A2B is competitive with Qwen3.6-35B-A3B on tool use.
- **Models:** both at Q4_K_M, on stock llama.cpp, on a 32 GB M1 Pro MacBook.
- **Tasks:** [`items_public/tool_items.py`](items_public/tool_items.py), 75 fresh multi-step tool-calling tasks in five mock environments, each graded by code on the final state, plus 6 calibration tasks. A separate private set of 40 tasks is not published; only its hash is.
- **Amendment 1 (2026-10-23, before any test model saw a public task):** a harder supplement, [`items_public/tool_items_hard.py`](items_public/tool_items_hard.py), 25 tasks plus 3 calibration tasks, added after the pilot showed a ceiling risk. The verdict stays on the original 75; the supplement is a pre-registered secondary analysis. Details, pilot results and deviations are in `PREREG.md`.
- **Harness:** [`harness/`](harness/).

To check the frozen hashes against the files:

```bash
grep -E '^    [0-9a-f]{64}  ' PREREG.md | grep -v items_private | sed 's/^    //' | tail -r | awk '!seen[$2]++' | shasum -a 256 -c
```

This checks each file against its most recent recorded hash (three harness files were updated by Amendment 1; their original hashes remain in `PREREG.md`).

Questions or corrections: hello@heldoutlab.com. [Subscribe](https://heldoutlab.substack.com/subscribe) for the next tests.

Code is MIT-licensed (see `LICENSE`); task texts and results are CC BY 4.0. Tasks, graders and harness were written with help from Claude (Anthropic).
