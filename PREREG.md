# Pre-registration — LFM2.5-24B-A2B vs Qwen3.6-35B-A3B at tool calling

Written 2026-10-22; rule, prediction and the decisions below fixed 2026-10-23; **frozen 2026-10-23 before any test model was run** (hashes at the end). Pushed to the public repository before the pilot; GitHub's push time is the timestamp.
Brief: `../../briefs/2026-10-19-brief.md`. Public repo for this test: **github.com/heldoutlab/lfm25-tools-test**.

All facts below were read on 2026-10-22 unless a date says otherwise. UNVERIFIED marks anything not confirmed at a primary source.

## Claim under test

The brief's wording: "LFM2.5-24B-A2B beats Qwen3.6-35B-A3B at tool calling and agent work, with fewer active parameters (2B vs 3B) and much faster."

What the vendor actually says:
- Model card (https://huggingface.co/LiquidAI/LFM2.5-24B-A2B): "Competitive with Qwen3.6-35B-A3B, a model with ~50% more total and active parameters, on agentic and tool-use tasks, scoring 48.09 on LiveClaw versus 38.19." And: "Runs up to 1.7× faster than comparable MoEs on-device, with 184 tok/s on an Apple M5 Max".
- Launch post, 2026-10-13 (https://www.liquid.ai/blog/lfm2-5-24b-a2b): it "leads or matches the competition on five of the eight tool-use and agentic benchmarks".

So the vendor says "competitive" and "leads or matches", not "beats". "Beats" is how the brief and the coverage put it. The post should test the vendor's words and say which wording it tests.

**Wording tested (author's decision, 2026-10-22):** the vendor's own — LFM2.5-24B-A2B is *competitive with* Qwen3.6-35B-A3B at tool calling. "Beats" is not tested.

Two smaller corrections to the brief:
- The model card gives **2.45B** active parameters (24.0B total), not 2B.
- The vendor's accuracy numbers come from **vLLM**, with settings that change per benchmark. Only the speed numbers use llama.cpp and Q4_K_M ("Collected on Pipette ... with llama.cpp, Q4_K_M quantization, and a 4K-token input context", model card). The card does not say which Qwen GGUF the speed chart used.

### Vendor's table, tool-use and agentic rows (model card, read 2026-10-22)

| Benchmark | LFM2.5-24B-A2B | Qwen3.6-35B-A3B | Vendor setting |
|---|---|---|---|
| AutomationBench | 27.69 | 8.72 | temperature 0.6, top_p 0.95, 50 turns |
| BFCLv4 | 65.01 | 65.00 | temperature 0.001, 4096 output tokens |
| ToolSandbox | 80.71 | 75.02 | temperature 0, 1024 output tokens, GPT-4o user simulator |
| LiveClaw | 48.09 | 38.19 | temperature 1.0, 8192 output tokens |
| PinchBench | 83.18 | 89.93 | not stated ("other evals": temperature 0.6, 32768 tokens) |
| τ³-Bench Banking | 8.04 | 7.63 | temperature 0, GPT-5.2 user model |
| WebArena Verified | 26.75 | 37.62 | temperature 0.6, top_p 0.95, 40 actions |
| SWE-Bench Verified (OpenCode) | 51.00 | 66.20 | not stated ("other evals") |

All runs: vLLM, BF16 assumed (the card does not say the precision for accuracy runs — UNVERIFIED). LFM2.5 leads on 5 of these 8 rows; on BFCLv4 the lead is 0.01 points.

## Arms

Both arms are 4-bit Q4_K_M GGUF files, one model loaded at a time, on the same llama.cpp binary.

| Arm | Model | Repo / file | Bytes | GB | sha256 (= Hugging Face LFS oid) |
|---|---|---|---|---|---|
| A | Qwen3.6-35B-A3B | `bartowski/Qwen_Qwen3.6-35B-A3B-GGUF` / `Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf` | 22,285,080,192 | 22.29 | `b46fedd33e0bfb0cae308aa3c158d0a4b2c4a1d2185a1ed6f093cdaf39064772` |
| B | LFM2.5-24B-A2B | `LiquidAI/LFM2.5-24B-A2B-GGUF` / `LFM2.5-24B-A2B-Q4_K_M.gguf` | 14,526,258,464 | 14.53 | `d3334405b96228eb9908cdebb5c6194908f2da64282bda852afc3a3910c64e7d` |

- Neither file is split into parts. Total download: **36,811,338,656 bytes = 36.81 GB (34.28 GiB)**. The brief's estimates (15–16 GB and 22–23 GB) were close; LFM's file is smaller than estimated.
- Sources: `https://huggingface.co/api/models/<repo>/tree/main` (sizes and LFS oids). Repo commits: bartowski Q4_K_M uploaded 2026-05-20 (commit `6dd29f0ec4`, repo head `5c2410d715`); LiquidAI GGUF repo head `0912d33141` (2026-10-13).
- File headers (first 24 MB read with an HTTP range request; no weights downloaded): both report `general.file_type = 15` (Q4_K_M). LFM: architecture `lfm2moe`, tokenizer pre-type `lfm2`, 128,000-token vocabulary. Qwen: architecture `qwen35moe`, tokenizer pre-type `qwen35`.
- **Chat templates.** The LiquidAI GGUF carries a template byte-identical to the base model's `chat_template.jinja` (sha256 prefix `ea663864491d`). The bartowski Qwen GGUF carries a template byte-identical to `Qwen/Qwen3.6-35B-A3B/chat_template.jinja` (prefix `e84f32a23fdd`).
- **Why these two files.** Qwen publishes no GGUF of this model (its org lists only the safetensors and FP8 repos). The most-downloaded GGUF repo is `unsloth/Qwen3.6-35B-A3B-GGUF` (1,222,953 downloads in the last 30 days, vs 189,449 for bartowski), but unsloth has no plain Q4_K_M: its file is `UD-Q4_K_M` (22,134,528,992 bytes, sha256 `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61`), a dynamic mixed quant with a modified chat template (unsloth adds developer-role and tool-call "fixes"). bartowski's file is a plain Q4_K_M with the official template, which matches arm B (plain Q4_K_M, official template). For LFM the vendor's own repo is the most used (148,626 downloads in 30 days; unsloth 63,032; bartowski 5,846). The author may overrule the Qwen choice (OPEN-QUESTIONS.md).
- Not tested: LiquidAI's `LFM2.5-24B-A2B-QAD-Q4_0.gguf` (13,578,190,880 bytes), which the vendor says "was trained to recover the accuracy lost to Q4_0 quantization". The result covers Q4_K_M only.

### Licences
- LFM2.5-24B-A2B: **LFM Open License v1.0** (`LICENSE` in both the base and GGUF repos; same text). Apache-2.0-style grant. Commercial use is licensed only for entities under USD 10 million annual revenue. **Nothing restricts publishing evaluation results or benchmarks.** Trademark clause allows "reasonable and customary use in describing the origin of the Work".
- Qwen3.6-35B-A3B: **Apache 2.0**. No restriction on publishing results.

### Sampler and thinking, per each card
- **Qwen3.6** (card, "Best Practices"): thinking mode for general tasks: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 1.5, repetition_penalty 1.0. Thinking is on by default and can be turned off (`chat_template_kwargs: {"enable_thinking": false}`). The template keeps the reasoning of every assistant turn after the last user message ("interleaved thinking"); a `preserve_thinking` option keeps older reasoning too. This test: thinking on, card sampler above.
- **LFM2.5** (card and `generation_config.json`): temperature 0.1, top_k 50, repetition_penalty 1.05. Nothing else is set. The card: "pure reasoning model that always thinks before it answers"; the template ends every generation prompt with `<|im_start|>assistant\n<think>`. There is **no switch and no budget setting** in the template. This test: card sampler, with top_p 1.0 and min_p 0.0 sent explicitly (the "off" values transformers and vLLM use when nothing is set), because llama-server would otherwise apply its own top_p 0.95 and min_p 0.05.
- Both GGUFs also carry `general.sampling.*` metadata (LFM: temp 0.1, top_k 50, repeat penalty 1.05; Qwen: temp 1.0, top_p 0.95, top_k 20). The harness sends every field with every request so this metadata never decides anything.
- Known difference that cannot be removed: llama.cpp applies `repeat_penalty` to the last 64 tokens; transformers applies `repetition_penalty` to the whole context.

## Runtime

**Stock llama.cpp (ggml-org/llama.cpp), not the Prism fork.** Pinned (decided 2026-10-23, following Claude's recommendation that a newer build fixes neither parsing issue): the Homebrew build already installed, **0.4.1, build 10964, commit `b29c606e2`** (committed 2026-09-14; `llama-server --version` on this Mac, 2026-10-22). The author may pin a newer build instead (OPEN-QUESTIONS.md). Latest at the time of writing: release **v0.7.0** = nightly **b11669** (2026-10-15, commit `620d2225cc`; macOS arm64 archive `llama-b11669-bin-macos-arm64.tar.gz`, 12,191,582 bytes, sha256 `dbed55493d72bc1ba2babc8770d669c2289b5bc4ee9fb2a4dfc158d45b7d13fd`); newest nightly b11819 (2026-10-22). Homebrew offers 0.7.0 as an upgrade; nothing was upgraded.

What the source says (read at tags b10964 and b11819 on GitHub):
- **LFM2.5 is supported for loading.** `lfm2moe` and tokenizer `lfm2` existed long before this model; the GGUF uses no new architecture or tokenizer type. UNVERIFIED on this Mac until the pilot loads it.
- **LFM2.5 tool calls are parsed natively.** `common/chat.cpp` detects the LFM2.5 template (it contains `List of tools: [` and no `<|tool_list_start|>`) and uses `common/parsers/lfm2.cpp`, a PEG parser for `<|tool_call_start|>[name(arg=value, ...)]<|tool_call_end|>` with optional `<think>` reasoning. It went in with PR #20251 (merged 2026-03-09) and the fix #21242 (2026-04-01). The parser at b10964 and at b11819 differs only by a refactor. The upstream test suite covers the LFM2.5-8B-A1B template, not the 24B template; the 24B template uses the same markers.
- **Qwen3.6 tool calls are parsed natively** by the Qwen3-Coder-style XML path (`<tool_call><function=…><parameter=…>`).

Open problems that bear directly on this test:
1. **#30490 (open, 2026-10-14)**: LFM2.5-24B-A2B "intermittently emits tool calls in the Qwen3-Coder-style XML shape as plain text"; the server returns them as content with no `tool_calls`, so the call is silently lost. The reporter saw it **after a run of tool errors**. Most turns parse correctly. https://github.com/ggml-org/llama.cpp/issues/30490
2. **#30118 (open, 2026-10-07)**: Qwen3.6 sometimes omits the opening `<tool_call>`; on builds after PR #27679 (merged 2026-08-25, **included in b10964**) the call leaks into content (9 of 100 requests in the reporter's test; 0 of 100 on b10433). A maintainer said a simple fallback was not usable and the issue awaits #29307. https://github.com/ggml-org/llama.cpp/issues/30118
3. **#26763 (closed as not planned, 2026-08-09)**: Qwen3.6 parameter values can capture the following call when the model omits a newline before `</parameter>`. Status in b10964 UNVERIFIED.
4. **#29002 (open)**: for LFM2-family hybrid models, co-batching corrupts the short-conv state; a comment (2026-09-29, LFM2.5-8B-A1B, one slot) reports 13.6% of multi-turn conversations corrupted with context checkpoints on and 5.0% with them off. Anecdotal; not reproduced. Qwen3.6 is also a hybrid model (Gated DeltaNet layers), so both arms use recurrent state and context checkpoints. https://github.com/ggml-org/llama.cpp/issues/29002

**Verdict on runtime support:** both models load and both native tool-call formats are parsed by mainline llama.cpp, including the installed Homebrew build. Neither parser is clean: each arm has an open issue where its calls leak into content as text. The tasks in this test produce tool errors on purpose (rules are enforced by the write tools), which is the trigger reported in #30490. The design answers this with a strict score (native parser only) and a lenient score that recovers the leaked shapes by fixed rules. No change after b10964 fixes either issue, so a newer pin would not remove the risk.

## Tasks

Written fresh for this test by two agents on 2026-10-22; none come from tests/bonsai2 (those are being published) and none have been shown to any model.
- **Public set:** 75 tasks, ids T01…, in `items_public/tool_items.py`. Published with the post.
- **Private set:** 40 tasks, ids T01…, in `items_private/tool_items.py`. Never published (git-ignored); used for calibration scores only, never in the verdict.
- **Calibration set:** 6 tasks, TP1…TP6, in `items_public/tool_items.py`. Used only by the pilot.
- Every task runs against a mock environment (no real systems; no banking, payments or money movement) and is graded by code on the final state. Each task file self-verifies: the reference action sequence passes, a trap sequence fails, the untouched state fails, and fuzzed calls never raise.
- No task is edited after the first test model sees it. A task later found broken (ambiguous prompt or wrong check) is dropped for both arms and listed in the write-up (`results/dropped_items.json`); it is never re-keyed in favour of an arm.
- The author reviews one environment's tasks critically before the freeze (PLAN.md).

## Settings

- Same for both arms: system prompt "You are a helpful assistant"; tools passed as OpenAI `tools` (the template from the GGUF renders them, `--jinja`); `parallel_tool_calls: true` sent explicitly (llama-server's default comes from each template's detected capabilities and could differ); tool choice auto; context 65,536; one slot (`-np 1`); `-ngl 99 -fa on`; output cap 16,384 tokens per assistant turn; at most 40 assistant turns per task (references need 7–16 calls).
- Reasoning: each turn's `reasoning_content` is sent back with that assistant turn inside the tool loop. Both templates render reasoning for turns after the last user message, so this is each vendor's intended default. Every task has a single user message.
- Seeds: main 1001, noise 2002, private 1001, pilot 7; each turn n uses seed + n.
- Arm A sampler: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 1.5, repeat_penalty 1.0, thinking on.
- Arm B sampler: temperature 0.1, top_k 50, repeat_penalty 1.05, top_p 1.0, min_p 0.0, presence_penalty 0.0.
- The rendered prompt of the first task of each arm is saved (`results/<run>/<arm>.rendered_prompt.txt`, via `/apply-template`), with the server command, `--version` output, binary sha256, `/props` build info and the template's sha256 in each manifest.
- The runner refuses to start if a model file's sha256 differs from the table above or the binary is not the pinned build.

## Runs

1. **Pilot** — calibration tasks TP1–TP6 only. Used to check that both models load, that a multi-turn tool loop renders and parses, the memory fit, the timing, and the task difficulty. May be run first with a model outside the test (`--outside`) and then with arms A and B. No test task is used.
2. **Main** — public set, both arms, seed 1001.
3. **Noise** — public set again, both arms, seed 2002. If the pilot projects too much machine time, the noise run uses the pre-registered 20-task subset (the first 20 public tasks in sha256-of-id order) instead.
4. **Private** — private set, both arms, seed 1001. Calibration scores only; results stay in `results/private/`.

## Scoring

- **Strict:** final state correct and no repair of any kind was needed.
- **Lenient:** final state correct, where the runner may (a) repair malformed argument JSON by fixed rules; (b) recover calls the server left in the reply text, in three fixed shapes tried in order for both arms — JSON in `<tool_call>` tags or a json fence (unchanged from bonsai2), Qwen3-Coder XML with or without the opening `<tool_call>` (the shape in #30118 and #30490), and the LFM Pythonic list with or without the special tokens (a bare list counts only when the whole reply is the list); (c) retry once after a server error. Recovery reads only the visible reply, never the reasoning. Values are parsed as literals; nothing is ever evaluated.
- Reported per arm: strict and lenient percent solved, overall, per environment and per difficulty label; paired difference B − A with a stratified (by environment) bootstrap 95% interval, 10,000 resamples; strict minus lenient for each arm.
- Speed: wall-clock seconds and completion tokens per task (mean and median) and **per solved task** (total over all tasks divided by tasks solved; strict and lenient); the ratio of seconds per solved task A ÷ B with a paired bootstrap interval; decode and prompt tokens per second from the server's timings.
- Failure counts: repair events by kind (each recovered text shape counted separately), tasks needing any repair, server errors, tasks with a turn cut off at the output cap, tasks still calling tools at the turn limit, and tool errors returned by the environments.
- `harness/regrade.py` re-runs every frozen check on the saved final states before analysis and must report "ALL SCORES REPRODUCE".

## Pass/fail rule — OPTIONS (the author chooses one before anything runs)

Gap = arm B (LFM2.5) − arm A (Qwen3.6), percentage points, public set, main run. The three candidates are the brief's, adapted only where the brief referred to things this test no longer has. `harness/analyze.py` implements all three and applies only the one set in `config.RULE`.

1. **"Beats Qwen at tool calls" (strict score).** Holds if LFM2.5 is ahead or within 3 points overall. Fails if it trails by 10 points or more and the interval excludes zero. Anything in between is "not resolved". *Adaptation needed:* the brief added "and ahead on round 2"; there are no rounds now. Either drop that clause or replace it with "and ahead on the tasks labelled very hard" (`config.RULE1_SUBSET_DIFFICULTIES`).
2. **Combined claim.** Holds if LFM2.5 is within 5 points of Qwen on accuracy (strict; gap ≥ −5) **and** at least 1.5× faster in seconds per solved task. *As written in the brief this has no "not resolved" band and no interval;* the author may want to add one (for example: fails only if the speed ratio's interval lies below 1.5 or the gap is −10 or worse with the interval excluding zero).
3. **Parser-fair rule.** Judge on the lenient score, with option 1's thresholds. Any strict-versus-lenient gap is reported as a tooling finding, not a model finding. *The brief gave no thresholds for this option;* option 1's are used unless the author writes others.

**Chosen rule (author's decision, 2026-10-22): option 3, parser-fair.** Judged on the lenient score with option 1's thresholds: *competitive* (claim holds) if LFM2.5 is ahead of Qwen3.6 or within 3 points overall; *not competitive* if it trails by 10 points or more and the 95% interval excludes zero; *not resolved* otherwise. Option 1's "round 2" clause is dropped (`config.RULE1_SUBSET_DIFFICULTIES = []`). The strict-versus-lenient gap for each model is reported separately as a finding about llama.cpp's tool-call parsing, not about the models.

## Noise floor

For each arm, the absolute change in percent solved between seed 1001 and seed 2002 on the re-run tasks, strict and lenient, plus the number of tasks whose result flipped. The floor is the larger of the two arms. Arm B runs at temperature 0.1 and arm A at 1.0, so arm A's run-to-run change will likely be larger, and the floor will mostly reflect Qwen.

## Sample-size limits, in plain words

With N paired tasks, the 95% interval on the gap is roughly ±196 × √(d / N) points, where d is the share of tasks on which the two models disagree (one solves, the other does not).

| Public tasks | d = 0.20 | d = 0.35 |
|---|---|---|
| 40 | ±14 points | ±18 points |
| 64 | ±11 points | ±14 points |
| 80 | ±10 points | ±13 points |

The brief's "±10 points with 64 tasks" holds only if the models disagree on fewer than 1 task in 6. In plain words: with about 40 public tasks this test can show a gap of roughly 15–20 points; it cannot show a 10-point gap with confidence, and it cannot settle the vendor's 5.7-point ToolSandbox gap or the 0.01-point BFCLv4 gap. Under option 1, a 10-point deficit will usually come out "not resolved", not "fails". The private set does not add power to the published result. The post must say this.

## Time and cost (estimates — UNVERIFIED until the pilot)

- Task-runs: 6 calibration × 2 arms + N_PUBLIC × 2 arms × 2 seeds + N_PRIVATE × 2 arms.
- Minutes per task: guess 2–10 (references need 7–16 calls, every turn thinks, reasoning is kept in context). For reference, bonsai2's dense Qwen3.8 27B took 251 s per round-1 tool task at ~5 turns; these MoEs have 2.45B and ~3B active parameters but these tasks are about twice as long.
- Example with N_PUBLIC = N_PRIVATE = 40: 252 task-runs → about 8 hours at 2 minutes, 21 hours at 5, 42 hours at 10. Within PLAN.md's 2–5 days per test.
- Cost: $0 in money. Downloads: 36.81 GB. Free disk on 2026-10-22: 72 GiB.
- If too slow: cut the noise run to the 20-task subset first; never cut the private run's arms unevenly.

## Risks and how the design answers them

- **Tool-call parsing (both arms).** Open issues #30490 (LFM2.5 calls as XML text) and #30118 (Qwen3.6 calls without the opener) predict calls lost to content. Strict and lenient scores are reported side by side; every recovered call is counted by shape; the rendered prompt and raw replies are saved. If the strict score collapses for one arm and the lenient score does not, that is a tooling finding and the post says so.
- **Memory (arm A).** The Qwen file is 22.29 GB on a 32 GB M1 Pro. With a 65,536-token context (about 1.3 GB of KV cache for either model: 10 full-attention layers × 20,480 bytes per token) and compute buffers it likely exceeds macOS's default GPU working-set limit (about two-thirds of RAM, ~22.9 GB — UNVERIFIED; the server log prints the real value). The pilot checks this before anything else. **Fixed remedy (decided 2026-10-23, following Claude's recommendation):** if arm A does not load or fit at 65,536 tokens, the context is lowered **for both arms** to the largest of 49,152 or 32,768 that arm A fits, before the main run; the model files, quantization and all other settings stay the same. Tasks that overflow the smaller context count as failed in both scores for both arms, and the overflow count is reported. The change is recorded under "Deviations after freezing".
- **Hybrid-model state corruption (#29002).** Both arms are hybrid. Sending reasoning back keeps each new prompt identical to what the model generated, so the server rarely rolls back to a checkpoint. The pilot looks for reasoning tags in content and replies that stop mid-sentence.
- **Wrong template or sampler.** Templates are the official ones (hashes above); every sampler field is sent explicitly; the rendered prompt is logged.
- **Wrong quant / runtime.** The result covers 4-bit Q4_K_M under llama.cpp on a laptop, the vendor's own setting for its speed claim. It does not cover vLLM at BF16, where the vendor measured accuracy, or LFM's QAD Q4_0 file.
- **"Your tasks aren't AutomationBench."** Intended: fresh tasks that cannot be in either model's training data. The vendor's table is quoted as the known result.
- **Speed measured fairly.** Time per solved task, not tokens per second, because LFM2.5 always thinks and thinking length differs between models. Both run one at a time on an otherwise idle machine; the machine must not be used during runs.
- **Low temperature for LFM.** Arm B runs at 0.1 (its card), arm A at 1.0 (its card). This follows each vendor and is reported; the noise run shows how much each arm moves between seeds.
- **Context overflow.** Long trajectories with reasoning kept can exceed 65,536 tokens; the server then errors, the task counts as failed in both scores, and the count is reported per arm.
- **Ceiling or floor.** Unknown difficulty for these new tasks; the pilot on the 6 calibration tasks (and an outside model) is the check. Results are also reported per difficulty label.
- **Same author for tasks and harness.** Tasks and checks are fixed and hashed before any run; the public set and all transcripts are published.
- **Crowded lane.** One vendor claim, one skill, tested on a laptop. Keep the post that narrow.

## Prediction (author writes this before anything runs)

**What do you expect to happen?** (written by the author, 2026-10-23)

I expect LFM2.5 to score lower by about 20 points on the lenient score, because these tasks need 7–16 dependent calls against 14-tool catalogs with policy rules, longer and stricter chains than I think the vendor's benchmarks stress.

**What result would prove you wrong?** (written by the author, 2026-10-23)

What would prove me wrong is the gap is 5 points or lower especially in complex tasks.

*Definition fixed with the author before any run:* "complex tasks" are the public tasks whose reference solution needs **12 or more tool calls** (43 of the 75). `harness/analyze.py` reports the gap, with its interval, separately for these and for the other 32 (`COMPLEX_MIN_CALLS = 12`). The prediction is not part of the pass/fail rule; it is scored in the write-up.

## Frozen hashes

Frozen 2026-10-23, before any test model was run (sha256). The private set is not published; its hash is, so it can be shown later that it did not change.

    c947665e6e4f0fe349e01b365f94e5f688a1ee462465f84a942c93c851817f2e  items_public/tool_items.py
    dad6591c786eee2e0f6e7f62887aafce3204847f4414a5461f4864805600191a  items_private/tool_items.py
    1b2733d04ac9e235691141a333d7bec76d459cd7dba3b6280abc12d9783376b0  fetch.py
    ece8b4eb26f69c852342fbf2302ac66f2e7a395f6acc741cebe43dd6a6214f00  harness/analyze.py
    024ee93a51afbed372e0aac38e4884bf8a1281c47ea306ffd5c0510fa74f684b  harness/config.py
    2956857e54e4d5f2b20e0bc1098b112e5c88ffa7d4b01cd4c689a0eabc0f3200  harness/grade.py
    bf1cb27b02f21a9aff68aeafae5f7a3a918bd6ea05c08867db036fdd0cb547b4  harness/regrade.py
    8407fb7ea56d3764f17f0a2e25036ef90329638f58210fcbff4e4dce9d1c79fc  harness/run.py

## Amendment 1 — 2026-10-23, pushed before either test model saw any public test task

**What happened after the freeze.**
1. **Order of steps (my error).** The lab's rule is to calibrate difficulty on a model outside the comparison *before* freezing. This pre-registration was frozen and pushed first and calibrated second. The public test tasks T01–T75 are unchanged.
2. **Model folder.** `fetch.py` saved the models to `~/models/lfm25-test`, but the harness expects `~/models/lfm25-tools-test`. The folder was renamed; no frozen file changed. Both model files match their pre-registered sha256.
3. **Pilot on the six calibration tasks (TP1–TP6).** Outside model Muse Glimmer 30B: 6 of 6. Arm A (Qwen3.6): 6 of 6. Arm B (LFM2.5): 3 of 6. No tool-call repairs were needed on either arm; no truncations or server errors. Arm A loads and runs at the full 65,536-token context, so the memory fallback is not used. About 3.5 minutes per task (A) and 3 (B).
4. **Calibration on the separate private set** (40 tasks, never published; calibration only): arm A solved more than 90%, which signals a ceiling risk for arm A on the public set.

**What is added.** A harder public supplement, written after the pilot and before any test model saw it: **25 tasks H01–H25** plus **3 calibration tasks HP1–HP3**, in `items_public/tool_items_hard.py`. It reuses the same five environments (imported from the frozen `tool_items.py`, which is not modified). Every task needs 12–20 reference calls and combines at least four of: a rule found only through the policy tool, a mid-task rule error that forces re-planning, state that must be re-read after a write, a batch with look-alikes, a computed argument, ordering; four tasks (H03, H10, H12, H22) require deliberately not doing something the prompt makes tempting. Calibration on the outside model: Muse Glimmer 30B solved HP1–HP3 (3 of 3), taking about twice the turns and time of the original calibration tasks. Three tasks are too few to show the supplement is not also near the ceiling for arm A; the write-up will say so.

**How it is analysed (fixed now, before running).**
- **The verdict does not change.** The pass/fail rule above is applied to the original 75 public tasks exactly as frozen.
- **Secondary, pre-registered:** the same comparison (lenient and strict, with intervals) on the 25 supplement tasks, reported next to the verdict.
- **The author's prediction** ("especially in complex tasks") is scored on all complex tasks together: the original public tasks with 12 or more reference calls (43) plus all 25 supplement tasks (each needs 12 or more), 68 tasks in total (`complex_all` in `harness/analyze.py`).
- **Runs:** `hard` (supplement, both arms, seed 1001) is added. The noise re-run stays on the original 75 tasks. Run order: main A, main B, hard A, hard B, then noise A and B.

**Files added or changed by this amendment (sha256).** The frozen `items_public/tool_items.py` (c947665e…) and `fetch.py` are unchanged.

    536640c6c47268ba108af0c01dde4321ec510a4aef4f464fcfa961cbbc4e74fa  items_public/tool_items_hard.py
    79d599766952efde09b74f074bf7cd8d967491014697203b8e5d54cc061ea2e1  harness/config.py
    34f9147b38da0bd9ac22024e38fd5713eed3b6c96e19f9a4739931c82d545c80  harness/run.py
    0f884ffe20ba054ab8d5c87757096bc5dfe8c0b85dd626e996f12c604af60b7f  harness/analyze.py

## Deviations after freezing

(None yet.)
