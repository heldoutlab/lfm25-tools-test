"""Fixed settings for the LFM2.5 tool-calling test. Everything here is frozen by PREREG.md before the main run.

Adapted from tests/bonsai2/harness/config.py: tools family only, two arms, stock llama.cpp, a public and a private set.
"""
import hashlib
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ITEMS_PUBLIC = os.path.join(ROOT, "items_public", "tool_items.py")
ITEMS_PRIVATE = os.path.join(ROOT, "items_private", "tool_items.py")
# Amendment 2026-10-23 (PREREG.md): harder public supplement, a separate file so the frozen set is untouched.
ITEMS_PUBLIC_HARD = os.path.join(ROOT, "items_public", "tool_items_hard.py")
RESULTS_DIR = os.path.join(ROOT, "results")
PRIVATE_RESULTS_DIR = os.path.join(RESULTS_DIR, "private")  # git-ignored: never published
MODELS_DIR = os.path.expanduser("~/models/lfm25-tools-test")

# Stock llama.cpp, the same binary for both arms. Default: the Homebrew build already on this Mac
# (0.4.1, build 10964, commit b29c606e2, 2026-09-14). PREREG.md "Runtime" says which build is pinned; if the author
# pins another one, change this path and SERVER_BUILD before the freeze. The runner records the binary's sha256,
# its --version output and /props build_info in every manifest, and refuses to start if SERVER_BUILD is set and
# the --version output does not contain it.
SERVER_BIN = "/opt/homebrew/bin/llama-server"
SERVER_BUILD = "b29c606e2"

SYSTEM_PROMPT = "You are a helpful assistant"
CONTEXT = 65536  # long trajectories (references are 7-16 calls) and reasoning kept inside the tool loop
PORT = 8792

# Every sampler field is sent explicitly with every request, so neither llama-server's own defaults
# (top_p 0.95, min_p 0.05) nor the GGUF's general.sampling.* metadata silently fill a gap.
# Qwen3.6 card, "Thinking mode for general tasks" (Best Practices section, read 2026-10-22).
QWEN_SAMPLER = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                "presence_penalty": 1.5, "repeat_penalty": 1.0}
# LFM2.5 card / generation_config.json: temperature 0.1, top_k 50, repetition_penalty 1.05 and nothing else.
# top_p 1.0 and min_p 0.0 are the "off" values (what transformers and vLLM use when the card sets nothing).
# Note: llama.cpp applies repeat_penalty to the last 64 tokens only (repeat_last_n), transformers to the whole context.
LFM_SAMPLER = {"temperature": 0.1, "top_k": 50, "repeat_penalty": 1.05, "top_p": 1.0, "min_p": 0.0,
               "presence_penalty": 0.0}

ARMS = {
    "A": {
        "label": "Qwen3.6-35B-A3B Q4_K_M (bartowski, 22.29 GB)",
        "file": "Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf",
        "repo": "bartowski/Qwen_Qwen3.6-35B-A3B-GGUF",
        "size": 22285080192,
        "sha256": "b46fedd33e0bfb0cae308aa3c158d0a4b2c4a1d2185a1ed6f093cdaf39064772",
        "sampler": QWEN_SAMPLER,
        # Thinking on is the template default; sent explicitly so the setting is on record.
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
    },
    "B": {
        "label": "LFM2.5-24B-A2B Q4_K_M (LiquidAI official, 14.53 GB)",
        "file": "LFM2.5-24B-A2B-Q4_K_M.gguf",
        "repo": "LiquidAI/LFM2.5-24B-A2B-GGUF",
        "size": 14526258464,
        "sha256": "d3334405b96228eb9908cdebb5c6194908f2da64282bda852afc3a3910c64e7d",
        "sampler": LFM_SAMPLER,
        # Always thinks; the template has no switch.
        "extra_body": {},
    },
}

# Reasoning produced inside the current tool loop is sent back as `reasoning_content`. Both chat templates render
# thinking for assistant turns after the last user message (Qwen3.6's "interleaved thinking" default; the LFM2.5
# template's keep_thinking rule). Every task has one user message, so this keeps the whole loop's reasoning.
ECHO_REASONING = True
# llama-server's default for parallel_tool_calls comes from each template's detected capabilities, so it could differ
# between arms. Sent explicitly, the same for both. Both formats can carry several calls in one turn.
PARALLEL_TOOL_CALLS = True

MAX_TOKENS = 16384  # per assistant turn; a turn cut off here is recorded as truncated
MAX_TOOL_TURNS = 40  # references need 7-16 calls; room for exploratory reads and recovery from tool errors
REQUEST_TIMEOUT_S = 3 * 3600

SEEDS = {"main": 1001, "noise": 2002, "private": 1001, "pilot": 7, "hard": 1001, "hard_pilot": 7}
NOISE_FALLBACK_N = 20  # the noise run covers every public task; this subset is the pre-registered fallback


def stable_order(ids):
    """Fixed pseudo-random order that does not depend on anything run-time: sort by sha256 of the id."""
    return sorted(ids, key=lambda i: hashlib.sha256(i.encode()).hexdigest())


# ---------- pass/fail rule ----------
# The author picks one of the options in PREREG.md before anything runs and writes its number here ("1", "2" or "3").
# analyze.py applies only this rule; with None it prints the scores and no verdict.
RULE = "3"  # author's choice, 2026-10-22: parser-fair (lenient score, option 1 thresholds)
# Option 1 had an "and ahead on round 2" clause. There are no rounds in this test. If the author keeps a subset
# clause, list the difficulty labels it covers here (for example ["very hard"]); an empty list drops the clause.
RULE1_SUBSET_DIFFICULTIES = []
