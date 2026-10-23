#!/opt/homebrew/bin/python3.12
"""Run one arm on one task set against a local llama-server. Resumable: finished tasks are skipped.

  run.py --arm A --run pilot          # calibration tasks TP1.. only (timing, harness and template checks)
  run.py --outside X --model-path M --run pilot   # calibration tasks on a model that is not a test arm
  run.py --arm B --run main           # the public set, seed 1001
  run.py --arm B --run noise          # the public set again, seed 2002 (--fallback: the 20-task subset)
  run.py --arm A --run private        # the private set, seed 1001; results stay in results/private (never published)

Adapted from tests/bonsai2/harness/run.py: tools family only, two task files, reasoning kept inside the tool loop,
sampler sent in full with every request, rendered prompt logged, speed and malformed-call counts recorded.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
import grade

# ---------- tasks ----------


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_set(which):
    """Returns ({task_id: task}, ENVS, module) for "public", "private" or "hard". Each task carries its own ENVS."""
    path = {"public": cfg.ITEMS_PUBLIC, "private": cfg.ITEMS_PRIVATE, "hard": cfg.ITEMS_PUBLIC_HARD}[which]
    mod = load_module(path, "tool_items_" + which)
    tasks = {}
    for t in mod.TASKS:
        if t["id"] in tasks:
            raise SystemExit("duplicate task id %s in %s" % (t["id"], path))
        tasks[t["id"]] = t
    return tasks, mod.ENVS, mod


def is_calibration(task_id):
    return task_id.startswith(("TP", "HP"))


def select(run, fallback=False):
    """Returns (ordered ids, tasks, envs). Calibration tasks (TP*, HP*) are used only by the pilots."""
    which = "private" if run == "private" else "hard" if run in ("hard", "hard_pilot") else "public"
    tasks, envs, _ = load_set(which)
    ids = [i for i in tasks if is_calibration(i) == (run in ("pilot", "hard_pilot"))]
    order = cfg.stable_order(ids)
    if run == "noise" and fallback:
        order = order[: cfg.NOISE_FALLBACK_N]
    return order, tasks, envs


# ---------- server ----------


def http(path, body=None, timeout=60):
    url = "http://127.0.0.1:%d%s" % (cfg.PORT, path)
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


SAMPLER_FLAGS = {"temperature": "--temp", "top_p": "--top-p", "top_k": "--top-k", "min_p": "--min-p",
                 "repeat_penalty": "--repeat-penalty", "presence_penalty": "--presence-penalty"}


def start_server(model_path, server_bin, sampler, log_path):
    cmd = [server_bin, "-m", model_path, "--host", "127.0.0.1", "--port", str(cfg.PORT),
           "-ngl", "99", "-fa", "on", "-c", str(cfg.CONTEXT), "-np", "1", "--jinja"]
    for key, flag in SAMPLER_FLAGS.items():  # server defaults match the request, so nothing differs if a field is dropped
        if key in sampler:
            cmd += [flag, str(sampler[key])]
    log = open(log_path, "a")
    log.write("\n# %s\n# %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), " ".join(cmd)))
    log.flush()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    deadline = time.time() + 900
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit("llama-server exited during load; see " + log_path)
        try:
            if http("/health", timeout=5).get("status") == "ok":
                return proc, cmd
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(3)
    proc.terminate()
    raise SystemExit("llama-server did not become healthy in 15 minutes")


def server_version(server_bin):
    try:
        p = subprocess.run([server_bin, "--version"], capture_output=True, text=True, timeout=60)
        return (p.stdout + p.stderr).strip()[-500:]
    except (OSError, subprocess.TimeoutExpired) as e:
        return "unavailable: %s" % e


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------- one chat turn ----------


def chat(arm, messages, seed, tools):
    body = {"model": "local", "messages": messages, "seed": seed, "max_tokens": cfg.MAX_TOKENS, "stream": False,
            "tools": tools, "parallel_tool_calls": cfg.PARALLEL_TOOL_CALLS}
    body.update(arm["sampler"])
    body.update(copy.deepcopy(arm.get("extra_body", {})))
    t0 = time.time()
    try:
        resp = http("/v1/chat/completions", body, timeout=cfg.REQUEST_TIMEOUT_S)
    except urllib.error.HTTPError as e:
        return {"http_error": e.code, "error_body": e.read().decode(errors="replace")[:2000], "wall_s": round(time.time() - t0, 1)}
    choice = resp["choices"][0]
    msg = choice["message"]
    return {
        "content": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content") or "",
        "tool_calls": msg.get("tool_calls") or [],
        "finish_reason": choice.get("finish_reason"),
        "usage": resp.get("usage", {}),
        "timings": resp.get("timings", {}),
        "wall_s": round(time.time() - t0, 1),
    }


def base_messages(prompt):
    return [{"role": "system", "content": cfg.SYSTEM_PROMPT}, {"role": "user", "content": prompt}]


def tokens(turns):
    return sum(t.get("usage", {}).get("completion_tokens", 0) for t in turns)


def log_rendered_prompt(path, messages, tools, arm):
    """The prompt exactly as the server's template renders it (brief: log it for the first task of each model)."""
    body = {"messages": messages, "tools": tools}
    body.update(copy.deepcopy(arm.get("extra_body", {})))
    try:
        rendered = http("/apply-template", body).get("prompt", "")
    except Exception as e:
        rendered = "apply-template failed: %s" % e
    with open(path, "w") as f:
        f.write(rendered if isinstance(rendered, str) else json.dumps(rendered, indent=1))


# ---------- one task ----------


def run_tools(arm, task, envs, seed, rendered_path=None):
    """One trajectory. `correct` is the strict score (no repair was ever needed); `correct_lenient` is the
    final-state check regardless of repairs. Reasoning from this loop is sent back when cfg.ECHO_REASONING."""
    env = envs[task["env"]](copy.deepcopy(task["init"]))
    messages = base_messages(task["prompt"])
    turns, repairs = [], []
    truncated = server_error = hit_turn_limit = False
    tool_errors = 0
    for n in range(cfg.MAX_TOOL_TURNS):
        turn = chat(arm, messages, seed + n, env.TOOLS)
        if "http_error" in turn:  # known failure mode: server rejects or fails to parse a call. Lenient: one retry.
            turns.append(turn)
            repairs.append("http_%s_retry" % turn["http_error"])
            turn = chat(arm, messages, seed + n + 5000, env.TOOLS)
            if "http_error" in turn:
                turns.append(turn)
                server_error = True
                break
        turns.append(turn)
        truncated = truncated or turn["finish_reason"] == "length"
        calls = []
        for i, tc in enumerate(turn["tool_calls"]):
            fn = tc.get("function", {})
            args, repaired = grade.parse_arguments(fn.get("arguments"))
            if repaired:
                repairs.append("arguments_repaired")
            calls.append({"id": tc.get("id") or "call_%d_%d" % (n, i), "name": fn.get("name"), "arguments": args, "raw": fn.get("arguments")})
        if not calls:
            recovered, shape = grade.calls_from_text(turn["content"], env.TOOLS)
            if recovered:
                repairs.append("call_recovered_from_text_%s" % shape)
                calls = [{"id": "call_%d_%d" % (n, i), "name": c["name"], "arguments": c["arguments"], "raw": None} for i, c in enumerate(recovered)]
        if not calls:
            break
        if n == cfg.MAX_TOOL_TURNS - 1:
            hit_turn_limit = True  # still calling tools when the turn limit ran out (looping or slow)
        assistant = {"role": "assistant", "content": turn["content"], "tool_calls": [
            {"id": c["id"], "type": "function", "function": {"name": c["name"] or "", "arguments": json.dumps(c["arguments"] if c["arguments"] is not None else {})}}
            for c in calls]}
        if cfg.ECHO_REASONING and turn["reasoning"]:
            assistant["reasoning_content"] = turn["reasoning"]
        messages.append(assistant)
        for c in calls:
            if c["arguments"] is None:
                result = {"error": "arguments were not valid JSON"}
            else:
                try:
                    result = env.call(c["name"], c["arguments"])
                except Exception as e:  # envs are fuzzed never to raise; belt and braces
                    result = {"error": "tool failed: %s" % type(e).__name__}
            if isinstance(result, dict) and "error" in result:
                tool_errors += 1
            c["result"] = result
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(result)})
        turn["executed"] = calls
    if rendered_path:
        log_rendered_prompt(rendered_path, messages, env.TOOLS, arm)
    try:
        state_ok = bool(task["check"](env.state))
    except Exception as e:
        state_ok = False
        repairs.append("check_raised_%s" % type(e).__name__)
    return {"turns": turns, "truncated": truncated, "server_error": server_error, "hit_turn_limit": hit_turn_limit,
            "repairs": repairs, "tool_errors": tool_errors, "correct": state_ok and not repairs,
            "correct_lenient": state_ok, "grader": {"final_state": env.state}}


# ---------- main ----------


def main():
    ap = argparse.ArgumentParser()
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--arm", choices=sorted(cfg.ARMS))
    who.add_argument("--outside", default="", help="pilot only: name of a model that is not a test arm (needs --model-path)")
    ap.add_argument("--run", required=True, choices=["pilot", "main", "noise", "private", "hard", "hard_pilot"])
    ap.add_argument("--fallback", action="store_true", help="noise run only: the pre-registered 20-task subset")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-name", default="", help="results subfolder (default: the run name)")
    ap.add_argument("--model-path", default="", help="outside model, or smoke tests: override the arm's model file")
    ap.add_argument("--sampler", default="{}", help="outside model only: sampler as JSON (default: the GGUF's own)")
    ap.add_argument("--server-bin", default="", help="smoke tests only: override the server binary")
    ap.add_argument("--skip-hash", action="store_true")
    args = ap.parse_args()

    if args.outside:
        if args.run not in ("pilot", "hard_pilot") or not args.model_path:
            raise SystemExit("--outside is for the pilot only and needs --model-path")
        arm_name = "X-" + args.outside
        arm = {"label": "outside: " + args.outside, "sampler": json.loads(args.sampler), "extra_body": {}}
    else:
        arm_name = args.arm
        arm = cfg.ARMS[args.arm]
    seed = cfg.SEEDS[args.run]
    order, tasks, envs = select(args.run, args.fallback)
    if args.limit:
        order = order[: args.limit]

    base = cfg.PRIVATE_RESULTS_DIR if args.run == "private" else cfg.RESULTS_DIR
    out_dir = os.path.join(base, args.out_name or args.run)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "%s.jsonl" % arm_name)
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    todo = [i for i in order if i not in done]
    print("%s / %s: %d tasks, %d already done" % (arm_name, args.run, len(order), len(order) - len(todo)), flush=True)
    if not todo:
        return

    model_path = args.model_path or os.path.join(cfg.MODELS_DIR, arm["file"])
    server_bin = args.server_bin or cfg.SERVER_BIN
    version = server_version(server_bin)
    if cfg.SERVER_BUILD and not args.server_bin and cfg.SERVER_BUILD not in version:
        raise SystemExit("server binary is not the pinned build %s: %s" % (cfg.SERVER_BUILD, version))
    manifest = {"arm": arm_name, "label": arm["label"], "run": args.run, "seed": seed, "sampler": arm["sampler"],
                "extra_body": arm.get("extra_body", {}), "echo_reasoning": cfg.ECHO_REASONING,
                "parallel_tool_calls": cfg.PARALLEL_TOOL_CALLS,
                "max_tokens": cfg.MAX_TOKENS, "max_tool_turns": cfg.MAX_TOOL_TURNS, "context": cfg.CONTEXT,
                "model_path": model_path, "server_bin": server_bin, "server_version": version,
                "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    if not args.skip_hash:
        manifest["model_sha256"] = sha256_file(model_path)
        if not args.model_path and not args.outside and manifest["model_sha256"] != arm["sha256"]:
            raise SystemExit("model file hash does not match the pre-registered hash: " + model_path)
        manifest["server_sha256"] = sha256_file(server_bin)
    proc, cmd = start_server(model_path, server_bin, arm["sampler"], os.path.join(out_dir, "%s.server.log" % arm_name))
    manifest["server_cmd"] = cmd
    try:
        props = http("/props")
        manifest["props"] = {k: v for k, v in props.items() if k in ("build_info", "model_path", "default_generation_settings")}
        template = props.get("chat_template") or ""
        manifest["chat_template_sha256"] = hashlib.sha256(template.encode()).hexdigest()
    except Exception:
        pass
    with open(os.path.join(out_dir, "%s.manifest.jsonl" % arm_name), "a") as f:
        f.write(json.dumps(manifest) + "\n")

    rendered_path = os.path.join(out_dir, "%s.rendered_prompt.txt" % arm_name)
    try:
        for k, task_id in enumerate(todo, 1):
            t0 = time.time()
            first = not os.path.exists(rendered_path)
            rec = run_tools(arm, tasks[task_id], envs, seed, rendered_path if first else None)
            rec.update(id=task_id, env=tasks[task_id]["env"], arm=arm_name, run=args.run, seed=seed,
                       completion_tokens=tokens(rec["turns"]), wall_s=round(time.time() - t0, 1))
            with open(out_path, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            print("[%d/%d] %s strict=%s lenient=%s tokens=%d turns=%d %.0fs%s%s" % (
                k, len(todo), task_id, rec["correct"], rec["correct_lenient"], rec["completion_tokens"],
                len(rec["turns"]), rec["wall_s"], " TRUNCATED" if rec["truncated"] else "",
                " REPAIRS=" + ",".join(rec["repairs"]) if rec["repairs"] else ""), flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
