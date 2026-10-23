#!/opt/homebrew/bin/python3.12
"""Score the runs and apply the pre-registered rule mechanically. Written before any model was run.

  analyze.py            # prints the report and writes results/report.json (public set, both arms)
  analyze.py --private  # calibration scores on the private set; writes results/private/report_private.json only
  analyze.py --pilot    # descriptive numbers for the calibration tasks; writes results/pilot/report_pilot.json

Adapted from tests/bonsai2/harness/analyze.py: one family (tools), strata are environments, the gap is
B - A (LFM2.5 minus Qwen3.6, positive = LFM2.5 ahead), speed per solved task, malformed-call counts, and the
three candidate rules from the brief (only the one named in config.RULE is applied).
"""
import argparse
import json
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
import run as runner

BOOT = 10000

# Tasks found broken after the run are dropped for every arm (PREREG.md, "Tasks"); listed in results/dropped_items.json.
_dropped_path = os.path.join(cfg.RESULTS_DIR, "dropped_items.json")
DROPPED = set(json.load(open(_dropped_path))) if os.path.exists(_dropped_path) else set()


def load(run, arm, base=None):
    path = os.path.join(base or cfg.RESULTS_DIR, run, "%s.jsonl" % arm)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {r["id"]: r for r in map(json.loads, filter(str.strip, f)) if r["id"] not in DROPPED}


COMPLEX_MIN_CALLS = 12  # PREREG.md, "Prediction": a task counts as complex if its reference solution needs 12+ calls


def difficulties(which):
    """task id -> difficulty label, from the task dict or the module's DIFFICULTY table (either layout is accepted).
    Tasks without a label (the public set) are split by reference length, the objective definition of "complex"."""
    tasks, _, mod = runner.load_set(which)
    table = getattr(mod, "DIFFICULTY", {})

    def by_length(t):
        return "complex (%d+ calls)" % COMPLEX_MIN_CALLS if len(t["reference"]) >= COMPLEX_MIN_CALLS \
            else "standard (under %d calls)" % COMPLEX_MIN_CALLS
    return {i: t.get("difficulty") or table.get(i) or by_length(t) for i, t in tasks.items()}


def score(rec, lenient=False):
    return 1.0 if (rec.get("correct_lenient") if lenient else rec["correct"]) else 0.0


def pct(xs):
    return 100.0 * sum(xs) / len(xs) if xs else float("nan")


def acc(recs, ids, lenient=False):
    return pct([score(recs[i], lenient) for i in ids])


def strata(recs, ids, key="env", labels=None):
    out = {}
    for i in ids:
        k = labels[i] if labels is not None else recs[i].get(key, "?")
        out.setdefault(k, []).append(i)
    return out


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, max(0, int(p * len(v))))]


def paired_boot(x, y, groups, lenient=False, seed=0):
    """Stratified paired bootstrap over tasks. 95% interval for accuracy(x) - accuracy(y), in points."""
    rng = random.Random(seed)
    ids = [i for members in groups.values() for i in members]
    sx = {i: score(x[i], lenient) for i in ids}
    sy = {i: score(y[i], lenient) for i in ids}
    diffs = []
    for _ in range(BOOT):
        sample = [rng.choice(members) for members in groups.values() for _ in members]
        diffs.append(pct([sx[i] for i in sample]) - pct([sy[i] for i in sample]))
    return q(diffs, 0.025), q(diffs, 0.975)


def compare(x, y, ids, labels, lenient=False):
    """x = arm B (LFM2.5), y = arm A (Qwen3.6). Positive diff = LFM2.5 ahead."""
    groups = strata(x, ids)
    out = {"n": len(ids), "B": acc(x, ids, lenient), "A": acc(y, ids, lenient)}
    out["diff_B_minus_A"] = out["B"] - out["A"]
    out["diff_ci"] = paired_boot(x, y, groups, lenient)
    out["envs"], out["difficulty"] = {}, {}
    for name, part, dest in (("env", groups, out["envs"]), ("difficulty", strata(x, ids, labels=labels), out["difficulty"])):
        for k, members in part.items():
            fx, fy = acc(x, members, lenient), acc(y, members, lenient)
            dest[k] = {"n": len(members), "B": fx, "A": fy, "diff_B_minus_A": fx - fy,
                       "diff_ci": paired_boot(x, y, {k: members}, lenient)}
    return out


def seconds_per_solved(recs, ids, lenient=False):
    solved = sum(score(recs[i], lenient) for i in ids)
    return sum(recs[i]["wall_s"] for i in ids) / solved if solved else float("inf")


def speed_ratio(a, b, ids, lenient=False, seed=0):
    """Seconds per solved task, A / B (above 1 = LFM2.5 faster per solved task), with a paired bootstrap interval."""
    point = seconds_per_solved(a, ids, lenient) / seconds_per_solved(b, ids, lenient)
    rng = random.Random(seed)
    ratios = []
    for _ in range(BOOT):
        sample = [rng.choice(ids) for _ in ids]
        sa = sum(score(a[i], lenient) for i in sample)
        sb = sum(score(b[i], lenient) for i in sample)
        if sa and sb:
            ratios.append((sum(a[i]["wall_s"] for i in sample) / sa) / (sum(b[i]["wall_s"] for i in sample) / sb))
    return {"ratio_A_over_B": point, "ci": (q(ratios, 0.025), q(ratios, 0.975)) if ratios else (float("nan"),) * 2,
            "resamples_with_zero_solved": BOOT - len(ratios)}


def describe(recs, labels):
    ids = list(recs)
    n = max(len(ids), 1)
    repairs = {}
    for r in recs.values():
        for kind in r.get("repairs", []):
            repairs[kind] = repairs.get(kind, 0) + 1
    timings = [t.get("timings", {}) for r in recs.values() for t in r["turns"]]
    pred_n = sum(t.get("predicted_n", 0) for t in timings)
    pred_ms = sum(t.get("predicted_ms", 0.0) for t in timings)
    prompt_n = sum(t.get("prompt_n", 0) for t in timings)
    prompt_ms = sum(t.get("prompt_ms", 0.0) for t in timings)
    toks = [r["completion_tokens"] for r in recs.values()]
    secs = [r["wall_s"] for r in recs.values()]
    solved_s = sum(score(r) for r in recs.values())
    solved_l = sum(score(r, True) for r in recs.values())
    return {
        "n": len(ids), "accuracy_strict": acc(recs, ids), "accuracy_lenient": acc(recs, ids, True),
        "envs": {k: {"strict": acc(recs, v), "lenient": acc(recs, v, True), "n": len(v)} for k, v in strata(recs, ids).items()},
        "difficulty": {k: {"strict": acc(recs, v), "lenient": acc(recs, v, True), "n": len(v)}
                       for k, v in strata(recs, ids, labels=labels).items()},
        "truncated_tasks": sum(1 for r in recs.values() if r.get("truncated")),
        "server_error_tasks": sum(1 for r in recs.values() if r.get("server_error")),
        "turn_limit_tasks": sum(1 for r in recs.values() if r.get("hit_turn_limit")),
        "tasks_needing_repair": sum(1 for r in recs.values() if r.get("repairs")),
        "repair_events": repairs,
        "tool_errors_returned": sum(r.get("tool_errors", 0) for r in recs.values()),
        "mean_turns": sum(len(r["turns"]) for r in recs.values()) / n,
        "mean_tokens": sum(toks) / n, "median_tokens": statistics.median(toks) if toks else float("nan"),
        "mean_seconds": sum(secs) / n, "median_seconds": statistics.median(secs) if secs else float("nan"),
        "seconds_per_solved_strict": sum(secs) / solved_s if solved_s else float("inf"),
        "seconds_per_solved_lenient": sum(secs) / solved_l if solved_l else float("inf"),
        "tokens_per_solved_strict": sum(toks) / solved_s if solved_s else float("inf"),
        "tokens_per_solved_lenient": sum(toks) / solved_l if solved_l else float("inf"),
        "decode_tok_per_s": 1000.0 * pred_n / pred_ms if pred_ms else float("nan"),
        "prompt_tok_per_s": 1000.0 * prompt_n / prompt_ms if prompt_ms else float("nan"),
        "hours": sum(secs) / 3600.0,
    }


def noise_floor(main, noise):
    """Per arm: |accuracy(seed 1001) - accuracy(seed 2002)| on the re-run tasks. Floor = the larger of the two arms."""
    out = {"strict": 0.0, "lenient": 0.0, "per_arm": {}}
    for arm in ("A", "B"):
        m, n = main.get(arm, {}), noise.get(arm, {})
        ids = [i for i in n if i in m]
        if not ids:
            continue
        d = {"n": len(ids), "strict": abs(acc(m, ids) - acc(n, ids)), "lenient": abs(acc(m, ids, True) - acc(n, ids, True)),
             "flips_strict": sum(1 for i in ids if score(m[i]) != score(n[i])),
             "flips_lenient": sum(1 for i in ids if score(m[i], True) != score(n[i], True))}
        out["per_arm"][arm] = d
        out["strict"] = max(out["strict"], d["strict"])
        out["lenient"] = max(out["lenient"], d["lenient"])
    return out


# ---------- the three candidate rules (PREREG.md); only config.RULE is applied ----------


def rule_1(cmp, lenient_label="strict"):
    """Holds if B is ahead or within 3 points overall (and ahead on the subset, if one is configured).
    Fails if B trails by 10 points or more and the interval excludes zero. Otherwise not resolved."""
    gap, (lo, hi) = cmp["diff_B_minus_A"], cmp["diff_ci"]
    sub = None
    if cfg.RULE1_SUBSET_DIFFICULTIES:
        parts = [cmp["difficulty"][d] for d in cfg.RULE1_SUBSET_DIFFICULTIES if d in cmp["difficulty"]]
        nsub = sum(p["n"] for p in parts)
        sub = sum(p["diff_B_minus_A"] * p["n"] for p in parts) / nsub if nsub else float("nan")
    if gap >= -3 and (sub is None or sub > 0):
        headline = "HOLDS"
    elif gap <= -10 and hi < 0:
        headline = "FAILS"
    else:
        headline = "NOT RESOLVED"
    return {"rule": "1", "score": lenient_label, "gap_B_minus_A": gap, "ci": (lo, hi), "subset_gap": sub, "headline": headline}


def rule_2(cmp, speed):
    """Holds if B is within 5 points of A (gap >= -5) and at least 1.5x faster in seconds per solved task (strict)."""
    ok = cmp["diff_B_minus_A"] >= -5 and speed["ratio_A_over_B"] >= 1.5
    return {"rule": "2", "gap_B_minus_A": cmp["diff_B_minus_A"], "ci": cmp["diff_ci"], "speed": speed,
            "headline": "HOLDS" if ok else "DOES NOT HOLD"}


def verdict(strict_cmp, lenient_cmp, speed):
    if cfg.RULE == "1":
        return rule_1(strict_cmp)
    if cfg.RULE == "2":
        return rule_2(strict_cmp, speed)
    if cfg.RULE == "3":  # rule 1's thresholds on the lenient score; the strict-lenient gap is reported as a tooling finding
        return rule_1(lenient_cmp, "lenient")
    return {"headline": "NO RULE CHOSEN (config.RULE is None): scores only"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()

    if args.private:
        labels = difficulties("private")
        recs = {a: load("private", a, cfg.PRIVATE_RESULTS_DIR) for a in cfg.ARMS}
        report = {"note": "private set: calibration scores only, never published",
                  "arms": {a: describe(r, labels) for a, r in recs.items() if r}}
        path = os.path.join(cfg.PRIVATE_RESULTS_DIR, "report_private.json")
    elif args.pilot:
        labels = difficulties("public")
        pilot_dir = os.path.join(cfg.RESULTS_DIR, "pilot")
        arms = sorted(f[:-6] for f in os.listdir(pilot_dir) if f.endswith(".jsonl")) if os.path.isdir(pilot_dir) else []
        report = {"arms": {a: describe(load("pilot", a), labels) for a in arms}}
        path = os.path.join(pilot_dir, "report_pilot.json")
    else:
        labels = difficulties("public")
        main_runs = {a: load("main", a) for a in cfg.ARMS}
        noise = {a: load("noise", a) for a in cfg.ARMS}
        report = {"arms": {a: describe(r, labels) for a, r in main_runs.items() if r}}
        A, B = main_runs["A"], main_runs["B"]
        common = sorted(i for i in A if i in B)
        if common:
            report["B_minus_A_strict"] = compare(B, A, common, labels)
            report["B_minus_A_lenient"] = compare(B, A, common, labels, lenient=True)
            report["strict_minus_lenient_points"] = {a: report["arms"][a]["accuracy_strict"] - report["arms"][a]["accuracy_lenient"] for a in "AB"}
            report["speed_strict"] = speed_ratio(A, B, common)
            report["speed_lenient"] = speed_ratio(A, B, common, lenient=True)
        report["noise_floor"] = noise_floor(main_runs, noise)
        # Amendment 2026-10-23: the harder supplement is a pre-registered secondary analysis; the verdict stays on the 75.
        hard = {a: load("hard", a) for a in cfg.ARMS}
        hard_common = sorted(i for i in hard["A"] if i in hard["B"])
        if hard_common:
            hard_labels = difficulties("hard")
            report["hard_arms"] = {a: describe(r, hard_labels) for a, r in hard.items() if r}
            report["hard_B_minus_A_lenient"] = compare(hard["B"], hard["A"], hard_common, hard_labels, lenient=True)
            report["hard_B_minus_A_strict"] = compare(hard["B"], hard["A"], hard_common, hard_labels)
            # For the author's prediction: every complex task, i.e. public tasks with 12+ reference calls plus the supplement.
            cx = [i for i in common if labels.get(i, "").startswith("complex")]
            both = dict(labels); both.update(hard_labels)
            AA = dict(A)
            AA.update(hard["A"])
            BB = dict(B)
            BB.update(hard["B"])
            ids = sorted(cx + hard_common)
            report["complex_all_B_minus_A_lenient"] = compare(BB, AA, ids, both, lenient=True)
        if common:
            report["verdict"] = verdict(report["B_minus_A_strict"], report["B_minus_A_lenient"], report["speed_strict"])
        path = os.path.join(cfg.RESULTS_DIR, "report.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
