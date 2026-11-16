"""Non-invasive live monitor for Run 4b. Reads train_run4b.log (the running
training process writes to it, never reads it), extracts every logged eval
point, and writes run4b_status.json for the dashboard to poll. Does not
touch the training process, its GPU usage, or its files in any way -- pure
log-tailing, safe to start/stop/crash independently of the actual run.
"""

import json
import re
import time
from pathlib import Path

LOG_PATH = Path("train_run4b.log")
SHADOW_PATH = Path("eval_shadow_run4b.json")
OUT_PATH = Path("run4b_status.json")

START_STEP = 46751
CONTINUE_STEPS = 25000
ROLLBACK_CHECK_STEP = CONTINUE_STEPS // 2
BEST_VAL_TO_BEAT = 1.3006

EVAL_RE = re.compile(
    r"global step (\d+): train loss ([\d.]+), val loss ([\d.]+) \(best this run: ([\d.]+)\)"
)
FINAL_RE = re.compile(r"FINAL global step (\d+): train loss ([\d.]+), val loss ([\d.]+)")
DONE_RE = re.compile(r"Done\. Total wall clock: ([\d.]+)h, global step reached: (\d+)")
PBAR_RE = re.compile(r"gstep=(\d+), loss=([\d.]+)")

print("Watching train_run4b.log -> run4b_status.json (Ctrl+C to stop; safe to kill any time)")

start_time = time.time()

while True:
    text = LOG_PATH.read_text(errors="ignore") if LOG_PATH.exists() else ""

    by_step = {}
    for m in EVAL_RE.finditer(text):
        gstep, train_l, val_l, best_l = m.groups()
        by_step[int(gstep)] = {
            "step": int(gstep),
            "train_loss": float(train_l),
            "val_loss": float(val_l),
            "source": "log",
        }

    # merge in shadow evaluator results (CPU-only, independent re-eval of each
    # checkpoint) for any step the buffered log hasn't surfaced yet -- see
    # eval_shadow_run4b.py. Log-derived values win if both exist for a step.
    if SHADOW_PATH.exists():
        try:
            shadow = json.loads(SHADOW_PATH.read_text())
        except json.JSONDecodeError:
            shadow = []
        for s in shadow:
            if s["step"] not in by_step:
                by_step[s["step"]] = {
                    "step": s["step"],
                    "train_loss": None,
                    "val_loss": s["val_loss"],
                    "source": "shadow",
                }

    history = sorted(by_step.values(), key=lambda h: h["step"])
    running_best = None
    for h in history:
        if running_best is None or h["val_loss"] < running_best:
            running_best = h["val_loss"]
        h["best_val"] = running_best

    finished = bool(DONE_RE.search(text))
    final_match = FINAL_RE.search(text)

    # latest live step/loss from the tqdm progress line (updates every step, not just every 250)
    pbar_matches = PBAR_RE.findall(text)
    latest_step, latest_loss = (pbar_matches[-1] if pbar_matches else (None, None))

    candidates = [h["step"] for h in history] + ([int(latest_step)] if latest_step else [])
    current_step = max(candidates) if candidates else START_STEP
    best_val = min((h["val_loss"] for h in history), default=None)
    local_step = current_step - START_STEP
    progress_pct = round(100 * local_step / CONTINUE_STEPS, 2)

    rollback_note = None
    if local_step >= ROLLBACK_CHECK_STEP and best_val is not None:
        margin = BEST_VAL_TO_BEAT - best_val
        rollback_note = "ON TRACK (beating target)" if margin > 0 else "NOT YET BEATEN -- plateau likely"

    status = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "watcher_uptime_min": round((time.time() - start_time) / 60, 1),
        "run": "Run 4b",
        "start_step": START_STEP,
        "budget_steps": CONTINUE_STEPS,
        "current_step": current_step,
        "local_step": local_step,
        "progress_pct": progress_pct,
        "latest_live_loss": float(latest_loss) if latest_loss else None,
        "best_val_this_run": best_val,
        "target_to_beat": BEST_VAL_TO_BEAT,
        "rollback_check_step": START_STEP + ROLLBACK_CHECK_STEP,
        "rollback_note": rollback_note,
        "finished": finished,
        "final": {
            "step": int(final_match.group(1)),
            "train_loss": float(final_match.group(2)),
            "val_loss": float(final_match.group(3)),
        } if final_match else None,
        "history": history,
    }

    OUT_PATH.write_text(json.dumps(status, indent=2))

    if finished:
        print("Run finished -- writing final status and exiting watcher.")
        break

    time.sleep(8)
