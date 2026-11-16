#!/bin/bash
# Waits for Run 3c (PID 80330) to finish, then decides whether Run 4 (the
# 305.6M-param jump) is actually justified -- per the Run 3b eval report's
# Next Experiment Plan: success = beating 1.4632 by more than 0.01 (i.e.
# best val < 1.4522). If Run 3c didn't clear that bar, the plateau is
# confirmed and Run 4 launches automatically. If it did clear the bar,
# Run 4 is deliberately NOT auto-launched -- more training at the current
# scale would still be justified, and burning ~3 days of compute on an
# unnecessary jump isn't. This decision is written to run_next_stage.log
# either way so it's clear what happened and why.

cd "$(dirname "$0")/.." || exit 1
LOG="run_next_stage.log"
RUN3C_PID=80330
THRESHOLD="1.4522"   # 1.4632 - 0.01, per the report's success criteria

echo "$(date): watching Run 3c (PID $RUN3C_PID)..." > "$LOG"

while kill -0 "$RUN3C_PID" 2>/dev/null; do
    sleep 60
done

echo "$(date): Run 3c (PID $RUN3C_PID) has exited." >> "$LOG"

BEST_VAL=$(grep -oE "Best val loss this run: [0-9.]+" train_run3c.log | tail -1 | grep -oE "[0-9.]+$")

if [ -z "$BEST_VAL" ]; then
    echo "$(date): Could not find a final best-val-loss line in train_run3c.log -- Run 3c may have crashed or been killed early. NOT launching Run 4 automatically. Check train_run3c.log manually." >> "$LOG"
    exit 1
fi

echo "$(date): Run 3c best val loss = $BEST_VAL (threshold to justify Run 4: >= $THRESHOLD)" >> "$LOG"

CLEARS_THRESHOLD=$(python3 -c "print(1 if float('$BEST_VAL') >= float('$THRESHOLD') else 0)")

if [ "$CLEARS_THRESHOLD" = "1" ]; then
    echo "$(date): Run 3c did NOT beat 1.4632 by more than 0.01 (best=$BEST_VAL) -- plateau confirmed. Launching Run 4 (18L/18H/1152E, 305.6M params, 50,000-step budget)." >> "$LOG"
    source .venv/bin/activate
    nohup python train_run4.py > train_run4.log 2>&1 &
    echo "$(date): Run 4 launched, PID $!" >> "$LOG"
else
    echo "$(date): Run 3c BEAT 1.4632 by more than 0.01 (best=$BEST_VAL) -- the plateau was not real, more training at the current 33.5M scale is still justified. Run 4 was NOT launched automatically. Decide manually whether to continue at this scale or still scale up." >> "$LOG"
fi
