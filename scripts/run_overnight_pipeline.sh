#!/bin/bash
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

MOE_PID=61528

echo "$(date): waiting for MoE run (PID $MOE_PID) to finish..."
while kill -0 "$MOE_PID" 2>/dev/null; do
    sleep 60
done
echo "$(date): MoE run finished."

echo "$(date): running MoE evaluation (forced-lobe + free-router samples)..."
python3 evaluate_moe.py

echo "$(date): re-encoding training data with EOS-separator fix..."
python3 encode_data.py

echo "$(date): starting Run 3 (bigger model, capacity test)..."
python3 train_run3.py

echo "$(date): overnight pipeline complete."
