#!/bin/bash
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

DENSE_PID=57559

echo "$(date): waiting for dense run (PID $DENSE_PID) to finish..."
while kill -0 "$DENSE_PID" 2>/dev/null; do
    sleep 60
done
echo "$(date): dense run finished. Starting MoE pipeline."

python3 prepare_vault_data.py
python3 encode_vault_data.py
python3 train_moe.py

echo "$(date): MoE pipeline complete."
