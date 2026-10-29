#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

{
    for s in 0 1 2 3 4; do
        for r in 0.8 0.9 1.0 1.1 1.3; do
            printf '\nRunning seed=%s ratio=%s\n' "$s" "$r"
            /home/colligo/phase-space-diffusion-threshold/.venv/bin/python -u elbo_train.py "$s" "$r"
        done
    done
} 2>&1 | tee -a elbo_train.log
