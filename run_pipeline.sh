#!/usr/bin/env bash
# Task 5/6: run training + inference twice for each data version.
# Usage: ./run_pipeline.sh gs://my-bucket [versions...]   (default versions: v1)
set -euo pipefail

BUCKET="${1:?usage: ./run_pipeline.sh gs://bucket-name [v1 v2 ...]}"
shift
VERSIONS=("${@:-v1}")

for version in "${VERSIONS[@]}"; do
  python src/prepare_data.py --bucket "$BUCKET" --version "$version"
  for run in 1 2; do
    echo "=== $version: run $run ==="
    run_id=$(python src/train.py --bucket "$BUCKET" --version "$version" | tail -n 1)
    python src/inference.py --bucket "$BUCKET" --version "$version" --run-id "$run_id"
    sleep 1  # guarantee distinct timestamp folders
  done
done

python src/compare_runs.py --bucket "$BUCKET"
