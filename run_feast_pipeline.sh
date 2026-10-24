#!/usr/bin/env bash
# Build the Feast source, register definitions, materialize the online store,
# train from historical features, and run a small online inference smoke test.
# Usage: ./run_feast_pipeline.sh [v1|v2]
set -euo pipefail

VERSION="${1:-v1}"

python src/feast_prepare.py --version "$VERSION"
(cd feature_repo && feast apply)

python - <<'PY'
from datetime import datetime, timezone
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo")
store.materialize(
    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    end_date=datetime.now(timezone.utc),
    force=True,
)
print("Materialized Feast online store")
PY

python src/feast_train.py
python src/feast_inference.py --ids 1 51 101
