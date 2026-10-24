"""Create the timestamped, entity-enriched source used by Feast.

Usage:
    python src/feast_prepare.py --version v1
"""

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FEATURE_REPO_DATA = ROOT / "feature_repo" / "data" / "iris_features.parquet"


def build_feature_data(version: str = "v1") -> pd.DataFrame:
    source = ROOT / "data" / version / "iris.csv"
    frame = pd.read_csv(source)
    frame.insert(0, "iris_id", range(1, len(frame) + 1))
    timestamps = pd.date_range(
        start="2024-01-01T00:00:00Z", periods=len(frame), freq="min"
    )
    frame["event_timestamp"] = timestamps
    frame["created_timestamp"] = timestamps
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()

    FEATURE_REPO_DATA.parent.mkdir(parents=True, exist_ok=True)
    frame = build_feature_data(args.version)
    frame.to_parquet(FEATURE_REPO_DATA, index=False)
    print(f"Wrote {len(frame)} rows to {FEATURE_REPO_DATA}")


if __name__ == "__main__":
    main()
