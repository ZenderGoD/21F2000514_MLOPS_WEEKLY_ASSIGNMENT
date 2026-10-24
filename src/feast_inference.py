"""Run inference using features retrieved from Feast's online store.

Usage:
    python src/feast_inference.py --ids 1 2 3
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from feast import FeatureStore


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "feature_repo"
MODEL_PATH = ROOT / "artifacts" / "feast_model.joblib"
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
FEATURE_REFS = [f"iris_features:{name}" for name in FEATURES]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="+", type=int, required=True)
    args = parser.parse_args()

    store = FeatureStore(repo_path=str(REPO))
    result = store.get_online_features(
        features=FEATURE_REFS,
        entity_rows=[{"iris_id": sample_id} for sample_id in args.ids],
    ).to_dict()
    features = pd.DataFrame({name: result[name] for name in FEATURES})

    model = joblib.load(MODEL_PATH)
    predictions = model.predict(features)
    output = pd.DataFrame({"iris_id": args.ids, "predicted_species": predictions})
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
