"""Train the IRIS model using Feast's offline store.

The model never reads ``data/*/iris.csv`` directly.  It builds an entity
dataframe and asks Feast for the historical feature values instead.
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from feast import FeatureStore
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "feature_repo"
MODEL_PATH = ROOT / "artifacts" / "feast_model.joblib"
METRICS_PATH = ROOT / "artifacts" / "feast_metrics.json"
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
FEATURE_REFS = [f"iris_features:{name}" for name in FEATURES]


def fetch_training_frame(store: FeatureStore) -> pd.DataFrame:
    source = pd.read_parquet(REPO / "data" / "iris_features.parquet")
    entities = source[["iris_id", "event_timestamp", "species"]]
    return store.get_historical_features(
        entity_df=entities,
        features=FEATURE_REFS,
    ).to_df()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-depth", type=int, default=3)
    args = parser.parse_args()

    store = FeatureStore(repo_path=str(REPO))
    training = fetch_training_frame(store)
    train, validation = train_test_split(
        training, test_size=0.2, stratify=training["species"], random_state=42
    )

    model = DecisionTreeClassifier(max_depth=args.max_depth, random_state=1)
    model.fit(train[FEATURES], train["species"])
    accuracy = metrics.accuracy_score(
        validation["species"], model.predict(validation[FEATURES])
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(
        json.dumps(
            {
                "feature_source": "feast_offline_store",
                "rows_retrieved": len(training),
                "train_rows": len(train),
                "validation_rows": len(validation),
                "validation_accuracy": round(float(accuracy), 4),
            },
            indent=2,
        )
    )
    print(f"Retrieved {len(training)} rows from Feast offline store")
    print(f"Validation accuracy: {accuracy:.3f}")
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
