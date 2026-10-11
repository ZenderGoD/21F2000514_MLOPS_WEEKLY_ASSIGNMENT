"""Task 3: fetch training data from GCS, train the IRIS classifier, and store
artifacts in a folder named after the training execution timestamp.

Usage:
    python src/train.py --bucket gs://my-bucket --version v1

Writes to gs://my-bucket/artifacts/<version>/<YYYY-MM-DDTHH-MM-SS>/:
    model.joblib   trained model
    metrics.json   validation accuracy + run metadata
    train.log      training log
"""
import argparse
import io
import json
import logging
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from storage import join, open_file

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "species"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True, help="gs://bucket-name (or a local dir for testing)")
    parser.add_argument("--version", default="v1", help="data version to train on")
    parser.add_argument("--max-depth", type=int, default=3)
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = join(args.bucket, "artifacts", args.version, run_id)

    log_buffer = io.StringIO()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.StreamHandler(log_buffer)],
    )
    log = logging.getLogger("train")

    train_path = join(args.bucket, "data", args.version, "train.csv")
    log.info("Run %s: reading training data from %s", run_id, train_path)
    with open_file(train_path) as f:
        data = pd.read_csv(f)
    log.info("Loaded %d rows", len(data))

    # Hold out part of the training data for validation; eval.csv is reserved for inference.
    train, val = train_test_split(data, test_size=0.2, stratify=data[TARGET], random_state=55)
    model = DecisionTreeClassifier(max_depth=args.max_depth, random_state=1)
    model.fit(train[FEATURES], train[TARGET])

    val_accuracy = metrics.accuracy_score(val[TARGET], model.predict(val[FEATURES]))
    log.info("Validation accuracy: %.3f", val_accuracy)

    with open_file(join(run_dir, "model.joblib"), "wb") as f:
        joblib.dump(model, f)
    run_info = {
        "run_id": run_id,
        "data_version": args.version,
        "train_rows": len(train),
        "val_rows": len(val),
        "model": "DecisionTreeClassifier",
        "params": {"max_depth": args.max_depth},
        "val_accuracy": round(val_accuracy, 4),
    }
    with open_file(join(run_dir, "metrics.json"), "w") as f:
        json.dump(run_info, f, indent=2)

    log.info("Saved artifacts to %s", run_dir)
    with open_file(join(run_dir, "train.log"), "w") as f:
        f.write(log_buffer.getvalue())

    print(run_id)


if __name__ == "__main__":
    main()
