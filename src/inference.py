"""Task 4: fetch a trained model from the GCS artifacts folder and run inference
on the evaluation set.

Usage:
    python src/inference.py --bucket gs://my-bucket --version v1            # latest run
    python src/inference.py --bucket gs://my-bucket --version v1 --run-id 2026-10-12T10-30-00

Writes into the same run folder:
    predictions.csv        eval rows with predicted species
    inference_metrics.json accuracy + classification report on the eval set
"""
import argparse
import json

import joblib
import pandas as pd
from sklearn import metrics

from storage import join, list_dirs, open_file
from train import FEATURES, TARGET


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True, help="gs://bucket-name (or a local dir for testing)")
    parser.add_argument("--version", default="v1", help="data version the model was trained on")
    parser.add_argument("--run-id", help="timestamp folder of the training run (default: latest)")
    args = parser.parse_args()

    runs_root = join(args.bucket, "artifacts", args.version)
    run_id = args.run_id or (list_dirs(runs_root) or [None])[-1]
    if run_id is None:
        raise SystemExit(f"No training runs found under {runs_root}")
    run_dir = join(runs_root, run_id)

    print(f"Loading model from {run_dir}/model.joblib")
    with open_file(join(run_dir, "model.joblib"), "rb") as f:
        model = joblib.load(f)

    with open_file(join(args.bucket, "data", args.version, "eval.csv")) as f:
        evaluation = pd.read_csv(f)

    evaluation["predicted_species"] = model.predict(evaluation[FEATURES])
    accuracy = metrics.accuracy_score(evaluation[TARGET], evaluation["predicted_species"])
    report = metrics.classification_report(
        evaluation[TARGET], evaluation["predicted_species"], output_dict=True
    )
    print(f"Eval accuracy for run {run_id}: {accuracy:.3f}")

    with open_file(join(run_dir, "predictions.csv"), "w") as f:
        evaluation.to_csv(f, index=False)
    with open_file(join(run_dir, "inference_metrics.json"), "w") as f:
        json.dump(
            {"run_id": run_id, "data_version": args.version, "eval_rows": len(evaluation),
             "eval_accuracy": round(accuracy, 4), "classification_report": report},
            f, indent=2,
        )
    print(f"Saved predictions and metrics to {run_dir}/")


if __name__ == "__main__":
    main()
