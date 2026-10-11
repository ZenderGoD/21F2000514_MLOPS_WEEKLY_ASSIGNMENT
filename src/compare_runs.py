"""Task 6: summarise every training run (all data versions) found in the bucket."""
import argparse
import json

import pandas as pd

from storage import join, list_dirs, open_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    args = parser.parse_args()

    rows = []
    artifacts = join(args.bucket, "artifacts")
    for version in list_dirs(artifacts):
        for run_id in list_dirs(join(artifacts, version)):
            run_dir = join(artifacts, version, run_id)
            with open_file(join(run_dir, "metrics.json")) as f:
                row = json.load(f)
            try:
                with open_file(join(run_dir, "inference_metrics.json")) as f:
                    row["eval_accuracy"] = json.load(f)["eval_accuracy"]
            except FileNotFoundError:
                row["eval_accuracy"] = None
            rows.append(row)

    table = pd.DataFrame(rows, columns=["data_version", "run_id", "train_rows", "val_accuracy", "eval_accuracy"])
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
