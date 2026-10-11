"""Task 2: split the IRIS dataset into train/eval sets and upload them to GCS.

Usage:
    python src/prepare_data.py --bucket gs://my-bucket --version v1

Writes:
    gs://my-bucket/data/<version>/train.csv
    gs://my-bucket/data/<version>/eval.csv
"""
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from storage import join, open_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True, help="gs://bucket-name (or a local dir for testing)")
    parser.add_argument("--version", default="v1", help="data version folder under data/")
    parser.add_argument("--eval-size", type=float, default=0.2)
    args = parser.parse_args()

    df = pd.read_csv(f"data/{args.version}/iris.csv")
    train, evaluation = train_test_split(
        df, test_size=args.eval_size, stratify=df["species"], random_state=42
    )

    prefix = join(args.bucket, "data", args.version)
    for name, split in [("train.csv", train), ("eval.csv", evaluation)]:
        with open_file(join(prefix, name), "w") as f:
            split.to_csv(f, index=False)
    print(f"Uploaded {len(train)} train rows and {len(evaluation)} eval rows to {prefix}/")


if __name__ == "__main__":
    main()
