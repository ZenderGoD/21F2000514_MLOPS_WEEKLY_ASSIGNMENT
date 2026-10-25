# IRIS Feast feature repository

`definitions.py` is the single source of truth for the IRIS entity and
feature view. `data/iris_features.parquet` is generated from the raw IRIS CSV
with `src/feast_prepare.py`; it adds the `iris_id`, `event_timestamp`, and
`created_timestamp` columns needed for Feast.

From the repository root:

```bash
python src/feast_prepare.py --version v1
cd feature_repo
feast apply
feast materialize-incremental "$(date -u +%Y-%m-%dT%H:%M:%S)"
cd ..
python src/feast_train.py
python src/feast_inference.py --ids 1 51 101
```

`registry.db` and `online_store.db` are local Feast runtime state and are
intentionally ignored by Git. They are recreated by `feast apply` and
materialization.

For the optional BigQuery task, copy `feature_store.bigquery.yaml.example`,
replace the bucket, dataset, and project placeholders, enable the BigQuery API,
and run `feast apply` from the feature repository. BigQuery is useful for a
shared, scalable historical store; the local SQLite setup is cheaper and
simpler for this small demonstration.
