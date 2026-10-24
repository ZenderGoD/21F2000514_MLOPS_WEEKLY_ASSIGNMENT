"""Feast definitions for the IRIS feature store.

The parquet file is generated from ``data/v1/iris.csv`` by
``src/feast_prepare.py``.  It adds a stable entity id and an event timestamp,
which are required for historical (point-in-time correct) retrieval.
"""

from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32


FEATURE_DATA = Path(__file__).resolve().parent / "data" / "iris_features.parquet"

iris_source = FileSource(
    name="iris_source",
    path=str(FEATURE_DATA),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

iris = Entity(
    name="iris",
    join_keys=["iris_id"],
    value_type=ValueType.INT64,
    description="Stable identifier for an IRIS sample.",
)

iris_features = FeatureView(
    name="iris_features",
    entities=[iris],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width", dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width", dtype=Float32),
    ],
    online=True,
    source=iris_source,
)
