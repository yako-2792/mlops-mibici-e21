"""Load the served model and turn validated requests into its input frame.

`MODEL_URI` is anything `mlflow.pyfunc.load_model` accepts:

- a directory written by `make export-model`, the default and what the image ships;
- `models:/<name>@<alias>` or `models:/<name>/<version>`, read from the registry at startup
  through `MLFLOW_TRACKING_URI`.
"""

from collections.abc import Sequence
import os
from pathlib import Path

from mlflow import MlflowClient
import mlflow.pyfunc
import numpy as np
import pandas as pd

from api.schemas import DemandFeatures, ModelInfo

DEFAULT_MODEL_URI = "models/p1-demanda"
REGISTRY_FILE = "registry.json"
REGISTRY_SCHEME = "models:/"
WEEKEND_START = 5
FEATURES = ("station_id", "hour", "day_of_week", "is_weekend")


class ServedModel:
    """A pyfunc model, the dtypes of its signature, and its registry lineage."""

    def __init__(self, model: mlflow.pyfunc.PyFuncModel, info: ModelInfo):
        schema = model.metadata.get_input_schema()
        if schema is None or tuple(schema.input_names()) != FEATURES:
            found = None if schema is None else schema.input_names()
            raise ValueError(f"The model must take the columns {list(FEATURES)}, not {found}.")
        self._model = model
        self._dtypes = {column.name: column.type.to_pandas() for column in schema.inputs}
        self.info = info

    def predict(self, instances: Sequence[DemandFeatures]) -> list[float]:
        """Return the expected departures of each instance, in order.

        Columns are cast to the dtypes of the signature, because MLflow rejects an int64
        column where the model was trained on int32.
        """
        frame = pd.DataFrame([instance.model_dump() for instance in instances])
        frame["is_weekend"] = (frame["day_of_week"] >= WEEKEND_START).astype(int)
        frame = frame[list(FEATURES)].astype(self._dtypes)
        return np.asarray(self._model.predict(frame), dtype=float).tolist()


def load_model(uri: str | None = None) -> ServedModel:
    """Load the model at `uri`, or at `MODEL_URI` when it is not given."""
    uri = uri or os.getenv("MODEL_URI") or DEFAULT_MODEL_URI
    return ServedModel(mlflow.pyfunc.load_model(uri), registry_info(uri))


def registry_info(uri: str) -> ModelInfo:
    """Return the registry lineage of a model URI.

    An exported directory carries it in `registry.json`. A `models:/` URI is resolved
    against the registry.
    """
    if not uri.startswith(REGISTRY_SCHEME):
        return ModelInfo.model_validate_json((Path(uri) / REGISTRY_FILE).read_text())

    client = MlflowClient()
    reference = uri.removeprefix(REGISTRY_SCHEME)
    if "@" in reference:
        name, alias = reference.split("@", 1)
        version = client.get_model_version_by_alias(name, alias)
    else:
        name, number = reference.rsplit("/", 1)
        alias = None
        version = client.get_model_version(name, number)
    return ModelInfo(name=name, version=str(version.version), alias=alias, run_id=version.run_id)
