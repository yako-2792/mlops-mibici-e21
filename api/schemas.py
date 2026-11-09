"""Request and response contracts of the API.

Inputs are strict: a number sent as a string, a float for an integer, or an unknown field
is rejected with a 422 instead of being coerced. `is_weekend` is not accepted, because the
service derives it from `day_of_week` exactly as the training features did.
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

MAX_INSTANCES = 1000


class DemandFeatures(BaseModel):
    """One station and hour to forecast."""

    model_config = ConfigDict(extra="forbid", strict=True)

    station_id: Annotated[int, Field(gt=0, description="MiBici station id.", examples=[42])]
    hour: Annotated[int, Field(ge=0, le=23, description="Hour of departure, 0 to 23.")]
    day_of_week: Annotated[int, Field(ge=0, le=6, description="Day of week, Monday is 0.")]


class PredictRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        json_schema_extra={
            "examples": [
                {
                    "instances": [
                        {"station_id": 42, "hour": 8, "day_of_week": 0},
                        {"station_id": 42, "hour": 18, "day_of_week": 5},
                    ]
                }
            ]
        },
    )

    instances: Annotated[list[DemandFeatures], Field(min_length=1, max_length=MAX_INSTANCES)]


class Prediction(DemandFeatures):
    predicted_departures: Annotated[
        float,
        Field(ge=0, allow_inf_nan=False, description="Expected departures in that hour."),
    ]


class ModelInfo(BaseModel):
    """Registry lineage of the model being served.

    `alias` is the alias the version was resolved from, or `None` for a pinned version.
    """

    name: str
    version: str
    alias: str | None
    run_id: str


class PredictResponse(BaseModel):
    model_name: str
    model_version: str
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: str
    model_version: str
