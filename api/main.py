"""FastAPI application that forecasts hourly departures per MiBici station.

uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from loguru import logger
from pydantic import ValidationError

from api.model import ServedModel, load_model
from api.schemas import (
    HealthResponse,
    ModelInfo,
    Prediction,
    PredictRequest,
    PredictResponse,
)


def create_app(loader: Callable[[], ServedModel] = load_model) -> FastAPI:
    """Build the application. The model is loaded once, when the application starts.

    - `loader`: returns the model to serve. Tests pass a model saved in a temporary directory.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.model = loader()
        info = app.state.model.info
        logger.info(f"Serving {info.name} version {info.version}")
        yield

    app = FastAPI(
        title="MiBici demand API",
        description="Hourly departures per station, from the P1 champion of the registry.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health(request: Request) -> HealthResponse:
        return HealthResponse(status="ok", model_version=request.app.state.model.info.version)

    @app.get("/model")
    def model_info(request: Request) -> ModelInfo:
        return request.app.state.model.info

    @app.post("/predict")
    def predict(body: PredictRequest, request: Request) -> PredictResponse:
        model: ServedModel = request.app.state.model
        values = model.predict(body.instances)
        try:
            predictions = [
                Prediction(**instance.model_dump(), predicted_departures=value)
                for instance, value in zip(body.instances, values, strict=True)
            ]
        except ValidationError as error:
            logger.error(f"The model returned an invalid prediction: {error}")
            raise HTTPException(500, "The model returned an invalid prediction.") from error
        return PredictResponse(
            model_name=model.info.name,
            model_version=model.info.version,
            predictions=predictions,
        )

    return app


app = create_app()
