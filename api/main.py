from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routers import dashboard, misc, predict
from api.services.data_service import data_service
from api.services.model_service import model_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загрузка модели и датасета один раз при запуске."""
    model_service.load()
    data_service.load()

    app.state.model_service = model_service
    app.state.data_service = data_service

    yield


app = FastAPI(
    title="Real Estate Price Prediction API",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(misc.router)
app.include_router(predict.router)
app.include_router(dashboard.router)
