from fastapi import APIRouter

from api.config import APP_VERSION
from api.schemas import AppInfo
from api.services.model_service import model_service

router = APIRouter(
    prefix="/api",
    tags=["misc"],
)


@router.get("/ping")
def ping():
    return {"status": "ok"}


@router.get(
    "/info",
    response_model=AppInfo,
)
def info():
    return AppInfo(
        version=APP_VERSION,
        description=(
            "Приложение прогнозирует стоимость объекта недвижимости "
            "на основе его характеристик: площадь, комнаты, район, "
            "этаж, отопление и другие параметры."
        ),
        model_name=model_service.model_name,
        model_limitations=(
            "Точность прогноза зависит от полноты исходных данных. "
            "Для нетиповых объектов, редких районов и аномальных "
            "характеристик прогноз может быть менее точным."
        ),
    )
