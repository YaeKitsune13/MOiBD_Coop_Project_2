from fastapi import APIRouter, HTTPException

from api.schemas import PredictRequest, PredictResponse
from api.services.model_service import model_service

router = APIRouter(
    prefix="/api",
    tags=["predict"],
)


@router.post(
    "/predict",
    response_model=PredictResponse,
)
def predict(payload: PredictRequest):
    try:
        result = model_service.predict(payload)

    except RuntimeError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
        )

    except (KeyError, ValueError, TypeError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Не удалось выполнить прогноз: {e}",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка модели: {e}",
        )

    return result
