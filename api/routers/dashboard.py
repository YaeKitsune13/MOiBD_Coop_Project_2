import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from api.config import MODEL_COMPARISON_PATH
from api.schemas import DatasetSummary
from api.services.data_service import data_service

router = APIRouter(
    prefix="/api/dashboard",
    tags=["dashboard"],
)


def _json_safe_dataframe(frame: pd.DataFrame) -> list[dict]:
    safe = frame.copy()
    safe = safe.astype(object).where(pd.notna(safe), None)
    return safe.to_dict(orient="records")


@router.get("/models")
def list_models():
    if not MODEL_COMPARISON_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="model_comparison.csv ещё не готов",
        )

    df = pd.read_csv(
        MODEL_COMPARISON_PATH,
        low_memory=False,
    )

    return _json_safe_dataframe(df)


@router.get(
    "/dataset/summary",
    response_model=DatasetSummary,
)
def dataset_summary():
    return data_service.summary()


@router.get("/dataset/sample")
def dataset_sample():
    return data_service.sample()


@router.get("/interactive/correlation")
def correlation_matrix():
    return data_service.correlation_matrix()


@router.get("/interactive/avg-by-category")
def avg_price_by_category(
    column: str = Query(
        ...,
        description=(
            "Например: room_count, heating_type, listing_type"
        ),
    ),
    agg: str = Query(
        "mean",
        pattern="^(mean|median)$",
    ),
):
    try:
        return data_service.avg_price_by_category(
            column,
            agg,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )
