import json
import math
import joblib  # Импортируем joblib

import pandas as pd

from api.config import MODEL_PATH, MODEL_META_PATH
from api.schemas import (
    PredictRequest,
    LISTING_TYPE_TO_RAW,
    FURNISHED_TO_RAW,
    HEATING_TYPE_TO_RAW,
)


class ModelService:
    def __init__(self):
        self.model = None
        self.meta: dict = {}
        self.mae: float = 0.0
        self.model_name: str = "unknown"

    def load(self) -> None:
        """Загружает модель один раз при старте приложения через joblib."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Файл модели не найден: {MODEL_PATH}"
            )

        # Используем joblib для загрузки.
        # Он корректно обрабатывает сжатие, если оно было применено при сохранении.
        try:
            self.model = joblib.load(MODEL_PATH)
        except Exception as e:
            raise RuntimeError(f"Не удалось загрузить модель через joblib: {e}")

        if not hasattr(self.model, "predict"):
            raise TypeError(
                "Загруженный объект модели не имеет метода predict()."
            )

        # Загрузка метаданных
        if MODEL_META_PATH.exists():
            with open(MODEL_META_PATH, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        else:
            self.meta = {}

        mae_value = self.meta.get(
            "mae",
            self.meta.get("MAE", 0.0),
        )

        try:
            self.mae = float(mae_value)
        except (TypeError, ValueError):
            self.mae = 0.0

        self.model_name = str(
            self.meta.get(
                "model_name",
                self.meta.get("name", "final_model"),
            )
        )

    def _to_dataframe(
        self,
        payload: PredictRequest,
    ) -> pd.DataFrame:
        """Преобразует запрос в формат, ожидаемый пайплайном."""
        row = {
            "size": payload.size,
            "room_count": payload.room_count,
            "building_age": payload.building_age,
            "total_floor_count": payload.total_floor_count,
            "floor_no": payload.floor_no,
            "listing_type": LISTING_TYPE_TO_RAW[payload.listing_type],
            "furnished": FURNISHED_TO_RAW[payload.furnished],
            "heating_type": HEATING_TYPE_TO_RAW[payload.heating_type],
            "address": payload.address,
        }

        return pd.DataFrame([row])

    def predict(self, payload: PredictRequest) -> dict:
        if self.model is None:
            raise RuntimeError(
                "Модель не загружена. "
                "Проверьте вызов ModelService.load() при старте."
            )

        df = self._to_dataframe(payload)

        prediction = self.model.predict(df)

        if len(prediction) == 0:
            raise RuntimeError(
                "Модель не вернула результат прогнозирования."
            )

        price = float(prediction[0])

        if not math.isfinite(price):
            raise RuntimeError(
                "Модель вернула некорректную стоимость."
            )

        price_low = price - self.mae
        price_high = price + self.mae

        return {
            "predicted_price": round(price, 2),
            "price_range_low": round(price_low, 2),
            "price_range_high": round(price_high, 2),
            "currency": "TRY",
            "model_name": self.model_name,
        }


model_service = ModelService()
