import json
import math
from datetime import datetime

import joblib
import numpy as np
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

        self.district_te_map: dict = {}
        self.district_prior: float = 0.0

    def load(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Файл модели не найден: {MODEL_PATH}"
            )

        try:
            self.model = joblib.load(MODEL_PATH)
        except Exception as e:
            raise RuntimeError(
                f"Не удалось загрузить модель через joblib: {e}"
            )

        if not hasattr(self.model, "predict"):
            raise TypeError(
                "Загруженный объект модели не имеет метода predict()."
            )
        if MODEL_META_PATH.exists():
            with open(MODEL_META_PATH, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        else:
            self.meta = {}

        metrics = self.meta.get("metrics", {})

        mae_value = metrics.get(
            "MAE",
            self.meta.get("mae", self.meta.get("MAE", 0.0))
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
        district_map_path = (
            MODEL_PATH.parent / "district_te_map.json"
        )

        if district_map_path.exists():
            with open(
                district_map_path,
                "r",
                encoding="utf-8",
            ) as f:
                data = json.load(f)

            self.district_te_map = data.get("mapping", {})
            self.district_prior = float(
                data.get("prior", 0.0)
            )

        else:
            raise FileNotFoundError(
                f"Файл district TE map не найден: "
                f"{district_map_path}"
            )

    @staticmethod
    def _building_age_to_num(value: int) -> float:
        value = int(value)

        if value <= 5:
            return float(value)
        if value <= 10:
            return 8.0
        if value <= 15:
            return 13.0
        if value <= 20:
            return 18.0
        if value <= 25:
            return 23.0
        if value <= 30:
            return 28.0

        return 35.0

    @staticmethod
    def _total_floor_to_num(value: int) -> float:
        value = int(value)

        if 1 <= value <= 9:
            return float(value)

        if 10 <= value <= 20:
            return 15.0

        if value >= 21:
            return 25.0

        return 1.0

    @staticmethod
    def _parse_room_count(value: str) -> tuple[float, float]:
        parts = str(value).replace(" ", "").split("+")

        try:
            living = float(parts[0])

            extra = (
                float(parts[1])
                if len(parts) > 1
                else 0.0
            )

            return living, extra

        except Exception:
            return np.nan, np.nan

    @staticmethod
    def _parse_address(address: str) -> tuple[str, str]:
        parts = str(address).split("/")

        city = parts[0].strip() if len(parts) > 0 else ""
        district = parts[1].strip() if len(parts) > 1 else ""

        return city, district

    @staticmethod
    def _dummy_columns(
        df: pd.DataFrame,
        column: str,
        expected_columns: list[str],
    ) -> pd.DataFrame:

        value = str(df.iloc[0][column])

        for feature in expected_columns:
            prefix = f"{column}_"

            if not feature.startswith(prefix):
                continue

            category = feature[len(prefix):]

            df[feature] = (
                1 if value == category else 0
            )

        df = df.drop(columns=[column])

        return df

    def _to_dataframe(
        self,
        payload: PredictRequest,
    ) -> pd.DataFrame:
        try:
            expected_columns = list(
                self.model.named_steps["prep"].feature_names_in_
            )
        except Exception as e:
            raise RuntimeError(
                f"Не удалось получить feature_names_in_ модели: {e}"
            )
        city, district = self._parse_address(
            payload.address
        )

        rooms_living, rooms_extra = (
            self._parse_room_count(
                payload.room_count
            )
        )

        building_age_num = (
            self._building_age_to_num(
                payload.building_age
            )
        )

        total_floor_num = (
            self._total_floor_to_num(
                payload.total_floor_count
            )
        )

        floor_num = float(payload.floor_no)

        tom = 0

        start_date = datetime.now()

        start_year = start_date.year
        start_month = start_date.month
        start_dow = start_date.weekday()

        sub_type_features = [
            c
            for c in expected_columns
            if c.startswith("sub_type_")
        ]

        sub_type_categories = [
            c[len("sub_type_"):]
            for c in sub_type_features
        ]

        sub_type = "Daire"

        if (
            sub_type_categories
            and sub_type in sub_type_categories
        ):
            sub_type = "Daire"

        row = {
            "listing_type": LISTING_TYPE_TO_RAW[
                payload.listing_type
            ],

            "tom": tom,

            "size": float(payload.size),

            "building_age_num": building_age_num,

            "total_floor_num": total_floor_num,

            "floor_num": floor_num,

            "rooms_living": rooms_living,

            "rooms_extra": rooms_extra,

            "start_year": start_year,

            "start_month": start_month,

            "start_dow": start_dow,

            "sub_type": sub_type,

            "heating_type": HEATING_TYPE_TO_RAW[
                payload.heating_type
            ],

            "city": city,

            "district": district,

        }

        df = pd.DataFrame([row])

        df = self._dummy_columns(
            df,
            "sub_type",
            expected_columns,
        )

        df = self._dummy_columns(
            df,
            "heating_type",
            expected_columns,
        )

        df = self._dummy_columns(
            df,
            "city",
            expected_columns,
        )

        df["floor_ratio"] = (
            df["floor_num"]
            / df["total_floor_num"].replace(0, 1)
        )

        df["rooms_total"] = (
            df["rooms_living"]
            + df["rooms_extra"]
        )

        df["size_per_room"] = (
            df["size"]
            / df["rooms_total"].replace(0, 1)
        )

        df["district_te"] = (
            district
            and self.district_te_map.get(
                district,
                self.district_prior,
            )
        )

        if df["district_te"].iloc[0] is False:
            df["district_te"] = self.district_prior

        df = df.drop(columns=["district"])

        for column in expected_columns:
            if column not in df.columns:
                df[column] = 0

        # Удаляем всё лишнее.
        df = df[
            expected_columns
        ]

        # Числовой тип.
        df = df.apply(
            pd.to_numeric,
            errors="coerce",
        )

        return df

    def predict(
        self,
        payload: PredictRequest,
    ) -> dict:

        if self.model is None:
            raise RuntimeError(
                "Модель не загружена. "
                "Проверьте вызов ModelService.load() "
                "при старте."
            )

        df = self._to_dataframe(payload)

        if df.shape[1] != 123:
            raise RuntimeError(
                f"Неверное количество признаков: "
                f"{df.shape[1]}, ожидалось 123."
            )

        prediction = self.model.predict(df)

        if len(prediction) == 0:
            raise RuntimeError(
                "Модель не вернула результат прогнозирования."
            )

        price_log = float(prediction[0])

        if not math.isfinite(price_log):
            raise RuntimeError(
                "Модель вернула некорректное значение."
            )

        price = float(np.expm1(price_log))

        if not math.isfinite(price) or price < 0:
            raise RuntimeError(
                "После обратного преобразования "
                "получена некорректная стоимость."
            )

        price_low = max(0.0, price - self.mae)
        price_high = price + self.mae

        return {
            "predicted_price": round(price, 2),
            "price_range_low": round(price_low, 2),
            "price_range_high": round(price_high, 2),
            "currency": "TRY",
            "model_name": self.model_name,
        }


model_service = ModelService()
