import math
import pandas as pd

from api.config import DATA_PATH, DASHBOARD_SAMPLE_SIZE

FURNISHED_MAP = {
    "Eşyalı": "Меблировано",
    "Eşyasız": "Без мебели",
    "Sadece Beyaz Eşya": "Только бытовая техника",
    "Sadece Mutfak": "Только кухня",
}

LISTING_TYPE_MAP = {
    "Satılık": "Продажа",
    "Kiralık": "Аренда",
}

HEATING_TYPE_MAP = {
    "Kalorifer (Doğalgaz)": "Центральное (газ)",
    "Kalorifer (Kömür)": "Центральное (уголь)",
    "Kombi (Elektrikli)": "Электрический котёл",
    "Klima": "Кондиционер",
    "Kombi (Doğalgaz)": "Газовый котёл",
    "Merkezi Sistem": "Центральное отопление",
    "Merkezi Sistem (Isı Payı Ölçer)": "Центральное со счётчиком",
    "Yerden Isıtma": "Тёплый пол",
    "Soba (Kömür)": "Печь (уголь)",
    "Soba (Doğalgaz)": "Печь (газ)",
    "Güneş Enerjisi": "Солнечная энергия",
    "Jeotermal": "Геотермальная",
    "Fancoil": "Фанкойл",
    "Kat Kaloriferi": "Поэтажное центральное",
    "Kalorifer (Akaryakıt)": "Центральное (жидкое топливо)",
    "Yok": "Отсутствует",
}


def _json_safe_value(value):
    if value is None:
        return None

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


def _json_safe_records(frame: pd.DataFrame) -> list[dict]:
    safe = frame.copy()
    safe = safe.astype(object).where(pd.notna(safe), None)

    records = safe.to_dict(orient="records")

    return [
        {
            key: _json_safe_value(value)
            for key, value in record.items()
        }
        for record in records
    ]


class DataService:
    def __init__(self):
        self.df: pd.DataFrame | None = None

    def load(self) -> None:
        df = pd.read_csv(DATA_PATH, low_memory=False)

        if "furnished" in df.columns:
            df["furnished"] = df["furnished"].replace(FURNISHED_MAP)

        if "listing_type" in df.columns:
            df["listing_type"] = df["listing_type"].replace(LISTING_TYPE_MAP)

        if "heating_type" in df.columns:
            df["heating_type"] = df["heating_type"].replace(HEATING_TYPE_MAP)

        self.df = df

    def _require_loaded(self) -> pd.DataFrame:
        if self.df is None:
            raise RuntimeError("Датасет не загружен.")
        return self.df

    def summary(self) -> dict:
        df = self._require_loaded()

        return {
            "row_count": int(len(df)),
            "avg_price": float(df["price"].mean()),
            "median_price": float(df["price"].median()),
            "min_price": float(df["price"].min()),
            "max_price": float(df["price"].max()),
            "avg_size": float(df["size"].mean()),
        }

    def sample(self, n: int = DASHBOARD_SAMPLE_SIZE) -> list[dict]:
        df = self._require_loaded()

        n = len(df)

        cols = [
            c
            for c in [
                "size",
                "building_age",
                "room_count",
                "price",
                "heating_type",
                "listing_type",
            ]
            if c in df.columns
        ]

        sample_df = df[cols].sample(
            n=n,
            random_state=42,
        )

        return _json_safe_records(sample_df)

    def correlation_matrix(self) -> dict:
        df = self._require_loaded()

        numeric_df = df.select_dtypes(include="number")
        corr = numeric_df.corr(numeric_only=True).round(3)

        safe_corr = corr.astype(object).where(
            pd.notna(corr),
            None,
        )

        return {
            "columns": list(corr.columns),
            "values": safe_corr.values.tolist(),
        }

    def avg_price_by_category(
        self,
        column: str,
        agg: str = "mean",
    ) -> list[dict]:
        df = self._require_loaded()

        if column not in df.columns:
            raise ValueError(f"Неизвестная колонка: {column}")

        grouped = (
            df.groupby(column)["price"]
            .agg(agg)
            .reset_index()
        )

        return _json_safe_records(grouped)


data_service = DataService()
