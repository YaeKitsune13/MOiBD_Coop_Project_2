from enum import Enum

from pydantic import BaseModel, Field, field_validator


class ListingType(str, Enum):
    sale = "Продажа"
    rent = "Аренда"


class Furnished(str, Enum):
    furnished = "Меблировано"
    unfurnished = "Без мебели"
    appliances_only = "Только бытовая техника"
    kitchen_only = "Только кухня"


class HeatingType(str, Enum):
    central_gas = "Центральное (газ)"
    combi_gas = "Газовый котёл"
    ac = "Кондиционер"
    central_system = "Центральное отопление"
    underfloor = "Тёплый пол"
    stove_coal = "Печь (уголь)"
    solar = "Солнечная энергия"
    none = "Отсутствует"


LISTING_TYPE_TO_RAW = {
    ListingType.sale: "Satılık",
    ListingType.rent: "Kiralık",
}

FURNISHED_TO_RAW = {
    Furnished.furnished: "Eşyalı",
    Furnished.unfurnished: "Eşyasız",
    Furnished.appliances_only: "Sadece Beyaz Eşya",
    Furnished.kitchen_only: "Sadece Mutfak",
}

HEATING_TYPE_TO_RAW = {
    HeatingType.central_gas: "Kalorifer (Doğalgaz)",
    HeatingType.combi_gas: "Kombi (Doğalgaz)",
    HeatingType.ac: "Klima",
    HeatingType.central_system: "Merkezi Sistem",
    HeatingType.underfloor: "Yerden Isıtma",
    HeatingType.stove_coal: "Soba (Kömür)",
    HeatingType.solar: "Güneş Enerjisi",
    HeatingType.none: "Yok",
}


class PredictRequest(BaseModel):
    size: float = Field(
        ...,
        gt=0,
        le=2000,
        description="Площадь, м²",
    )

    room_count: str = Field(
        ...,
        description='Формат "N+1", например "2+1"',
    )

    building_age: int = Field(
        ...,
        ge=0,
        le=150,
        description="Возраст здания, лет",
    )

    total_floor_count: int = Field(
        ...,
        ge=1,
        le=100,
        description="Этажей в здании",
    )

    floor_no: int = Field(
        ...,
        ge=-3,
        le=100,
        description="Номер этажа",
    )

    listing_type: ListingType
    furnished: Furnished
    heating_type: HeatingType

    address: str = Field(
        ...,
        min_length=2,
        max_length=200,
    )

    @field_validator("room_count")
    @classmethod
    def validate_room_count(cls, v: str) -> str:
        if "+" not in v:
            raise ValueError(
                'room_count должен быть в формате "N+M", например "2+1"'
            )

        parts = v.split("+")

        if len(parts) != 2 or not all(
            p.strip().isdigit()
            for p in parts
        ):
            raise ValueError(
                'room_count должен быть в формате "N+M", например "2+1"'
            )

        return v

    @field_validator("floor_no")
    @classmethod
    def validate_floor_vs_total(cls, v: int, info) -> int:
        total = info.data.get("total_floor_count")

        if total is not None and v > total:
            raise ValueError(
                "floor_no не может быть больше total_floor_count"
            )

        return v


class PredictResponse(BaseModel):
    predicted_price: float
    price_range_low: float
    price_range_high: float
    currency: str = "TRY"
    model_name: str


class ModelMetric(BaseModel):
    name: str
    mae: float
    rmse: float
    r2: float


class DatasetSummary(BaseModel):
    row_count: int
    avg_price: float
    median_price: float
    min_price: float
    max_price: float
    avg_size: float


class AppInfo(BaseModel):
    version: str
    description: str
    model_name: str
    model_limitations: str
