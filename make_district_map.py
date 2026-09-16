import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "api" / "real_estate_data.csv"
OUTPUT_PATH = BASE_DIR / "models" / "district_te_map.json"


print("Loading CSV...")
df = pd.read_csv(DATA_PATH, low_memory=False)

print(f"Original rows: {len(df):,}")

df = df.drop(columns=["type"], errors="ignore")

df = df[df["price_currency"] == "TRY"].reset_index(drop=True)

df = df.drop(columns=["price_currency"])

print(f"After TRY filter: {len(df):,}")

age_map = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6-10 arası": 8,
    "11-15 arası": 13,
    "16-20 arası": 18,
    "21-25 arası": 23,
    "26-30 arası": 28,
    "31 ve üzeri": 35,
}

df["building_age_num"] = df["building_age"].map(age_map)
df = df.drop(columns=["building_age"])

df["building_age_num"] = df["building_age_num"].fillna(
    df["building_age_num"].median()
)

floor_map = {
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10-20 arası": 15,
    "20 ve üzeri": 25,
    "Müstakil": 1,
}

df["total_floor_num"] = df["total_floor_count"].map(floor_map)

df = df.drop(columns=["total_floor_count"])

def parse_floor(x):
    x = str(x).strip()

    if x.isdigit():
        return int(x)

    special = {
        "Yüksek Giriş": 1,
        "Giriş": 0,
        "Kot 1": -1,
        "Kot 2": -2,
        "Asma Kat": 1,
        "Müstakil": 1,
        "Çatı Katı": 99,
    }

    return special.get(x, np.nan)


df["floor_num"] = df["floor_no"].apply(parse_floor)

df["floor_num"] = df["floor_num"].fillna(
    df["floor_num"].median()
)

df = df.drop(columns=["floor_no"])

def parse_rooms(x):
    parts = str(x).replace(" ", "").split("+")

    try:
        return (
            int(parts[0]),
            int(parts[1]) if len(parts) > 1 else 0
        )
    except Exception:
        return np.nan, np.nan


df[["rooms_living", "rooms_extra"]] = df["room_count"].apply(
    lambda x: pd.Series(parse_rooms(x))
)

df["rooms_living"] = df["rooms_living"].fillna(
    df["rooms_living"].median()
)

df["rooms_extra"] = df["rooms_extra"].fillna(0)

df = df.drop(columns=["room_count"])

df["start_date"] = pd.to_datetime(
    df["start_date"],
    format="%m/%d/%y",
    errors="coerce"
)

df["start_year"] = df["start_date"].dt.year
df["start_month"] = df["start_date"].dt.month
df["start_dow"] = df["start_date"].dt.dayofweek

df = df.drop(columns=["start_date"])

addr_split = df["address"].str.split(
    "/",
    n=2,
    expand=True
)

df["city"] = addr_split[0].str.strip()

df["district"] = (
    addr_split[1].str.strip()
    if addr_split.shape[1] > 1
    else "Unknown"
)

df = df.drop(columns=["address"])

df["floor_ratio"] = (
    df["floor_num"] /
    df["total_floor_num"].replace(0, 1)
)

df["rooms_total"] = (
    df["rooms_living"] +
    df["rooms_extra"]
)

df["size_per_room"] = (
    df["size"] /
    df["rooms_total"].replace(0, 1)
)

df["price_log"] = np.log1p(df["price"])

df = df[
    (df["size"] > 0) &
    (df["price"] > 0)
].reset_index(drop=True)

X = df.drop(
    columns=["price", "price_log"],
    errors="ignore"
)

y = df["price_log"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42
)


print(f"Train: {len(X_train):,}")
print(f"Val:   {len(X_val):,}")
print(f"Test:  {len(X_test):,}")

prior = y_train.mean()
m = 10

agg = (
    pd.DataFrame({
        "district": X_train["district"],
        "y": y_train
    })
    .groupby("district")["y"]
    .agg(["mean", "count"])
)

agg = agg[agg["count"] >= 5]

smooth = (
    (agg["mean"] * agg["count"] + prior * m)
    / (agg["count"] + m)
)

mapping = {
    "prior": float(prior),
    "smoothing": m,
    "min_count": 5,
    "mapping": {
        str(k): float(v)
        for k, v in smooth.items()
    }
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(
        mapping,
        f,
        ensure_ascii=False,
        indent=2
    )


print()
print("=" * 60)
print("DONE")
print("=" * 60)
print(f"Districts: {len(smooth):,}")
print(f"Prior:     {prior:.6f}")
print(f"Output:    {OUTPUT_PATH}")
