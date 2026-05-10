from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import h3
import pandas as pd


LoaderFn = Callable[[], dict[str, pd.DataFrame]]
PreprocessFn = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass(frozen=True)
class TaskSpec:
    key: str
    name: str
    task_type: str
    target: str
    categorical_cols: tuple[str, ...]
    drop_cols: tuple[str, ...]
    loader: LoaderFn
    preprocess: PreprocessFn

    def load_splits(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        data = self.loader()
        if "train" not in data or "test" not in data:
            raise ValueError(f"Task '{self.key}' loader must return 'train' and 'test' splits")

        train_df = self.preprocess(data["train"].copy())
        test_df = self.preprocess(data["test"].copy())

        if "h3_index" not in train_df.columns or "h3_index" not in test_df.columns:
            raise ValueError(f"Task '{self.key}' preprocessing must produce an h3_index column")

        return train_df, test_df


def _load_airbnb() -> dict[str, pd.DataFrame]:
    from srai.datasets import AirbnbMulticityDataset

    dataset_loader = AirbnbMulticityDataset()
    return dataset_loader.load()


def _prep_airbnb(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["last_review"] = pd.to_datetime(out["last_review"])
    ref_date = pd.Timestamp("2024-01-01")
    out["days_since_review"] = (ref_date - out["last_review"]).dt.days
    out["days_since_review"] = out["days_since_review"].fillna(3650)
    out["name_length"] = out["name"].str.len().fillna(0)
    out["h3_index"] = [h3.latlng_to_cell(y, x, 9) for x, y in zip(out.geometry.x, out.geometry.y)]
    return out


def _load_king_county() -> dict[str, pd.DataFrame]:
    from srai.datasets import HouseSalesInKingCountyDataset

    dataset_loader = HouseSalesInKingCountyDataset()
    return dataset_loader.load()


def _prep_king_county(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["h3_index"] = [h3.latlng_to_cell(y, x, 9) for x, y in zip(out.geometry.x, out.geometry.y)]
    out["date"] = pd.to_datetime(out["date"])
    out["sale_year"] = out["date"].dt.year
    out["sale_month"] = out["date"].dt.month
    out["house_age"] = 2016 - out["yr_built"]
    out["has_basement"] = (out["sqft_basement"] > 0).astype(int)
    return out


def _load_sf_crime() -> dict[str, pd.DataFrame]:
    from srai.datasets import PoliceDepartmentIncidentsDataset

    dataset_loader = PoliceDepartmentIncidentsDataset()
    dataset_loader.target = "count"
    return dataset_loader.load()


def _prep_sf_crime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["h3_index"] = [h3.latlng_to_cell(y, x, 9) for x, y in zip(out.geometry.x, out.geometry.y)]
    out["incident_datetime"] = pd.to_datetime(out["Incident Datetime"], format="mixed")
    out["hour"] = out["incident_datetime"].dt.hour
    out["is_night"] = out["hour"].apply(lambda x: 1 if (x >= 22 or x <= 6) else 0)
    out["is_weekend"] = out["incident_datetime"].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)

    agg_funcs = {
        "h3_index": "count",
        "is_night": "mean",
        "is_weekend": "mean",
        "Police District": lambda x: x.mode()[0] if not x.mode().empty else "Unknown",
    }
    hex_data = out.groupby("h3_index").agg(agg_funcs).rename(columns={"h3_index": "count"}).reset_index()

    top_crimes = ["Larceny Theft", "Malicious Mischief", "Assault", "Motor Vehicle Theft", "Non-Criminal"]
    type_counts = (
        out[out["Incident Category"].isin(top_crimes)]
        .pivot_table(index="h3_index", columns="Incident Category", aggfunc="size", fill_value=0)
        .add_prefix("type_")
    )
    hex_data = hex_data.join(type_counts, on="h3_index").fillna(0)
    hex_data["lat"] = hex_data["h3_index"].apply(lambda x: h3.cell_to_latlng(x)[0])
    hex_data["lon"] = hex_data["h3_index"].apply(lambda x: h3.cell_to_latlng(x)[1])
    return hex_data


def _load_chicago_crime() -> dict[str, pd.DataFrame]:
    from srai.datasets import ChicagoCrimeDataset

    dataset_loader = ChicagoCrimeDataset()
    dataset_loader.target = "count"
    return dataset_loader.load()


def _prep_chicago_crime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["h3_index"] = [h3.latlng_to_cell(y, x, 9) for x, y in zip(out.geometry.x, out.geometry.y)]

    date_col = "Date" if "Date" in out.columns else "date"
    out["date_clean"] = pd.to_datetime(out[date_col], format="mixed")
    out["hour"] = out["date_clean"].dt.hour
    out["is_night"] = out["hour"].apply(lambda x: 1 if (x >= 22 or x <= 6) else 0)

    arrest_col = "Arrest" if "Arrest" in out.columns else "arrest"
    out["arrest_flag"] = out[arrest_col].astype(int) if arrest_col in out.columns else 0

    dist_col = "District" if "District" in out.columns else "district"
    agg_funcs = {
        "h3_index": "count",
        "is_night": "mean",
        "arrest_flag": "mean",
        dist_col: lambda x: x.mode()[0] if not x.mode().empty else "Unknown",
    }
    hex_data = out.groupby("h3_index").agg(agg_funcs).rename(columns={"h3_index": "count"}).reset_index()

    type_col = "Primary Type" if "Primary Type" in out.columns else "primary_type"
    top_crimes = ["THEFT", "BATTERY", "CRIMINAL DAMAGE", "NARCOTICS", "ASSAULT"]
    type_counts = (
        out[out[type_col].isin(top_crimes)]
        .pivot_table(index="h3_index", columns=type_col, aggfunc="size", fill_value=0)
        .add_prefix("type_")
    )

    hex_data = hex_data.join(type_counts, on="h3_index").fillna(0)
    hex_data["lat"] = hex_data["h3_index"].apply(lambda x: h3.cell_to_latlng(x)[0])
    hex_data["lon"] = hex_data["h3_index"].apply(lambda x: h3.cell_to_latlng(x)[1])
    if dist_col != "district":
        hex_data = hex_data.rename(columns={dist_col: "district"})
    return hex_data


def _load_philadelphia_crime() -> dict[str, pd.DataFrame]:
    from srai.datasets import PhiladelphiaCrimeDataset

    dataset_loader = PhiladelphiaCrimeDataset()
    dataset_loader.target = "count"
    return dataset_loader.load()


def _prep_philadelphia_crime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["h3_index"] = [h3.latlng_to_cell(y, x, 9) for x, y in zip(out.geometry.x, out.geometry.y)]
    date_col = "dispatch_date_time" if "dispatch_date_time" in out.columns else "dispatch_date"
    out["date"] = pd.to_datetime(out[date_col])
    out["hour"] = out["date"].dt.hour
    out["is_night"] = out["hour"].apply(lambda x: 1 if (x >= 22 or x <= 6) else 0)

    agg_funcs = {
        "h3_index": "count",
        "is_night": "mean",
        "dc_dist": lambda x: x.mode()[0] if not x.mode().empty else "Unknown",
    }
    hex_data = out.groupby("h3_index").agg(agg_funcs).rename(columns={"h3_index": "count"}).reset_index()

    actual_top = out["text_general_code"].value_counts().head(5).index.tolist()
    type_counts = (
        out[out["text_general_code"].isin(actual_top)]
        .pivot_table(index="h3_index", columns="text_general_code", aggfunc="size", fill_value=0)
        .add_prefix("type_")
    )
    hex_data = hex_data.join(type_counts, on="h3_index").fillna(0)
    hex_data["lat"] = hex_data["h3_index"].apply(lambda x: h3.cell_to_latlng(x)[0])
    hex_data["lon"] = hex_data["h3_index"].apply(lambda x: h3.cell_to_latlng(x)[1])
    return hex_data


def _load_beijing_housing() -> dict[str, pd.DataFrame]:
    import geopandas as gpd
    import kagglehub
    from shapely.geometry import Point
    from sklearn.model_selection import train_test_split

    dataset_handle = "ruiqurm/lianjia"
    dataset_path = kagglehub.dataset_download(dataset_handle)

    target_file = ""
    for filename in os.listdir(dataset_path):
        if filename.lower().endswith(".csv") and "new.csv" in filename.lower():
            target_file = os.path.join(dataset_path, filename)
            break
    if not target_file:
        for filename in os.listdir(dataset_path):
            if filename.lower().endswith(".csv"):
                target_file = os.path.join(dataset_path, filename)
                break
    if not target_file:
        raise FileNotFoundError(f"No CSV files found in {dataset_path}")

    try:
        raw = pd.read_csv(target_file, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        raw = pd.read_csv(target_file, encoding="gb18030", low_memory=False)

    raw = raw.dropna(subset=["Lng", "Lat"])
    raw = raw[(raw["Lng"] > 70) & (raw["Lng"] < 140) & (raw["Lat"] > 10) & (raw["Lat"] < 60)]

    geometry = [Point(xy) for xy in zip(raw.Lng, raw.Lat)]
    gdf = gpd.GeoDataFrame(raw, geometry=geometry, crs="EPSG:4326")

    gdf["constructionTime"] = pd.to_numeric(gdf["constructionTime"], errors="coerce")
    gdf["constructionTime"] = gdf["constructionTime"].fillna(gdf["constructionTime"].median())
    gdf = gdf.dropna(subset=["totalPrice"])

    if "price" in gdf.columns:
        gdf = gdf.drop(columns=["price"])

    train_df, test_df = train_test_split(gdf, test_size=0.2, random_state=42)
    return {"train": train_df, "test": test_df}


def _prep_beijing_housing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = ["square", "livingRoom", "drawingRoom", "kitchen", "bathRoom", "communityAverage"]
    for column in numeric_cols:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0)

    if "constructionTime" in out.columns:
        out["building_age"] = 2024 - out["constructionTime"]
        out = out.drop(columns=["constructionTime"])

    out["h3_index"] = [h3.latlng_to_cell(y, x, 9) for x, y in zip(out.geometry.x, out.geometry.y)]
    return out


TASK_REGISTRY: dict[str, TaskSpec] = {
    "airbnb": TaskSpec(
        key="airbnb",
        name="Airbnb Multicity Dataset",
        task_type="regression",
        target="price",
        categorical_cols=("room_type", "city"),
        drop_cols=("id", "host_id", "name", "host_name", "neighbourhood", "last_review", "date"),
        loader=_load_airbnb,
        preprocess=_prep_airbnb,
    ),
    "king_county": TaskSpec(
        key="king_county",
        name="King County House Sales Dataset",
        task_type="regression",
        target="price",
        categorical_cols=("zipcode",),
        drop_cols=("id", "date"),
        loader=_load_king_county,
        preprocess=_prep_king_county,
    ),
    "san_francisco_crime": TaskSpec(
        key="san_francisco_crime",
        name="SF Crime",
        task_type="regression",
        target="count",
        categorical_cols=("Police District",),
        drop_cols=("h3_index",),
        loader=_load_sf_crime,
        preprocess=_prep_sf_crime,
    ),
    "chicago_crime": TaskSpec(
        key="chicago_crime",
        name="Chicago Crime Dataset",
        task_type="regression",
        target="count",
        categorical_cols=("district",),
        drop_cols=("h3_index",),
        loader=_load_chicago_crime,
        preprocess=_prep_chicago_crime,
    ),
    "philadelphia_crime": TaskSpec(
        key="philadelphia_crime",
        name="Philadelphia Crime",
        task_type="regression",
        target="count",
        categorical_cols=("dc_dist",),
        drop_cols=("h3_index",),
        loader=_load_philadelphia_crime,
        preprocess=_prep_philadelphia_crime,
    ),
    "beijing_housing": TaskSpec(
        key="beijing_housing",
        name="Beijing Housing",
        task_type="regression",
        target="totalPrice",
        categorical_cols=("buildingType", "renovationCondition", "buildingStructure", "elevator", "district", "subway"),
        drop_cols=("url", "id", "Cid", "tradeTime", "DOM", "floor"),
        loader=_load_beijing_housing,
        preprocess=_prep_beijing_housing,
    ),
}
