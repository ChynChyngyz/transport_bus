import datetime
from functools import lru_cache
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    from shapely.geometry import LineString
except ImportError as impexc:
    print(impexc)
    print("You must install GeoPandas to use this module.")
    raise

DATE_FORMAT = "%Y%m%d"


@lru_cache(maxsize=2**18)
def parse_time(val: str) -> Any:
    """
    Функция `parse_time` принимает строку, представляющую значение времени в формате "hh:mm:ss", и
    возвращает эквивалентное время в секундах как numpy int или возвращает входное значение, если оно
    уже является numpy int или int.

    Аргументы:
    val (str): Параметр `val` — это строка, представляющая значение времени в формате "hh:mm:ss".

    Возвращает:
    значение типа np.float32.
    """
    if isinstance(val, float):
        return val
    
    try:
        val = str(val).strip()
        h, m, s = map(float, val.split(":"))
        return np.float32(h * 3600 + m * 60 + s)
    except ValueError:
        return np.nan


@lru_cache(maxsize=2**18)
def parse_date(val: str) -> datetime.date:
    """
    Функция `parse_date` принимает строку или объект `datetime.date` в качестве входных данных и возвращает
    объект `datetime.date`.

    Аргументы:
    val (str): Параметр `val` — это строка, представляющая дату.

    Возвращает:
    объект `datetime.date`.
    """
    if isinstance(val, datetime.date):
        return val
    return datetime.datetime.strptime(val, DATE_FORMAT).date()


@lru_cache(maxsize=2**18)
def parse_float(val: Any) -> float:
    if isinstance(val, float):
        return val
    try:
        return float(val)
    except ValueError:
        return np.nan

@lru_cache(maxsize=2**18)
def parse_integer(val: Any) -> Union[int, float]:
    if isinstance(val, int) or isinstance(val,float):
        return val
    try:
        return int(val)
    except ValueError:
        return np.nan


vparse_float = lambda x : pd.to_numeric(x, errors="coerce", downcast =None) # np.vectorize(parse_float)
vparse_int = lambda x : pd.to_numeric(x, errors="coerce", downcast ="integer") # np.vectorize(parse_integer)
vparse_time = np.vectorize(parse_time)
vparse_date = np.vectorize(parse_date)

DEFAULT_CRS = "EPSG:4326"


def build_shapes(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Функция принимает pandas DataFrame, содержащий точки формы, и возвращает GeoDataFrame с
    идентификаторами форм и соответствующими геометриями.

    Аргументы:
    df (pd.DataFrame): Параметр `df` — это pandas DataFrame, содержащий информацию
    о формах. Ожидается, что он будет иметь следующие столбцы:

    Возвращает:
    объект GeoDataFrame.
    """
    if df.empty:
        return gpd.GeoDataFrame({"shape_id": [], "geometry": []})

    data: Dict[str, List] = {"shape_id": [], "geometry": []}
    for shape_id, shape in df.sort_values("shape_pt_sequence").groupby("shape_id"):
        data["shape_id"].append(shape_id)
        data["geometry"].append(LineString(list(zip(shape.shape_pt_lon, shape.shape_pt_lat))))

    return gpd.GeoDataFrame(data, crs=DEFAULT_CRS)


def build_stops(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Функция `build_stops` принимает pandas DataFrame `df` и возвращает GeoDataFrame с
    теми же данными, но с новым столбцом геометрии, созданным из столбцов `stop_lon` и `stop_lat`.

    Аргументы:
    df (pd.DataFrame): Pandas DataFrame, содержащий информацию об остановках. Ожидается, что
    он будет иметь столбцы с именами "stop_lon" и "stop_lat", которые представляют координаты долготы и
    широты каждой остановки соответственно. DataFrame также может содержать другие столбцы
    с дополнительной информацией об остановках.

    Возвращает:
    GeoDataFrame со столбцом геометрии, содержащим точки, созданные из столбцов stop_lon и stop_lat
    входного DataFrame. Затем столбцы stop_lon и stop_lat удаляются из
    DataFrame перед возвратом окончательного GeoDataFrame.
    """
    if df.empty:
        return gpd.GeoDataFrame(df, geometry=[], crs=DEFAULT_CRS)

    df = gpd.GeoDataFrame(
        df, crs=DEFAULT_CRS, geometry=gpd.points_from_xy(df.stop_lon, df.stop_lat)
    )

    df.drop(["stop_lon", "stop_lat"], axis=1, inplace=True)

    return gpd.GeoDataFrame(df, crs=DEFAULT_CRS)


def transforms_dict() -> Dict[str, Dict[str, Any]]:
    """
    Функция `transforms_dict` возвращает словарь, который определяет требуемые столбцы и
    конвертеры для каждого файла в транзитном потоке данных.

    Возвращает:
    словарь, содержащий информацию о различных текстовых файлах и их требуемых столбцах и
    конвертерах.
    """
    return_dict = {
        "agency.txt": {
            "usecols": {
                "agency_name": "str",
                "agency_url": "str",
                "agency_timezone": "str",
                "agency_lang": "str",
                "agency_phone": "int",
                "agency_fare_url": "str",
                "agency_email": "str",
            },
            "required_columns": ("agency_name", "agency_url", "agency_timezone"),
        },
        "calendar.txt": {
            "usecols": {
                "service_id": "str",
                "start_date": "str",
                "end_date": "str",
                "monday": "bool",
                "tuesday": "bool",
                "wednesday": "bool",
                "thursday": "bool",
                "friday": "bool",
                "saturday": "bool",
                "sunday": "bool",
            },
            "converters": {
                "start_date": vparse_date,
                "end_date": vparse_date,
                "monday": vparse_float,
                "tuesday": vparse_float,
                "wednesday": vparse_float,
                "thursday": vparse_float,
                "friday": vparse_float,
                "saturday": vparse_float,
                "sunday": vparse_float,
            },
            "required_columns": (
                "service_id",
                "monday",
                "tuesday",
                "wednesday",
                "thursday",
                "friday",
                "saturday",
                "sunday",
                "start_date",
                "end_date",
            ),
        },
        "calendar_dates.txt": {
            "usecols": {"service_id": "str", "date": "str", "exception_type": "int8"},
            "converters": {
                "date": vparse_date,
                "exception_type": vparse_float,
            },
            "required_columns": ("service_id", "date", "exception_type"),
        },
        "fare_attributes.txt": {
            "usecols": {
                "fare_id": "str",
                "price": "float",
                "currency_type": "str",
                "payment_method": "str",
                "transfers": "str",
                "transfer_duration": "float16",
            },
            "converters": {
                "price": vparse_float,
                "payment_method": vparse_float,
                "transfer_duration": vparse_float,
            },
            "required_columns": (
                "fare_id",
                "price",
                "currency_type",
                "payment_method",
                "transfers",
            ),
        },
        "fare_rules.txt": {
            "usecols": {
                "fare_id": "str",
                "route_id": "str",
                "origin_id": "str",
                "destination_id": "str",
                "contains_id": "str",
            },
            "required_columns": ("fare_id",),
        },
        "feed_info.txt": {
            "usecols": {
                "feed_publisher_name": "str",
                "feed_publisher_url": "str",
                "feed_lang": "str",
                "feed_start_date": "str",
                "feed_end_date": "str",
            },
            "converters": {
                "feed_start_date": vparse_date,
                "feed_end_date": vparse_date,
            },
            "required_columns": (
                "feed_publisher_name",
                "feed_publisher_url",
                "feed_lang",
            ),
        },
        "frequencies.txt": {
            "usecols": {
                "trip_id": "str",
                "start_time": "float32",
                "end_time": "float32",
                "headway_secs": "float32",
                "exact_times": "bool",
            },
            "converters": {
                "headway_secs": vparse_float,
                "exact_times": vparse_float,
                "start_time": vparse_time,
                "end_time": vparse_time,
            },
            "required_columns": (
                "trip_id",
                "start_time",
                "end_time",
                "headway_secs",
            ),
        },
        "routes.txt": {
            "usecols": {
                "route_id": "str",
                "route_short_name": "str",
                "route_long_name": "str",
                "route_type": "int8",
                # "route_color": "str",
                # "route_text_color": "str",
            },
            "converters": {
                "route_type": vparse_float,
            },
            "required_columns": (
                "route_id",
                "route_short_name",
                "route_long_name",
                "route_type",
            ),
        },
        "shapes.txt": {
            "usecols": {
                "shape_id": "str",
                "shape_pt_lat": "float32",
                "shape_pt_lon": "float32",
                "shape_pt_sequence": "int16",
                # "shape_dist_traveled":"float32",
            },
            "converters": {
                "shape_pt_lat": vparse_float,
                "shape_pt_lon": vparse_float,
                "shape_pt_sequence": vparse_float,
                # "shape_dist_traveled": vparse_float,
            },
            "required_columns": (
                "shape_id",
                "shape_pt_lat",
                "shape_pt_lon",
                "shape_pt_sequence",
            ),
            "transformations": [build_shapes],
        },
        "stops.txt": {
            "usecols": {
                "stop_id": "str",
                "stop_name": "str",
                "stop_lat": "float32",
                "stop_lon": "float32",
                "location_type": "int8",
                "wheelchair_boarding":"int8",
            },
            "converters": {
                "stop_lat": vparse_float,
                "stop_lon": vparse_float,
                "location_type": vparse_int,
                "wheelchair_boarding": vparse_int,
            },
            "required_columns": (
                "stop_id",
                "stop_name",
                "stop_lat",
                "stop_lon",
            ),
            "transformations": [build_stops],
        },
        "stop_times.txt": {
            "usecols": {
                "trip_id": "str",
                "arrival_time": "float32",
                # "departure_time",
                "stop_id": "str",
                "stop_sequence": vparse_int,
                "pickup_type": vparse_int,
                "drop_off_type": vparse_int,
                "shape_dist_traveled": vparse_float,
                "timepoint":"bool",
            },
            "converters": {
                "arrival_time": vparse_time,
                "departure_time": vparse_time,
                "pickup_type": vparse_int,
                "drop_off_type": vparse_int,
                "shape_dist_traveled": vparse_float,
                "stop_sequence": vparse_int,
                "timepoint": vparse_int,
            },
            "required_columns": (
                "trip_id",
                "arrival_time",
                "departure_time",
                "stop_id",
                "stop_sequence",
            ),
        },
        "transfers.txt": {
            "usecols": ["from_stop_id", "to_stop_id", "transfer_type", "min_transfer_time"],
            "converters": {
                "transfer_type": vparse_float,
                "min_transfer_time": vparse_float,
            },
            "required_columns": ("from_stop_id", "to_stop_id", "transfer_type"),
        },
        "trips.txt": {
            "usecols": {
                "route_id": "str",
                "shape_id": "str",
                "service_id": "str",
                "trip_id": "str",
                "direction_id": "bool",
                # "wheelchair_accessible": "int8",
                # "bikes_allowed":"int8",
            },
            "converters": {
                "direction_id": vparse_float,
                # "wheelchair_accessible": vparse_float,
                "bikes_allowed": vparse_float,
            },
            "required_columns": ("route_id", "service_id", "trip_id"),
        },
    }
    return return_dict
