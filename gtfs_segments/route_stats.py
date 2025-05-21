from datetime import timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import utm
from numpy.typing import NDArray
from shapely.geometry import LineString

from .geom_utils import code
from .partridge_mod.gtfs import Feed


def get_zone_epsg_from_ls(geom: LineString) -> int:
    """
    > Функция принимает фрейм данных со столбцом геометрии и возвращает код EPSG для зы UTM,
    содержащей геометрию

    Аргументы:
    stop_df: фрейм данных со столбцом геометрии

    Возвращает:
    Код EPSG для зы UTM, в которой находится остановка.
    """
    lon = geom.coords[0][0]
    lat = geom.coords[0][1]
    zone = utm.from_latlon(lat, lon)
    return code(zone, lat)


def get_sec(time_str: str) -> int:
    """
    принимает строку в формате чч:мм:сс и возвращает количество секунд

    Аргументы:
    time_str: Строка времени для преобразования в секунды.

    Возвращает:
    общее количество секунд в строке времени.
    """
    h, m, s = time_str.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def get_trips_len(df: pd.DataFrame, time: int) -> int:
    """
    > Возвращает количество поездок, которые в данный момент активны в указанное время

    Аргументы:
    df: датафрейм поездок
    time: время в секундах

    Возвращает:
    Количество поездок, которые активны в указанное время.
    """
    return len(df[(df.start_time <= time) & (df.end_time >= time)].trip_id.unique())


def get_trips(df: pd.DataFrame, time: int) -> pd.DataFrame:
    """
    > Возвращает все поездки, которые в данный момент активны в указанное время

    Аргументы:
    df: датафрейм, содержащий поездки
    time: время в секундах

    Возвращает:
    датафрейм всех поездок, которые начинаются до времени и заканчиваются после времени.
    """
    return df[(df.start_time <= time) & (df.end_time >= time)]


def get_peak_time(df: pd.DataFrame) -> List:
    """
    Функция `get_peak_time` принимает фрейм данных автобусных поездок и возвращает количество автобусов и время, в которое ходит больше всего автобусов.

    Аргументы:
    df: Параметр `df` — это фрейм данных, содержащий информацию об автобусах. Скорее всего, имеет столбцы,
    такие как `start_time` и `end_time`, которые представляют время начала и окчания каждой автобусной поездки.

    Возвращает:
    список, содержащий количество автобусов и время, в которое ходит больше всего автобусов.
    """
    best = 0
    peak_time = 0
    start_time = int(min(df.start_time))
    end_time = int(max(df.end_time))
    for time in range(start_time, end_time, 60):
        no_buses = get_trips_len(df, time)
        if no_buses >= best:
            peak_time = str(timedelta(seconds=time))
            best = no_buses
    return [best, peak_time]


def get_service_length(df: pd.DataFrame) -> int:
    """
    Функция принимает канал GTFS и route_id, сортирует фрейм данных по stop_sequence, корректирует значения shape_dist_traveled при необходимости и возвращает максимальное значение shape_dist_traveled.

    Аргументы:
    df: Параметр `df` — это фрейм данных, содержащий данные о времени остановок из канала GTFS.

    Возвращает:
    Максимальное значение столбца 'shape_dist_traveled' для поездки.
    """
    df = df.sort_values(["stop_sequence"])
    if df.iloc[0]["pickup_type"] == 1:
        sp_dist = df.iloc[1]["shape_dist_traveled"]
        df.shape_dist_traveled = df.shape_dist_traveled - sp_dist
        df.drop(index=df.iloc[0, :].index.tolist(), inplace=True, axis=0)
    return df.shape_dist_traveled.max()


def get_route_grp(route_df: pd.DataFrame) -> pd.DataFrame:
    """
    принимает фрейм данных с информацией о маршруте и возвращает фрейм данных с информацией о маршруте с
    первой и последней остановкой для каждой поездки

    Аргументы:
    route_df: фрейм данных, содержащий информацию о маршруте

    Возвращает:
    фрейм данных с первой и последней остановкой каждой поездки.
    """
    route_df = route_df.sort_values(["stop_sequence"])
    route_df_grp = route_df.groupby(["trip_id"]).first()
    route_df_grp["start_time"] = route_df.groupby(["trip_id"]).first().arrival_time
    route_df_grp["end_time"] = route_df.groupby(["trip_id"]).last().arrival_time
    route_df_grp = route_df_grp.reset_index()
    col_filter = np.array(
        [
            "trip_id",
            "route_id",
            "direction_id",
            "start_time",
            "end_time",
            "pickup_type",
            "drop_off_type",
            "shape_dist_traveled",
        ]
    )
    col_subset = col_filter[np.in1d(col_filter, route_df_grp.columns)]
    return route_df_grp[col_subset]


def get_all_peak_times(df_dir: pd.DataFrame) -> Dict[str, NDArray[Any]]:
    """
    берет фрейм данных автобусных поездок и возвращает пиковое количество автобусов и время пика
    
    Аргументы:
    df: фрейм данных маршрута, для которого вы хотите получить пиковое время
    
    Возвращает:
    Словарь с пиковым количеством автобусов в каждом направлении и пиковым временем.
    """
    best = 0
    peak_times = []
    df = get_route_grp(df_dir)
    start_time = int(min(df.start_time))
    end_time = int(max(df.end_time))
    for time in range(start_time, end_time, 60):
        no_buses = get_trips_len(df, time)
        if no_buses == best:
            peak_times.append(time)
        if no_buses > best:
            peak_times = [time]
    peak_time_condensed = np.array([str(timedelta(seconds=peak_times[0]))], dtype="object")
    for i, time in enumerate(peak_times):
        if i > 0:
            if (time - peak_times[i - 1]) != 60:
                time = str(timedelta(seconds=time))
                peak_time_condensed[-1] = (
                    str(peak_time_condensed[-1]) + "-" + str(timedelta(seconds=peak_times[i - 1]))
                )
                peak_time_condensed = np.append(peak_time_condensed, str(time))
            else:
                if i == len(peak_times) - 1:
                    peak_time_condensed[-1] = str(
                        peak_time_condensed[-1] + "-" + str(timedelta(seconds=peak_times[i]))
                    )
    return {"peak_buses": peak_time_condensed}


def get_average_headway(df_dir: pd.DataFrame) -> Dict[str, float]:
    """
    Для каждого маршрута найдите форму с наибольшим количеством поездок, затем найдите остановку с наибольшим количеством поездок на этой форме, затем найдите средний интервал для этой остановки

    Аргументы:
    df_route: фрейм данных, содержащий маршрут, который вы хотите проанализировать

    Возвращает:
    Словарь со средним интервалом для каждого направления.
    """
    hdw_0 = np.array([0])
    if len(df_dir) > 1:
        shape_0 = df_dir.groupby("shape_id").count().trip_id.idxmax()
        df_dir = df_dir[df_dir.shape_id == shape_0]
        stop_id0 = df_dir.stop_id.unique()[0]
        hdw_0 = (
            df_dir[df_dir.stop_id == stop_id0]
            .sort_values(["arrival_time"])
            .arrival_time.astype(int)
            .diff()
        )
    return {"headway": np.round(hdw_0[hdw_0 <= 3 * 60 * 60].mean() / 3600, 2)}


def get_average_speed(df_dir: pd.DataFrame, route_dict: dict) -> Dict[str, float]:
    """
    Принимает фрейм данных маршрута и словарь информации о маршруте и возвращает словарь
    средних скоростей для каждого направления

    Аргументы:
    df_route: фрейм данных маршрута
    route_dict: словарь, содержащий длину маршрута и общее время для каждого направления

    Возвращает:
    Словарь со средней скоростью для каждого направления.
    """
    ret_dict = {}
    if len(df_dir) > 1:
        ret_dict["average_speed"] = np.round(
            route_dict["route_length"] / route_dict["total_time"], 2
        )
    return ret_dict


def get_route_time(df_dir: pd.DataFrame) -> Dict[str, float]:
    """
    Для каждого маршрута найдите shape_id, который имеет наибольшее количество поездок, затем найдите trip_id, который имеет этот shape_id, затем найдите Arrival_time первой и последней остановки этого trip_id, затем вычтите
    два, чтобы получить общее время маршрута

    Аргументы:
    df_route: dataframe маршрута

    Возвращает:
    Словарь с общим временем для каждого направления.
    """
    time_0 = 0
    if len(df_dir) > 1:
        shape_0 = df_dir.groupby("shape_id").count().trip_id.idxmax()
        trip_0 = df_dir[df_dir.trip_id == df_dir[df_dir.shape_id == shape_0].trip_id.unique()[0]]
        time_0 = trip_0.arrival_time.max() - trip_0.arrival_time.min()
    return {"total_time": np.round(time_0 / 3600, 2) if time_0 != 0 else 0}


def get_bus_spacing(route_dict: dict) -> Dict[str, float]:
    """
    Принимает фрейм данных маршрута и словарь информации о маршруте и возвращает словарь
    минимального расстояния между автобусами на маршруте

    Аргументы:
    df_route: фрейм данных всех поездок по заданному маршруту
    route_dict: словарь, содержащий длину маршрута и количество автобусов для каждого направления

    Возвращает:
    Словарь с ключами 'spacing dir 0' и 'spacing dir 1'
    """
    return {"bus_spacing": np.round(route_dict["route_length"] / route_dict["n_bus_avg"], 3)}


def average_active_buses(df_dir: pd.DataFrame) -> Dict[str, np.floating[Any]]:
    """
    Рассчитайте среднее количество активных автобусов за интервал времени.

    Аргументы:
    df_dir (pd.DataFrame): Входной DataFrame, содержащий данные шины.

    Возвращает:
    Dict[str, np.floating[Any]]: Словарь со средним количеством активных автобусов.
    """
    n_buses = []
    df = get_route_grp(df_dir)
    start_time = int(min(df.start_time))
    end_time = int(max(df.end_time))
    for time in range(start_time, end_time, 5 * 60):
        no_buses = get_trips_len(df, time)
        n_buses.append(no_buses)
    n_buses = np.array(n_buses)
    return {"n_bus_avg": np.round(np.mean(n_buses[n_buses > 0]), 3)}


def get_stop_spacing(df_dir: pd.DataFrame, route_dict: dict) -> Dict[str, float]:
    """
    Для каждого маршрута найдите форму с наибольшим количеством поездок, а затем найдите количество остановок в этой поездке.
    Затем разделите длину маршрута на количество остановок, чтобы получить интервал между остановками

    Аргументы:
    df_route: фрейм данных остановок для данного маршрута
    route_dict: словарь, содержащий route_id, длину маршрута в каждом направлении и количество
    поездок в каждом направлении

    Возвращает:
    Словарь с интервалом между остановками для каждого направления.
    """
    spc_0 = 0
    if len(df_dir) > 1:
        shape_0 = df_dir.groupby("shape_id").count().trip_id.idxmax()
        n_stops = len(
            df_dir[df_dir.trip_id == df_dir[df_dir.shape_id == shape_0].trip_id.unique()[0]]
        )
        spc_0 = route_dict["route_length"] / n_stops
    return {"stop_spacing": np.round(spc_0, 2) if spc_0 != 0 else 0}


def get_route_lens(df_dir: pd.DataFrame, df_shapes: LineString) -> Dict[str, float]:
    """
    Принимает фрейм данных поездок для заданного маршрута и фрейм данных фигур для заданного маршрута и
    возвращает словарь с длиной маршрута в каждом направлении

    Аргументы:
    df_route: фрейм данных файла routes.txt
    df_shapes: фрейм данных фигур

    Возвращает:
    Словарь с длиной маршрута для каждого направления.
    """
    epsg_zone = get_zone_epsg_from_ls(df_shapes.iloc[0]["geometry"])
    len_0 = 0
    if len(df_dir) > 1:
        shape_0 = df_dir.groupby("shape_id").count().trip_id.idxmax()
        len_0 = (
            df_shapes.loc[df_shapes.shape_id == shape_0].to_crs(epsg_zone).geometry.length.iloc[0]
        )
    return {"route_length": np.round(len_0 / 1000, 2)}


def get_route_stats(feed: Feed, peak_time: bool = False) -> pd.DataFrame:
    """
    Принимает канал GTFS и route_id и возвращает фрейм данных со следующими столбцами:

    - route_id
    - route_length
    - route_time
    - average_headway
    - peak_times
    - average_speed
    - min_spacing
    - stop_spacing

    Функция немного длинная, но не слишком сложная.

    Первое, что она делает, — объединяет таблицы stop_times и trips. Это необходимо, поскольку в таблице stop_times нет столбца route_id.

    Следующее, что она делает, — создает пустой словарь с именем route_dict. Он будет использоваться для хранения
    результатов анализа.

    Для каждого маршрута она создает новый словарь с именем ret_dict. Это будет использоваться для хранения результатов
    анализа для текущего

    Аргументы:
    feed: объект фида GTFS
    route_id: route_id маршрута, который вы хотите проанализировать

    Возвращает:
    Фрейм данных с route_id в качестве первого столбца, а остальные столбцы — это статистика для маршрута.
    """
    df_merge = feed.stop_times.merge(feed.trips, how="left", on="trip_id")
    df_shapes = feed.shapes
    route_list = []
    for route in df_merge.route_id.unique():
        df_route = df_merge[df_merge.route_id == route]
        for direction in df_route.direction_id.unique():
            df_dir = df_route[df_route.direction_id == direction]
            ret_dict = {}
            ret_dict["route"] = route
            ret_dict["direction"] = direction
            ret_dict.update(get_route_lens(df_dir, df_shapes))
            ret_dict.update(get_route_time(df_dir))
            ret_dict.update(get_average_headway(df_dir))
            ret_dict.update(get_average_speed(df_dir, ret_dict))
            ret_dict.update(average_active_buses(df_dir))
            ret_dict.update(get_bus_spacing(ret_dict))
            ret_dict.update(get_stop_spacing(df_dir, ret_dict))
            if peak_time:
                ret_dict.update(get_all_peak_times(df_dir))

            route_list.append(ret_dict)
    df = pd.DataFrame.from_records(route_list)
    return df.reset_index(drop=True)
