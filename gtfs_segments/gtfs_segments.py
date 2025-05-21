import os
from typing import List, Optional, Set

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from .geom_utils import (
    get_zone_epsg,
    make_gdf,
    nearest_points,
    nearest_points_parallel,
    ret_high_res_shape,
)
from .mobility import summary_stats_mobility
from .partridge_func import get_bus_feed
from .partridge_mod.gtfs import Feed
from .utils import download_write_file, export_segments, failed_pipeline, plot_hist


def merge_trip_geom(trip_df: pd.DataFrame, shape_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Он принимает dataframe поездок и dataframe фигур и возвращает geodataframe поездок с
    геометрией фигур

    Аргументы:
    trip_df: dataframe поездок
    shape_df: GeoDataFrame файла shapes.txt

    Возвращает:
    A GeoDataFrame
    """
    trips_with_no_shape_id = list(trip_df[trip_df["shape_id"].isna()].trip_id)
    if len(trips_with_no_shape_id) > 0:
        print("Excluding Trips with no shape_id:", trips_with_no_shape_id)
        trip_df = trip_df[~trip_df["trip_id"].isin(trips_with_no_shape_id)]

    non_existent_shape_id = set(trip_df["shape_id"]) - set(shape_df["shape_id"])
    if len(non_existent_shape_id) > 0:
        trips_with_no_corresponding_shape = list(trip_df[trip_df["shape_id"].isin(non_existent_shape_id)].trip_id)
        print("Excluding Trips with non-existent shape_ids in shapes.txt:", trips_with_no_corresponding_shape)
        trip_df = trip_df[~trip_df["shape_id"].isin(non_existent_shape_id)]

    # `direction_id` и `shape_id` необязательны
    if "direction_id" in trip_df.columns:
        # Проверьте, указаны ли direction_ids как null
        if trip_df["direction_id"].isnull().sum() == 0:
            grp = trip_df.groupby(["route_id", "shape_id", "direction_id"])
        else:
            grp = trip_df.groupby(["route_id", "shape_id"])
    else:
        grp = trip_df.groupby(["route_id", "shape_id"])
    trip_df = grp.first().reset_index()
    trip_df["traversals"] = grp.count().reset_index(drop=True)["trip_id"]
    subset_list = np.array(
        ["route_id", "trip_id", "shape_id", "service_id", "direction_id", "traversals"]
    )
    col_subset = subset_list[np.in1d(subset_list, trip_df.columns)]
    trip_df = trip_df[col_subset]
    trip_df = trip_df.dropna(how="all", axis=1)
    trip_df = shape_df.merge(trip_df, on="shape_id", how="left")
    return make_gdf(trip_df)


def make_segments_unique(df: gpd.GeoDataFrame, traversal_threshold: int = 1) -> gpd.GeoDataFrame:
    # Вычислить количество уникальных округленных расстояний для каждого route_id и segment_id
    unique_counts = df.groupby(["route_id", "segment_id"])["distance"].apply(
        lambda x: x.round().nunique()
    )

    # Фильтровать строки, где количество уникальных значений больше 1
    filtered_df = df[
        df.set_index(["route_id", "segment_id"]).index.isin(unique_counts[unique_counts > 1].index)
    ].copy()

    # Создать функцию модификации сегмента
    def modify_segment(segment_id: str, count: int) -> str:
        seg_split = str(segment_id).split("-")
        return seg_split[0] + "-" + seg_split[1] + "-" + str(count + 1)

    # Применить функцию модификации к segment_id
    filtered_df["modification"] = filtered_df.groupby(["route_id", "segment_id"]).cumcount()
    filtered_df["segment_id"] = filtered_df.apply(
        lambda row: modify_segment(row["segment_id"], row["modification"])
        if row["modification"] != 0
        else row["segment_id"],
        axis=1,
    )

    # Объединить измененные сегменты обратно в исходный DataFrame.
    df = pd.concat([df[~df.index.isin(filtered_df.index)], filtered_df], ignore_index=True)

    # Агрегировать обходы и фильтровать по пороговому значению обхода
    grp_again = df.groupby(["route_id", "segment_id"])
    df = grp_again.first().reset_index()
    df["traversals"] = grp_again["traversals"].sum().values
    df = df[df.traversals > traversal_threshold].reset_index(drop=True)
    return make_gdf(df)


def filter_stop_df(stop_df: pd.DataFrame, trip_ids: Set, stop_loc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Он принимает фрейм данных остановок и список идентификаторов поездок и возвращает фрейм данных остановок, которые находятся в
    списке идентификаторов поездок

    Аргументы:
    stop_df: фрейм данных всех остановок
    trip_ids: список trip_id, по которому вы хотите отфильтровать stop_df

    Возвращает:
    Фрейм данных с trip_id, s top_id и stop_sequence для поездок в списке trip_ids.
    """
    missing_stop_locs = set(stop_df.stop_id) - set(stop_loc_df.stop_id)
    if len(missing_stop_locs) > 0:
        print("Missing stop locations for:", missing_stop_locs)
        missing_trips = stop_df[stop_df.stop_id.isin(missing_stop_locs)].trip_id.unique()
        for trip in missing_trips:
            trip_ids.discard(trip)
            print(
                "Removed the trip_id:", trip, "as stop locations are missing for stops in the trip"
            )
    # Отфильтруйте stop_df, чтобы включить только trip_ids в список trip_ids
    stop_df = stop_df[stop_df.trip_id.isin(trip_ids)].reset_index(drop=True)
    stop_df = stop_df.sort_values(["trip_id", "stop_sequence"]).reset_index(drop=True)
    stop_df["main_index"] = stop_df.index
    stop_df_grp = stop_df.groupby("trip_id")
    drop_inds = []
    if "pickup_type" in stop_df.columns:
        grp_f = stop_df_grp.first()
        drop_inds.append(grp_f.loc[grp_f["pickup_type"] == 1, "main_index"])
    if "drop_off_type" in stop_df.columns:
        grp_l = stop_df_grp.last()
        drop_inds.append(
            grp_l.loc[grp_l["drop_off_type"] == 1, "main_index"]
        )
    if len(drop_inds) > 0 and len(drop_inds[0]) > 0:
        stop_df = stop_df[~stop_df["main_index"].isin(drop_inds)].reset_index(drop=True)
    stop_df = stop_df[["trip_id", "stop_id", "stop_sequence", "arrival_time"]]

    stop_df = stop_df.sort_values(["trip_id", "stop_sequence"]).reset_index(drop=True)
    return stop_df


def merge_stop_geom(stop_df: pd.DataFrame, stop_loc_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    > Объединить stop_loc_df с stop_df, а затем преобразовать результат в GeoDataFrame

    Аргументы:
    stop_df: dataframe остановок
    stop_loc_df: GeoDataFrame остановок

    Возвращает:
    GeoDataFrame
    """
    stop_df["start"] = stop_df.copy().merge(stop_loc_df, how="left", on="stop_id")["geometry"]
    return stop_df


def create_segments(stop_df: gpd.GeoDataFrame, parallel: bool = False) -> pd.DataFrame:
    """
    Эта функция создает сегменты между остановками на основе их близости и возвращает GeoDataFrame.

    Аргументы:
    stop_df: Pandas DataFrame, содержащий информацию об остановках в транспортной сети, включая
    их stop_id, координаты и trip_id.

    Возвращает:
    GeoDataFrame с сегментами, созданными из входных данных stop_df.
    """
    if parallel:
        stop_df = nearest_points_parallel(stop_df)
    else:
        stop_df = nearest_points(stop_df)
    stop_df = stop_df.rename({"stop_id": "stop_id1", "arrival_time": "arrival_time1"}, axis=1)
    grp = (
        pd.DataFrame(stop_df).groupby("trip_id", group_keys=False).shift(-1).reset_index(drop=True)
    )
    stop_df[["stop_id2", "end", "snap_end_id", "arrival_time2"]] = grp[
        ["stop_id1", "start", "snap_start_id", "arrival_time1"]
    ]
    stop_df["segment_id"] = stop_df.apply(
        lambda row: str(row["stop_id1"]) + "-" + str(row["stop_id2"]) + "-1", axis=1
    )
    stop_df = stop_df.dropna().reset_index(drop=True)
    stop_df.snap_end_id = stop_df.snap_end_id.astype(int)
    stop_df = stop_df[stop_df["snap_end_id"] > stop_df["snap_start_id"]].reset_index(drop=True)
    stop_df["geometry"] = stop_df.apply(
        lambda row: LineString(
            row["geometry"].coords[row["snap_start_id"] : row["snap_end_id"] + 1]
        ),
        axis=1,
    )
    return stop_df


def process_feed_stops(feed: Feed) -> gpd.GeoDataFrame:
    """
    Он берет канал GTFS, объединяет данные о поездке и форме, фильтрует данные stop_times, чтобы включить только

    поездки, которые есть в канале, объединяет данные stop_times с данными об остановках, создает сегмент для
    каждой пары остановок, получает зону EPSG для канала, создает GeoDataFrame и вычисляет длину

    каждого сегмента

    Аргументы:
    канал: объект канала GTFS
    max_spacing: максимальное расстояние между остановками в метрах. Если остановка находится дальше этого расстояния
    от предыдущей остановки, она будет удалена.

    Возвращает:
    GeoDataFrame со следующими столбцами:
    """
    trip_df = merge_trip_geom(feed.trips, feed.shapes)
    trip_ids = set(trip_df.trip_id.unique())
    stop_loc_df = feed.stops[["stop_id", "geometry"]]
    stop_df = filter_stop_df(feed.stop_times, trip_ids, stop_loc_df)
    stop_df = merge_stop_geom(stop_df, stop_loc_df)
    stop_df = stop_df.merge(trip_df, on="trip_id", how="left")
    stops = stop_df.groupby("shape_id").count().reset_index()["geometry"]
    stop_df = stop_df.groupby("shape_id").first().reset_index()
    stop_df["n_stops"] = stops
    epsg_zone = get_zone_epsg(stop_df)
    if epsg_zone is not None:
        stop_df["distance"] = stop_df.geometry.to_crs(epsg_zone).length
        stop_df["mean_distance"] = stop_df["distance"] / stop_df["n_stops"]
    return make_gdf(stop_df)


def process_feed(
    feed: Feed, parallel: bool = False, max_spacing: Optional[float] = None
) -> gpd.GeoDataFrame:
    """
    Функция `process_feed` принимает фид и необязательное максимальное расстояние в качестве входных данных, выполняет различные
    операции по обработке и фильтрации данных фида и возвращает GeoDataFrame, содержащий
    обработанные данные.

    Аргументы:
    feed: Параметр `feed` — это структура данных, которая содержит информацию о транспортной сети.
    Вероятно, она включает такие данные, как формы (геометрические представления маршрутов), поездки (последовательности
    остановок), время остановок (время прибытия и отправления на остановках) и остановки (местоположения остановок).
    [Необязательно] max_spacing: Параметр `max_spacing` — это необязательный параметр, который указывает максимальное
    расстояние между остановками. Если он указан, функция отфильтрует остановки, которые находятся дальше друг от друга, чем
    указанное максимальное расстояние.

    Возвращает:
    GeoDataFrame, содержащий информацию об остановках и сегментах в фиде с сегментами, меньшими, чем значения max_spacing.
    """
    shapes = ret_high_res_shape(feed.shapes, feed.trips, spat_res=5)
    trip_df = merge_trip_geom(feed.trips, shapes)
    trip_ids = set(trip_df.trip_id.unique())
    stop_loc_df = feed.stops[["stop_id", "geometry"]]
    stop_df = filter_stop_df(feed.stop_times, trip_ids, stop_loc_df)
    stop_df = merge_stop_geom(stop_df, stop_loc_df)
    stop_df = stop_df.merge(trip_df, on="trip_id", how="left")
    stop_df = create_segments(stop_df, parallel=parallel)
    stop_df = make_gdf(stop_df)
    epsg_zone = get_zone_epsg(stop_df)
    if epsg_zone is not None:
        stop_df["distance"] = stop_df.set_geometry("geometry").to_crs(epsg_zone).geometry.length
        stop_df["distance"] = stop_df["distance"].round(2)  # round to 2 decimal places
    stop_df["traversal_time"] = (stop_df["arrival_time2"] - stop_df["arrival_time1"]).astype(
        "float"
    )
    stop_df["speed"] = stop_df["distance"].div(stop_df["traversal_time"])
    stop_df = make_segments_unique(stop_df, traversal_threshold=0)
    subset_list = np.array(
        [
            "segment_id",
            "route_id",
            "direction_id",
            "trip_id",
            "traversals",
            "distance",
            "stop_id1",
            "stop_id2",
            "traversal_time",
            "speed",
            "geometry",
        ]
    )
    col_subset = subset_list[np.in1d(subset_list, stop_df.columns)]
    stop_df = stop_df[col_subset]
    if max_spacing is not None:
        stop_df = stop_df[stop_df["distance"] <= max_spacing]
    return make_gdf(stop_df)


def inspect_feed(feed: Feed) -> str:
    """
    Он проверяет, есть ли в фиде какие-либо автобусные маршруты и есть ли столбец `shape_id` в таблице `trips`

    Аргументы:
    фид: Объект фида, который вы хотите проверить.

    Возвращает:
    Сообщение
    """
    message = "Valid GTFS Feed"
    if len(feed.stop_times) == 0:
        message = "No Bus Routes in "
    if "shape_id" not in feed.trips.columns:
        message = "Missing `shape_id` column in "
    return message


def get_gtfs_segments(
    path: str,
    agency_id: Optional[str] = None,
    threshold: Optional[int] = 1,
    max_spacing: Optional[float] = None,
    parallel: bool = False,
) -> gpd.GeoDataFrame:
    """
    Функция `get_gtfs_segments` принимает путь к файлу потока GTFS, необязательное имя агентства,
    пороговое значение и необязательное значение максимального интервала и возвращает обработанные сегменты GTFS.

    Аргументы:
    path: Параметр path — это путь к файлу данных GTFS (General Transit Feed Specification).
    Это формат данных, используемый агентствами общественного транспорта для предоставления расписания и географической
    информации о своих услугах.
    [Необязательно] agency_id: идентификатор агентства агентства, для которого вы хотите получить поток автобусов. Если этот
    параметр не указан, функция получит поток автобусов для всех транспортных агентств. Вы можете передать
    список идентификаторов агентства, чтобы получить поток автобусов для нескольких транспортных агентств.
    [Необязательно] threshold: Параметр threshold используется для фильтрации автобусных поездок, которые имеют меньше остановок, чем
    указанный порог. Поездки с меньшим количеством остановок, чем порог, будут исключены из результата.
    По умолчанию 1
    [Необязательно] max_spacing: Параметр `max_spacing` используется для указания максимального расстояния между двумя
    последовательными остановками в сегменте. Если расстояние между двумя остановками превышает значение `max_spacing`,
    сегмент разбивается на несколько сегментов.

    Возвращает:
    GeoDataFrame, содержащий информацию об остановках и сегментах в фиде с сегментами,
    меньше, чем значения max_spacing. Каждая строка содержит следующие столбцы:
    - segment_id: идентификатор сегмента, созданный gtfs-segments
    - stop_id1: идентификатор `stop_id` начальной остановки сегмента.
    Идентификатор тот же, который агентство выбрало в файле stops.txt своего пакета GTFS.
    - stop_id2: идентификатор `stop_id` конечной остановки сегмента.
    - route_id: тот же идентификатор маршрута, что указан в файле routes.txt агентства.
    - direction_id: идентификатор направления маршрута.
    - traversals: количество раз, когда указанный маршрут пересекает сегмент в течение «интервала измерения».
    Выбранный «интервал измерения» — это самый загруженный день в расписании GTFS: день, в который выполняется больше всего автобусных рейсов.
    - расстояние: длина сегмента в метрах.
    - геометрия: LINESTRING сегмента (формат для кодирования географических путей).
    Все геометрии перепроецируются на Меркатор (EPSG:4326/WGS84) для сохранения согласованности.
    """
    feed = get_bus_feed(path, agency_id=agency_id, threshold=threshold, parallel=parallel)
    df = process_feed(feed, parallel=parallel)
    if max_spacing is not None:
        print("Using max_spacing {:.0f} to filter segments".format(max_spacing))
        df = df[df["distance"] <= max_spacing]
    return df


def pipeline_gtfs(filename: str, url: str, bounds: List, max_spacing: float) -> str:
    """
    Он берет файл GTFS, загружает его, читает его, обрабатывает его, а затем выводит кучу файлов.

    Давайте рассмотрим функцию шаг за шагом.

    Сначала мы определяем функцию и даем ей имя. Мы также даем ей несколько аргументов:

    - filename: имя файла, в который мы хотим сохранить вывод.
    - url: URL-адрес файла GTFS, который мы хотим загрузить.
    - bounds: ограничивающая рамка области, которую мы хотим проанализировать.
    - max_spacing: максимальный интервал, который мы хотим проанализировать.

    Затем мы создаем папку для сохранения вывода.

    Затем мы загружаем файл GTFS и сохраняем его в только что созданной папке.

    Затем мы считываем файл GTFS с помощью функции `get_bus_feed`.

    Args:
    filename: имя файла, в который вы хотите сохранить вывод
    url: URL-адрес файла GTFS
    bounds: ограничивающий прямоугольник области, которую вы хотите проанализировать. Это в формате
    [min_lat,min_lon,max_lat,max_lon]
    max_spacing: максимальное расстояние между остановками, которое вы хотите учитывать.

    Возвращает:
    Успех или неудача конвейера
    """
    folder_path = os.path.join("output_files", filename)
    gtfs_file_loc = download_write_file(url, folder_path)

    feed = get_bus_feed(gtfs_file_loc)
    message = inspect_feed(feed)
    if message != "Valid GTFS Feed":
        return failed_pipeline(message, filename, folder_path)

    df = process_feed(feed)
    df_sub = df[df["distance"] < 3000].copy().reset_index(drop=True)
    if len(df_sub) == 0:
        return failed_pipeline("Only Long Bus Routes in ", filename, folder_path)
    summary_stats_mobility(df, folder_path, filename, url, bounds, max_spacing, export=True)

    plot_hist(
        df,
        file_path=os.path.join(folder_path, "spacings.png"),
        title=filename.split(".")[0],
        max_spacing=max_spacing,
        save_fig=True,
    )
    export_segments(
        df, os.path.join(folder_path, "geojson"), output_format="geojson", geometry=True
    )
    export_segments(
        df,
        os.path.join(folder_path, "spacings_with_geometry"),
        output_format="csv",
        geometry=True,
    )
    export_segments(df, os.path.join(folder_path, "spacings"), output_format="csv", geometry=False)
    return "Success for " + filename
