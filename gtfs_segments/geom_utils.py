from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional, Tuple

import contextily as cx
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utm
from matplotlib.figure import Figure
from pyproj import Geod
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.ops import split

geod = Geod(ellps="WGS84")


def split_route(row: pd.Series) -> str:
    route = row["geometry"]
    if row["snapped_start_id"]:
        try:
            route = split(route, row["start"]).geoms[1]
        except IndexError:
            pass
    if row["snapped_end_id"]:
        route = split(route, row["end"]).geoms[0]
    return route.wkt


def nearest_snap(route_string: LineString, stop_point: Point) -> str:
    route = np.array(route_string.coords)
    point = np.array(stop_point.coords)
    ckd_tree = cKDTree(route)
    return Point(route[ckd_tree.query(point, k=1)[1]][0]).wkt


def make_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(df)
    gdf = gdf.set_crs(epsg=4326, allow_override=True)
    return gdf


def code(zone: List, lat: float) -> int:
    if lat < 0:
        epsg_code = 32700 + zone[2]
    else:
        epsg_code = 32600 + zone[2]
    return epsg_code


def get_zone_epsg(stop_df: gpd.GeoDataFrame) -> int:
    lon = stop_df.start[0].x
    lat = stop_df.start[0].y
    zone = utm.from_latlon(lat, lon)
    return code(zone, lat)


def view_spacings(
    gdf: gpd.GeoDataFrame,
    basemap: bool = False,
    map_provider: str = cx.providers.CartoDB.Positron,
    show_stops: bool = False,
    level: str = "whole",
    axis: str = "on",
    dpi: int = 300,
    **kwargs: Any,
) -> Figure:
    _, ax = plt.subplots(figsize=(10, 10), dpi=dpi)
    crs = gdf.crs
    # Фильтр по направлению и уровню
    if "direction" in kwargs:
        gdf = gdf[gdf.direction_id == kwargs["direction"]].copy()
    if level == "whole":
        markersize = 20
        ax = gdf.plot(
            ax=ax,
            color="#34495e",
            linewidth=0.5,
            edgecolor="black",
            label="Bus network",
            zorder=1,
        )
    elif level == "route":
        markersize = 40
        assert "route" in kwargs, "Please provide a route_id in route attibute"
        kwargs["route"] = [kwargs["route"]] if isinstance(kwargs["route"], str) else kwargs["route"]
        gdf = gdf[gdf.route_id.isin(kwargs["route"])].copy()
    elif level == "segment":
        markersize = 60
        assert "segment" in kwargs, "Please provide a segment_id in segment attibute"
        kwargs["segment"] = [kwargs["segment"]] if isinstance(kwargs["segment"], str) else kwargs["segment"]
        gdf = gdf[gdf.segment_id.isin(kwargs["segment"])].copy()
    else:
        raise ValueError("level must be either whole, route, or segment")

    # Постройте график интервалов
    if "route" in kwargs:
        kwargs["route"] = [kwargs["route"]] if isinstance(kwargs["route"], str) else kwargs["route"]
        gdf = gdf[gdf.route_id.isin(kwargs["route"])].copy()
        if len(kwargs["route"]) > 1:
            ax = gdf.plot(
                ax=ax,
                linewidth=2,
                column="route_id",
                label="Route ID:" + str(kwargs["route"]),
                zorder=2,
                cmap="tab20",
                legend=True,
            )
        else:
            ax = gdf.plot(
                ax=ax,
                linewidth=2,
                color="#2ecc71",
                label="Route ID:" + str(kwargs["route"]),
                zorder=2,
            )
    if "segment" in kwargs:
        try:
            kwargs["segment"] = [kwargs["segment"]] if isinstance(kwargs["segment"], str) else kwargs["segment"]
            gdf = gdf[gdf.segment_id.isin(kwargs["segment"])].copy()
        except ValueError as e:
            raise ValueError(f"No such segment exists. Check if direction_id is incorrect {e}")
        ax = gdf.plot(
            ax=ax,
            linewidth=2.5,
            color="#000000",
            label="Segment ID: " + str(kwargs["segment"]),
            zorder=3,
        )
    if show_stops:
        geo_series = gdf.geometry.apply(lambda line: Point(line.coords[0]))
        geo_series = pd.concat([geo_series, gpd.GeoSeries(Point(gdf.iloc[-1].geometry.coords[-1]))])
        geo_series.set_crs(crs=gdf.crs).plot(
            ax=ax,
            color="#FFD700",
            edgecolor="#000000",
            linewidth=1,
            markersize=markersize,
            alpha=0.95,
            zorder=10,
        )

    if basemap:
        df = gpd.GeoDataFrame(gdf, crs=crs)
        cx.add_basemap(ax, crs=df.crs, source=map_provider, attribution_size=5)
    plt.axis(axis)
    if level != "segment":
        plt.legend(loc="best")
    else:
        ax.legend().set_visible(False)
    return ax


def view_spacings_interactive(
    gdf: gpd.GeoDataFrame,
    basemap: bool = True,
    show_stops: bool = False,
    level: str = "whole",
    **kwargs: Any,
) -> folium.Map:
    """
    Создает интерактивную карту Folium для визуализации расстояний между остановками.

    Параметры:
    gdf (gpd.GeoDataFrame): GeoDataFrame, содержащий данные о расстояний между остановками.
    basemap (bool, необязательно): добавлять ли базовую карту на карту. По умолчанию True.
    show_stops (bool, необязательно): показывать ли остановки на карте. По умолчанию False.
    level (str, необязательно): уровень, на котором следует фильтровать данные. Может быть «whole», «route» или «segment».
    По умолчанию «whole».
    **kwargs: дополнительные ключевые аргументы для фильтрации данных на основе уровня.

    Возвращает:
    folium.Map: сгенерированная карта Folium.

    Вызывает:
    AssertionError: если не предоставлены требуемые атрибуты для фильтрации данных.

    Пример использования:
    gdf = gpd.GeoDataFrame(...)
    map = view_spacings_interactive(gdf, basemap=True, show_stops=True, level='route', route='123')
    """
    if "direction" in kwargs:
        gdf = gdf[gdf.direction_id == kwargs["direction"]].copy()
    # При необходимости конвертируйте CRS в EPSG:4326
    if gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)

    # Инициализировать карту Folium
    bounds = gdf.total_bounds
    map_center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    fmap = folium.Map(location=map_center, control_scale=True, zoom_start=12)

    # Фильтр и построение графика на основе уровня
    if level == "route":
        assert "route" in kwargs, "Please provide a route_id in route attribute"
        kwargs["route"] = [kwargs["route"]] if isinstance(kwargs["route"], str) else kwargs["route"]
        gdf = gdf[gdf.route_id.isin(kwargs["route"])].copy()
    elif level == "segment":
        assert "segment" in kwargs, "Please provide a segment_id in segment attribute"
        kwargs["segment"] = [kwargs["segment"]] if isinstance(kwargs["segment"], str) else kwargs["segment"]
        gdf = gdf[gdf.segment_id.isin(kwargs["segment"])].copy()

    # Добавить линии на карту
    tooltip = folium.GeoJsonTooltip(fields=["segment_id", "distance"])
    popup = folium.GeoJsonPopup(fields=gdf.drop(columns=["geometry"]).columns.tolist())

    def style_function(x: Any) -> dict[str, Any]:
        if "route" in kwargs:
            return {
                "color": (
                    "#2ecc71" if x["properties"]["route_id"] == kwargs["route"] else "#34495e"
                ),
                "weight": (5 if x["properties"]["route_id"] == kwargs["route"] else 2),
            }
        if "segment" in kwargs:
            return {
                "color": (
                    "#000000" if x["properties"]["segment_id"] == kwargs["segment"] else "#34495e"
                ),
                "weight": (5 if x["properties"]["segment_id"] == kwargs["segment"] else 2),
                "z_index": 1000,
            }
        return {"color": "#34495e", "weight": 2}

    folium.GeoJson(
        gdf, tooltip=tooltip, popup=popup, zoom_on_click=True, style_function=style_function
    ).add_to(fmap)

    # Показать остановки
    if show_stops:
        if "route" in kwargs:
            kwargs["route"] = [kwargs["route"]] if isinstance(kwargs["route"], str) else kwargs["route"]
            gdf = gdf[gdf.route_id.isin(kwargs["route"])].copy()
        if "segment" in kwargs:
            kwargs["segment"] = [kwargs["segment"]] if isinstance(kwargs["segment"], str) else kwargs["segment"]
            gdf = gdf[gdf.segment_id == kwargs["segment"]].copy()
        stop_ids = {}
        for _, row in gdf.iterrows():
            stop_ids[row["stop_id1"]] = Point(row["geometry"].coords[0])
            stop_ids[row["stop_id2"]] = Point(row["geometry"].coords[-1])
        for stop_id, point in stop_ids.items():
            folium.CircleMarker(
                location=[point.y, point.x],
                radius=(6 if "segment" in kwargs else 4 if "route" in kwargs else 2),
                scale_radius=True,
                weight=1,
                fill_opacity=0.9,
                color="#000000",
                fill_color="#FFD700",
                fill=True,
                tooltip=str(stop_id),
            ).add_to(fmap)

    # Добавить базовую карту (basemap)
    if basemap:
        folium.TileLayer("CartoDB positron", name="Light Map", control=False).add_to(fmap)

    return fmap


def increase_resolution(geom: LineString, spat_res: int = 5) -> LineString:
    """
    Эта функция увеличивает разрешение геометрии LineString, добавляя
    точки вдоль линии с указанным пространственным разрешением.

    Аргументы:
    geom: Входная геометрия, которую необходимо изменить (в данном случае LineString).
    spat_res: пространственное разрешение, которое является желаемым расстоянием между последовательными точками
    на LineString. Если расстояние между двумя последовательными точками больше, чем
    пространственное разрешение, функция добавит дополнительные точки к LineString,
    чтобы увеличить его разрешение. По умолчанию 5

    Возвращает:
    объект LineString с повышенным разрешением на основе входного пространственного разрешения.
    """
    coords = geom.coords
    coord_pairs = np.array([coords[i : i + 2] for i in range(len(coords) - 1)])
    coord_dists = np.array(
        [geod.geometry_length(LineString(coords[i : i + 2])) for i in range(len(coords) - 1)]
    )
    new_ls = []
    for i, dists in enumerate(coord_dists):
        pair = coord_pairs[i]
        if dists > spat_res:
            factor = int(np.ceil(dists / spat_res))
            ret_points = [tuple(pair[0])]
            for j in range(1, factor):
                new_point = (
                    pair[0][0] + (pair[1][0] - pair[0][0]) * j / factor,
                    pair[0][1] + (pair[1][1] - pair[0][1]) * j / factor,
                )
                ret_points.append(new_point)
            for pt in ret_points:
                new_ls.append(pt)
        else:
            new_ls.append(tuple(pair[0]))
    new_ls.append(tuple(coord_pairs[-1][1]))
    return LineString(new_ls)


def ret_high_res_shape(
    shapes: gpd.GeoDataFrame, trips: pd.DataFrame, spat_res: int = 5
) -> gpd.GeoDataFrame:
    """
    Эта функция увеличивает разрешение геометрий в заданном фрейме данных фигур на заданное пространственное разрешение.

    Аргументы:

    shapes: pandas DataFrame, содержащий столбец с именем «geometry», содержащий объекты геометрической формы

    spat_res: пространственное разрешение, которое является размером каждого пикселя или ячейки в растровом наборе данных. В этой
    функции он используется для увеличения разрешения входных фигур путем создания большего количества вершин в
    их геометриях. Значение по умолчанию — 5, что означает, что разрешение будет увеличено путем добавления вершин. По умолчанию — 5

    Возвращает:
    GeoDataFrame со столбцом геометрии, обновленным для получения фигур с более высоким разрешением.
    """
    shape_ids = trips.shape_id.unique()
    shapes = shapes[shapes.shape_id.isin(shape_ids)].copy()
    high_res_shapes = [
        increase_resolution(row["geometry"], spat_res) for i, row in shapes.iterrows()
    ]
    shapes.geometry = high_res_shapes
    return shapes


def ret_high_res_shape_parallel(shapes: gpd.GeoDataFrame, spat_res: int = 5) -> gpd.GeoDataFrame:
    """
    Эта функция увеличивает разрешение геометрий в заданном фрейме данных фигур на заданное пространственное разрешение.

    Аргументы:

    shapes: pandas DataFrame, содержащий столбец с именем «geometry», содержащий объекты геометрической формы

    spat_res: пространственное разрешение, которое является размером каждого пикселя или ячейки в растровом наборе данных. В этой
    функции он используется для увеличения разрешения входных фигур путем создания большего количества вершин в
    их геометриях. Значение по умолчанию — 5, что означает, что разрешение будет увеличено путем добавления вершин. По умолчанию — 5

    Возвращает:
    GeoDataFrame со столбцом геометрии, обновленным для получения фигур с более высоким разрешением.
    """

    def process_shape(row: pd.core.series.Series) -> LineString:
        return increase_resolution(row["geometry"], spat_res)

    high_res_shapes = []
    with ThreadPoolExecutor(max_workers=None) as executor:
        high_res_shapes = list(executor.map(process_shape, shapes.to_dict("records")))

    shapes.geometry = high_res_shapes
    return shapes


def nearest_points(stop_df: gpd.GeoDataFrame, k_neighbors: int = 3) -> pd.DataFrame:
    """
    Функция берет фрейм данных остановок и привязывает их к ближайшим точкам на линейной геометрии,
    с возможностью указать количество ближайших соседей для рассмотрения.

    Аргументы:
    stop_df: фрейм данных pandas, содержащий информацию об остановках вдоль набора поездок, включая
    идентификатор поездки, местоположение остановки (как объект Shapely Point) и геометрию поездки (как объект Shapely
    LineString)
    k_neighbors: количество ближайших соседей для рассмотрения при привязке остановок к линейной геометрии.
    Значение по умолчанию — 3. По умолчанию — 3

    Возвращает:
    фрейм данных stop_df с дополнительным столбцом 'snap_start_id', который содержит индексы
    ближайших точек на маршруте поездки для каждой остановки. Если какие-либо поездки не удалось привязать, они исключаются из
    возвращаемого фрейма данных.
    """
    stop_df["snap_start_id"] = -1
    geo_const = 6371000 * np.pi / 180
    failed_trips = []
    count = 0
    total_trip_count = 0
    defective_trip_count = 0
    for name, group in stop_df.groupby("trip_id"):
        # print(name)
        count += 1
        total_trip_count += len(group)
        neighbors = k_neighbors
        geom_line = group["geometry"].iloc[0]
        # print(len(geom_line.coords))
        tree = cKDTree(data=np.array(geom_line.coords))
        stops = [x.coords[0] for x in group["start"]]
        if len(stops) <= 1:
            failed_trips.append(name)
            print("Excluding Trip: " + name + " because of too few stops")
            defective_trip_count += len(group)
            continue
        failed_trip = False
        solution_found = False
        while not solution_found:
            np_dist, np_inds = tree.query(stops, workers=-1, k=neighbors)
            # Приблизительное расстояние в метрах
            np_dist = np_dist * geo_const
            prev_point = min(np_inds[0])
            points = [prev_point]
            for i, nps in enumerate(np_inds[1:]):
                condition = (nps > prev_point) & (nps < max(np_inds[i + 1]))
                points_valid = nps[condition]
                if len(points_valid) > 0:
                    points_score = (np.power(points_valid - prev_point, 3)) * np.power(
                        np_dist[i + 1, condition], 1
                    )
                    prev_point = nps[condition][np.argmin(points_score)]
                    points.append(prev_point)
                else:
                    # Действительные баллы не найдены
                    if neighbors < len(stops):
                        neighbors = min(neighbors + 2, len(stops))
                        break
                    else:
                        failed_trips.append(name)
                        failed_trip = True
                        solution_found = True
                        print("Excluding Trip: " + name + " because of failed snap!")
                        defective_trip_count += len(group)
                        break
            if len(points) == len(stops):
                solution_found = True
        if len(points) != len(set(points)):
            print("Processing", count, len(stop_df.trip_id.unique()))
            print("Points defective")

        if not failed_trip:
            stop_df.loc[stop_df.trip_id == name, "snap_start_id"] = points

    print("Total trips processed: ", total_trip_count)
    if defective_trip_count > 0:
        percent_defective = defective_trip_count / total_trip_count * 100
        print("Total defective trips: ", defective_trip_count)
        print(f"Percentage defective trips: {percent_defective:.2f}%",
        )
    stop_df = stop_df[~stop_df.trip_id.isin(failed_trips)].reset_index(drop=True)
    return stop_df


# def process_trip_group(
#     name: str, group: pd.core.groupby.DataFrameGroupBy, k_neighbors: int, geo_const: float
# ) -> Tuple:
#     neighbors = k_neighbors
#     geom_line = group["geometry"].iloc[0]
#     tree = cKDTree(data=np.array(geom_line.coords))
#     stops = [x.coords[0] for x in group["start"]]
#     n_stops = len(stops)
#     if n_stops <= 1:
#         return name, None, True  # Failed trip due to too few stops

#     failed_trip = False
#     solution_found = False
#     points = []
#     while not solution_found:
#         np_dist, np_inds = tree.query(stops, workers=-1, k=neighbors)
#         np_dist = np_dist * geo_const  # Approx distance in meters
#         prev_point = min(np_inds[0])
#         points = [prev_point]
#         for i, nps in enumerate(np_inds[1:]):
#             condition = (nps > prev_point) & (nps < max(np_inds[i + 1]))
#             points_valid = nps[condition]
#             if len(points_valid) > 0:
#                 points_score = np.power(points_valid - prev_point, 3) * np.power(
#                     np_dist[i + 1, condition], 1
#                 )
#                 prev_point = nps[condition][np.argmin(points_score)]
#                 points.append(prev_point)
#             else:
#                 # Capping the number of nearest neighbors to 11
#                 if neighbors < min(n_stops, 11):
#                     neighbors = min(neighbors + 2, n_stops)
#                     break
#                 else:
#                     failed_trip = True
#                     solution_found = True
#                     break
#         if len(points) == n_stops:
#             solution_found = True

#     if failed_trip:
#         return name, None, True
#     else:
#         return name, points, False


def process_trip_group(
    name: str, group: pd.core.groupby.DataFrameGroupBy, k_neighbors: int, geo_const: float
) -> Tuple:
    neighbors = k_neighbors
    geom_line = group["geometry"].iloc[0]
    tree = cKDTree(data=np.array(geom_line.coords))
    stops = [x.coords[0] for x in group["start"]]
    n_stops = len(stops)
    MAX_NEIGHBORS = min(n_stops, 9)
    if n_stops <= 1:
        return name, None, True

    failed_trip = False
    solution_found = False
    points = []
    np_dist_all, np_inds_all = tree.query(stops, workers=-1, k=MAX_NEIGHBORS)
    np_dist_all = np_dist_all * geo_const  # Приблизительное расстояние в метрах
    while not solution_found:
        np_inds = np_inds_all[:, :neighbors]
        np_dist = np_dist_all[:, :neighbors]
        prev_point = min(np_inds[0])
        points = [prev_point]
        for i, nps in enumerate(np_inds[1:]):
            condition = (nps > prev_point) & (nps < max(np_inds[i + 1]))
            points_valid = nps[condition]
            if len(points_valid) > 0:
                points_score = np.power(points_valid - prev_point, 3) * np.power(
                    np_dist[i + 1, condition], 1
                )
                prev_point = nps[condition][np.argmin(points_score)]
                points.append(prev_point)
            else:
                # Ограничение числа ближайших соседей до 11
                if neighbors < MAX_NEIGHBORS:
                    neighbors = min(neighbors + 2, n_stops)
                    break
                else:
                    failed_trip = True
                    solution_found = True
                    break
        if len(points) == n_stops:
            solution_found = True

    if failed_trip:
        return name, None, True
    else:
        return name, points, False


def nearest_points_parallel(stop_df: gpd.GeoDataFrame, k_neighbors: int = 5) -> pd.DataFrame:
    stop_df["snap_start_id"] = -1
    geo_const = 6371000 * np.pi / 180
    failed_trips = []
    defective_trip_count = 0
    with ThreadPoolExecutor(max_workers=None) as executor:
        results = executor.map(
            lambda x: process_trip_group(x[0], x[1], k_neighbors, geo_const),
            stop_df.groupby("trip_id"),
        )

    for name, points, failed in results:
        if failed:
            failed_trips.append(name)
        else:
            stop_df.loc[stop_df.trip_id == name, "snap_start_id"] = points
    defective_trip_count = (
        stop_df[stop_df.trip_id.isin(failed_trips)].groupby("trip_id").first().traversals.sum()
    )
    total_trip_count = len(stop_df)
    stop_df = stop_df[~stop_df.trip_id.isin(failed_trips)].reset_index(drop=True)

    print("Total trips processed:", total_trip_count)
    if defective_trip_count > 0:
        print("Total defective trips:", defective_trip_count)
        print(
            "Percentage defective trips:{:.2f}".format(
                defective_trip_count / total_trip_count * 100
            )
        )
    return stop_df


def view_heatmap(
    gdf: gpd.GeoDataFrame,
    column: str = "distance",
    cmap: Optional[str] = "RdYlBu",
    light_mode: bool = True,
    interactive: bool = False,
) -> Any:
    """
    Создает визуализацию тепловой карты GeoDataFrame.

    Параметры:
    gdf (gpd.GeoDataFrame): GeoDataFrame, содержащий данные для визуализации.
    cmap (Необязательно[str], необязательно): Цветовая карта, используемая для тепловой карты. По умолчанию "RdYlBu".
    light_mode (bool, необязательно): Указывает, использовать ли базовую карту в светлом режиме. По умолчанию True.
    interactive (bool, необязательно): Указывает, создавать ли интерактивную карту. По умолчанию False.

    Возвращает:
    Any: Сгенерированная визуализация тепловой карты.
    """
    df_filtered = gdf.copy()
    df_filtered[column] = pd.to_numeric(df_filtered[column])
    if column == "distance":
        MAX_RANGE = gdf["distance"].max()
        df_filtered = gdf[(gdf["distance"] >= 30)].copy()
        bins = [125, 200, 400, 600, 800, 1000, 1200, 1500, 2000, MAX_RANGE]
    else:
        df_filtered = df_filtered[(df_filtered[column] >= df_filtered[column].quantile(0.01))]
        df_filtered = df_filtered[(df_filtered[column] <= df_filtered[column].quantile(1 - 0.01))]
    if interactive:
        if column == "distance":
            fmap = df_filtered.explore(
                column=column,
                scheme="UserDefined",
                tooltip=["segment_id", "distance"],
                tiles="CartoDB Positron" if light_mode else "CartoDB Dark Matter",
                legend=True,
                cmap=cmap,  # YlOrRd
                classification_kwds=dict(bins=bins),
                legend_kwds=dict(colorbar=False, fmt="{:.0f}"),
                style_kwds=dict(opacity=0.75, fillOpacity=0.75),
                popup=True,
            )
        else:
            fmap = df_filtered.explore(
                column=column,
                cmap=cmap,  # YlOrRd
                tooltip=["segment_id", column],
                tiles="CartoDB Positron" if light_mode else "CartoDB Dark Matter",
                legend=True,
                style_kwds=dict(opacity=0.75, fillOpacity=0.75),
                popup=True,
                scheme="Quantiles",
                legend_kwds=dict(colorbar=False, fmt="{:.0f}"),
            )
        return fmap
    else:
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        if column == "distance":
            df_filtered.plot(
                column=column,
                scheme="UserDefined",
                cmap=cmap,  # YlOrRd
                kind="geo",
                ax=ax,
                legend=True,
                classification_kwds=dict(bins=bins),
                legend_kwds=dict(
                    fmt="{:.0f}", loc="upper left", bbox_to_anchor=(0, 1), interval=True
                ),
                alpha=0.75,
            )
        else:
            df_filtered.plot(
                column=column,
                cmap=cmap,  # YlOrRd
                kind="geo",
                ax=ax,
                legend=True,
                alpha=0.275,
                scheme="Quantiles",
            )
        map_provider = (
            cx.providers.CartoDB.Positron if light_mode else cx.providers.CartoDB.DarkMatter
        )
        cx.add_basemap(ax, crs=gdf.crs, source=map_provider, attribution_size=5)
        plt.axis("off")
        plt.close()
        return fig
