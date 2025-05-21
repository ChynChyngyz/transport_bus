import os
import shutil
import traceback
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.stats import gaussian_kde
from shapely.geometry import Point

# Plot style
plt.style.use("ggplot")


def plot_hist(
    df: pd.DataFrame, save_fig: bool = False, show_mean: bool = False, **kwargs: Any
) -> plt.Figure:
    """
    Он берет dataframe с двумя столбцами, один с расстоянием между остановками, а другой с
    количеством проходов между этими остановками, и строит взвешенную гистограмму расстояний

    Аргументы:
    df: dataframe, содержащий данные
    save_fig: если True, рисунок будет сохранен в file_path. По умолчанию False
    show_mean: если True, будет показано среднее значение распределения. По умолчанию False

    Возвращает:
    Axis matplotlib
    """
    if "max_spacing" not in kwargs.keys():
        max_spacing = 3000
        print("Using max_spacing = 3000")
    else:
        max_spacing = kwargs["max_spacing"]
    if "ax" in kwargs.keys():
        ax = kwargs["ax"]
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
    df = df[df["distance"] < max_spacing]
    data = np.hstack([np.repeat(x, y) for x, y in zip(df["distance"], df.traversals)])
    plt.hist(
        data,
        range=(0, max_spacing),
        density=True,
        bins=int(max_spacing / 50),
        fc=(0, 105 / 255, 160 / 255, 0.4),
        ec="white",
        lw=0.8,
    )
    x = np.arange(0, max_spacing, 5)
    plt.plot(x, gaussian_kde(data)(x), lw=1.5, color=(0, 85 / 255, 120 / 255, 1))
    # sns.histplot(data,binwidth=50,stat = "density",kde=True,ax=ax)
    plt.xlim([0, max_spacing])
    plt.xlabel("Stop Spacing [m]")
    plt.ylabel("Density - Traversal Weighted")
    plt.title("Histogram of Spacing")
    if show_mean:
        plt.axvline(np.mean(data), color="k", linestyle="dashed", linewidth=2)
        _, max_ylim = plt.ylim()
        plt.text(
            np.mean(data) * 1.1,
            max_ylim * 0.9,
            "Mean: {:.0f}".format(np.mean(data)),
            fontsize=12,
        )
    if "title" in kwargs.keys():
        plt.title(kwargs["title"])
    if save_fig:
        assert "file_path" in kwargs.keys(), "Please pass in the `file_path`"
        plt.savefig(kwargs["file_path"], dpi=300)
    plt.close()
    return fig


def summary_stats(
    df: pd.DataFrame, max_spacing: float = 3000, min_spacing: float = 10, export: bool = False, **kwargs: Any
) -> pd.DataFrame:
    """
    принимает фрейм данных и возвращает фрейм данных со сводной статистикой.
    Max_spacing и min_spacing служат порогом для удаления выбросов.

    Аргументы:
    df: фрейм данных, для которого вы хотите получить сводную статистику.
    max_spacing: максимальный интервал между двумя остановками. По умолчанию 3000[м]
    min_spacing: минимальный интервал между двумя остановками. По умолчанию 10[м]
    export: если True, сводка будет экспортирована в файл CSV. По умолчанию False

    Возвращает:
    фрейм данных со сводной статистикой
    """
    print("Using max_spacing = ", max_spacing)
    print("Using min_spacing = ", min_spacing)
    percent_spacing = round(
        df[df["distance"] > max_spacing]["traversals"].sum() / df["traversals"].sum() * 100,
        3,
    )
    df = df[(df["distance"] <= max_spacing) & (df["distance"] >= min_spacing)]
    seg_weighted_mean = (
        df.groupby(["segment_id", "distance"]).first().reset_index()["distance"].mean()
    )
    seg_weighted_median = (
        df.groupby(["segment_id", "distance"]).first().reset_index()["distance"].median()
    )
    route_weighted_mean = (
        df.groupby(["route_id", "segment_id", "distance"]).first().reset_index()["distance"].mean()
    )
    route_weighted_median = (
        df.groupby(["route_id", "segment_id", "distance"])
        .first()
        .reset_index()["distance"]
        .median()
    )
    weighted_data = np.hstack([np.repeat(x, y) for x, y in zip(df["distance"], df.traversals)])

    df_dict = {
        "Segment Weighted Mean": np.round(seg_weighted_mean, 2),
        "Route Weighted Mean": np.round(route_weighted_mean, 2),
        "Traversal Weighted Mean": np.round(np.mean(weighted_data), 3),
        "Segment Weighted Median": np.round(seg_weighted_median, 2),
        "Route Weighted Median": np.round(route_weighted_median, 2),
        "Traversal Weighted Median": np.round(np.median(weighted_data), 2),
        "Traversal Weighted Std": np.round(np.std(weighted_data), 3),
        "Traversal Weighted 25 % Quantile": np.round(np.quantile(weighted_data, 0.25), 3),
        "Traversal Weighted 50 % Quantile": np.round(np.quantile(weighted_data, 0.50), 3),
        "Traversal Weighted 75 % Quantile": np.round(np.quantile(weighted_data, 0.75), 3),
        "No of Segments": int(len(df.segment_id.unique())),
        "No of Routes": int(len(df.route_id.unique())),
        "No of Traversals": int(sum(df.traversals)),
        "Max Spacing": int(max_spacing),
        "% Segments w/ spacing > max_spacing": percent_spacing,
    }
    summary_df = pd.DataFrame([df_dict])
    # df.set_index(summary_df.columns[0],inplace=True)
    if export:
        assert "file_path" in kwargs.keys(), "Please pass in the `file_path`"
        summary_df.to_csv(kwargs["file_path"], index=False)
        print("Saved the summary in " + kwargs["file_path"])
    summary_df = summary_df.T
    return summary_df


def export_segments(
    df: pd.DataFrame, file_path: str, output_format: str, geometry: bool = True
) -> None:
    """
    Эта функция принимает GeoDataFrame сегментов, путь к файлу, формат вывода и логическое значение
    для включения или невключения геометрии в вывод.

    Если формат вывода — GeoJSON, функция выведет GeoDataFrame в файл GeoJSON.

    Если формат вывода — CSV, функция выведет GeoDataFrame в файл CSV. Если
    логическое значение геометрии установлено в True, функция выведет файл CSV со столбцом геометрии. Если
    логическое значение геометрии установлено в False, функция выведет файл CSV без столбца геометрии.

    Функция также добавит дополнительные столбцы в файл CSV, включая начальную и конечную точки
    сегментов, начальную и конечную долготу и широту сегментов и расстояние
    сегментов.

    Функция также добавит столбец в файл CSV, который указывает количество раз, когда сегмент
    был пройден.

    Args:
    df: dataframe, содержащий сегменты
    file_path: путь к файлу, в который вы хотите экспортировать.
    output_format: geojson или csv
    [Необязательно] geometry: если True, вывод будет включать геометрию сегментов. Если False, вывод будет включать только начальную и конечную точки сегментов. По умолчанию True
    """
    # Output to GeoJSON
    if output_format == "geojson":
        df.to_file(file_path, driver="GeoJSON")
    elif output_format == "csv":
        s_df = df.copy()
        geom_list = s_df.geometry.apply(lambda g: np.array(g.coords))
        s_df["start_point"] = [Point(g[0]).wkt for g in geom_list]
        s_df["end_point"] = [Point(g[-1]).wkt for g in geom_list]
        sg_df = s_df.copy()
        s_df["start_lon"] = [g[0][0] for g in geom_list]
        s_df["start_lat"] = [g[0][1] for g in geom_list]
        s_df["end_lon"] = [g[-1][0] for g in geom_list]
        s_df["end_lat"] = [g[-1][1] for g in geom_list]
        if geometry:
            # Output With LS
            sg_df.to_csv(file_path, index=False)
        else:
            d_df = s_df.drop(columns=["geometry", "start_point", "end_point"])
            # Output without LS
            d_df.to_csv(file_path, index=False)


def process(pipeline_gtfs: Any, row: pd.core.series.Series, max_spacing: float) -> Any:
    """
    Он принимает конвейер, строку из sources_df и max_spacing и возвращает вывод
    конвейера

    Аргументы:
    pipeline_gtfs: Это функция, которая будет использоваться для обработки данных GTFS.
    row: Это строка в фрейме данных sources_df. Она содержит имя поставщика, URL-адрес
    файла gtfs и ограничивающий прямоугольник области, которую охватывает файл gtfs.
    max_spacing: Максимально допустимый интервал между двумя последовательными остановками.

    Возвращает:
    Возвращаемое значение — кортеж в форме (имя_файла, путь_к_папке, df)
    """
    filename = row["provider"]
    url = row["urls.latest"]
    bounds = [
        [row["minimum_longitude"], row["minimum_latitude"]],
        [row["maximum_longitude"], row["maximum_latitude"]],
    ]
    print(filename)
    try:
        return pipeline_gtfs(filename, url, bounds, max_spacing)
    except Exception as e:
        traceback.print_exc()
        raise ValueError(f"Failed for {filename}") from e


def failed_pipeline(message: str, filename: str, folder_path: str) -> str:
    """
    "Если путь к папке существует, удалить его и вернуть сообщение об ошибке".

    Аргументы:
    message: Возвращаемое сообщение
    filename: Имя обрабатываемого файла
    folder_path: Путь к папке, в которой находится файл

    Возвращает:
    строку, которая является конкатенацией сообщения и имени файла, что указывает на ошибку
    """

    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    return message + " : " + filename


def download_write_file(url: str, folder_path: str) -> str:
    """
    принимает URL и путь к папке в качестве входных данных, создает новую папку, если она не существует, загружает
    файл с URL и записывает файл в путь к папке

    Аргументы:
    url: URL файла GTFS, который вы хотите загрузить
    folder_path: Путь к папке, в которой вы хотите сохранить файл GTFS.

    Возвращает:
    Местоположение загруженного файла.
    """
    # Create a new directory if it does not exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # Download file from URL
    gtfs_file_loc = os.path.join(folder_path, "gtfs.zip")

    try:
        r = requests.get(url, allow_redirects=True, timeout=300)
        # Write file locally
        file = open(gtfs_file_loc, "wb")
        file.write(r.content)
        file.close()
    except requests.exceptions.RequestException as e:
        print(e)
        raise ValueError(f"Failed to download {url}") from e
    return gtfs_file_loc
