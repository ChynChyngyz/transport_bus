import os
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from thefuzz import fuzz
from .utils import download_write_file

MOBILITY_SOURCES_LINK = "https://bit.ly/catalogs-csv"


def fetch_gtfs_source(
    place: str = "ALL",
    country_code: Optional[str] = "US",
    active: Optional[bool] = True,
    use_fuzz: bool = False,
) -> Any:
    """
    Извлекает источники данных GTFS из файла данных мобильности и создает фрейм данных.

    Аргументы:
    place (str): Место, для которого вы хотите получить данные GTFS. Это может быть город, штат или страна. По умолчанию "ALL".
    country_code (str): Код страны для фильтрации источников данных. По умолчанию "US".
    active (bool): Если True, будут загружены только активные каналы. Если False, будут загружены все каналы. По умолчанию True.

    Возвращает:
    pd.DataFrame: Фрейм данных с источниками данных GTFS.

    Примеры:
    >>> fetch_gtfs_source()
    Возвращает все источники данных GTFS из Кыргызстана.

    >>> fetch_gtfs_source(place="New York")
    Возвращает источники данных GTFS для места "Бишкек" в Кыргызстане.
    """
    abb_df = pd.read_json(ABBREV_LINK)
    sources_df = pd.read_csv(MOBILITY_SOURCES_LINK)

    if country_code != "ALL":
        sources_df = sources_df[sources_df["location.country_code"] == country_code]
    sources_df = sources_df[sources_df["data_type"] == "gtfs"]
    # Download only active feeds
    if active:
        sources_df = sources_df[sources_df["status"].isin(["active", np.nan, None])]
        sources_df.drop(["status"], axis=1, inplace=True)
    sources_df = pd.merge(
        sources_df,
        abb_df,
        how="left",
        left_on="location.subdivision_name",
        right_on="state",
    )
    # sources_df = sources_df[~sources_df.state_code.isna()]
    sources_df["location.municipality"] = sources_df["location.municipality"].astype("str")
    sources_df.drop(
        [
            "entity_type",
            "mdb_source_id",
            "data_type",
            "note",
            "static_reference",
            "urls.direct_download",
            "urls.authentication_type",
            "urls.license",
            "location.bounding_box.extracted_on",
            "urls.authentication_info",
            "urls.api_key_parameter_name",
            "features",
        ],
        axis=1,
        inplace=True,
    )
    file_names = []
    for i, row in sources_df.iterrows():
        if row["location.municipality"] != "nan":
            if (
                len(
                    sources_df[
                        (sources_df["location.municipality"] == row["location.municipality"])
                        & (sources_df["provider"] == row["provider"])
                    ]
                )
                <= 1
            ):
                f_name = (
                    str(row["location.municipality"])
                    + "-"
                    + str(row["provider"])
                    + "-"
                    + str(row["state_code"])
                )
            else:
                f_name = (
                    str(row["location.municipality"])
                    + "-"
                    + str(row["provider"])
                    + "-"
                    + str(row["name"])
                    + "-"
                    + str(row["state_code"])
                )
        else:
            if (
                len(
                    sources_df[
                        (
                            sources_df["location.subdivision_name"]
                            == row["location.subdivision_name"]
                        )
                        & (sources_df["provider"] == row["provider"])
                    ]
                )
                <= 1
            ):
                f_name = (
                    str(row["location.subdivision_name"])
                    + "-"
                    + str(row["provider"])
                    + "-"
                    + str(row["state_code"])
                )
            else:
                f_name = (
                    str(row["location.subdivision_name"])
                    + "-"
                    + str(row["provider"])
                    + "-"
                    + str(row["name"])
                    + "-"
                    + str(row["state_code"])
                )
        f_name = f_name.replace("/", "").strip()
        file_names.append(f_name)
    sources_df.drop(
        [
            "provider",
            "location.municipality",
            "location.subdivision_name",
            "name",
            "state_code",
            "state",
        ],
        axis=1,
        inplace=True,
    )
    sources_df.insert(0, "provider", file_names)
    sources_df.columns = sources_df.columns.str.replace("location.bounding_box.", "", regex=True)
    sources_df.rename(
        columns={
            "location.country_code": "country_code",
            "minimum_longitude": "min_lon",
            "maximum_longitude": "max_lon",
            "minimum_latitude": "min_lat",
            "maximum_latitude": "max_lat",
            "urls.latest": "url",
        },
        inplace=True,
    )
    if place == "ALL":
        return sources_df.reset_index(drop=True)
    else:
        if use_fuzz:
            sources_df = fuzzy_match(place, sources_df)
        else:
            sources_df = sources_df[
                sources_df.apply(
                    lambda row: row.astype(str).str.contains(place.lower(), case=False).any(),
                    axis=1,
                )
            ]
        if len(sources_df) == 0:
            print("No sources found for the given place")
        else:
            return sources_df.reset_index(drop=True)


def fuzzy_match(place: str, sources_df: pd.DataFrame) -> pd.DataFrame:
    sources_df["fuzz_ratio"] = sources_df["provider"].apply(
        lambda x: fuzz.partial_token_sort_ratio(x.lower(), place.lower())
    )
    sources_df = sources_df[sources_df["fuzz_ratio"] >= 75]

    return sources_df.drop("fuzz_ratio", axis=1).reset_index(drop=True)


def summary_stats_mobility(
    df: pd.DataFrame,
    folder_path: str,
    filename: str,
    link: str,
    bounds: List,
    max_spacing: float = 3000,
    export: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Принимает фрейм данных, путь к папке, имя файла, самый загруженный день, ссылку, ограничивающий прямоугольник, максимальный
    интервал и логическое значение для экспорта сводки в CSV.

    Затем он вычисляет процент сегментов, у которых интервал больше максимального.
    Затем он фильтрует фрейм данных, чтобы включить только сегменты с интервалом меньше максимального.
    Затем он вычисляет среднее взвешенное значение сегмента, среднее взвешенное значение маршрута, среднее взвешенное значение обхода, среднеквадратичное отклонение
    взвешенного значения обхода, 25-й процентиль взвешенного значения обхода, 50-й процентиль взвешенного значения обхода, 75-й процентиль взвешенного значения обхода, количество сегментов, количество маршрутов, количество обходов и
    максимальный интервал. Затем он создает словарь со всеми указанными выше значениями и создает фрейм данных
    из словаря. Затем он экспортирует фрейм данных в CSV, если логическое значение экспорта равно true. Если экспортное логическое значение равно false, он транспонирует фрейм данных и возвращает его.

    Args:
    df: dataframe, содержащий данные о мобильности
    folder_path: Путь к папке, в которой вы хотите сохранить файл summary.csv.
    filename: Имя файла, в котором вы хотите сохранить данные.
    b_day: Самый загруженный день недели
    link: Ссылка на карту, которую вы хотите использовать.
    bounds: Ограничивающий прямоугольник области, которую вы хотите проанализировать.
    max_spacing: Максимальное расстояние между двумя остановками, которое вы хотите рассмотреть. По умолчанию 3000
    export: Если True, сводка будет сохранена как CSV-файл в folder_path. Если False, сводка
    будет возвращена как dataframe. По умолчанию False

    Возвращает:
    Dataframe со сводной статистикой данных о мобильности.
    """
    percent_spacing = round(
        df[df["distance"] > max_spacing]["traversals"].sum() / df["traversals"].sum() * 100,
        3,
    )
    df = df[df["distance"] <= max_spacing]
    csv_path = os.path.join(folder_path, "summary.csv")
    seg_weighted_mean = (
        df.groupby(["segment_id", "distance"])
        .first()
        .reset_index()["distance"]
        .apply(pd.Series)
        .mean()
        .round(2)
    )
    seg_weighted_median = (
        df.groupby(["segment_id", "distance"])
        .first()
        .reset_index()["distance"]
        .apply(pd.Series)
        .median()
        .round(2)
    )
    route_weighted_mean = (
        df.groupby(["route_id", "segment_id", "distance"])
        .first()
        .reset_index()["distance"]
        .apply(pd.Series)
        .mean()
        .round(2)
    )
    route_weighted_median = (
        df.groupby(["route_id", "segment_id", "distance"])
        .first()
        .reset_index()["distance"]
        .apply(pd.Series)
        .median()
        .round(2)
    )
    weighted_data = np.hstack([np.repeat(x, y) for x, y in zip(df["distance"], df.traversals)])
    df_dict = {
        "Name": filename,
        "Link": link,
        "Min Latitude": bounds[0][1],
        "Min Longitude": bounds[0][0],
        "Max Latitude": bounds[1][1],
        "Max Longitude": bounds[1][0],
        "Segment Weighted Mean": seg_weighted_mean,
        "Route Weighted Mean": route_weighted_mean,
        "Traversal Weighted Mean": round(np.mean(weighted_data), 3),
        "Segment Weighted Median": seg_weighted_median,
        "Route Weighted Median": route_weighted_median,
        "Traversal Weighted Median": round(np.median(weighted_data), 2),
        "Traversal Weighted Std": round(np.std(weighted_data), 3),
        "Traversal Weighted 25 % Quantile": round(np.quantile(weighted_data, 0.25), 3),
        "Traversal Weighted 50 % Quantile": round(np.quantile(weighted_data, 0.5), 3),
        "Traversal Weighted 75 % Quantile": round(np.quantile(weighted_data, 0.75), 3),
        "No of Segments": len(df.segment_id.unique()),
        "No of Routes": len(df.route_id.unique()),
        "No of Traversals": sum(df.traversals),
        "Max Spacing": max_spacing,
        "% Segments w/ spacing > max_spacing": percent_spacing,
    }
    summary_df = pd.DataFrame([df_dict])
    if export:
        try:
            summary_df.to_csv(csv_path, index=False)
            print("Saved the summary.csv in " + folder_path)
        except FileNotFoundError as e:
            print("Error saving the summary.csv: " + str(e))
        return None
    else:
        summary_df = summary_df.T
        return summary_df


def download_latest_data(sources_df: pd.DataFrame, out_folder_path: str) -> None:
    """
    Перебирает строки фрейма данных и для каждой строки пытается загрузить файл с URL-адреса в столбце `urls.latest` и записать его в папку, указанную в столбце `provider`

    Аргументы:
    sources_df: Это фрейм данных, содержащий URL-адреса для данных.
    out_folder_path: Путь к папке, в которой вы хотите сохранить данные.
    """
    for i, row in sources_df.iterrows():
        try:
            download_write_file(row["url"], os.path.join(out_folder_path, row["provider"]))
        except Exception as e:
            print("Error downloading the file for " + row["provider"] + " : " + str(e))
            continue
    print("Downloaded the latest data")
