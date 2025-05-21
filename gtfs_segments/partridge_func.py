import os
from typing import Optional

import pandas as pd

import gtfs_segments.partridge_mod as ptg

from .partridge_mod.gtfs import Feed, parallel_read


def get_bus_feed(
    path: str, agency_id: Optional[str] = None, threshold: Optional[int] = 1, parallel: bool = False
) -> Feed:
    """
    Функция `get_bus_feed` извлекает данные автобусного потока из указанного пути с возможностью фильтрации
    по названию агентства и возвращает самую загруженную дату и объект потока GTFS.

    Аргументы:
    path (str): Параметр `path` — это строка, представляющая путь к файлу GTFS. Этот файл
    содержит данные автобусного потока.
    agency_id (All): Параметр `agency_id` — это необязательный параметр, позволяющий фильтровать
    данные автобусного потока по названию агентства. Он используется для указания идентификатора транспортного агентства, для которого
    вы хотите получить данные автобусного потока. Если вы укажете `agency_id`, функция вернет только
    data
    threshold (int): Параметр `threshold` используется для фильтрации идентификаторов служб, которые имеют низкую
    частоту. Он установлен на значение по умолчанию 1, но вы можете изменить его на другое значение, если необходимо.
    Идентификаторы служб с суммой времени остановок, превышающей порог, будут включены в возвращаемый
    bus. По умолчанию 1

    Возвращает:
    Кортеж, содержащий самую загруженную дату и объект фида GTFS. Объект фида GTFS содержит
    информацию о маршрутах, остановках, времени остановок, поездках и формах для расписания транспортного агентства.
    """
    b_day, bday_service_ids = ptg.read_busiest_date(path)
    print("Using the busiest day:", b_day)
    all_days_s_ids_df = get_all_days_s_ids(path)
    series = all_days_s_ids_df[bday_service_ids].sum(axis=0) > threshold
    service_ids = series[series].index.values
    route_types = [3, 700, 702, 703, 704, 705]  # 701 is regional
    removed_service_ids = set(bday_service_ids) - set(service_ids)
    if len(removed_service_ids) > 0:
        print("Service IDs eliminated due to low frequency:", removed_service_ids)
    if agency_id is not None:
        view = {
            "routes.txt": {"route_type": route_types},  # Only bus routes
            "trips.txt": {"service_id": service_ids},  # Busiest day only
            "agency.txt": {"agency_id": agency_id},
        }
    else:
        view = {
            "routes.txt": {"route_type": route_types},  # Only bus routes
            "trips.txt": {"service_id": service_ids},  # Busiest day only
        }
    feed = ptg.load_geo_feed(path, view=view)
    if parallel:
        num_cores = os.cpu_count()
        print(":: Processing Feed in Parallel :: Number of cores:", num_cores)
        parallel_read(feed)
    return feed


def get_all_days_s_ids(path: str) -> pd.DataFrame:
    """
    Считывает даты по идентификаторам служб из указанного пути, создает DataFrame, заполняет его датами и
    идентификаторами служб и заполняет недостающие значения значением False.

    Аргументы:
    путь: путь к файлу GTFS

    Возвращает:
    DataFrame, содержащий даты и идентификаторы служб.
    """
    dates_by_service_ids = ptg.read_dates_by_service_ids(path)
    data = dates_by_service_ids
    # Create a DataFrame
    data_frame = pd.DataFrame(columns=sorted(list({col for row in data.keys() for col in row})))

    # Просмотрите данные и заполните DataFrame.
    for service_ids, dates in data.items():
        for date_value in dates:
            data_frame.loc[date_value, list(service_ids)] = True

    # Заполните пропущенные значения значением False
    data_frame.fillna(False, inplace=True)
    return data_frame
