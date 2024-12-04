import numpy as np


def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    ed_dist: euclidean distance between ts1 and ts2
    """
    
    # Проверка на равенство длины временных рядов
    if len(ts1) != len(ts2):
        raise ValueError("Time series must be of the same length.")

    ed_dist = np.sqrt(np.sum((ts1 - ts2) ** 2))  # Евклидово расстояние

    return ed_dist


def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the normalized Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    norm_ed_dist: normalized Euclidean distance between ts1 and ts2
    """
    
    # Проверка на равенство длины временных рядов
    if len(ts1) != len(ts2):
        raise ValueError("Time series must be of the same length.")

    ed_dist = ED_distance(ts1, ts2)  # Вычисляем евклидово расстояние
    norm_ed_dist = ed_dist / np.sqrt(len(ts1))  # Нормализованное расстояние

    return norm_ed_dist


def DTW_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate DTW distance without Sakoe-Chiba constraint.

    Parameters
    ----------
    ts1: first time series
    ts2: second time series
    
    Returns
    -------
    dtw_dist: DTW distance between ts1 and ts2
    """

    n = len(ts1)
    m = len(ts2)

    # Создаем матрицу DTW
    dtw_matrix = np.zeros((n + 1, m + 1))
    dtw_matrix[0, 1:] = np.inf
    dtw_matrix[1:, 0] = np.inf
    dtw_matrix[0, 0] = 0

    # Заполняем матрицу DTW без учета ограничения Сако-Чиба
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (ts1[i - 1] - ts2[j - 1]) ** 2
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],    # Прямо
                                           dtw_matrix[i, j - 1],    # Вниз
                                           dtw_matrix[i - 1, j - 1])  # По диагонали

    return dtw_matrix[n, m]



