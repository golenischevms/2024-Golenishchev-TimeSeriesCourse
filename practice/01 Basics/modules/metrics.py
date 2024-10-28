import numpy as np

def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance between two time series.

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    ed_dist: Euclidean distance between ts1 and ts2
    """
    # Вычисление Евклидова расстояния
    ed_dist = np.sqrt(np.sum((ts1 - ts2) ** 2))
    return ed_dist


def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the normalized Euclidean distance between two time series.

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    norm_ed_dist: normalized Euclidean distance between ts1 and ts2
    """
    # Вычисление среднего и стандартного отклонения
    mu_ts1 = np.mean(ts1)
    mu_ts2 = np.mean(ts2)
    sigma_ts1 = np.std(ts1)
    sigma_ts2 = np.std(ts2)
    
    print(f"mu_ts1: {mu_ts1}, mu_ts2: {mu_ts2}")
    print(f"sigma_ts1: {sigma_ts1}, sigma_ts2: {sigma_ts2}")
    
    # Проверка на деление на ноль
    if sigma_ts1 == 0 or sigma_ts2 == 0:
        raise ValueError("Стандартное отклонение не должно быть равно нулю.")
    
    # Вычисление скалярного произведения
    inner_product = np.dot(ts1, ts2)
    print(f"Скалярное произведение: {inner_product}")
    
    # Вычисление длины временных рядов
    n = len(ts1)

    # Вычисление нормализованного евклидова расстояния
    norm_ed_dist = np.sqrt(
        np.abs(2 * n * (1 - (inner_product - n * mu_ts1 * mu_ts2) / (n * sigma_ts1 * sigma_ts2)))
    )
    
    return norm_ed_dist



def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1.0) -> float:
    """
    Calculate DTW distance with an optional warping window.

    Parameters
    ----------
    ts1: first time series
    ts2: second time series
    r: warping window size, expressed as a proportion of the series length
    
    Returns
    -------
    dtw_dist: DTW distance between ts1 and ts2
    """
    n = len(ts1)
    m = len(ts2)
    window = max(int(r * n), abs(n - m))  # Применение окна деформации

    # Инициализация матрицы стоимости пути
    d = np.full((n + 1, m + 1), np.inf)
    d[0, 0] = 0

    # Заполнение матрицы стоимости
    for i in range(1, n + 1):
        for j in range(max(1, i - window), min(m + 1, i + window + 1)):
            cost = (ts1[i - 1] - ts2[j - 1]) ** 2
            d[i, j] = cost + min(d[i-1, j],    # вставка
                                 d[i, j-1],    # удаление
                                 d[i-1, j-1])  # совпадение

    # Финальное DTW расстояние с применением квадратного корня
    dtw_dist = np.sqrt(d[n, m])
    return dtw_dist