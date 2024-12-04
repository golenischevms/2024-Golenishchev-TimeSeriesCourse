import numpy as np
import math
from modules.utils import z_normalize
from modules.metrics import ED_distance, norm_ED_distance


def brute_force(ts: np.ndarray, query: np.ndarray, is_normalize: bool = True) -> np.ndarray:
    """
    Вычисление профиля расстояний с использованием алгоритма грубой силы с оптимизациями

    Параметры
    ----------
    ts: временной ряд
    query: запрос, длина которого меньше длины временного ряда
    is_normalize: нормализовать ли временной ряд и запрос

    Возвращает
    -------
    dist_profile: оптимизированный профиль расстояний между запросом и временным рядом
    """
    n = len(ts)  # Длина временного ряда
    m = len(query)  # Длина запроса
    N = n - m + 1  # Количество подпоследовательностей, с которыми будем сравнивать запрос
    dist_profile = np.zeros(N)  # Массив для хранения профиля расстояний

    # Предварительно нормализуем запрос, если это необходимо
    if is_normalize:
        query = z_normalize(query)

    # Вычисляем кумулятивные суммы и суммы квадратов для временного ряда
    cumsum = np.cumsum(ts)  # Кумулятивная сумма
    cumsum2 = np.cumsum(ts**2)  # Кумулятивная сумма квадратов

    # Проходим по каждой подпоследовательности временного ряда
    for i in range(N):
        # Извлекаем текущую подпоследовательность
        subseq = ts[i:i + m]
        
        if is_normalize:
            # Вычисляем сумму и сумму квадратов текущей подпоследовательности через кумулятивные суммы
            sum_seq = cumsum[i + m - 1] - (cumsum[i - 1] if i > 0 else 0)
            sum_seq2 = cumsum2[i + m - 1] - (cumsum2[i - 1] if i > 0 else 0)
            # Находим среднее и стандартное отклонение
            mean_seq = sum_seq / m
            std_seq = math.sqrt(sum_seq2 / m - mean_seq**2)
            
            # Нормализуем подпоследовательность, если стандартное отклонение не равно нулю
            if std_seq > 0:
                subseq = (subseq - mean_seq) / std_seq
            else:
                # Если стандартное отклонение равно нулю, задаём подпоследовательность как нулевой массив
                subseq = np.zeros_like(subseq)

        # Вычисляем евклидово расстояние между нормализованным запросом и подпоследовательностью
        dist_profile[i] = np.linalg.norm(query - subseq)

    return dist_profile

