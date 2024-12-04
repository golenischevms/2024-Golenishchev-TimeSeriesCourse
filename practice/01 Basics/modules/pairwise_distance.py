import numpy as np
from modules.metrics import ED_distance, norm_ED_distance, DTW_distance
from modules.utils import z_normalize

class PairwiseDistance:
    """
    Distance matrix between time series 

    Parameters
    ----------
    metric: distance metric between two time series
            Options: {euclidean, dtw}
    is_normalize: normalize or not time series
    """

    def __init__(self, metric: str = 'euclidean', is_normalize: bool = False) -> None:
        self.metric: str = metric
        self.is_normalize: bool = is_normalize

    @property
    def distance_metric(self) -> str:
        """Return the distance metric

        Returns
        -------
            string with metric which is used to calculate distances between set of time series
        """
        norm_str = "normalized " if self.is_normalize else "non-normalized "
        return norm_str + self.metric + " distance"

    def _choose_distance(self):
        """ Choose distance function for calculation of matrix
        
        Returns
        -------
        dist_func: function reference
        """
        # Выбор функции для вычисления расстояний на основе метрики и нормализации
        if self.metric == 'euclidean':
            dist_func = norm_ED_distance if self.is_normalize else ED_distance
        elif self.metric == 'dtw':
            dist_func = DTW_distance
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        return dist_func

    def calculate(self, input_data: np.ndarray) -> np.ndarray:
        """ Calculate distance matrix
        
        Parameters
        ----------
        input_data: time series set (2D array where each row is a time series)
        
        Returns
        -------
        matrix_values: distance matrix
        """
        n_series = input_data.shape[0]
        matrix_values = np.zeros((n_series, n_series))
        
        # Выбор функции для вычисления расстояний
        dist_func = self._choose_distance()

        # Нормализация временных рядов, если указано
        if self.is_normalize:
            input_data = np.array([z_normalize(series) for series in input_data])

        # Вычисление верхнего треугольника матрицы расстояний
        for i in range(n_series):
            for j in range(i + 1, n_series):
                distance = dist_func(input_data[i], input_data[j])
                matrix_values[i, j] = distance
                matrix_values[j, i] = distance  # Симметричность матрицы

        return matrix_values
