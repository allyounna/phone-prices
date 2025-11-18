"""Модуль для загрузки данных."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_PRICE_CATEGORIES = 4


class DataLoader:
    """Класс для загрузки и валидации данных."""

    def __init__(self, config):
        """Инициализация загрузки данных.

        Args:
            config: Конфигурация загрузки

        """
        self.config = config
        self.expected_features = [
            "battery_power",
            "blue",
            "clock_speed",
            "dual_sim",
            "fc",
            "four_g",
            "int_memory",
            "m_dep",
            "mobile_wt",
            "n_cores",
            "pc",
            "px_height",
            "px_width",
            "ram",
            "sc_h",
            "sc_w",
            "talk_time",
            "three_g",
            "touch_screen",
            "wifi",
            "price_range",
        ]

    def load_data(self, data_path):
        """Загрузка данных с валидацией.

        Args:
            data_path: Путь к файлу с данными

        Returns:
            DataFrame с загруженными данными

        """
        logger.info(f"Загрузка данных из {data_path}")

        if not Path(data_path).exists():
            raise FileNotFoundError(f"Файл {data_path} не найден")

        df = pd.read_csv(data_path)

        # Валидация данных
        self._validate_data(df)

        logger.info(f"Данные успешно загружены. Размер: {df.shape}")
        return df

    def _validate_data(self, df):
        """Валидация структуры данных.

        Args:
            df: DataFrame с данными

        Returns:
            True в случае успеха

        """
        missing_features = set(self.expected_features) - set(df.columns)
        if missing_features:
            raise ValueError(f"Отсутствуют признаки: {missing_features}")

        # Проверка на пропущенные значения
        if df.isnull().sum().sum() > 0:
            missing_counts = df.isnull().sum()
            missing_features = missing_counts[missing_counts > 0]
            logger.warning(f"Обнаружены пропущенные значения: {missing_features.to_dict()}")

        # Проверка целевой переменной
        if df["price_range"].nunique() != EXPECTED_PRICE_CATEGORIES:
            logger.warning("Неожиданное количество уникальных значений в price_range")

        return True
