"""Модуль для предобработки данных."""

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Класс для предобработки данных."""

    def __init__(self, config):
        """
        Инициализация препроцессора данных.

        Args:
            config: Конфигурация препроцессора

        """
        self.config = config
        self.scaler = StandardScaler()
        self.feature_names = None

    def preprocess(self, df):
        """
        Полный пайплайн предобработки.

        Args:
            df: Датасет с данными

        Returns:
            Обработанный датасет, разделенный на train и test

        """
        logger.info("Начало предобработки данных")

        # Разделение на признаки и целевую переменную
        x = df.drop("price_range", axis=1)
        y = df["price_range"]

        # Сохранение имен признаков
        self.feature_names = x.columns.tolist()

        # Обработка пропущенных значений
        x = self._handle_missing_values(x)

        # Разделение данных
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=self.config["data"]["test_size"],
            random_state=self.config["data"]["random_state"],
            stratify=y,
        )

        # Масштабирование (для всех моделей, так как RandomForest тоже может выиграть)
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)

        # Преобразование обратно в DataFrame для совместимости
        x_train_scaled = pd.DataFrame(x_train_scaled, columns=self.feature_names)
        x_test_scaled = pd.DataFrame(x_test_scaled, columns=self.feature_names)

        logger.info("Предобработка данных завершена")
        return x_train_scaled, x_test_scaled, x_train, x_test, y_train, y_test

    def _handle_missing_values(self, x):
        """
        Обработка пропущенных значений.

        Args:
            x: Данные, который нужно заполнить

        Returns:
            Обработанный столбец

        """
        if x.isnull().sum().sum() > 0:
            # Заполнение медианой для числовых признаков
            numeric_features = x.select_dtypes(include=[np.number]).columns
            x[numeric_features] = x[numeric_features].fillna(x[numeric_features].median())

            # Заполнение модой для категориальных признаков
            categorical_features = x.select_dtypes(include=["object"]).columns
            for col in categorical_features:
                x[col] = x[col].fillna(x[col].mode()[0] if len(x[col].mode()) > 0 else "unknown")

        return x
