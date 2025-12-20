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

    def preprocess(self, df: pd.DataFrame):
        """
        Полный пайплайн предобработки.

        Args:
            df: Датасет с данными

        Returns:
            Обработанный датасет, разделенный на train и test

        """
        logger.info("Начало предобработки данных")

        # Разделение на признаки и целевую переменную
        x: pd.DataFrame = df.drop(columns=["price_range"])
        y: pd.Series = pd.Series(df["price_range"])

        # Сохраняем имена признаков (явная типизация)
        self.feature_names: list[str] = list(x.columns)

        # Обработка пропущенных значений
        x_processed: pd.DataFrame = self._handle_missing_values(x)

        # Разделение данных
        x_train, x_test, y_train, y_test = train_test_split(
            x_processed,
            y,
            test_size=self.config["data"]["test_size"],
            random_state=self.config["data"]["random_state"],
            stratify=y,
        )

        # Масштабирование
        x_train_scaled_array = self.scaler.fit_transform(x_train)
        x_test_scaled_array = self.scaler.transform(x_test)

        # Явно задаем Axes (Index) для columns
        columns: pd.Index = pd.Index(self.feature_names)

        # Преобразование обратно в DataFrame
        x_train_scaled: pd.DataFrame = pd.DataFrame(
            x_train_scaled_array,
            columns=columns,
            index=pd.Axes(x_train.index),
        )

        x_test_scaled: pd.DataFrame = pd.DataFrame(
            x_test_scaled_array,
            columns=columns,
            index=pd.Axes(x_test.index),
        )

        logger.info("Предобработка данных завершена")

        return (
            x_train_scaled,
            x_test_scaled,
            x_train,
            x_test,
            y_train,
            y_test,
        )

    def preprocess_single(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Предобработка данных для одиночного inference без разделения на train/test.

        Args:
            df: DataFrame с входными данными (1 или более строк)

        Returns:
            DataFrame после всех преобразований, совместимый с обученной моделью.

        """
        logger.info("Начало предобработки данных для inference")

        # Работаем с копией, чтобы не модифицировать входной DataFrame
        df_input: pd.DataFrame = df.copy()

        # Удаляем целевую переменную, если она присутствует
        if "price_range" in df_input.columns:
            df_input = df_input.drop(columns=["price_range"])

        # Сохраняем имена признаков (явная типизация для type checker)
        self.feature_names: list[str] = list(df_input.columns)

        # Обработка пропусков
        df_processed: pd.DataFrame = self._handle_missing_values(df_input)

        # Масштабирование (используется обученный scaler)
        scaled_array = self.scaler.transform(df_processed)

        # Преобразуем результат обратно в DataFrame
        df_scaled: pd.DataFrame = pd.DataFrame(
            scaled_array,
            columns=pd.Axes(self.feature_names),
            index=df_processed.index,
        )

        logger.info("Предобработка данных для inference завершена")

        return df_scaled

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
