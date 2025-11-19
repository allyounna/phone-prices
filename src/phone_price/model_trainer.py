"""Модуль для обучения моделей машинного обучения."""

import logging

import pytorch_lightning as pl
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from .wrapper import SklearnLightningWrapper

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Класс для обучения sklearn моделей с использованием PyTorch Lightning."""

    def __init__(self, config):
        """
        Инициализация trainer.

        Args:
            config: Конфигурация trainer

        """
        self.config = config

    def train_single_model(self, model_name, x_train, x_test, y_train, y_test):
        """
        Обучение выбранной модели с использованием PyTorch Lightning.

        Args:
            model_name: Название модели
            x_train: Тренировочные данные
            x_test: Тестовые данные
            y_train: Тренировочные таргеты
            y_test: Тестовые таргеты

        Returns:
            Обученная модель, результаты обучения

        """
        logger.info(f"Обучение модели: {model_name}")

        # Подготовка данных в формате PyTorch
        train_loader, val_loader = self._prepare_data(x_train, x_test, y_train, y_test)

        # Создание модели-обертки
        model_params = self.config["model"]["params"].get(model_name, {})
        model = SklearnLightningWrapper(
            model_name=model_name,
            model_params=model_params,
            random_state=self.config["data"]["random_state"],
        )

        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="auto",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            num_sanity_val_steps=0,
        )

        # Обучение модели
        logger.info(f"Запуск обучения модели {model_name}")
        trainer.fit(model, train_loader, val_loader)

        # Финальная оценка модели
        results = self._evaluate_model(model, x_train, x_test, y_train, y_test)

        return model.model, results  # Возвращаем sklearn модель и результаты

    def _prepare_data(self, x_train, x_test, y_train, y_test):
        """
        Подготовка данных в формате PyTorch DataLoader.

        Args:
            x_train: Тренировочные данные
            x_test: Тестовые данные
            y_train: Тренировочные таргеты
            y_test: Тестовые таргеты

        Returns:
            лоадеры для обучения и валидации

        """
        # Преобразование в тензоры PyTorch
        x_train_tensor = torch.FloatTensor(x_train.values if hasattr(x_train, "values") else x_train)
        x_test_tensor = torch.FloatTensor(x_test.values if hasattr(x_test, "values") else x_test)
        y_train_tensor = torch.LongTensor(y_train.values if hasattr(y_train, "values") else y_train)
        y_test_tensor = torch.LongTensor(y_test.values if hasattr(y_test, "values") else y_test)

        # Создание datasets
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(x_test_tensor, y_test_tensor)

        batch_size = min(32, len(x_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def _evaluate_model(self, model, x_train, x_test, y_train, y_test):
        """
        Оценка модели после обучения.

        Args:
            model: модель
            x_train: Тренировочные данные
            x_test: Тестовые данные
            y_train: Тренировочные таргеты
            y_test: Тестовые таргеты

        Returns:
            результаты оценки

        """
        # Используем sklearn модель для предсказаний
        sklearn_model = model.model

        # Предсказания
        y_pred_train = sklearn_model.predict(x_train)
        y_pred_test = sklearn_model.predict(x_test)

        # Метрики
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)

        # Сохранение результатов
        results = {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "classification_report": classification_report(y_test, y_pred_test, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_pred_test).tolist(),
        }

        logger.info(f"Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

        return results
