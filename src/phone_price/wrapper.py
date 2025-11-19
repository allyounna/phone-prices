"""Модуль-обертка для sklearn моделей в PyTorch Lightning."""

import logging

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


class SklearnLightningWrapper(pl.LightningModule):
    """Обертка для sklearn моделей в PyTorch Lightning."""

    def __init__(self, model_name, model_params, random_state=42):
        """
        Инициализация обертки.

        Args:
            model_name: название модели, которую будем обучать
            model_params: параметры модели
            random_state: случайное состояние для воспроизводимости

        Returns:
            ничего

        """
        super().__init__()
        self.model_name = model_name
        self.model_params = model_params
        self.random_state = random_state
        self.model = self._create_model()
        self.is_trained = False

        # Для хранения данных для обучения
        self.train_features = []
        self.train_targets = []

        # Для хранения данных для валидации
        self.val_features = []
        self.val_targets = []

        # Отключаем автоматическую оптимизацию для sklearn моделей
        self.automatic_optimization = False

    def _create_model(self):
        """Создание sklearn модели на основе конфигурации."""
        if self.model_name == "random_forest":
            model = RandomForestClassifier(
                n_estimators=self.model_params.get("n_estimators", 100),
                max_depth=self.model_params.get("max_depth", None),
                random_state=self.random_state,
            )
        elif self.model_name == "svm":
            model = SVC(
                kernel=self.model_params.get("kernel", "rbf"),
                C=self.model_params.get("C", 1.0),
                random_state=self.random_state,
                probability=True,
            )
        elif self.model_name == "logistic_regression":
            model = LogisticRegression(
                max_iter=self.model_params.get("max_iter", 1000),
                C=self.model_params.get("C", 1.0),
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Неизвестная модель: {self.model_name}")

        logger.info(f"Создана модель {self.model_name} с параметрами: {self.model_params}")
        return model

    def forward(self, x):
        """Предсказание для батча данных."""
        if not self.is_trained:
            raise RuntimeError("Модель не обучена. Сначала вызовите fit.")

        x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
        return self.model.predict(x_np)

    def predict_proba(self, x):
        """Вероятности предсказаний."""
        if not self.is_trained:
            raise RuntimeError("Модель не обучена. Сначала вызовите fit.")

        x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(x_np)
        return None

    def training_step(self, batch, batch_idx):
        """Тренировочный шаг - собираем данные для обучения."""
        x, y = batch
        x_np = x.cpu().numpy()
        y_np = y.cpu().numpy()

        # Сохраняем данные для обучения (признаки и цели)
        self.train_features.append(x_np)
        self.train_targets.append(y_np)

        # Не возвращаем loss, так как отключена автоматическая оптимизация

    def validation_step(self, batch, batch_idx):
        """Валидационный шаг - собираем данные для оценки."""
        if not self.is_trained:
            # Если модель еще не обучена, пропускаем валидацию
            return None

        x, y = batch
        x_np = x.cpu().numpy()
        y_np = y.cpu().numpy()

        # Сохраняем данные для валидации
        self.val_features.append(x_np)
        self.val_targets.append(y_np)

        return None

    def on_train_epoch_start(self):
        """В начале каждой эпохи очищаем собранные данные."""
        self.train_features = []
        self.train_targets = []

    def on_train_epoch_end(self):
        """В конце тренировочной эпохи обучаем модель на всех собранных данных."""
        if len(self.train_features) > 0:
            # Объединяем все батчи в один массив
            x_train = np.vstack(self.train_features)
            y_train = np.hstack(self.train_targets)

            # Обучаем модель на всех данных
            logger.info(f"Обучение модели {self.model_name} на {len(x_train)} примерах")
            self.model.fit(x_train, y_train)
            self.is_trained = True

    def on_validation_epoch_start(self):
        """В начале валидационной эпохи очищаем собранные данные."""
        self.val_features = []
        self.val_targets = []

    def on_validation_epoch_end(self):
        """В конце валидационной эпохи вычисляем метрики."""
        if not self.is_trained or len(self.val_features) == 0:
            return

        # Объединяем все батчи валидации
        x_val = np.vstack(self.val_features)
        y_val = np.hstack(self.val_targets)

        # Проверяем, что есть данные для оценки
        if len(x_val) == 0:
            return

        # Делаем предсказания на валидационных данных
        y_pred = self.model.predict(x_val)
        accuracy = accuracy_score(y_val, y_pred)

        self.log("val_accuracy", float(accuracy), prog_bar=True)  # Явное преобразование

    def configure_optimizers(self):
        """Sklearn модели не требуют оптимизаторов."""
        return None
