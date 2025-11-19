"""Mobile Price Classification - Training Script."""

import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import yaml

logger = logging.getLogger(__name__)


class ModelSaver:
    """Класс для сохранения обученной модели и артефактов."""

    def __init__(self, models_dir="models"):
        """
        Инициализация сохранения данных.

        Args:
            models_dir: Директория, в которую сохранять модель

        """
        self.models_dir = Path(models_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_path = self.models_dir / self.timestamp
        self._create_model_dir()

    def _create_model_dir(self):
        """Создание директории для модели."""
        self.model_path.mkdir(parents=True, exist_ok=True)
        logger.info(f" Создана директория для модели: {self.model_path}")

    def save_model(self, model, model_name, results, preprocessor, config):
        """
        Сохранение модели и связанных артефактов.

        Args:
            model: модель
            model_name: название модели
            results: Результаты тренировки
            preprocessor: Препоцессор данных для модели
            config: конфиг модели

        Returns:
            Путь до сохраненной модели

        """
        logger.info(f" Сохранение модели {model_name}")

        # Сохранение модели
        model_filename = f"{model_name}_model.pkl"
        model_filepath = self.model_path / model_filename
        joblib.dump(model, model_filepath)
        logger.info(f" Модель сохранена: {model_filepath}")

        # Сохранение preprocessor
        preprocessor_filepath = self.model_path / "preprocessor.pkl"
        joblib.dump(preprocessor, preprocessor_filepath)
        logger.info(f" Preprocessor сохранен: {preprocessor_filepath}")

        # Сохранение конфигурации
        config_filepath = self.model_path / "config.yaml"
        with open(config_filepath, "w") as f:
            yaml.dump(config, f)
        logger.info(f" Конфигурация сохранена: {config_filepath}")

        # Сохранение результатов
        results_filepath = self.model_path / "training_results.json"

        # Конвертируем для JSON сериализации
        serializable_results = {
            "model_name": model_name,
            "train_accuracy": float(results["train_accuracy"]),
            "test_accuracy": float(results["test_accuracy"]),
            "classification_report": results["classification_report"],
            "confusion_matrix": results["confusion_matrix"],
        }

        with open(results_filepath, "w") as f:
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Результаты сохранены: {results_filepath}")

        return self.model_path
