"""Mobile Price Classification - Training Script."""

import logging
from typing import cast

import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf

import mlflow
from mlflow.sklearn import autolog
from phone_price.data_loader import DataLoader
from phone_price.data_preprocessor import DataPreprocessor
from phone_price.model_saver import ModelSaver
from phone_price.model_trainer import ModelTrainer
from phone_price.utils import setup_logging

logger = logging.getLogger(__name__)


def run_training_pipeline(config: dict):
    """
    Запуск полного пайплайна обучения.

    Args:
        data_path: путь до тренировочных данных
        config_path: конфиг обучения
        models_dir: Директория, в которую сохранять модель

    """
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("price_prediction")

    logger.info("Запуск обучения")

    model_name = config["model"]["name"]
    logger.info(f"Выбрана модель: {model_name}")

    # Инициализация компонентов
    with mlflow.start_run(run_name="baseline_model"):
        autolog(registered_model_name="PriceModel")
        data_loader = DataLoader(config)
        preprocessor = DataPreprocessor(config)
        model_trainer = ModelTrainer(config)

        # Всегда инициализируем model_saver, но используем только если нужно
        model_saver = ModelSaver(config["models_dir"]) if config.get("model", {}).get("save", True) else None

        try:
            # Этап 1: Загрузка данных
            logger.info("Загрузка данных.")
            df = data_loader.load_data(config["data_path"])

            # Этап 2: Предобработка
            logger.info("Предобработка данных..")
            x_train_scaled, x_test_scaled, x_train, x_test, y_train, y_test = preprocessor.preprocess(df)

            # Этап 3: Обучение модели
            logger.info(f"Обучение модели {model_name}....")
            model, results = model_trainer.train_single_model(
                model_name,
                x_train_scaled,
                x_test_scaled,
                y_train,
                y_test,
            )

            # Финальный вывод
            logger.info(f"Обучение завершено! Модель: {model_name}")
            logger.info(f"Train Accuracy: {results['train_accuracy']:.4f}")
            logger.info(f"Test Accuracy: {results['test_accuracy']:.4f}")
            mlflow.log_dict(results, "training_metrics.json")

            # mlflow.log_artifact("model.pkl")

            if model_saver is not None:
                # Этап 4: Сохранение модели
                logger.info("Сохранение модели....")
                model_path = model_saver.save_model(model, model_name, results, preprocessor, config)
                logger.info(f"Модель сохранена в: {model_path}")
            else:
                logger.info("Сохранение модели пропущено (config['model']['save'] = False)")

        except Exception as e:
            logger.error(f"Ошибка: {str(e)}")
            raise


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    """Load script."""
    cfg_dict = cast(dict, OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    # Настройка логирования
    setup_logging()
    # Запуск пайплайна
    run_training_pipeline(cfg_dict)


if __name__ == "__main__":
    main()
