"""Mobile Price Classification - Prediction Script."""

import logging
import time
from typing import cast

import hydra
import pandas as pd
from omegaconf import DictConfig
from omegaconf import OmegaConf

import mlflow
from phone_price.data_preprocessor import DataPreprocessor
from phone_price.utils import load_trained_model_and_preprocessor
from phone_price.utils import setup_logging

mlflow.set_tracking_uri("http://localhost:5000")

logger = logging.getLogger(__name__)


def predict(model, preprocessor: DataPreprocessor, input_data):
    """
    Выполнение предсказаний.

    Args:
        model: модель
        preprocessor: препроцессор
        input_data: денные для предсказаний

    Returns:
        Предсказания, вероятности предсказаний

    """
    start = time.time()
    # Предобработка входных данных
    processed_data = preprocessor.preprocess_single(input_data)

    # Предсказание
    predictions = model.predict(processed_data)
    prediction_probas = model.predict_proba(processed_data) if hasattr(model, "predict_proba") else None
    latency = time.time() - start
    with mlflow.start_run(run_name="inference", nested=True):
        mlflow.log_metric("latency_ms", latency * 1000)
        mlflow.log_metric("n_rows", len(input_data))
    return predictions, prediction_probas


@hydra.main(version_base=None, config_path="configs", config_name="predict")
def main(cfg: DictConfig):
    """Load script."""
    args = cast(dict, OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/PriceModel/{args['model_stage']}",
    )
    setup_logging()

    try:
        # Загрузка данных
        logger.info(f"Загрузка данных из {args['input_data']}")
        input_df = pd.read_csv(args["input_data"])

        # Загрузка модели и preprocessor
        logger.info(f"Загрузка модели из {args['models_dir']}")
        model, preprocessor, model_path, model_name = load_trained_model_and_preprocessor(args["models_dir"])
        print(preprocessor)
        # Выполнение предсказаний
        predictions, probabilities = predict(model, preprocessor, input_df)

        # Вывод результатов
        print(f"Результаты предсказаний (модель: {model_name}):")
        print("=" * 50)

        price_categories = {
            0: "Низкая стоимость",
            1: "Средняя стоимость",
            2: "Высокая стоимость",
            3: "Очень высокая стоимость",
        }

        for i in range(len(predictions)):
            pred = predictions[i]
            proba = probabilities[i] if probabilities is not None else None
            category = price_categories.get(pred, "Неизвестно")
            print(f"Образец {i + 1}: {category} (класс {pred})")
            if proba is not None:
                print(f"Вероятности: {dict(enumerate([f'{p:.3f}' for p in proba]))}")

        # Сохранение результатов
        results_df = input_df.copy()
        results_df["predicted_price_range"] = predictions
        results_df["predicted_category"] = [price_categories[p] for p in predictions]

        output_path = f"predictions_{model_path.name}.csv"
        results_df.to_csv(output_path, index=False)
        logger.info(f"Результаты сохранены в: {output_path}")

    except Exception as e:
        logger.error(f"Ошибка: {str(e)}")
        raise


if __name__ == "__main__":
    main()
