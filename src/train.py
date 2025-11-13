
"""
Mobile Price Classification - Training Script
"""
import argparse
import logging
from phone_price.data_loader import DataLoader
from phone_price.data_preprocessor import DataPreprocessor
from phone_price.model_trainer import ModelTrainer
from phone_price.model_saver import ModelSaver
from phone_price.utils import setup_logging, load_config

logger = logging.getLogger(__name__)

def run_training_pipeline(data_path, config_path=None, models_dir="models"):
    """Запуск полного пайплайна обучения"""
    logger.info("Запуск обучения")
    
    # Загрузка конфигурации
    config = load_config(config_path)
    
    # Проверка конфигурации модели
    if 'model' not in config:
        raise ValueError("Конфигурация должна содержать раздел 'model'")
    
    model_name = config['model']['name']
    logger.info(f"Выбрана модель: {model_name}")
    
    # Инициализация компонентов
    data_loader = DataLoader(config)
    preprocessor = DataPreprocessor(config)
    model_trainer = ModelTrainer(config)
    if config['model']['save']:
        model_saver = ModelSaver(models_dir)
    
    try:
        # Этап 1: Загрузка данных
        logger.info("Загрузка данных.")
        df = data_loader.load_data(data_path)
        
        # Этап 2: Предобработка
        logger.info("Предобработка данных..")
        X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test = preprocessor.preprocess(df)
        
        # Этап 3: Обучение модели
        logger.info(f"Обучение модели {model_name}....")
        model, results = model_trainer.train_single_model(
            model_name, X_train_scaled, X_test_scaled, y_train, y_test
        )
        
        # Финальный вывод
        logger.info(f" Обучение завершено! Модель: {model_name}")
        logger.info(f" Train Accuracy: {results['train_accuracy']:.4f}")
        logger.info(f" Test Accuracy: {results['test_accuracy']:.4f}")
        
        if config['model']['save']:
            # Этап 4: Сохранение модели
            logger.info("Сохранение модели....")
            model_path = model_saver.save_model(model, model_name, results, preprocessor, config)
            logger.info(f" Модель сохранена в: {model_path}")
            
        
    except Exception as e:
        logger.error(f" Ошибка: {str(e)}")
        raise

def main():
    """Основная функция для запуска обучения из командной строки"""
    parser = argparse.ArgumentParser(description='Mobile Price Classification Training')
    
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to the CSV data file')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                       help='Path to configuration YAML file')
    parser.add_argument('--models_dir', type=str, default='models',
                       help='Directory to save trained models')
    
    args = parser.parse_args()
    
    # Настройка логирования
    setup_logging()
    
    # Запуск пайплайна
    run_training_pipeline(args.data_path, args.config, args.models_dir)

if __name__ == "__main__":
    main()