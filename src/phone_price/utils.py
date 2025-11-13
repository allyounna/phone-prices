import yaml
import os
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

def setup_logging():
    """Настройка логирования"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )

def load_config(config_path=None):
    """Загрузка конфигурации"""
    default_config = {
        'data': {
            'test_size': 0.2,
            'random_state': 42
        },
        'model': {
            'name': 'random_forest',
            'params': {
                'random_forest': {
                    'n_estimators': 100,
                    'max_depth': None
                },
                'svm': {
                    'kernel': 'rbf',
                    'C': 1.0
                },
                'logistic_regression': {
                    'max_iter': 1000,
                    'C': 1.0
                }
            }
        }
    }
    
    if config_path and os.path.exists(config_path):
        logger.info(f"Загрузка конфигурации из {config_path}")
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
            # Рекурсивное обновление конфигурации
            def update_dict(d, u):
                for k, v in u.items():
                    if isinstance(v, dict) and k in d:
                        d[k] = update_dict(d[k], v)
                    else:
                        d[k] = v
                return d
            default_config = update_dict(default_config, user_config)
    else:
        logger.info("Использование конфигурации по умолчанию")
    
    return default_config

def get_latest_model_path(models_dir="models"):
    """Получение пути к последней обученной модели"""
    models_path = Path(models_dir)
    if not models_path.exists():
        raise FileNotFoundError(f"Директория {models_dir} не существует")
    
    # Ищем поддиректории с timestamp
    model_dirs = [d for d in models_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if not model_dirs:
        raise FileNotFoundError(f"Модели не найдены в {models_dir}")
    
    # Сортируем по имени (timestamp) и берем последнюю
    latest_dir = sorted(model_dirs)[-1]
    return latest_dir

def load_trained_model_and_preprocessor(models_dir="models"):
    """Загрузка обученной модели и preprocessor"""
    model_path = get_latest_model_path(models_dir)
    
    # Ищем файлы модели
    model_files = list(model_path.glob("*_model.pkl"))
    if not model_files:
        raise FileNotFoundError(f"Файлы модели не найдены в {model_path}")
    
    # Загружаем модель
    model_file = model_files[0]
    model = joblib.load(model_file)
    
    # Загружаем preprocessor
    preprocessor_file = model_path / "preprocessor.pkl"
    if not preprocessor_file.exists():
        raise FileNotFoundError(f"Preprocessor не найден: {preprocessor_file}")
    
    preprocessor = joblib.load(preprocessor_file)
    
    # Загружаем конфигурацию для получения имени модели
    config_file = model_path / "config.yaml"
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        model_name = config.get('model', {}).get('name', 'unknown')
    else:
        model_name = 'unknown'
    
    logger.info(f"Загружена модель: {model_name} из {model_file.name}")
    return model, preprocessor, model_path, model_name