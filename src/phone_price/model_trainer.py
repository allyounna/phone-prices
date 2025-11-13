from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Класс для обучения одной выбранной модели"""
    
    def __init__(self, config):
        self.config = config
    
    def train_single_model(self, model_name, X_train, X_test, y_train, y_test):
        """Обучение одной выбранной модели"""
        logger.info(f"Обучение модели: {model_name}")
        
        # Создание модели с параметрами из конфига
        model = self._create_model(model_name)
        
        # Обучение модели
        model.fit(X_train, y_train)
        
        # Оценка модели
        results = self._evaluate_model(model, X_train, X_test, y_train, y_test)
        
        return model, results
    
    def _create_model(self, model_name):
        """Создание модели на основе конфигурации"""
        model_params = self.config['model']['params'].get(model_name, {})
        
        if model_name == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=model_params.get('n_estimators', 100),
                max_depth=model_params.get('max_depth', None),
                random_state=self.config['data']['random_state']
            )
        elif model_name == 'svm':
            model = SVC(
                kernel=model_params.get('kernel', 'rbf'),
                C=model_params.get('C', 1.0),
                random_state=self.config['data']['random_state']
            )
        elif model_name == 'logistic_regression':
            model = LogisticRegression(
                max_iter=model_params.get('max_iter', 1000),
                C=model_params.get('C', 1.0),
                random_state=self.config['data']['random_state']
            )
        else:
            raise ValueError(f"Неизвестная модель: {model_name}")
        
        logger.info(f"Создана модель {model_name} с параметрами: {model_params}")
        return model
    
    def _evaluate_model(self, model, X_train, X_test, y_train, y_test):
        """Оценка модели"""
        # Предсказания
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Метрики
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        # Сохранение результатов
        results = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'classification_report': classification_report(y_test, y_pred_test, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist()
        }
        
        logger.info(f"Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
        
        return results