import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Класс для предобработки данных"""
    
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def preprocess(self, df):
        """Полный пайплайн предобработки"""
        logger.info("Начало предобработки данных")
        
        # Разделение на признаки и целевую переменную
        X = df.drop('price_range', axis=1)
        y = df['price_range']
        
        # Сохранение имен признаков
        self.feature_names = X.columns.tolist()
        
        # Обработка пропущенных значений
        X = self._handle_missing_values(X)
        
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state'],
            stratify=y
        )
        
        # Масштабирование (для всех моделей, так как RandomForest тоже может выиграть)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Преобразование обратно в DataFrame для совместимости
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_names)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_names)
        
        logger.info("Предобработка данных завершена")
        return X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test
    
    def _handle_missing_values(self, X):
        """Обработка пропущенных значений"""
        if X.isnull().sum().sum() > 0:
            # Заполнение медианой для числовых признаков
            numeric_features = X.select_dtypes(include=[np.number]).columns
            X[numeric_features] = X[numeric_features].fillna(X[numeric_features].median())
            
            # Заполнение модой для категориальных признаков
            categorical_features = X.select_dtypes(include=['object']).columns
            for col in categorical_features:
                X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'unknown')
        
        return X