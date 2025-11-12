import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import argparse
import joblib
import os

class MobilePriceClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
    
    def load_data(self, data_path):
        """Загрузка и первоначальный анализ данных"""
        print("📊 Загрузка данных...")
        self.df = pd.read_csv(data_path)
        print(f"Размер dataset: {self.df.shape}")
        print("\nПервые 5 строк:")
        print(self.df.head())
        print("\nИнформация о данных:")
        print(self.df.info())
        print("\nСтатистика:")
        print(self.df.describe())
        return self.df
    
    def explore_data(self):
        """Разведочный анализ данных"""
        print("\n🔍 Разведочный анализ данных...")
        
        # Распределение целевой переменной
        plt.figure(figsize=(10, 6))
        sns.countplot(x='price_range', data=self.df)
        plt.title('Распределение ценовых категорий')
        plt.savefig('price_distribution.png')
        plt.close()
        
        # Корреляционная матрица
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.df.corr()
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
        plt.title('Корреляционная матрица')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png')
        plt.close()
        
        # Анализ важных признаков
        plt.figure(figsize=(10, 6))
        top_features = correlation_matrix['price_range'].sort_values(ascending=False)[1:11]
        sns.barplot(x=top_features.values, y=top_features.index)
        plt.title('Топ-10 признаков, влияющих на цену')
        plt.savefig('top_features.png')
        plt.close()
    
    def preprocess_data(self):
        """Предобработка данных"""
        print("\n⚙️ Предобработка данных...")
        
        # Разделение на признаки и целевую переменную
        X = self.df.drop('price_range', axis=1)
        y = self.df['price_range']
        
        # Разделение на train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Масштабирование признаков
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Train set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_models(self):
        """Обучение нескольких моделей"""
        print("\n🤖 Обучение моделей...")
        
        # Определение моделей
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='rbf', random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        for name, model in models.items():
            print(f"Обучение {name}...")
            if name == 'svm' or name == 'logistic_regression':
                model.fit(self.X_train_scaled, self.y_train)
            else:
                model.fit(self.X_train, self.y_train)
            
            self.models[name] = model
        
        return self.models
    
    def evaluate_models(self):
        """Оценка всех моделей"""
        print("\n📊 Оценка моделей...")
        
        for name, model in self.models.items():
            print(f"\n--- {name.upper()} ---")
            
            # Выбор данных в зависимости от модели
            if name == 'svm' or name == 'logistic_regression':
                X_test = self.X_test_scaled
                X_train = self.X_train_scaled
            else:
                X_test = self.X_test
                X_train = self.X_train
            
            # Предсказания
            y_pred = model.predict(X_test)
            y_pred_train = model.predict(X_train)
            
            # Метрики
            test_accuracy = accuracy_score(self.y_test, y_pred)
            train_accuracy = accuracy_score(self.y_train, y_pred_train)
            
            print(f"Accuracy (train): {train_accuracy:.4f}")
            print(f"Accuracy (test): {test_accuracy:.4f}")
            print(f"\nClassification Report:\n{classification_report(self.y_test, y_pred)}")
            
            # Матрица ошибок
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(self.y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Матрица ошибок - {name}')
            plt.ylabel('Истинные значения')
            plt.xlabel('Предсказанные значения')
            plt.savefig(f'confusion_matrix_{name}.png')
            plt.close()
            
            # Сохранение результатов
            self.results[name] = {
                'test_accuracy': test_accuracy,
                'train_accuracy': train_accuracy,
                'classification_report': classification_report(self.y_test, y_pred, output_dict=True)
            }
    
    def save_models(self, model_dir='models'):
        """Сохранение обученных моделей"""
        print(f"\n💾 Сохранение моделей в папку '{model_dir}'...")
        
        os.makedirs(model_dir, exist_ok=True)
        
        for name, model in self.models.items():
            model_path = os.path.join(model_dir, f'{name}_model.pkl')
            joblib.dump(model, model_path)
            print(f"Модель {name} сохранена как {model_path}")
        
        # Сохранение scaler
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        print(f"Scaler сохранен как {scaler_path}")
    
    def run_complete_pipeline(self, data_path):
        """Запуск полного пайплайна"""
        self.load_data(data_path)
        self.explore_data()
        self.preprocess_data()
        self.train_models()
        self.evaluate_models()
        self.save_models()
        
        # Вывод лучшей модели
        best_model = max(self.results.items(), key=lambda x: x[1]['test_accuracy'])
        print(f"\n🎉 Лучшая модель: {best_model[0]} с accuracy {best_model[1]['test_accuracy']:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Mobile Price Classification Pipeline')
    parser.add_argument('--data_path', type=str, required=True, 
                       help='Путь к CSV файлу с данными')
    parser.add_argument('--train', action='store_true',
                       help='Запустить полный процесс обучения')
    parser.add_argument('--explore', action='store_true',
                       help='Только разведочный анализ данных')
    parser.add_argument('--model_dir', type=str, default='models',
                       help='Папка для сохранения моделей')
    
    args = parser.parse_args()
    
    classifier = MobilePriceClassifier()
    
    if args.train:
        print("🚀 Запуск полного процесса обучения...")
        classifier.run_complete_pipeline(args.data_path)
    
    elif args.explore:
        print("🔍 Запуск только разведочного анализа...")
        classifier.load_data(args.data_path)
        classifier.explore_data()
    
    else:
        print("❌ Укажите --train для обучения или --explore для анализа данных")

if __name__ == "__main__":
    main()