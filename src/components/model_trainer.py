import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models, load_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Splitting training and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "KNN": KNeighborsRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=0),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Ada Boost": AdaBoostRegressor(),
            }

            params = {
                "Decision Tree": {'max_depth': [3, 5, 10]},
                "Random Forest": {'n_estimators': [8, 16, 32, 64, 128, 256]},
                "KNN": {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                },
                "Linear Regression": {},
                "XGBoost": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoost": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Ada Boost": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
            }

            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with score >= 0.6")

            logging.info(f"Best model found: {best_model_name} with R2 score: {best_model_score}")

            # Load preprocessor
            preprocessor = load_object(preprocessor_path)

            # Create full pipeline
            full_pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", best_model)
            ])

            # Fit pipeline on raw training data
            full_pipeline.fit(X_train, y_train)

            # Save full pipeline
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=full_pipeline,
            )

            prediction = full_pipeline.predict(X_test)
            r2_score_test = r2_score(y_test, prediction)

            logging.info(f"Test R2 score: {r2_score_test}")

            return r2_score_test

        except Exception as e:
            raise CustomException(e, sys)
