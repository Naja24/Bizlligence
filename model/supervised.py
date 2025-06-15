import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    mean_squared_error, r2_score, mean_absolute_error, confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.impute import SimpleImputer
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Dict, List, Tuple, Optional, Any
import joblib
import logging
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

class SupervisedLearning:
    """
    A class for handling supervised learning tasks including classification and regression.
    
    This class provides functionality for:
    - Data preprocessing
    - Training multiple models in parallel
    - Performance evaluation
    - Hyperparameter tuning
    - Making predictions with the best model
    """
    
    # Define available models for each task type
    CLASSIFICATION_MODELS = {
        'logistic_regression': LogisticRegression,
        'random_forest': RandomForestClassifier,
        'gradient_boosting': GradientBoostingClassifier,
        'decision_tree': DecisionTreeClassifier,
        'svc': SVC,
        'knn': KNeighborsClassifier,
        'mlp': MLPClassifier,
        'xgboost': xgb.XGBClassifier
    }
    
    REGRESSION_MODELS = {
        'linear_regression': LinearRegression,
        'ridge': Ridge,
        'lasso': Lasso,
        'random_forest': RandomForestRegressor,
        'gradient_boosting': GradientBoostingRegressor,
        'decision_tree': DecisionTreeRegressor,
        'svr': SVR,
        'knn': KNeighborsRegressor,
        'mlp': MLPRegressor,
        'xgboost': xgb.XGBRegressor
    }
    
    # Default model parameters
    DEFAULT_MODEL_PARAMS = {
        'logistic_regression': {'C': 1.0, 'max_iter': 1000},
        'random_forest': {'n_estimators': 300 , 'max_depth' : 7},
        'gradient_boosting': {'n_estimators': 100},
        'decision_tree': {'max_depth': 5},
        'svc': {'probability': True},
        'knn': {'n_neighbors': 5},
        'mlp': {'max_iter': 1000, 'early_stopping': True},
        'linear_regression': {},
        'ridge': {'alpha': 1.0},
        'lasso': {'alpha': 1.0},
        'svr': {'C': 1.0},
        'xgboost': {'n_estimators': 100, 'learning_rate': 0.1} 
    }
    
    def __init__(
        self, 
        task_type: str = 'auto', 
        models_to_train: List[str] = None,
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: bool = True
    ):
        """
        Initialize the SupervisedLearning class.
        """
        self.task_type = task_type
        self.models_to_train = models_to_train or ["random_forest", "logistic_regression", "xgboost"]
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Initialize attributes
        self.trained_models = {}
        self.model_pipelines = {}
        self.best_model_name = None
        self.best_model = None
        self.best_pipeline = None
        self.pipeline = None  # This will store the final best pipeline
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoder = None
        self.feature_importances = None
        self.evaluation_results = {}
        self.training_times = {}
        self.best_params = None
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('SupervisedLearning')
        
        if not self.verbose:
            self.logger.setLevel(logging.WARNING)
    
    def _log(self, message: str):
        """Log message if verbose is True"""
        if self.verbose:
            self.logger.info(message)
    
    def _get_default_param_grid(self, model_name):
        """
        Return default parameter grid for hyperparameter tuning based on model type.
        """
        param_grids = {
            'logistic_regression': {
                'C': [0.001 , 0.01, 0.05 , 0.1, 0.5, 1.0],
                'solver': ['liblinear', 'lbfgs'],
                'max_iter': [1000]
            },
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 7, 10],
                'min_samples_split': [2, 5, 10]
                # 'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'decision_tree': {
                'max_depth': [None, 5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'svc': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto', 0.1, 1]
            },
            'knn': {
                'n_neighbors': [3, 5, 7, 10],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]  # 1 for manhattan, 2 for euclidean
            },
            'mlp': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            },
            'linear_regression': {
                'fit_intercept': [True, False],
                'normalize': [True, False]
            },
            'ridge': {
                'alpha': [0.1, 1.0, 10.0],
                'solver': ['auto', 'svd', 'cholesky']
            },
            'lasso': {
                'alpha': [0.1, 1.0, 10.0],
                'selection': ['cyclic', 'random']
            },
            'svr': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto', 0.1, 1]
            },
            'xgboost': {  # Added parameter grid for XGBoost
                'n_estimators': [100,200,300],
                'learning_rate': [0.01, 0.05, 0.1, 0.5],
                'max_depth': [5, 7, 11],
                'subsample': [0.5 ,0.8 , 1.0],
                # 'colsample_bytree': [0.8, 1.0],
                'lambda' : [0.1 , 1 , 5],
                'gamma' : [0.1,0.5],
                # 'min_child_weight': [1, 3, 5]
            }
        }
        
        # Return default grid for the specified model
        if model_name in param_grids:
            self._log(f"Using default parameter grid for {model_name}")
            return param_grids[model_name]
        else:
            self._log(f"No default parameter grid for {model_name}. Using empty grid.")
            return {}
    
    def detect_task_type(self, y):
        """
        Automatically detect if the task is classification or regression.
        """
        y_series = pd.Series(y)
        n_unique = y_series.nunique()
        
        # Check if categorical
        if n_unique <= 10 or pd.api.types.is_categorical_dtype(y_series):
            return 'classification'
        
        # Check if it's binary with 0s and 1s or True/False
        if n_unique == 2 and set(pd.unique(y_series)).issubset({0, 1, True, False}):
            return 'classification'
        
        # Check if float or if more than 10 unique values
        if pd.api.types.is_float_dtype(y_series) or n_unique > 10:
            return 'regression'
        
        # Default to classification if unclear
        return 'classification'
    
    def split_data(
        self, 
        X, 
        y, 
        test_size: float = 0.2, 
        stratify: bool = True
    ):
        """
        Split data into training and testing sets.
        """
        # Auto-detect task type if set to 'auto'
        if self.task_type == 'auto':
            self.task_type = self.detect_task_type(y)
            self._log(f"Task type auto-detected as: {self.task_type}")
        
        # For classification tasks, convert categorical target to numeric
        if self.task_type == 'classification':
            if not pd.api.types.is_numeric_dtype(pd.Series(y)):
                self._log("Converting categorical target to numeric")
                self.label_encoder = LabelEncoder()
                y = self.label_encoder.fit_transform(y)
        
        # Determine stratify parameter
        stratify_param = None
        if stratify and self.task_type == 'classification':
            stratify_param = y
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, 
            stratify=stratify_param
        )
        
        self._log(f"Data split: {self.X_train.shape[0]} training samples, "
                  f"{self.X_test.shape[0]} testing samples")
        
        # Initialize model list based on task type
        self._initialize_models()
        
        return self
    
    def _initialize_models(self):
        """Initialize the models based on task type and user selection"""
        available_models = (
            self.CLASSIFICATION_MODELS if self.task_type == 'classification' 
            else self.REGRESSION_MODELS
        )
        print(available_models)
        
        if self.models_to_train is None:
            self.models_to_train = list(available_models.keys())
            self._log(f"No models specified, training all {len(self.models_to_train)} available models")
        else:
            valid_models = [m for m in self.models_to_train if m in available_models]
            invalid_models = set(self.models_to_train) - set(valid_models)
            
            if invalid_models:
                self._log(f"Warning: The following models are not available for {self.task_type}: {invalid_models}")
            
            self.models_to_train = valid_models
            self._log(f"Training {len(self.models_to_train)} models: {self.models_to_train}")
    
    def build_pipeline(self, model_name, preprocessing_steps=None):
        """
        Build a pipeline that combines preprocessing steps and a model.
        """
        steps = []
        
        if preprocessing_steps is None:
            steps.append(('scaler', StandardScaler()))
        else:
            steps.extend(preprocessing_steps)
        
        if self.task_type == 'classification':
            model_class = self.CLASSIFICATION_MODELS.get(model_name)
        else:
            model_class = self.REGRESSION_MODELS.get(model_name)
        
        default_params = self.DEFAULT_MODEL_PARAMS.get(model_name, {}).copy()
        
        if model_name in ['random_forest', 'gradient_boosting', 'decision_tree',
                          'logistic_regression', 'mlp', 'xgboost']:
            default_params['random_state'] = self.random_state

        try:
            model = model_class(random_state=self.random_state, **default_params)
        except TypeError:
            model = model_class(**default_params)
        
        steps.append(('model', model))
        
        return Pipeline(steps)
    
    def train(self, X=None, y=None, preprocessing_steps=None):
        """
        Train multiple models on the provided data.
        """
        X_train = self.X_train if X is None else X
        y_train = self.y_train if y is None else y
        
        if X_train is None or y_train is None:
            raise ValueError("No training data available. Call split_data() first or provide X and y.")
        
        if not self.models_to_train:
            raise ValueError("No models to train. Initialize with models_to_train parameter.")
        
        self._log(f"Training {len(self.models_to_train)} models in parallel...")
        
        self.trained_models = {}
        self.model_pipelines = {}
        self.training_times = {}
        
        if len(self.models_to_train) <= 2:
            for model_name in self.models_to_train:
                self._train_single_model(model_name, X_train, y_train, preprocessing_steps)
        else:
            with ProcessPoolExecutor(max_workers=(min(len(self.models_to_train), self.n_jobs) if self.n_jobs > 0 else None)) as executor:
                futures = {
                    executor.submit(self._train_single_model_parallel, model_name, X_train, y_train, preprocessing_steps): 
                    model_name for model_name in self.models_to_train
                }
                
                for future in as_completed(futures):
                    model_name = futures[future]
                    try:
                        result = future.result()
                        self.model_pipelines[model_name] = result['pipeline']
                        self.trained_models[model_name] = result['pipeline'].named_steps['model']
                        self.training_times[model_name] = result['training_time']
                    except Exception as e:
                        self._log(f"Error training {model_name}: {str(e)}")
        
        self._select_best_model(X_train, y_train)
        
        # Set the final pipeline to the best pipeline for consistency
        self.pipeline = self.best_pipeline
        
        return self
    
    def _train_single_model(self, model_name, X_train, y_train, preprocessing_steps=None):
        """Train a single model and store results"""
        try:
            self._log(f"Training {model_name}...")
            start_time = time.time()
            
            pipeline = self.build_pipeline(model_name, preprocessing_steps)
            pipeline.fit(X_train, y_train)
            
            self.model_pipelines[model_name] = pipeline
            self.trained_models[model_name] = pipeline.named_steps['model']
            training_time = time.time() - start_time
            self.training_times[model_name] = training_time
            
            self._log(f"{model_name} trained in {training_time:.2f} seconds")
            return True
        except Exception as e:
            self._log(f"Error training {model_name}: {str(e)}")
            return False
    
    def _train_single_model_parallel(self, model_name, X_train, y_train, preprocessing_steps=None):
        """Parallel version of training a single model"""
        try:
            start_time = time.time()
            pipeline = self.build_pipeline(model_name, preprocessing_steps)
            pipeline.fit(X_train, y_train)
            return {
                'pipeline': pipeline,
                'model': pipeline.named_steps['model'],
                'training_time': time.time() - start_time
            }
        except Exception as e:
            raise Exception(f"Error training {model_name}: {str(e)}")
    
    def _select_best_model(self, X_train, y_train, cv=3):
        """Select the best performing model using cross-validation"""
        if not self.trained_models:
            self._log("No trained models available to select from")
            return
        
        self._log("Evaluating models to select the best one...")
        
        scoring = 'f1_weighted' if self.task_type == 'classification' else 'neg_mean_squared_error'
        
        best_score = float('-inf')
        best_model_name = None
        cv_scores = {}
        for model_name, pipeline in self.model_pipelines.items():
            try:
                scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring)
                mean_score = np.mean(scores)
                cv_scores[model_name] = mean_score
                self._log(f"{model_name} CV {scoring}: {mean_score:.4f}")
                if mean_score > best_score:
                    best_score = mean_score
                    best_model_name = model_name
            except Exception as e:
                self._log(f"Error evaluating {model_name}: {str(e)}")
        
        if best_model_name:
            self.best_model_name = best_model_name
            self.best_model = self.trained_models[best_model_name]
            self.best_pipeline = self.model_pipelines[best_model_name]
            self._log(f"Best model: {best_model_name} with score: {best_score:.4f}")
            self._extract_feature_importances(X_train, best_model_name)
            self.best_params = {}  # Optionally, store hyperparameters if tuned
        return cv_scores
    
    def _extract_feature_importances(self, X, model_name):
        """Extract feature importances if available"""
        model = self.trained_models.get(model_name)
        if model is None:
            return
        
        try:
            if hasattr(model, 'feature_importances_'):
                features = X.columns if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
                self.feature_importances = pd.DataFrame({
                    'feature': features,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                self._log(f"Feature importances extracted from {model_name}")
        except Exception as e:
            self._log(f"Could not extract feature importances: {str(e)}")
    
    def evaluate(self, X=None, y=None, model_name=None):
        """
        Evaluate trained models on test data.
        """
        X_test = self.X_test if X is None else X
        y_test = self.y_test if y is None else y
        
        if X_test is None or y_test is None:
            raise ValueError("No test data available. Call split_data() first or provide X and y.")
        
        if not self.trained_models:
            raise ValueError("No trained models available. Call train() first.")
        
        models_to_evaluate = [model_name] if model_name else self.trained_models.keys()
        
        self._log(f"Evaluating {len(models_to_evaluate)} models on test data...")
        
        results = {}
        for name in models_to_evaluate:
            if name not in self.model_pipelines:
                self._log(f"Model {name} not found in trained models")
                continue
            
            pipeline = self.model_pipelines[name]
            self._log(f"Evaluating {name}...")
            y_pred = pipeline.predict(X_test)
            
            if self.task_type == 'classification':
                try:
                    y_prob = pipeline.predict_proba(X_test)
                    if len(np.unique(y_test)) == 2:
                        auc = roc_auc_score(y_test, y_prob[:, 1])
                    else:
                        auc = roc_auc_score(pd.get_dummies(y_test), y_prob, multi_class='ovr')
                except (AttributeError, ValueError):
                    auc = None
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted'),
                    'recall': recall_score(y_test, y_pred, average='weighted'),
                    'f1': f1_score(y_test, y_pred, average='weighted'),
                    'roc_auc': auc,
                    'confusion_matrix': confusion_matrix(y_test, y_pred),
                    'classification_report': classification_report(y_test, y_pred)
                }
            else:
                metrics = {
                    'mean_squared_error': mean_squared_error(y_test, y_pred),
                    'root_mean_squared_error': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'mean_absolute_error': mean_absolute_error(y_test, y_pred),
                    'r2_score': r2_score(y_test, y_pred)
                }
            results[name] = metrics
            for key, value in metrics.items():
                if isinstance(value, (int, float, np.number)):
                    self._log(f"{name} - {key}: {value:.4f}")
        self.evaluation_results = results
        return results
    
    def tune_hyperparameters(
        self, 
        model_name=None, 
        param_grid=None, 
        cv=5, 
        scoring=None, 
        n_jobs=-1
    ):
        """
        Perform hyperparameter tuning using GridSearchCV for a specific model.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("No training data available. Call split_data() first.")
        
        if model_name is None:
            if self.best_model_name is None:
                raise ValueError("No best model selected. Call train() first.")
            model_name = self.best_model_name
        
        if model_name not in self.model_pipelines:
            raise ValueError(f"Model {model_name} not found in trained models")
        
        pipeline = self.model_pipelines[model_name]
        
        if param_grid is None:
            param_grid = self._get_default_param_grid(model_name)
        
        if scoring is None:
            scoring = 'f1_weighted' if self.task_type == 'classification' else 'neg_mean_squared_error'
        
        self._log(f"Starting hyperparameter tuning for {model_name} with {cv}-fold cross-validation...")
        start_time = time.time()
        
        pipeline_param_grid = {}
        for param, values in param_grid.items():
            if not param.startswith('model__'):
                pipeline_param_grid[f'model__{param}'] = values
            else:
                pipeline_param_grid[param] = values
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid=pipeline_param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=2 if self.verbose else 0
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        model_best_params = {}
        for key, value in grid_search.best_params_.items():
            if key.startswith('model__'):
                model_best_params[key[7:]] = value
            else:
                model_best_params[key] = value
        
        self.model_pipelines[model_name] = grid_search.best_estimator_
        self.trained_models[model_name] = grid_search.best_estimator_.named_steps['model']
        
        if model_name == self.best_model_name:
            self.best_model = self.trained_models[model_name]
            self.best_pipeline = self.model_pipelines[model_name]
            self.pipeline = self.best_pipeline
        tuning_time = time.time() - start_time
        self._log(f"Hyperparameter tuning for {model_name} completed in {tuning_time:.2f} seconds")
        self._log(f"Best parameters: {model_best_params}")
        self._log(f"Best score ({scoring}): {grid_search.best_score_:.4f}")
        
        return self
    
    def predict(self, X, model_name=None):
        """
        Make predictions using a trained model.
        """
        if model_name is None:
            if self.best_pipeline is None:
                raise ValueError("No best model selected. Call train() first.")
            pipeline = self.best_pipeline
        else:
            if model_name not in self.model_pipelines:
                raise ValueError(f"Model {model_name} not found in trained models")
            pipeline = self.model_pipelines[model_name]
        
        self._log(f"Making predictions on {X.shape[0]} samples")
        predictions = pipeline.predict(X)
        
        if self.task_type == 'classification' and self.label_encoder is not None:
            predictions = self.label_encoder.inverse_transform(predictions)
            
        return predictions
    
    def predict_proba(self, X):
        """
        Make probability predictions for classification tasks.
        """
        if self.task_type != 'classification':
            raise ValueError("predict_proba is only available for classification tasks")
            
        if self.pipeline is None or not hasattr(self.pipeline, 'predict_proba'):
            raise ValueError("Model not trained or doesn't support probability predictions")
        
        self._log(f"Making probability predictions on {X.shape[0]} samples")
        return self.pipeline.predict_proba(X)
    
    def cross_validate(self, cv=5, scoring=None):
        """
        Perform cross-validation on the training data.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("No training data available. Call split_data() first.")
        
        if self.pipeline is None:
            raise ValueError("No pipeline available for cross-validation")
        
        if scoring is None:
            scoring = 'f1_weighted' if self.task_type == 'classification' else 'neg_mean_squared_error'
        
        self._log(f"Performing {cv}-fold cross-validation...")
        
        cv_scores = cross_val_score(
            self.pipeline, 
            self.X_train, 
            self.y_train, 
            cv=cv, 
            scoring=scoring
        )
        
        cv_results = {
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'scores': cv_scores,
            'scoring': scoring
        }
        
        self._log(f"Cross-validation {scoring}: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
        
        return cv_results
    
    def plot_feature_importance(self, top_n=20):
        """
        Plot feature importances if available.
        """
        if self.feature_importances is None:
            raise ValueError("Feature importances not available for this model")
        
        plot_data = self.feature_importances.head(top_n)
        plt.figure(figsize=(10, 8))
        ax = sns.barplot(x='importance', y='feature', data=plot_data)
        ax.set_title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        self._log(f"Feature importance plot created with top {top_n} features")
        return plt.gcf()
    
    def save_model(self, filepath):
        """
        Save the trained pipeline to disk.
        """
        if self.pipeline is None:
            raise ValueError("No trained model to save. Call train() first.")
        
        joblib.dump(self.pipeline, filepath)
        self._log(f"Model saved to {filepath}")
        return self
    
    @classmethod
    def load_model(cls, filepath):
        """
        Load a saved model from disk.
        """
        instance = cls()
        instance.pipeline = joblib.load(filepath)
        if hasattr(instance.pipeline, 'named_steps') and 'model' in instance.pipeline.named_steps:
            instance.best_model = instance.pipeline.named_steps['model']
            instance.best_pipeline = instance.pipeline
            instance.pipeline = instance.best_pipeline
            if hasattr(instance.best_model, 'predict_proba'):
                instance.task_type = 'classification'
            else:
                instance.task_type = 'regression'
        instance._log(f"Model loaded from {filepath}")
        return instance
    
    def get_summary(self):
        """
        Get a summary of the trained model and its performance.
        """
        if self.pipeline is None:
            raise ValueError("No trained model available. Call train() first.")
        
        # Use best_pipeline and best_model consistently
        summary = {
            'task_type': self.task_type,
            'model_type': str(type(self.best_model).__name__),
            'training_time': self.training_times.get(self.best_model_name, None),
            'evaluation_results': self.evaluation_results,
            'best_params': self.best_params,
        }
        
        if self.feature_importances is not None:
            summary['top_features'] = self.feature_importances.head(10).to_dict('records')
        
        return summary
