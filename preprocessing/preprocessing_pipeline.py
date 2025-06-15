import pandas as pd
from preprocessing.data_cleaning import DataCleaner
from preprocessing.feature_engineering import FeatureEngineer
from sklearn.preprocessing import LabelEncoder

class PreprocessingPipeline:
    def __init__(self, impute_type=None, strategy="mean", n_neighbors=5, task_type='supervised',
                 target_col=None,
                 numeric_features=None, categorical_features=None, 
                 scale_method='standard', normalize=True, use_poly=False, poly_degree=2):
        """
        Combines data cleaning and feature engineering. Separates target column and, for classification,
        label encodes it.
        """
        self.target_col = target_col
        self.task_type = task_type.lower()
        self.data_cleaner = DataCleaner(impute_type=impute_type, strategy=strategy, 
                                        n_neighbors=n_neighbors, task_type=task_type)
        self.feature_engineer = FeatureEngineer(task_type = task_type, use_poly= False, poly_degree = 3)
        if self.task_type == 'supervised' and self.target_col is not None:
            self.target_encoder = LabelEncoder()
        else:
            self.target_encoder = None

    def process(self, df: pd.DataFrame):
        """
        Cleans the data, applies feature engineering, and processes the target column.
        For classification tasks, label encodes the target.
        Returns:
          X_transformed: Processed features.
          y_processed: Processed target.
        """
        # Separate target if provided
        if self.target_col is not None and self.target_col in df.columns:
            y = df[self.target_col].copy()
            X = df.drop(columns=[self.target_col])
        else:
            X = df.copy()
            y = None

        # Data cleaning
        X_clean = self.data_cleaner.transform(X)
        y = y.loc[X_clean.index]
        # Feature engineering
        X_transformed = self.feature_engineer.transform(X_clean)
        
        if self.task_type == 'supervised' and y is not None:
            self.target_encoder.fit(y)
            y_processed = pd.Series(self.target_encoder.transform(y), name=self.target_col)
        else:
            y_processed = y
        
        return X_transformed, y_processed
