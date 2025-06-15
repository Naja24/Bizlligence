import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder , OneHotEncoder , OrdinalEncoder , TargetEncoder
import re
from datetime import datetime
import warnings
from typing import Union, List, Dict, Optional, Tuple

class DataCleaner:
    def __init__(self, 
                 impute_type=None, 
                 strategy="mean", 
                 n_neighbors=5, 
                 task_type='supervised',
                 datetime_features=True,
                 drop_high_missing=0.8,
                 drop_low_variance=0.01,
                 cardinality_threshold=100,
                 max_skewness=5,
                 text_cleaning=True,
                 detect_data_types=True,
                #  auto_encode_categoricals=True,
                 handle_imbalance=False):
        """
        Initialize DataCleaner with comprehensive data cleaning capabilities.
        
        Parameters:
          impute_type (str): 'knn' for KNNImputer, 'iterative' for IterativeImputer,
                            or None for SimpleImputer.
          strategy (str): Strategy for numeric imputation (e.g., "mean", "median", "most_frequent").
          n_neighbors (int): For KNNImputer.
          task_type (str): 'supervised' or 'unsupervised'.
          datetime_features (bool): Extract features from datetime columns.
          drop_high_missing (float): Drop columns with missing values ratio above this threshold.
          drop_low_variance (float): Drop columns with variance below this threshold.
          cardinality_threshold (int): Max unique values for categorical encoding.
          max_skewness (float): Apply transforms to features with skewness above this value.
          text_cleaning (bool): Whether to clean text columns.
          detect_data_types (bool): Automatically detect and convert data types.
          auto_encode_categoricals (bool): Automatically encode categorical features.
          handle_imbalance (bool): Apply techniques to handle class imbalance (supervised only).
        """
        self.task_type = task_type.lower()
        self.datetime_features = datetime_features
        self.drop_high_missing = drop_high_missing
        self.drop_low_variance = drop_low_variance
        self.cardinality_threshold = cardinality_threshold
        self.max_skewness = max_skewness
        self.text_cleaning = text_cleaning
        self.detect_data_types = detect_data_types
        # self.auto_encode_categoricals = auto_encode_categoricals
        self.handle_imbalance = handle_imbalance
        
        # Imputers
        if impute_type == 'knn':
            self.num_imputer = KNNImputer(n_neighbors=n_neighbors)
        elif impute_type == 'iterative':
            # Only import if needed to avoid dependency issues
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
            self.num_imputer = IterativeImputer(random_state=42)
        else:
            self.num_imputer = SimpleImputer(strategy=strategy)
        
        # For categorical imputation
        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        
        # For encoding
        self.encoders = {}
        
        # Track transformed columns
        self.dropped_columns = []
        self.datetime_columns = []
        self.text_columns = []
        self.high_cardinality_columns = []
        self.numeric_columns = []
        self.categorical_columns = []
        
        # Stats
        self.missing_stats = {}
        self.cardinality_stats = {}
        self.skewness_stats = {}
        
    def infer_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Automatically detect and convert column data types."""
        if not self.detect_data_types:
            return df
            
        df = df.copy()
        
        # Date detection pattern
        date_pattern = re.compile(r'(?:date|time|dt|created|modified|timestamp)', re.IGNORECASE)
        
        for col in df.columns:
            # Skip columns that are already datetime
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                self.datetime_columns.append(col)
                continue
                
            # Try to identify datetime columns by name and content
            if date_pattern.search(col):
                try:
                    # Convert to datetime if possible
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    self.datetime_columns.append(col)
                    continue
                except:
                    pass
            
            # Detect numeric columns with string format
            if df[col].dtype == 'object':
                # Check if column contains primarily numbers in string format
                try:
                    numeric_values = pd.to_numeric(df[col], errors='coerce')
                    # If >80% of values converted successfully, consider it numeric
                    if numeric_values.notna().mean() > 0.8:
                        df[col] = numeric_values
                        continue
                except:
                    pass
                    
                # Text column detection (longer strings, sentences)
                # Text vs. categorical detection for string columns
                if df[col].dtype == 'object':
                    # Get non-null values and convert to string
                    clean_values = df[col].dropna()
                    sample = clean_values.sample(min(100, len(clean_values))).astype(str)
                    
                    # Check various text indicators
                    avg_length = sample.str.len().mean()
                    has_sentences = sample.str.contains(r'[.!?]').mean() > 0.3
                    
                    # Calculate cardinality metrics
                    unique_count = clean_values.nunique()
                    unique_ratio = unique_count / len(clean_values) if len(clean_values) > 0 else 0
                    
                    # Conditions for text columns:
                    # 1. Long average length OR
                    # 2. Contains sentence endings AND has high cardinality
                    is_text = (avg_length > 50) or (has_sentences and unique_ratio > 0.5)
                    
                    # Conditions for categorical:
                    # Short text with limited unique values
                    is_categorical = (avg_length < 30) and (unique_count < 100) and (unique_ratio < 0.2)
                    
                    if is_text:
                        self.text_columns.append(col)
                    elif is_categorical:
                        self.categorical_columns.append(col)
        
        return df
        
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes duplicate rows from the DataFrame."""
        original_shape = df.shape
        df = df.drop_duplicates()
        if original_shape[0] > df.shape[0]:
            print(f"Removed {original_shape[0] - df.shape[0]} duplicate rows.")
        return df
        
    def drop_constant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns with constant value."""
        df = df.copy()
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            self.dropped_columns.extend(constant_cols)
            df = df.drop(columns=constant_cols)
            print(f"Dropped {len(constant_cols)} constant columns: {constant_cols}")
        return df
        
    def drop_high_missing_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns with high percentage of missing values."""
        df = df.copy()
        
        # Calculate missing percentage for each column
        missing_percentage = df.isnull().mean()
        self.missing_stats = missing_percentage.to_dict()
        
        # Identify columns to drop
        high_missing_cols = missing_percentage[missing_percentage > self.drop_high_missing].index.tolist()
        
        if high_missing_cols:
            self.dropped_columns.extend(high_missing_cols)
            df = df.drop(columns=high_missing_cols)
            print(f"Dropped {len(high_missing_cols)} columns with >{self.drop_high_missing*100}% missing values: {high_missing_cols}")
            
        return df
        
    def drop_low_variance_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop numeric columns with low variance."""
        df = df.copy()
        
        # Only consider numeric columns
        num_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(num_cols) == 0:
            return df
            
        # Calculate variance for each column
        variances = df[num_cols].var()
        low_variance_cols = variances[variances < self.drop_low_variance].index.tolist()
        
        if low_variance_cols:
            self.dropped_columns.extend(low_variance_cols)
            df = df.drop(columns=low_variance_cols)
            print(f"Dropped {len(low_variance_cols)} low variance columns: {low_variance_cols}")
            
        return df

    def detect_feature_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Automatically detect different types of features in the DataFrame.
        
        Returns:
            Tuple containing lists of numeric, categorical, datetime, and text columns
        """
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        # Detect datetime columns
        datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
        # Additional datetime detection
        date_pattern = r'(?:date|time|dt|created|modified|timestamp)'
        for col in df.columns:
            if col in datetime_cols:
                continue
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                # Check column name for datetime patterns
                if any(date_word in col.lower() for date_word in ['date', 'time', 'day', 'month', 'year']):
                    try:
                        pd.to_datetime(df[col], errors='raise')
                        datetime_cols.append(col)
                        if col in categorical_cols:
                            categorical_cols.remove(col)
                    except:
                        pass
        
        # Detect text columns (simple heuristic: string columns with average length > 50)
        text_cols = []
        for col in categorical_cols.copy():
            if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                # Check if it has strings with significant length
                if df[col].dropna().astype(str).str.len().mean() > 50:
                    text_cols.append(col)
                    categorical_cols.remove(col)
                    
        return numeric_cols, categorical_cols, datetime_cols, text_cols
        
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Imputes missing values using appropriate strategies for different column types.
        """
        df = df.copy()
        
        # Identify column types
        # num_cols, cat_cols = self.identify_column_types(df)
        num_cols, cat_cols, datetime_cols, text_cols = self.detect_feature_types(df)
        # Process numeric columns
        if num_cols:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Only process columns that have missing values
                missing_num_cols = [col for col in num_cols if df[col].isnull().any()]
                if missing_num_cols:
                    df[missing_num_cols] = self.num_imputer.fit_transform(df[missing_num_cols])
        
        # Process categorical columns
        if cat_cols:
            missing_cat_cols = [col for col in cat_cols if df[col].isnull().any()]
            if missing_cat_cols:
                df[missing_cat_cols] = self.cat_imputer.fit_transform(df[missing_cat_cols])
        
        return df
    
    def clean_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply basic text cleaning to text columns."""
        if not self.text_cleaning or not self.text_columns:
            return df
        
        df = df.copy()
        
        for col in self.text_columns:
            if col in df.columns:
                # Convert to string type
                df[col] = df[col].astype(str)
                
                # Basic text cleaning
                # Remove extra whitespace
                df[col] = df[col].str.strip().str.replace(r'\s+', ' ', regex=True)
                
                # Replace common special characters
                df[col] = df[col].str.replace(r'[^\w\s]', ' ', regex=True)
                
                # Convert to lowercase
                df[col] = df[col].str.lower()
        
        return df
        
    def handle_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from datetime columns."""
        if not self.datetime_features or not self.datetime_columns:
            return df
            
        df = df.copy()
        
        for col in self.datetime_columns:
            if col in df.columns:
                # Ensure column is datetime type
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    except:
                        continue
                
                # Extract useful datetime components
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
                df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                df[f'{col}_hour'] = df[col].dt.hour
                df[f'{col}_quarter'] = df[col].dt.quarter
                
                # Flag weekends
                df[f'{col}_is_weekend'] = df[col].dt.dayofweek >= 5
                
                # Month start/end flags
                df[f'{col}_is_month_start'] = df[col].dt.is_month_start
                df[f'{col}_is_month_end'] = df[col].dt.is_month_end
        
        return df
    
    def fix_skewed_features(self, df: pd.DataFrame, threshold=None) -> pd.DataFrame:
        """Apply transformations to fix highly skewed numeric features."""
        if threshold is None:
            threshold = self.max_skewness
            
        df = df.copy()
        num_cols = df.select_dtypes(include=[np.number]).columns
        
        # Skip if no numeric columns
        if len(num_cols) == 0:
            return df
            
        # Calculate skewness for each numeric column
        skewness = df[num_cols].skew()
        self.skewness_stats = skewness.to_dict()
        
        # Process highly skewed columns
        for col in num_cols:
            # Skip columns with NaN skewness (e.g., constant columns)
            if pd.isna(skewness[col]):
                continue
                
            # Apply log transformation to highly positive skewed data
            if skewness[col] > threshold:
                # Make sure all values are positive before log transform
                if df[col].min() <= 0:
                    shift = abs(df[col].min()) + 1  # Add 1 to avoid log(0)
                    df[f'{col}_log'] = np.log(df[col] + shift)
                else:
                    df[f'{col}_log'] = np.log(df[col])
                    
            # Apply exponential transformation to highly negative skewed data
            elif skewness[col] < -threshold:
                df[f'{col}_exp'] = np.exp(df[col] / df[col].max())  # Scale to avoid overflow
        
        return df
            
    def remove_outliers(self, df: pd.DataFrame, target_col=None, threshold=1.5) -> pd.DataFrame:
        """
        Removes outliers using appropriate methods for the task type.
        For supervised tasks, target_col can be provided to preserve its distribution.
        """
        df = df.copy()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if target_col in num_cols:
            num_cols.remove(target_col)  # Remove target column from list
        
        if not num_cols:
            return df  # No numeric columns to process
        
        original_shape = df.shape[0]
        
        if self.task_type == 'supervised':
            # Use IQR method
            for col in num_cols:
                if df[col].nunique() > 10:  # Only process columns with sufficient unique values
                    Q1, Q3 = df[col].quantile([0.25, 0.75])
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        else:
            # Use IsolationForest for multivariate outlier detection
            iso = IsolationForest(contamination=0.05, random_state=42)
            predictions = iso.fit_predict(df[num_cols])
            df = df[predictions == 1]
        
        removed = original_shape - df.shape[0]
        if removed > 0:
            print(f"Removed {removed} outliers ({removed/original_shape:.2%} of data)")
            
        return df
        
    def generate_cleaning_report(self) -> Dict:
        """Generate a report of the cleaning operations."""
        report = {
            "dropped_columns": self.dropped_columns,
            "datetime_columns": self.datetime_columns,
            "text_columns": self.text_columns,
            "high_cardinality_columns": self.high_cardinality_columns,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "missing_values_stats": self.missing_stats,
            "cardinality_stats": self.cardinality_stats,
            "skewness_stats": self.skewness_stats
        }
        return report
    
    def transform(self, df: pd.DataFrame, target_col=None, ordinal_features=None) -> pd.DataFrame:
        """
        Apply the complete data cleaning pipeline.
        
        Parameters:
            df: Input DataFrame
            target_col: Name of the target column for supervised tasks
            ordinal_features: List of ordinal features to preserve as-is
            
        Returns:
            Cleaned DataFrame
        """
        # Type inference if enabled
        if self.detect_data_types:
            df = self.infer_data_types(df)
        
        # Basic cleaning
        df = self.remove_duplicates(df)
        df = self.drop_constant_columns(df)
        df = self.drop_high_missing_columns(df)
        
        # Clean text columns if enabled
        if self.text_cleaning:
            df = self.clean_text_columns(df)
        
        # Handle datetime features if enabled
        if self.datetime_features:
            df = self.handle_datetime_features(df)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Remove outliers
        df = self.remove_outliers(df, target_col='Survived', threshold=5.0)

        
        # Fix skewed features
        df = self.fix_skewed_features(df)
        
        # Drop low variance columns (do this after fixing skewness)
        df = self.drop_low_variance_columns(df)
        
        # Encode categorical features if enabled
        # if self.auto_encode_categoricals:
        #     df = self.encode_categorical_features(df, target_col=target_col)
        print(self.generate_cleaning_report())
        return df