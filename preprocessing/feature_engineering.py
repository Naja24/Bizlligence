import pandas as pd
import category_encoders as ce  # For Target Encoder
import numpy as np
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder, StandardScaler, MinMaxScaler, OneHotEncoder, PolynomialFeatures, Normalizer
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_regression, mutual_info_classif, f_regression, f_classif
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import Optional, List, Tuple, Dict
import warnings

class FeatureEngineer:
    def __init__(self, 
                 task_type='supervised',
                 task_subtype='classification',
                 numeric_features: Optional[List[str]] = None, 
                 categorical_features: Optional[List[str]] = None, 
                 datetime_features: Optional[List[str]] = None,
                 text_features: Optional[List[str]] = None,
                 scale_method='standard', 
                 normalize=False, 
                 use_poly=False, 
                 poly_degree=2,
                 create_interactions=False,
                 handle_skewness=True,
                 feature_selection=False,
                 k_best_features=10,
                 use_pca=False,
                 pca_components=0.95,
                 create_clusters=False,
                 n_clusters=5,
                 binning=False,
                 n_bins=5,
                 discrete_num_col = None,
                 target_encoding=False,
                 auto_encode_categoricals=True,
                 drop_original_categorical: bool = False,  # Preserve original categorical columns
                 create_domain_features=True,
                 max_categories_for_onehot=10,
                 # New parameters for outlier removal:
                 outlier_removal: bool = True,
                 outlier_threshold: float = 5.0,
                 max_outlier_removal_ratio: float = 0.03,
                 # New parameter: minimum variance for scaling
                 min_variance: float = 0.01):
        """
        Initialize FeatureEngineer with comprehensive feature engineering capabilities.
        """
        self.task_type = task_type.lower()
        self.task_subtype = task_subtype.lower()
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.datetime_features = datetime_features
        self.text_features = text_features
        self.scale_method = scale_method.lower()
        self.normalize = normalize
        self.use_poly = use_poly
        self.poly_degree = poly_degree
        self.create_interactions = create_interactions
        self.handle_skewness = handle_skewness
        self.feature_selection = feature_selection
        self.k_best_features = k_best_features
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.create_clusters = create_clusters
        self.n_clusters = n_clusters
        self.binning = binning
        self.n_bins = n_bins
        self.discrete_num_col = discrete_num_col
        self.target_encoding = target_encoding
        self.create_domain_features = create_domain_features
        self.auto_encode_categoricals = auto_encode_categoricals
        self.drop_original_categorical = drop_original_categorical
        self.max_categories_for_onehot = max_categories_for_onehot
        
        self.outlier_removal = outlier_removal
        self.outlier_threshold = outlier_threshold
        self.max_outlier_removal_ratio = max_outlier_removal_ratio
        self.min_variance = min_variance
        
        self.pipeline = None
        self.cluster_models = {}
        self.feature_names = None
        self.feature_importances = None
        self.encoders = {}
        self.high_cardinality_columns = []
        
    def detect_feature_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str]]:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        for col in df.columns:
            if col in datetime_cols:
                continue
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                if any(keyword in col.lower() for keyword in ['date', 'time', 'day', 'month', 'year']):
                    try:
                        pd.to_datetime(df[col], errors='raise')
                        datetime_cols.append(col)
                        if col in categorical_cols:
                            categorical_cols.remove(col)
                    except:
                        pass
        text_cols = []
        for col in categorical_cols.copy():
            if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                if df[col].dropna().astype(str).str.len().mean() > 50:
                    text_cols.append(col)
                    categorical_cols.remove(col)
        return numeric_cols, categorical_cols, datetime_cols, text_cols
    
    def create_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.datetime_features:
            return df
        df_new = df.copy()
        datetime_features = []
        for col in self.datetime_features:
            if col in df_new.columns:
                if not pd.api.types.is_datetime64_any_dtype(df_new[col]):
                    try:
                        df_new[col] = pd.to_datetime(df_new[col], errors='coerce')
                    except:
                        continue
                df_new[f'{col}_year'] = df_new[col].dt.year
                df_new[f'{col}_month'] = df_new[col].dt.month
                df_new[f'{col}_day'] = df_new[col].dt.day
                df_new[f'{col}_dayofweek'] = df_new[col].dt.dayofweek
                df_new[f'{col}_hour'] = df_new[col].dt.hour
                df_new[f'{col}_quarter'] = df_new[col].dt.quarter
                df_new[f'{col}_sin_month'] = np.sin(2 * np.pi * df_new[col].dt.month / 12)
                df_new[f'{col}_cos_month'] = np.cos(2 * np.pi * df_new[col].dt.month / 12)
                df_new[f'{col}_sin_day'] = np.sin(2 * np.pi * df_new[col].dt.day / 31)
                df_new[f'{col}_cos_day'] = np.cos(2 * np.pi * df_new[col].dt.day / 31)
                df_new[f'{col}_sin_hour'] = np.sin(2 * np.pi * df_new[col].dt.hour / 24)
                df_new[f'{col}_cos_hour'] = np.cos(2 * np.pi * df_new[col].dt.hour / 24)
                df_new[f'{col}_is_weekend'] = df_new[col].dt.dayofweek >= 5
                df_new[f'{col}_is_month_start'] = df_new[col].dt.is_month_start
                df_new[f'{col}_is_month_end'] = df_new[col].dt.is_month_end
                df_new[f'{col}_is_quarter_start'] = df_new[col].dt.is_quarter_start
                df_new[f'{col}_is_quarter_end'] = df_new[col].dt.is_quarter_end
                datetime_features.extend([
                    f'{col}_year', f'{col}_month', f'{col}_day', f'{col}_dayofweek',
                    f'{col}_hour', f'{col}_quarter', f'{col}_sin_month', f'{col}_cos_month',
                    f'{col}_sin_day', f'{col}_cos_day', f'{col}_sin_hour', f'{col}_cos_hour',
                    f'{col}_is_weekend', f'{col}_is_month_start', f'{col}_is_month_end',
                    f'{col}_is_quarter_start', f'{col}_is_quarter_end'
                ])
        if self.numeric_features is not None:
            self.numeric_features.extend([f for f in datetime_features if f not in self.numeric_features and f in df_new.columns])
        return df_new
    
    def process_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.text_features:
            return df
        df_new = df.copy()
        for col in self.text_features:
            if col in df_new.columns:
                df_new[col] = df_new[col].fillna('').astype(str)
                df_new[f'{col}_char_count'] = df_new[col].str.len()
                df_new[f'{col}_word_count'] = df_new[col].str.split().str.len()
                df_new[f'{col}_avg_word_length'] = df_new[col].apply(lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0)
                df_new[f'{col}_uppercase_count'] = df_new[col].apply(lambda x: sum(1 for c in x if c.isupper()))
                df_new[f'{col}_special_char_count'] = df_new[col].apply(lambda x: sum(1 for c in x if not c.isalnum() and not c.isspace()))
                if self.numeric_features is not None:
                    self.numeric_features.extend([
                        f'{col}_char_count', f'{col}_word_count', f'{col}_avg_word_length',
                        f'{col}_uppercase_count', f'{col}_special_char_count'
                    ])
        return df_new
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.create_interactions or not self.numeric_features or len(self.numeric_features) < 2:
            return df
        df_new = df.copy()
        interaction_features = []
        max_combos = min(100, len(self.numeric_features) * (len(self.numeric_features) - 1) // 2)
        combos = 0
        sorted_features = self.numeric_features.copy()
        if hasattr(self, 'feature_importances') and self.feature_importances is not None:
            sorted_features = [f for f, _ in sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True) if f in self.numeric_features]
        for i, feat1 in enumerate(sorted_features):
            if feat1 not in df_new.columns:
                continue
            for feat2 in sorted_features[i+1:]:
                if feat2 not in df_new.columns:
                    continue
                df_new[f'{feat1}_plus_{feat2}'] = df_new[feat1] + df_new[feat2]
                interaction_features.append(f'{feat1}_plus_{feat2}')
                df_new[f'{feat1}_minus_{feat2}'] = df_new[feat1] - df_new[feat2]
                interaction_features.append(f'{feat1}_minus_{feat2}')
                df_new[f'{feat1}_mult_{feat2}'] = df_new[feat1] * df_new[feat2]
                interaction_features.append(f'{feat1}_mult_{feat2}')
                df_new[f'{feat1}_div_{feat2}'] = df_new[feat1] / (df_new[feat2] + 1e-10)
                interaction_features.append(f'{feat1}_div_{feat2}')
                combos += 4
                if combos >= max_combos:
                    break
            if combos >= max_combos:
                break
        if self.numeric_features is not None:
            self.numeric_features.extend(interaction_features)
        return df_new
    
    def create_cluster_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.create_clusters or not self.numeric_features or len(self.numeric_features) < 2:
            return df
        df_new = df.copy()
        numeric_data = df_new[self.numeric_features].copy()
        for col in numeric_data.columns:
            if numeric_data[col].isnull().any():
                numeric_data[col] = numeric_data[col].fillna(numeric_data[col].median())
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)
        self.cluster_models['kmeans'] = kmeans
        df_new['cluster_label'] = cluster_labels
        centers = kmeans.cluster_centers_
        for i in range(self.n_clusters):
            distances = np.linalg.norm(scaled_data - centers[i], axis=1)
            df_new[f'cluster_{i}_distance'] = distances
        if self.numeric_features is not None:
            self.numeric_features.append('cluster_label')
            self.numeric_features.extend([f'cluster_{i}_distance' for i in range(self.n_clusters)])
        return df_new
    
    def bin_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.binning or not self.numeric_features:
            return df
        df_new = df.copy()
        binned_features = []
        for col in self.numeric_features:
            if col in df_new.columns and df_new[col].nunique() > self.n_bins:
                try:
                    df_new[f'{col}_bin'] = pd.qcut(df_new[col], q=self.n_bins, labels=False, duplicates='drop').astype('float')
                    binned_features.append(f'{col}_bin')
                except:
                    continue
        if self.numeric_features is not None:
            self.numeric_features.extend(binned_features)
        return df_new
    
    def calculate_feature_importances(self, df: pd.DataFrame, target_col: str) -> Dict[str, float]:
        """
        Calculate feature importances, ensuring unique feature names and proper aggregation
        of duplicate importance values.
        
        Parameters:
        -----------
        df : pd.DataFrame
            The input dataframe with features
        target_col : str
            The name of the target column
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of feature names and their importance scores, sorted by importance
        """
        if self.task_type != 'supervised' or target_col not in df.columns:
            return {}
            
        # Create a clean DataFrame for importance calculation
        df_clean = df.copy()
        
        # Handle missing values
        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                else:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        
        # Select appropriate scoring function based on task type
        score_func = mutual_info_classif if self.task_subtype == 'classification' else mutual_info_regression
        
        # Get columns for importance calculation (exclude target)
        numeric_cols = [col for col in df_clean.columns if col != target_col and pd.api.types.is_numeric_dtype(df_clean[col])]
        
        # Skip calculation if no appropriate features
        if not numeric_cols:
            return {}
            
        # Extract features and target
        X = df_clean[numeric_cols].copy()
        y = df_clean[target_col]
        
        # Calculate feature importances
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            importances = score_func(X, y, random_state=42)
        
        # Create a dictionary with unique feature names
        importance_dict = {}
        
        # Process each feature, handling duplicates properly
        for col, imp in zip(numeric_cols, importances):
            # Extract base feature name (in case of transformations)
            base_name = col.split('_')[0] if '_' in col else col
            
            # Handle duplicate feature names by aggregating importances
            if base_name in importance_dict:
                # For duplicate base features, keep track of both
                # original feature and its variants, retaining the highest importance
                if col in importance_dict:
                    importance_dict[col] = max(importance_dict[col], imp)
                else:
                    importance_dict[col] = imp
            else:
                importance_dict[col] = imp
        
        # Sort by importance (descending)
        self.feature_importances = {k: v for k, v in sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)}
        
        # Store proper feature names to avoid confusion in later stages
        self.important_feature_names = list(self.feature_importances.keys())
        
        return self.feature_importances
    
    def create_domain_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.create_domain_features:
            return df
        df_new = df.copy()
        domain_features = []
        if self.numeric_features and len(self.numeric_features) >= 2:
            col_patterns = {}
            for col in self.numeric_features:
                if col not in df_new.columns:
                    continue
                base_name = col.split('_')[0] if '_' in col else col
                col_patterns.setdefault(base_name, []).append(col)
            for base_name, cols in col_patterns.items():
                if len(cols) >= 2:
                    for i, col1 in enumerate(cols):
                        if col1 not in df_new.columns:
                            continue
                        for col2 in cols[i+1:]:
                            if col2 not in df_new.columns:
                                continue
                            ratio_name = f'{col1}_to_{col2}_ratio'
                            df_new[ratio_name] = df_new[col1] / (df_new[col2] + 1e-10)
                            domain_features.append(ratio_name)
        if self.numeric_features is not None:
            self.numeric_features.extend(domain_features)
        return df_new
    
    def fix_skewed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.handle_skewness or not self.numeric_features:
            return df
        df_new = df.copy()
        transformed_features = []
        for col in self.numeric_features:
            if col in df_new.columns and df_new[col].nunique() > 5:
                if not pd.api.types.is_numeric_dtype(df_new[col]) or df_new[col].isnull().any():
                    continue
                try:
                    skew_value = df_new[col].skew()
                    if abs(skew_value) > 0.5:
                        if skew_value > 0:
                            min_val = df_new[col].min()
                            if min_val > 0:
                                df_new[f'{col}_log'] = np.log1p(df_new[col])
                            else:
                                shift = abs(min_val) + 1
                                df_new[f'{col}_log'] = np.log1p(df_new[col] + shift)
                            transformed_features.append(f'{col}_log')
                            df_new[f'{col}_sqrt'] = np.sqrt(df_new[col] - min_val + 0.01)
                            transformed_features.append(f'{col}_sqrt')
                        else:
                            df_new[f'{col}_squared'] = np.square(df_new[col])
                            transformed_features.append(f'{col}_squared')
                            max_val = df_new[col].max()
                            if max_val > 0 and df_new[col].min() >= 0:
                                df_new[f'{col}_reciprocal'] = 1 / (df_new[col] + 0.01)
                                transformed_features.append(f'{col}_reciprocal')
                except:
                    continue
        if self.numeric_features is not None:
            self.numeric_features.extend(transformed_features)
        return df_new
    
    def identify_ordinal_features(self, df: pd.DataFrame) -> List[str]:
        potential_ordinal_features = []
        ordinal_keywords = ['level', 'grade', 'stage', 'tier', 'rank', 'rating', 'score', 'class',
                            'priority', 'severity', 'status', 'phase', 'step', 'size', 'satisfaction',
                            'education', 'degree', 'frequency', 'intensity', 'experience', 'seniority']
        for col in self.categorical_features or []:
            if col not in df.columns:
                continue
            if any(keyword in col.lower() for keyword in ordinal_keywords):
                potential_ordinal_features.append(col)
                continue
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                values = df[col].dropna().unique()
                if len(values) > 10:
                    continue
                if all(str(v).replace('level', '').replace('grade', '').replace('tier', '').strip().isdigit() for v in values if str(v).strip()):
                    potential_ordinal_features.append(col)
                    continue
                common_ordered_sets = [
                    ['low', 'medium', 'high'],
                    ['small', 'medium', 'large'],
                    ['basic', 'intermediate', 'advanced'],
                    ['beginner', 'intermediate', 'expert'],
                    ['poor', 'fair', 'good', 'excellent'],
                    ['never', 'rarely', 'sometimes', 'often', 'always'],
                    ['strongly disagree', 'disagree', 'neutral', 'agree', 'strongly agree']
                ]
                values_lower = [str(v).lower() for v in values if str(v).strip()]
                for ordered_set in common_ordered_sets:
                    if all(val in ordered_set for val in values_lower) and len(values_lower) > 1:
                        potential_ordinal_features.append(col)
                        break
        return potential_ordinal_features

    # def encode_categorical_features(self, df: pd.DataFrame, target_col=None) -> pd.DataFrame:
    #     """
    #     Enhanced categorical feature encoding:
    #     - For numerical discrete features (low unique count relative to dataset size): treat as categorical and one-hot encode
    #     - For categorical ordinal features: apply ordinal encoding
    #     - For categorical nominal with low cardinality: apply one-hot encoding
    #     - For categorical nominal with high cardinality: apply target encoding
    #     - For target column: apply appropriate encoding based on task type
    #     """

    #     # For Numeric columsn with discrete entries
    #     discrete_threshold_pct = 0.05
    #     discrete_threshold = max(10 , int(len(df)*discrete_threshold_pct))
        
    #     # Detect different feature types
    #     num_cols, cat_cols, date_cols, text_cols = self.detect_feature_types(df)
        
    #     # Store detected categorical features
    #     self.categorical_features = cat_cols
        
    #     if not self.auto_encode_categoricals or not self.categorical_features:
    #         return df
    #     df_new = df.copy()
    #     encoded_df = pd.DataFrame(index=df_new.index)
    #     ordinal_features = self.identify_ordinal_features(df_new)

    #     # self.discrete_num_col = []

    #     # for col in num_cols:
    #     #     if col == target_col:
    #     #         continue
    #     #     if df_new[col].nunique < discrete_threshold:

    #     # For the target column, if classification, use label encoding.
    #     if target_col is not None and target_col in self.categorical_features and target_col in df_new.columns:
    #         le = LabelEncoder()
    #         df_new[target_col] = df_new[target_col].fillna('Unknown')
    #         df_new[target_col] = le.fit_transform(df_new[target_col].astype(str))
    #         self.encoders[f'target_{target_col}'] = le
        
    #     encoded_cat_df = pd.DataFrame(index=df_new.index)
    #     for col in self.categorical_features:
    #         if col not in df_new.columns or col == target_col:
    #             continue
    #         df_new[col] = df_new[col].fillna('Unknown')
    #         n_unique = df_new[col].nunique()
    #         if n_unique < 10:
    #             # One-Hot Encoding
    #             one_hot = pd.get_dummies(df_new[col], prefix=col, drop_first=True).astype(int)
    #             encoded_cat_df = pd.concat([encoded_cat_df, one_hot], axis=1)
    #         else:
    #             # Otherwise, label encode
    #             le = LabelEncoder()
    #             encoded = le.fit_transform(df_new[col].astype(str))
    #             encoded_cat_df[col] = encoded
    #             self.encoders[f'label_{col}'] = le
    #     return encoded_cat_df

    def encode_categorical_features(self, df: pd.DataFrame, target_col=None, task_type=None) -> pd.DataFrame:
        """
        Enhanced categorical feature encoding with fixed handling of numeric columns
        """
        # Set discrete threshold as percentage of dataset length
        if not hasattr(self, 'discrete_threshold_pct'):
            self.discrete_threshold_pct = 0.005  # Default: 0.5% of dataset length
        
        # Calculate discrete threshold based on dataset size
        discrete_threshold = max(10, int(len(df) * self.discrete_threshold_pct))
        
        # Detect different feature types
        num_cols, cat_cols, date_cols, text_cols = self.detect_feature_types(df)
        
        # Store detected categorical features
        self.categorical_features = cat_cols
        
        if not self.auto_encode_categoricals:
            return df
        
        df_new = df.copy()
        encoded_df = pd.DataFrame(index=df_new.index)
        
        # Initialize encoders dictionary if it doesn't exist
        if not hasattr(self, 'encoders'):
            self.encoders = {}
        
        # Track processed columns to avoid duplicates
        processed_columns = set()
        
        # Handle discrete numerical features (treat as categorical if low cardinality)
        self.discrete_num_cols = []
        for col in num_cols:
            if col == target_col:
                continue
            if df_new[col].nunique() < discrete_threshold:
                self.discrete_num_cols.append(col)
                # Treat as categorical and apply one-hot encoding
                df_new[col] = df_new[col].fillna(df_new[col].median())  # Fill NA with median for numerical
                one_hot = pd.get_dummies(df_new[col], prefix=col, drop_first=True).astype(int)
                encoded_df = pd.concat([encoded_df, one_hot], axis=1)
                processed_columns.add(col)  # Mark as processed
            else:
                # Keep continuous numerical features as is
                encoded_df[col] = df_new[col]
                processed_columns.add(col)  # Mark as processed
        
        # Handle target column separately
        if target_col is not None and target_col in df_new.columns:
            if task_type == 'classification':
                if target_col in cat_cols:
                    # Check if target is ordinal
                    if target_col in self.identify_ordinal_features(df_new):
                        # Use ordinal encoding for ordinal target
                        from sklearn.preprocessing import OrdinalEncoder
                        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                        # Use predefined mappings if available
                        if hasattr(self, 'ordinal_mappings') and target_col in self.ordinal_mappings:
                            categories = [self.ordinal_mappings[target_col]]
                            oe = OrdinalEncoder(categories=categories, handle_unknown='use_encoded_value', unknown_value=-1)
                        
                        df_new[target_col] = df_new[target_col].fillna('Unknown')
                        encoded = oe.fit_transform(df_new[[target_col]])
                        encoded_df[target_col] = encoded
                        self.encoders[f'target_ordinal_{target_col}'] = oe
                    else:
                        # Use label encoding for non-ordinal categorical target
                        le = LabelEncoder()
                        df_new[target_col] = df_new[target_col].fillna('Unknown')
                        df_new[target_col] = le.fit_transform(df_new[target_col].astype(str))
                        self.encoders[f'target_{target_col}'] = le
                        encoded_df[target_col] = df_new[target_col]
                else:
                    # For numerical target in classification, keep as is
                    encoded_df[target_col] = df_new[target_col]
            else:
                # Regression or other task
                encoded_df[target_col] = df_new[target_col]
            processed_columns.add(target_col)  # Mark target as processed
        
        # Identify ordinal categorical features
        ordinal_features = self.identify_ordinal_features(df_new)
        
        # Process categorical features
        for col in cat_cols:
            if col not in df_new.columns or col == target_col or col in processed_columns:
                continue
                
            df_new[col] = df_new[col].fillna('Unknown')
            n_unique = df_new[col].nunique()
            
            if col in ordinal_features:
                # Ordinal encoding for ordinal features
                from sklearn.preprocessing import OrdinalEncoder
                oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                # Use predefined ordinal_mappings if available for this column
                if hasattr(self, 'ordinal_mappings') and col in self.ordinal_mappings:
                    categories = [self.ordinal_mappings[col]]
                    oe = OrdinalEncoder(categories=categories, handle_unknown='use_encoded_value', unknown_value=-1)
                
                encoded = oe.fit_transform(df_new[[col]])
                encoded_df[col] = encoded
                self.encoders[f'ordinal_{col}'] = oe
                processed_columns.add(col)  # Mark as processed
                
            elif n_unique < 10:  # Low cardinality nominal features
                # One-Hot Encoding
                one_hot = pd.get_dummies(df_new[col], prefix=col, drop_first=True).astype(int)
                encoded_df = pd.concat([encoded_df, one_hot], axis=1)
                processed_columns.add(col)  # Mark as processed
                
            else:  # High cardinality nominal features
                # Target encoding
                if target_col is not None and task_type == 'classification' and target_col in df_new.columns:
                    # Simple target mean encoding with smoothing
                    target_means = df_new.groupby(col)[target_col].mean()
                    global_mean = df_new[target_col].mean()
                    
                    # Calculate adaptive smoothing factor based on dataset size
                    smoothing_factor = min(20, max(5, int(len(df) * 0.01)))  # Between 5 and 20
                    
                    # Apply smoothing: smaller groups are more affected by global mean
                    counts = df_new[col].value_counts()
                    smoothed_means = (target_means * counts + global_mean * smoothing_factor) / (counts + smoothing_factor)
                    
                    encoded_df[f'{col}_target_enc'] = df_new[col].map(smoothed_means).fillna(global_mean)
                    self.encoders[f'target_enc_{col}'] = {
                        'mapping': smoothed_means,
                        'default': global_mean
                    }
                else:
                    # Fallback to label encoding if target encoding not applicable
                    le = LabelEncoder()
                    encoded = le.fit_transform(df_new[col].astype(str))
                    encoded_df[col] = encoded
                    self.encoders[f'label_{col}'] = le
                processed_columns.add(col)  # Mark as processed
        
        # Add remaining columns like date and text if needed (that haven't been processed yet)
        for col in date_cols + text_cols:
            if col in df_new.columns and col not in encoded_df.columns and col not in processed_columns and col != target_col:
                # Date and text features would typically need special handling,
                # but for now just pass them through
                encoded_df[col] = df_new[col]
                processed_columns.add(col)  # Mark as processed
        
        return encoded_df

    def build_pipeline(self, df: pd.DataFrame) -> ColumnTransformer:
        """
        Build a pipeline for numeric features.
        Numeric columns with variance below min_variance are passed through without scaling.
        """
        # If not set, auto-detect features
        if self.numeric_features is None or self.categorical_features is None:
            detected_num, detected_cat, detected_dt, detected_text = self.detect_feature_types(df)
            if self.numeric_features is None:
                self.numeric_features = detected_num
            if self.categorical_features is None:
                self.categorical_features = detected_cat
            if self.datetime_features is None:
                self.datetime_features = detected_dt
            if self.text_features is None:
                self.text_features = detected_text
        
        # Split numeric features based on variance threshold
        high_var = []
        low_var = []
        for col in self.numeric_features:
            if col in df.columns:
                if df[col].var() < self.min_variance:
                    low_var.append(col)
                else:
                    high_var.append(col)
        
        # Build pipeline for high variance numeric columns
        if self.scale_method == 'minmax':
            scaler = MinMaxScaler()
        elif self.scale_method == 'robust':
            scaler = RobustScaler()
        elif self.scale_method == 'power':
            scaler = PowerTransformer(method='yeo-johnson')
        elif self.scale_method == 'quantile':
            scaler = QuantileTransformer(output_distribution='normal')
        else:
            scaler = StandardScaler()
        
        num_steps = [('scaler', scaler)]
        if self.normalize:
            num_steps.append(('normalizer', Normalizer()))
        if self.use_poly:
            num_steps.append(('poly', PolynomialFeatures(degree=self.poly_degree, include_bias=False, interaction_only=True)))
        if self.feature_selection and self.task_type == 'supervised':
            selector = SelectKBest(score_func=f_classif if self.task_subtype=='classification' else f_regression, k=self.k_best_features)
            num_steps.append(('selector', selector))
        if self.use_pca and len(high_var) > 1:
            if isinstance(self.pca_components, float) and 0 < self.pca_components < 1:
                num_steps.append(('pca', PCA(n_components=self.pca_components)))
            elif isinstance(self.pca_components, int) and self.pca_components > 0:
                n_components = min(self.pca_components, len(high_var))
                num_steps.append(('pca', PCA(n_components=n_components)))
        high_var_pipeline = Pipeline(steps=num_steps)
        
        # Create transformers for numeric data:
        transformers = []
        if high_var:
            transformers.append(('num', high_var_pipeline, high_var))
        if low_var:
            # For low variance, pass through without scaling.
            transformers.append(('low_var', 'passthrough', low_var))
        # If auto_encode_categoricals is False, add a transformer for categoricals
        if not self.auto_encode_categoricals and self.categorical_features:
            unencoded_cat = [col for col in self.categorical_features if col in df.columns and col not in self.numeric_features]
            if unencoded_cat:
                cat_steps = [('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]
                cat_pipeline = Pipeline(steps=cat_steps)
                transformers.append(('cat', cat_pipeline, unencoded_cat))
        preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
        return preprocessor

    def transform(self, df: pd.DataFrame, target_col=None) -> pd.DataFrame:
        """
        Modified transform method to avoid feature duplication
        """
        df_transformed = df.copy()
        
        # Step 1-7: Apply all feature transformations (unchanged)
        if self.datetime_features:
            df_transformed = self.create_datetime_features(df_transformed)
        if self.text_features:
            df_transformed = self.process_text_features(df_transformed)
        if self.create_domain_features:
            df_transformed = self.create_domain_specific_features(df_transformed)
        if self.handle_skewness:
            df_transformed = self.fix_skewed_features(df_transformed)
        if self.create_interactions:
            df_transformed = self.create_interaction_features(df_transformed)
        if self.create_clusters:
            df_transformed = self.create_cluster_features(df_transformed)
        if self.binning:
            df_transformed = self.bin_numeric_features(df_transformed)
        
        # Track which features will be handled by pipeline vs. categorical encoding
        pipeline_features = set()
        if self.numeric_features:
            pipeline_features.update(self.numeric_features)
        
        # Step 8: Build and apply numeric pipeline
        pipeline = self.build_pipeline(df_transformed)
        self.pipeline = pipeline
        
        # Extract target if present
        target_series = None
        if target_col is not None and target_col in df_transformed.columns:
            target_series = df_transformed[target_col].copy()
        
        # Apply pipeline transformation
        X_numeric = pipeline.fit_transform(df_transformed)
        
        # Get accurate feature names from pipeline
        feature_names = []
        if hasattr(pipeline, 'get_feature_names_out'):
            try:
                # This works for newer scikit-learn versions
                feature_names = pipeline.get_feature_names_out().tolist()
            except:
                # Fallback to manual feature name construction
                if 'num' in pipeline.named_transformers_:
                    num_trans = pipeline.named_transformers_['num']
                    valid_numeric = [col for col in self.numeric_features if col in df_transformed.columns]
                    if 'pca' in num_trans.named_steps:
                        pca_step = num_trans.named_steps['pca']
                        feature_names.extend([f'PC{i+1}' for i in range(pca_step.n_components_)])
                    elif 'poly' in num_trans.named_steps:
                        poly_step = num_trans.named_steps['poly']
                        feature_names.extend(poly_step.get_feature_names_out(valid_numeric).tolist())
                    else:
                        feature_names.extend(valid_numeric)
                if 'low_var' in pipeline.named_transformers_:
                    low_var_cols = pipeline.transformers_[1][2]
                    feature_names.extend(low_var_cols)
        
        # If feature names extraction failed, use generic names
        if len(feature_names) != X_numeric.shape[1]:
            feature_names = [f'feature_{i}' for i in range(X_numeric.shape[1])]
        
        # Create DataFrame with pipeline results
        result_df = pd.DataFrame(X_numeric, columns=feature_names, index=df_transformed.index)
        
        # Step 9: Handle categorical features separately to avoid duplication
        if self.auto_encode_categoricals and self.categorical_features:
            # Get list of categorical features that weren't processed by pipeline
            cat_features_to_encode = [
                col for col in self.categorical_features 
                if col in df_transformed.columns and col not in pipeline_features
            ]
            
            # Only encode categorical features that weren't already handled
            if cat_features_to_encode:
                # Create a dataframe with only categorical columns that need encoding
                cat_df = df_transformed[cat_features_to_encode].copy()
                if target_col:
                    cat_df[target_col] = df_transformed[target_col].copy()
                    
                # Apply categorical encoding
                encoded_cat_df = self.encode_categorical_features(cat_df, target_col)
                
                # Remove columns that might be duplicated across dataframes
                for col in result_df.columns:
                    if col in encoded_cat_df.columns:
                        encoded_cat_df = encoded_cat_df.drop(columns=[col])
                
                # Only concat if there are new columns to add
                if not encoded_cat_df.empty:
                    # Ensure indices align
                    result_df = pd.concat([
                        result_df,
                        encoded_cat_df
                    ], axis=1)
        
        # Add target column back if it was present
        if target_series is not None:
            result_df[target_col] = target_series.values
        
        # Store final feature names to ensure consistent naming
        self.feature_names = result_df.columns.tolist()
        
        return result_df