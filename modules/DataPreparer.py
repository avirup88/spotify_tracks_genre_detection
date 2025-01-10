import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN

class DataPreparer:
    
    def __init__(self, dataframe: pd.DataFrame, target: str, features: list = None):
        """
        Initialize the DataPreparer with a pandas DataFrame, target column, and optional feature columns.

        Parameters:
        - dataframe (pd.DataFrame): The dataset to be prepared.
        - target (str): The name of the target column.
        - features (list): Optional list of columns to use as features. If None, all columns except the target are used.
        """
        self.dataframe = dataframe.copy()
        self.target = target
        self.features = features if features else dataframe.drop(columns=[target]).columns

    #Function to handle nulls
    def handle_missing_values(self, strategy='mean', fill_value=None):
        """
        Handle missing values in the dataset.

        Parameters:
        - strategy (str): 'mean', 'median', 'most_frequent', or 'constant'.
        - fill_value: Value to fill for 'constant' strategy.
        """
        imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        self.dataframe[self.features] = imputer.fit_transform(self.dataframe[self.features])
        return self

    #Function to remove row-level duplicates
    def remove_duplicates(self):
        """Remove duplicate rows."""
        self.dataframe = self.dataframe.drop_duplicates()
        return self

    
    #Function to handle outliers
    def handle_outliers(self, clean_outliers=False):
        """
        Handle outliers using the IQR method.

        Parameters:
        - clean_outliers (bool): If True, cap outliers to IQR range.
        """
        numeric_cols = self.dataframe[self.features].select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = self.dataframe[col].quantile(0.25)
            Q3 = self.dataframe[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            if clean_outliers:
                self.dataframe[col] = np.where(self.dataframe[col] < lower_bound, lower_bound,
                                               np.where(self.dataframe[col] > upper_bound, upper_bound, self.dataframe[col]))
        return self

    #Function to encode categorical features
    def encode_categorical(self):
        """Encode categorical features in the specified feature columns."""
        
        # Select categorical columns only from the explicitly defined features
        categorical_cols = [col for col in self.features if self.dataframe[col].dtype in ['object', 'category']]
        
        # One-hot encode categorical columns
        
        if categorical_cols:
            
            self.dataframe = pd.get_dummies(self.dataframe, columns=categorical_cols, drop_first=False)
            
            # Update self.features to reflect the new one-hot encoded columns
            encoded_cols = [col for col in self.dataframe.columns if col.startswith(tuple(categorical_cols))]
            
            self.features = [col for col in self.features if col not in categorical_cols] + encoded_cols
            
        return self

    #Function to split data
    def split_data(self, test_size=0.2, random_state=42, scale_method=None):
        """
        Split the dataset into training and test sets, with optional scaling.
    
        Parameters:
        - test_size (float): Proportion of the dataset to include in the test split.
        - random_state (int): Random seed for reproducibility.
        - scale_method (str): 'standard' for StandardScaler, 'minmax' for MinMaxScaler, or None.
    
        Returns:
        - X_train, X_test, y_train, y_test: Split and optionally scaled data.
        - label_encoder: The LabelEncoder object for encoding/decoding target values (if used).
        - scaler: The scaler object used for scaling (if applied).
        """
        # Select the non-feature columns
        non_feature_cols = self.dataframe.drop(columns=self.features + [self.target], errors='ignore')

    
        # Separate features into numeric and non-numeric
        numeric_features = self.dataframe[self.features].select_dtypes(include=['number'])
        non_numeric_features = self.dataframe[self.features].select_dtypes(exclude=['number'])
    
        X_numeric = numeric_features
        X_non_numeric = non_numeric_features
        y = self.dataframe[self.target]
    
        # Check if label encoding is needed
        if y.dtype == 'object' or y.nunique() > 2:
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
        else:
            y_encoded = y
            label_encoder = None  # No encoding needed
    
        # Split the data
        if not non_feature_cols.empty:
            X_train_num, X_test_num, y_train, y_test, nf_train, nf_test = train_test_split(
                X_numeric, y_encoded, non_feature_cols, test_size=test_size, random_state=random_state, stratify=y_encoded
            )
            X_train_non_num, X_test_non_num = train_test_split(
                X_non_numeric, test_size=test_size, random_state=random_state, stratify=y_encoded
            )
        else:
            X_train_num, X_test_num, y_train, y_test = train_test_split(
                X_numeric, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
            )
            X_train_non_num, X_test_non_num = train_test_split(
                X_non_numeric, test_size=test_size, random_state=random_state, stratify=y_encoded
            )
            nf_train, nf_test = pd.DataFrame(), pd.DataFrame()  # Empty DataFrame for consistency
    
        # Apply scaling to numeric columns if specified
        scaler = None
        if scale_method == 'standard':
            scaler = StandardScaler()
        elif scale_method == 'minmax':
            scaler = MinMaxScaler()
    
        if scaler:
            X_train_num = scaler.fit_transform(X_train_num)  # Fit and transform on training data
            X_test_num = scaler.transform(X_test_num)       # Transform only on test data
    
       # Combine numeric and non-numeric features back
        X_train = pd.concat([
            pd.DataFrame(X_train_num, columns=numeric_features.columns, index=X_train_num if isinstance(X_train_num, pd.DataFrame) else None),
            X_train_non_num.reset_index(drop=True)
        ], axis=1)
    
        X_test = pd.concat([
            pd.DataFrame(X_test_num, columns=numeric_features.columns, index=X_test_num if isinstance(X_test_num, pd.DataFrame) else None),
            X_test_non_num.reset_index(drop=True)
        ], axis=1)
    
        return X_train, X_test, y_train, y_test, nf_train, nf_test, label_encoder, scaler

    #Function to handle class imbalance for the training samples
    def handle_class_imbalance(self, X_train, y_train, mode='over'):
            """
            Function to handle class imbalance
            X_train : Training Features
            y_train: Training Labels
            mode: 'over' performs oversampling of the minority class
                  'under' performs undersampling of the majority class
                  'adasyn' adaptive synthetic sampling technique
                  'combo' performs SMOTE technique for oversampling while performing Tomek Link Removal which is undersampling
            """
            #Oversampling
            if mode == 'over':
                
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
            #Undersampling
            elif mode == 'under':
                
                tomek = TomekLinks(sampling_strategy = 'not minority')
                X_resampled, y_resampled = tomek.fit_resample(X_train, y_train)

            #ADASYN technique
            elif mode == 'adasyn':
                
                adasyn = ADASYN(sampling_strategy='minority', random_state=42)
                X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)

            #Combination
            else:
            
                smote_tomek = SMOTETomek(random_state=42)
                X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
        
            return X_resampled, y_resampled

    #Function to get back the original data frame from the scaled and encoded labels with the predicted labels
    def revert_to_original(self, X_scaled, y_encoded,y_pred_encoded, nf_columns, scaler, label_encoder):
            """
            Revert scaled and encoded data, including non-feature columns, back to the original format.
    
            Parameters:
            - X_scaled: Scaled feature data.
            - y_encoded: Encoded target data.
            - nf_columns: Original ID columns corresponding to the data.
    
            Returns:
            - original_data: DataFrame with original features, target, and ID columns.
            """
        
            # Revert scaling
            if scaler is not None:
                X_original = scaler.inverse_transform(X_scaled[scaler.feature_names_in_])
                X_cat_features = X_scaled[list(set(self.features).difference(set(scaler.feature_names_in_)))]
                original_data = pd.concat([pd.DataFrame(X_original,columns = scaler.feature_names_in_), X_cat_features],axis=1)
            else:
                original_data = X_scaled  # No scaling applied
    
            # Revert label encoding
            if label_encoder is not None:
                y_original = label_encoder.inverse_transform(y_encoded)
                y_predict = label_encoder.inverse_transform(y_pred_encoded)
            else:
                y_original = y_encoded  # No encoding applied
                y_predict = y_pred_encoded  # No encoding applied

            # Combine features, target, and ID columns into a DataFrame
            #original_data = pd.DataFrame(X_original, columns=self.features)
            original_data['original_{}'.format(self.target)] = y_original
            original_data['predicted_{}'.format(self.target)] = y_predict
    
            # Add back ID columns (align indices)
            nf_columns.reset_index(drop=True, inplace=True)
            original_data.reset_index(drop=True, inplace=True)
            original_data = pd.concat([nf_columns, original_data], axis=1)
    
            return original_data

    #Function to return the pre-processed data frame
    def get_prepared_data(self):
        """Return the prepared DataFrame."""
        return self.dataframe
