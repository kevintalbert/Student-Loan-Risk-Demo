"""
Data Preprocessing Pipeline for Student Loan Risk Modeling

This module handles data cleaning, feature engineering, and preprocessing
for machine learning models that predict student loan delinquency risk.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class StudentLoanPreprocessor:
    """Preprocessing pipeline for student loan data."""
    
    def __init__(self):
        """Initialize the preprocessor with fitted transformers."""
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_names = []
        self.target_column = 'is_delinquent'
        
        # Define feature groups
        self.numerical_features = [
            'age', 'credit_score_at_origination', 'annual_income', 'dependents',
            'gpa', 'graduation_year', 'total_loan_amount', 'loan_count',
            'avg_loan_amount', 'avg_interest_rate', 'total_current_balance',
            'total_monthly_payment', 'debt_to_income_ratio', 'avg_days_late',
            'max_days_late', 'std_days_late', 'total_payments_made',
            'total_payment_count', 'total_scheduled', 'missed_payment_count',
            'payment_ratio', 'missed_payment_rate', 'recent_avg_days_late',
            'recent_missed_payments', 'risk_score'
        ]
        
        self.categorical_features = [
            'gender', 'state', 'employment_status', 'housing_status',
            'school_name', 'degree_type', 'major', 'school_type', 'completion_status'
        ]
    
    def create_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional engineered features for better prediction."""
        df = df.copy()
        
        # Age-based features
        df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 100], 
                                labels=['Young', 'Middle', 'Mature', 'Senior'])
        
        # Income-based features
        df['income_category'] = pd.cut(df['annual_income'], 
                                      bins=[0, 30000, 50000, 75000, 100000, np.inf],
                                      labels=['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High'])
        
        # Credit score categories
        df['credit_category'] = pd.cut(df['credit_score_at_origination'],
                                      bins=[0, 580, 670, 740, 800, 850],
                                      labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
        
        # Loan burden features
        df['high_debt_burden'] = (df['debt_to_income_ratio'] > 0.4).astype(int)
        df['multiple_loans'] = (df['loan_count'] > 1).astype(int)
        df['large_loan_amount'] = (df['total_loan_amount'] > 50000).astype(int)
        
        # Payment behavior features
        df['frequent_late_payments'] = (df['missed_payment_rate'] > 0.1).astype(int)
        df['recent_payment_issues'] = (df['recent_missed_payments'] > 0).astype(int)
        
        # Educational features
        df['high_gpa'] = (df['gpa'] > 3.5).astype(int)
        df['graduate_degree'] = df['degree_type'].isin(['Masters', 'Doctorate']).astype(int)
        df['completed_education'] = (df['completion_status'] == 'Completed').astype(int)
        
        # Geographic risk (simplified - some states have higher default rates)
        high_risk_states = ['NV', 'FL', 'AZ', 'CA', 'MI']
        df['high_risk_state'] = df['state'].isin(high_risk_states).astype(int)
        
        # Employment stability
        df['stable_employment'] = (df['employment_status'] == 'Employed Full-time').astype(int)
        
        # Time since graduation
        current_year = pd.Timestamp.now().year
        df['years_since_graduation'] = current_year - df['graduation_year']
        df['recent_graduate'] = (df['years_since_graduation'] <= 2).astype(int)
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate input data."""
        df = df.copy()
        
        # Handle missing values for numerical features
        numerical_cols = [col for col in self.numerical_features if col in df.columns]
        for col in numerical_cols:
            if col not in self.imputers:
                self.imputers[col] = SimpleImputer(strategy='median')
                df[col] = self.imputers[col].fit_transform(df[[col]]).flatten()
            else:
                df[col] = self.imputers[col].transform(df[[col]]).flatten()
        
        # Handle missing values for categorical features
        categorical_cols = [col for col in self.categorical_features if col in df.columns]
        for col in categorical_cols:
            if col not in self.imputers:
                self.imputers[col] = SimpleImputer(strategy='most_frequent')
                df[col] = self.imputers[col].fit_transform(df[[col]]).flatten()
            else:
                df[col] = self.imputers[col].transform(df[[col]]).flatten()
        
        # Clip extreme values
        df['annual_income'] = df['annual_income'].clip(0, 500000)
        df['debt_to_income_ratio'] = df['debt_to_income_ratio'].clip(0, 5)
        df['risk_score'] = df['risk_score'].clip(0, 100)
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features using one-hot encoding."""
        df = df.copy()
        
        categorical_cols = [col for col in self.categorical_features if col in df.columns]
        
        # Add engineered categorical features
        engineered_categorical = ['age_group', 'income_category', 'credit_category']
        categorical_cols.extend([col for col in engineered_categorical if col in df.columns])
        
        encoded_dfs = [df]
        
        for col in categorical_cols:
            if fit and col not in self.encoders:
                encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
                encoded_values = encoder.fit_transform(df[[col]])
                feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0][1:]]
                self.encoders[col] = encoder
            else:
                encoder = self.encoders[col]
                encoded_values = encoder.transform(df[[col]])
                feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0][1:]]
            
            encoded_df = pd.DataFrame(encoded_values, columns=feature_names, index=df.index)
            encoded_dfs.append(encoded_df)
        
        # Remove original categorical columns and concatenate encoded features
        df_encoded = pd.concat(encoded_dfs, axis=1)
        df_encoded = df_encoded.drop(columns=categorical_cols, errors='ignore')
        
        return df_encoded
    
    def scale_numerical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features using StandardScaler."""
        df = df.copy()
        
        # Get numerical columns (excluding target and already encoded features)
        numerical_cols = [col for col in df.columns 
                         if col in self.numerical_features or 
                         col.startswith(('high_', 'multiple_', 'large_', 'frequent_', 'recent_', 
                                       'completed_', 'stable_', 'years_since_'))]
        
        if fit and 'numerical' not in self.scalers:
            scaler = StandardScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            self.scalers['numerical'] = scaler
        elif 'numerical' in self.scalers:
            df[numerical_cols] = self.scalers['numerical'].transform(df[numerical_cols])
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Complete preprocessing pipeline."""
        
        # Remove ID columns and other non-feature columns
        columns_to_remove = ['borrower_id']  # Add other ID columns here if needed
        df = df.drop(columns=[col for col in columns_to_remove if col in df.columns], errors='ignore')
        
        # Create additional features
        df = self.create_additional_features(df)
        
        # Clean data
        df = self.clean_data(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df, fit=fit)
        
        # Scale numerical features
        df = self.scale_numerical_features(df, fit=fit)
        
        if fit:
            # Store feature names (excluding target)
            self.feature_names = [col for col in df.columns if col != self.target_column]
        
        return df
    
    def prepare_training_data(self, df: pd.DataFrame, test_size: float = 0.2, 
                            random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                           pd.Series, pd.Series]:
        """Prepare training and testing datasets."""
        
        # Prepare features
        df_processed = self.prepare_features(df, fit=True)
        
        # Separate features and target
        X = df_processed.drop(columns=[self.target_column])
        y = df_processed[self.target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Testing set: {len(X_test)} samples")
        print(f"Target distribution - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}")
        print(f"Number of features: {len(self.feature_names)}")
        
        return X_train, X_test, y_train, y_test
    
    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessors."""
        if not self.feature_names:
            raise ValueError("Preprocessor must be fitted first using prepare_training_data()")
        
        # Apply preprocessing pipeline
        df_processed = self.prepare_features(df, fit=False)
        
        # Ensure same features as training data
        X = df_processed.drop(columns=[self.target_column], errors='ignore')
        
        # Add missing features with zeros
        for feature in self.feature_names:
            if feature not in X.columns:
                X[feature] = 0
        
        # Reorder columns to match training data
        X = X[self.feature_names]
        
        return X
    
    def get_feature_importance_names(self) -> List[str]:
        """Get list of feature names for importance analysis."""
        return self.feature_names.copy()


def main():
    """Test the preprocessing pipeline."""
    
    # Load synthetic data
    import os
    data_path = '../data/synthetic/student_loan_master_dataset.csv'
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        
        # Initialize preprocessor
        preprocessor = StudentLoanPreprocessor()
        
        # Prepare training data
        X_train, X_test, y_train, y_test = preprocessor.prepare_training_data(df)
        
        print(f"\nPreprocessing completed successfully!")
        print(f"Feature names: {len(preprocessor.get_feature_importance_names())}")
        
    else:
        print(f"Data file not found: {data_path}")
        print("Run data_generator.py first to create the dataset.")


if __name__ == "__main__":
    main()
