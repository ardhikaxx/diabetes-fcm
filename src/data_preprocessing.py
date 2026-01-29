import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import yaml
import pickle

class DiabetesDataPreprocessor:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load diabetes dataset"""
        df = pd.read_csv(self.config['data']['path'])
        print(f"Dataset shape: {df.shape}")
        print(f"Missing values:\n{df.isnull().sum()}")
        
        # Check class distribution
        print(f"\nClass distribution:")
        print(f"Healthy (0): {sum(df['Outcome'] == 0)} samples")
        print(f"Diabetic (1): {sum(df['Outcome'] == 1)} samples")
        print(f"Ratio: {sum(df['Outcome'] == 1)/len(df):.2%}")
        
        return df
    
    def preprocess(self, df):
        """Preprocess the diabetes dataset with improved handling"""
        # Save target variable
        y = df['Outcome'].values.copy()
        
        # Create a copy of features
        X = df.drop('Outcome', axis=1).copy()
        
        # Handle zeros in features (common in diabetes dataset)
        zero_not_allowed = ['Glucose', 'BloodPressure', 'SkinThickness', 
                           'Insulin', 'BMI']
        
        # Add outcome back temporarily for group-based imputation
        X_temp = X.copy()
        X_temp['Outcome'] = y
        
        for col in zero_not_allowed:
            # Replace 0 with NaN
            X_temp[col] = X_temp[col].replace(0, np.nan)
            # Fill with median based on outcome
            X_temp[col] = X_temp.groupby('Outcome')[col].transform(
                lambda x: x.fillna(x.median())
            )
        
        # Remove outcome column
        X = X_temp.drop('Outcome', axis=1)
        
        # Handle any remaining NaN values
        X = X.fillna(X.median())
        
        # Feature engineering - add interaction terms
        X['Glucose_BMI'] = X['Glucose'] * X['BMI'] / 100
        X['Age_Glucose'] = X['Age'] * X['Glucose'] / 100
        X['BP_BMI'] = X['BloodPressure'] * X['BMI'] / 100
        
        # Scale features
        feature_names = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, feature_names
    
    def split_data(self, X, y):
        """Split data into train, validation, and test sets"""
        # First split: train+val vs test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, 
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state'],
            stratify=y
        )
        
        # Second split: train vs validation
        val_size = self.config['data'].get('validation_size', 0.1)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_size/(1-self.config['data']['test_size']),
            random_state=self.config['data']['random_state'],
            stratify=y_train_val
        )
        
        print(f"Train shape: {X_train.shape}")
        print(f"Validation shape: {X_val.shape}")
        print(f"Test shape: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_scaler(self, path='models/scaler.pkl'):
        """Save the fitted scaler"""
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load_scaler(self, path='models/scaler.pkl'):
        """Load the fitted scaler"""
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)
        return self.scaler