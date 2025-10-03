"""
Boosted Hybrid Model: Sinusoidal Regressor + XGBoost
This implements the method from your notebook for ensemble with TS2Vec.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb


def create_dataset_ts2vec_style(series, pred_len, drop=0):
    """
    Create dataset using TS2Vec's exact windowing logic
    
    Args:
        series: Time series data [timesteps]
        pred_len: Prediction horizon length
        drop: Number of samples to drop from beginning (padding compensation)
        
    Returns:
        X, y arrays with exact same logic as TS2Vec generate_pred_samples
    """
    n = len(series)
    
    # TS2Vec logic: remove last pred_len timesteps for features
    features_end = n - pred_len
    
    # Create labels using TS2Vec's stacking logic
    labels = []
    for i in range(pred_len):
        # Extract prediction targets for each step ahead
        label_slice = series[1+i:1+features_end+i]  # Offset by 1 like TS2Vec
        labels.append(label_slice)
    
    # Stack labels: [n_steps, pred_len] 
    labels = np.stack(labels, axis=1)  # Shape: [features_end-1, pred_len]
    
    # Create input features (sliding windows)
    X = []
    for i in range(len(labels)):
        if i + 1 < len(series):  # Ensure we have enough data
            X.append(series[i:i+1])  # Single timestep window (can be extended)
    
    X = np.array(X)
    
    # Apply drop (padding compensation) - same as TS2Vec
    if drop > 0:
        X = X[drop:]
        labels = labels[drop:]
    
    return X, labels


def create_dataset_with_lookback(series, input_len, pred_len, drop=0):
    """
    Create dataset with proper lookback window, aligned with TS2Vec logic
    
    Args:
        series: Time series data [timesteps]
        input_len: Length of input window (e.g., 168)
        pred_len: Prediction horizon length
        drop: Number of samples to drop from beginning (padding compensation)
        
    Returns:
        X, y arrays aligned with TS2Vec sample generation
    """
    n = len(series)
    
    # Calculate available samples using TS2Vec logic
    # TS2Vec removes last pred_len steps, then applies drop
    max_samples = n - pred_len - drop - input_len + 1
    
    if max_samples <= 0:
        return np.array([]), np.array([])
    
    X, y = [], []
    
    # Generate samples aligned with TS2Vec's indexing
    for i in range(max_samples):
        # Input window
        start_idx = i + drop
        end_idx = start_idx + input_len
        
        # Prediction targets (aligned with TS2Vec label generation)
        pred_start = end_idx
        pred_end = pred_start + pred_len
        
        if pred_end <= n:
            X.append(series[start_idx:end_idx])
            y.append(series[pred_start:pred_end])
    
    return np.array(X), np.array(y)


def add_sin_features(n_samples, horizon, period=24):
    """Generate sinusoidal features for given horizon"""
    t = np.arange(horizon)
    sin_feat = np.sin(2*np.pi*t/period)
    cos_feat = np.cos(2*np.pi*t/period)
    return np.vstack([sin_feat, cos_feat]).T


class BoostedHybridForecaster:
    """Sinusoidal + XGBoost ensemble for time series forecasting"""
    
    def __init__(self, input_len=168, period=24, xgb_params=None):
        self.input_len = input_len
        self.period = period
        self.linreg = LinearRegression()
        
        # Default XGBoost parameters optimized for residual modeling
        if xgb_params is None:
            xgb_params = {
                'n_estimators': 100,
                'learning_rate': 0.01,
                'max_depth': 5,
                'tree_method': 'hist',
                'random_state': 42
            }
        self.xgb_model = xgb.XGBRegressor(**xgb_params)
        
    def fit_for_horizons(self, series, train_slice, val_slice, horizons):
        """
        Train the boosted hybrid model for specific horizons
        
        Args:
            series: Normalized time series data [timesteps]
            train_slice: Training data slice
            val_slice: Validation data slice for early stopping
            horizons: List of specific horizons to train for
        """
        self.models = {}
        
        for horizon in horizons:
            print(f"Training hybrid model for horizon {horizon}...")
            
            # Create windowed dataset using TS2Vec-aligned logic
            X, y = create_dataset_with_lookback(series, self.input_len, horizon, drop=0)
            
            # Apply slicing to match TS2Vec's train/val splits
            # Convert slice indices to sample indices
            train_samples = min(len(X), train_slice.stop - self.input_len - horizon)
            val_start = max(0, val_slice.start - self.input_len - horizon)
            val_samples = min(len(X), val_slice.stop - self.input_len - horizon)
            
            if train_samples <= 0 or val_samples <= val_start:
                print(f"Insufficient data for horizon {horizon}")
                continue
                
            X_train = X[:train_samples]
            y_train = y[:train_samples]
            X_val = X[val_start:val_samples] if val_samples > val_start else X[:1]
            y_val = y[val_start:val_samples] if val_samples > val_start else y[:1]
            
            # Skip if not enough data
            if len(X_train) == 0 or len(y_train) == 0:
                print(f"No training data for horizon {horizon}")
                continue
                
            # 1. Train sinusoidal regressor
            sin_features = add_sin_features(len(y_train), horizon, self.period)
            X_sin = np.tile(sin_features, (len(y_train), 1))
            y_train_flat = y_train.flatten()
            
            linreg = LinearRegression()
            linreg.fit(X_sin, y_train_flat)
            
            # 2. Compute residuals
            y_train_pred_sin = linreg.predict(X_sin).reshape(len(y_train), horizon)
            residuals = y_train - y_train_pred_sin
            
            # 3. Train XGBoost on residuals
            X_train_flat = X_train.reshape(len(X_train), -1)
            residuals_flat = residuals.reshape(len(residuals), -1)
            
            xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.01, 
                max_depth=5,
                tree_method='hist',
                random_state=42
            )
            xgb_model.fit(X_train_flat, residuals_flat)
            
            # Store models
            self.models[horizon] = {
                'linreg': linreg,
                'xgb_model': xgb_model,
                'sin_features': sin_features
            }
            
    def predict(self, series, test_slice, horizon, padding=200):
        """
        Generate predictions for given horizon using TS2Vec-aligned logic
        
        Args:
            series: Normalized time series data
            test_slice: Test data slice
            horizon: Prediction horizon
            padding: Padding to match TS2Vec (default 200)
            
        Returns:
            Predictions array [n_samples, horizon]
        """
        if horizon not in self.models:
            raise ValueError(f"Model not trained for horizon {horizon}")
        
        model_dict = self.models[horizon]
        linreg = model_dict['linreg']
        xgb_model = model_dict['xgb_model']
        sin_features = model_dict['sin_features']
        
        # Create windowed dataset using TS2Vec-aligned logic with padding
        X, y = create_dataset_with_lookback(series, self.input_len, horizon, drop=padding)
        
        # Calculate test sample indices aligned with TS2Vec
        # TS2Vec test_slice refers to timesteps, we need to convert to sample indices
        test_start_sample = max(0, test_slice.start - self.input_len - horizon - padding)
        test_end_sample = min(len(X), test_slice.stop - self.input_len - horizon - padding)
        
        if test_end_sample <= test_start_sample:
            print(f"No test samples available for horizon {horizon}")
            return np.array([])
        
        X_test = X[test_start_sample:test_end_sample]
        
        if len(X_test) == 0:
            return np.array([])
        
        # 1. Sinusoidal prediction
        X_sin_test = np.tile(sin_features, (len(X_test), 1))
        y_pred_sin = linreg.predict(X_sin_test).reshape(len(X_test), horizon)
        
        # 2. Residual correction with XGBoost
        X_test_flat = X_test.reshape(len(X_test), -1)
        res_pred = xgb_model.predict(X_test_flat).reshape(len(X_test), horizon)
        
        # 3. Final prediction = sinusoidal + residual
        y_pred = y_pred_sin + res_pred
        
        return y_pred


def get_hybrid_predictions(data, train_slice, val_slice, test_slice, pred_lens, scaler):
    """
    Generate predictions using the Boosted Hybrid Model for ensemble with TS2Vec
    
    Args:
        data: Time series data [batch, time, features] 
        train_slice, val_slice, test_slice: Data slices
        pred_lens: List of prediction horizons (dataset-specific)
        scaler: Fitted scaler for inverse transform
        
    Returns:
        Dictionary with predictions for each horizon
    """
    # Extract univariate series (assuming last column is target)
    series = data[0, :, -1]  # [time_steps]
    
    # Initialize hybrid model
    hybrid_model = BoostedHybridForecaster(input_len=168, period=24)
    
    # Train the model for the specific horizons needed
    hybrid_model.fit_for_horizons(series, train_slice, val_slice, pred_lens)
    
    # Generate predictions for each horizon
    predictions = {}
    
    for pred_len in pred_lens:
        try:
            # Get hybrid predictions with TS2Vec padding alignment
            hybrid_preds = hybrid_model.predict(series, test_slice, pred_len, padding=200)
            
            if len(hybrid_preds) > 0:
                # Inverse transform predictions
                hybrid_preds_inv = scaler.inverse_transform(
                    hybrid_preds.reshape(-1, 1)
                ).reshape(hybrid_preds.shape)
                
                predictions[pred_len] = {
                    'norm': hybrid_preds,
                    'raw': hybrid_preds_inv
                }
            else:
                predictions[pred_len] = None
                
        except Exception as e:
            print(f"Hybrid model failed for horizon {pred_len}: {e}")
            predictions[pred_len] = None
            
    return predictions