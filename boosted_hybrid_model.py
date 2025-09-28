"""
Boosted Hybrid Model: Sinusoidal Regressor + XGBoost
This implements the method from your notebook for ensemble with TS2Vec.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb


def create_dataset(series, input_len=168, horizon=24):
    """Create sliding window dataset for forecasting"""
    X, y = [], []
    for i in range(len(series) - input_len - horizon):
        X.append(series[i:i+input_len])
        y.append(series[i+input_len:i+input_len+horizon])
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
        
    def fit(self, series, train_slice, val_slice):
        """
        Train the boosted hybrid model
        
        Args:
            series: Normalized time series data [timesteps]
            train_slice: Training data slice
            val_slice: Validation data slice for early stopping
        """
        # Create datasets for different horizons
        self.models = {}
        
        # Train models for common forecasting horizons
        horizons = [24, 48, 168, 336, 720]
        
        for horizon in horizons:
            print(f"Training hybrid model for horizon {horizon}...")
            
            # Create windowed dataset
            X, y = create_dataset(series, self.input_len, horizon)
            
            # Split according to provided slices
            X_train = X[train_slice.start:train_slice.stop]
            y_train = y[train_slice.start:train_slice.stop]
            X_val = X[val_slice.start:val_slice.stop]
            y_val = y[val_slice.start:val_slice.stop]
            
            # Skip if not enough data
            if len(X_train) == 0 or len(X_val) == 0:
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
            
    def predict(self, series, test_slice, horizon):
        """
        Generate predictions for given horizon
        
        Args:
            series: Normalized time series data
            test_slice: Test data slice
            horizon: Prediction horizon
            
        Returns:
            Predictions array [n_samples, horizon]
        """
        if horizon not in self.models:
            raise ValueError(f"Model not trained for horizon {horizon}")
        
        model_dict = self.models[horizon]
        linreg = model_dict['linreg']
        xgb_model = model_dict['xgb_model']
        sin_features = model_dict['sin_features']
        
        # Create windowed dataset
        X, y = create_dataset(series, self.input_len, horizon)
        
        # Align with TS2Vec's padding logic (padding=200)
        padding = 200
        adjusted_start = max(test_slice.start, padding)
        adjusted_stop = min(test_slice.stop, len(X))
        
        X_test = X[adjusted_start:adjusted_stop]
        
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
        pred_lens: List of prediction horizons
        scaler: Fitted scaler for inverse transform
        
    Returns:
        Dictionary with predictions for each horizon
    """
    # Extract univariate series (assuming last column is target)
    series = data[0, :, -1]  # [time_steps]
    
    # Initialize hybrid model
    hybrid_model = BoostedHybridForecaster(input_len=168, period=24)
    
    # Train the model
    hybrid_model.fit(series, train_slice, val_slice)
    
    # Generate predictions for each horizon
    predictions = {}
    
    for pred_len in pred_lens:
        try:
            # Get hybrid predictions
            hybrid_preds = hybrid_model.predict(series, test_slice, pred_len)
            
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