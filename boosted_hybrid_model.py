"""
Simple Sinusoidal Forecaster for TS2Vec Ensemble
This implements only the sinusoidal regression component for fast ensemble forecasting.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


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


def add_sin_features(n_samples, horizon, period=24):
    """Generate sinusoidal features for given horizon"""
    t = np.arange(horizon)
    sin_feat = np.sin(2*np.pi*t/period)
    cos_feat = np.cos(2*np.pi*t/period)
    return np.vstack([sin_feat, cos_feat]).T


class SimpleSinusoidalForecaster:
    """Simple sinusoidal regressor for time series forecasting"""
    
    def __init__(self, period=24):
        self.period = period
        self.models = {}
        
    def fit_for_horizons(self, series, train_slice, val_slice, horizons, padding=200):
        """
        Train sinusoidal regression models for specific horizons
        
        Args:
            series: Normalized time series data [timesteps]
            train_slice: Training data slice
            val_slice: Validation data slice (unused but kept for compatibility)
            horizons: List of specific horizons to train for
            padding: Padding value to match TS2Vec logic
        """
        self.models = {}
        
        for horizon in horizons:
            print(f"Training sinusoidal model for horizon {horizon}...")
            
            # Create windowed dataset using TS2Vec-style logic
            X, y = create_dataset_ts2vec_style(series, horizon, drop=padding)
            
            # Split according to provided slices (adjust for windowing)
            train_end = min(train_slice.stop, len(X))
            X_train = X[train_slice.start:train_end]
            y_train = y[train_slice.start:train_end]
            
            # Skip if not enough data
            if len(X_train) == 0:
                print(f"Skipping horizon {horizon} - not enough training data")
                continue
                
            # Train sinusoidal regressor
            sin_features = add_sin_features(len(y_train), horizon, self.period)
            X_sin = np.tile(sin_features, (len(y_train), 1))
            y_train_flat = y_train.flatten()
            
            linreg = LinearRegression()
            linreg.fit(X_sin, y_train_flat)
            
            # Store models
            self.models[horizon] = {
                'linreg': linreg,
                'sin_features': sin_features
            }
            print(f"Trained sinusoidal model for horizon {horizon} with {len(y_train)} samples")
            
    def predict_ts2vec_aligned(self, series, test_slice, horizon, padding=200):
        """
        Generate predictions aligned with TS2Vec output
        
        Args:
            series: Normalized time series data
            test_slice: Test data slice  
            horizon: Prediction horizon
            padding: Padding value to match TS2Vec
            
        Returns:
            Predictions array [n_samples, horizon] aligned with TS2Vec
        """
        if horizon not in self.models:
            raise ValueError(f"Model not trained for horizon {horizon}")
        
        model_dict = self.models[horizon]
        linreg = model_dict['linreg']
        sin_features = model_dict['sin_features']
        
        # Create windowed dataset using same logic as training
        X, y = create_dataset_ts2vec_style(series, horizon, drop=padding)
        
        # Align with test slice (adjust for windowing effects)
        test_end = min(test_slice.stop, len(X))
        X_test = X[test_slice.start:test_end]
        
        if len(X_test) == 0:
            return np.array([])
        
        # Generate sinusoidal predictions
        X_sin_test = np.tile(sin_features, (len(X_test), 1))
        y_pred = linreg.predict(X_sin_test).reshape(len(X_test), horizon)
        
        return y_pred


def get_hybrid_predictions(data, train_slice, val_slice, test_slice, pred_lens, scaler):
    """
    Generate predictions using Simple Sinusoidal Model for ensemble with TS2Vec
    
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
    
    # Initialize simple sinusoidal model
    simple_model = SimpleSinusoidalForecaster(period=24)
    
    # Train the model for the specific horizons needed
    simple_model.fit_for_horizons(series, train_slice, val_slice, pred_lens, padding=200)
    
    # Generate predictions for each horizon
    predictions = {}
    
    for pred_len in pred_lens:
        try:
            # Get sinusoidal predictions
            simple_preds = simple_model.predict_ts2vec_aligned(series, test_slice, pred_len, padding=200)
            
            if len(simple_preds) > 0:
                # Inverse transform predictions
                simple_preds_inv = scaler.inverse_transform(
                    simple_preds.reshape(-1, 1)
                ).reshape(simple_preds.shape)
                
                predictions[pred_len] = {
                    'norm': simple_preds,
                    'raw': simple_preds_inv
                }
            else:
                predictions[pred_len] = None
                
        except Exception as e:
            print(f"Sinusoidal model failed for horizon {pred_len}: {e}")
            predictions[pred_len] = None
            
    return predictions