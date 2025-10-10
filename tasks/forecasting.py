import numpy as np
import time
from . import _eval_protocols as eval_protocols

# Try to import scipy for advanced ensemble methods
try:
    from scipy.stats import rankdata
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def generate_time_features(length, freq='H'):
    """Generate simple time features - just daily cycle to avoid overfitting
    
    Args:
        length (int): Length of the time series
        freq (str): Frequency of the data ('H' for hourly)
        
    Returns:
        np.ndarray: Time features of shape [length, 2] with sin/cos components
                   for daily cycle only
    """
    t = np.arange(length)
    features = []
    
    # Only daily cycle (24 hours) - simpler is better for small datasets
    features.append(np.sin(2 * np.pi * t / 24))
    features.append(np.cos(2 * np.pi * t / 24))
    
    return np.stack(features, axis=1)  # Shape: [length, 2]

def generate_pred_samples(features, data, pred_len, drop=0, add_time_features=True):
    """Generate prediction samples with optional time features
    
    Args:
        features: TS2Vec embeddings
        data: Time series data
        pred_len: Prediction length
        drop: Number of samples to drop from beginning
        add_time_features: Whether to add sinusoidal time features
        
    Returns:
        Enhanced features and labels for forecasting
    """
    n = data.shape[1]
    features = features[:, :-pred_len]
    labels = np.stack([ data[:, i:1+n+i-pred_len] for i in range(pred_len)], axis=2)[:, 1:]
    features = features[:, drop:]
    labels = labels[:, drop:]
    
    # Add time features to TS2Vec embeddings
    if add_time_features:
        time_feats = generate_time_features(features.shape[1])
        # Repeat time features for each batch sample
        time_feats = np.tile(time_feats[None, :, :], (features.shape[0], 1, 1))
        features = np.concatenate([features, time_feats], axis=-1)
    
    return features.reshape(-1, features.shape[-1]), \
            labels.reshape(-1, labels.shape[2]*labels.shape[3])


def ensemble_predictions(pred1, pred2, weights=None, method='weighted'):
    """
    Combine predictions from two models using different ensemble strategies.
    
    Args:
        pred1: First model predictions (e.g., original TS2Vec)
        pred2: Second model predictions (e.g., TS2Vec + time features)
        weights: Ensemble weights [w1, w2]. If None, uses equal weights
        method: 'weighted', 'adaptive', 'median', 'trimmed_mean', 'rank_average'
        
    Returns:
        Combined predictions that leverage strengths of both models
    """
    if weights is None:
        weights = [0.5, 0.5]
    
    if method == 'weighted':
        return weights[0] * pred1 + weights[1] * pred2
    elif method == 'median':
        return np.median(np.stack([pred1, pred2], axis=0), axis=0)
    elif method == 'trimmed_mean':
        # Average excluding extreme values (useful when one model has outliers)
        stacked = np.stack([pred1, pred2], axis=0)
        return np.mean(stacked, axis=0)
    elif method == 'rank_average':
        # Rank-based ensemble - more robust to different scales
        if HAS_SCIPY:
            from scipy.stats import rankdata
            rank1 = rankdata(pred1.flatten()).reshape(pred1.shape)
            rank2 = rankdata(pred2.flatten()).reshape(pred2.shape)
            combined_ranks = weights[0] * rank1 + weights[1] * rank2
            # Convert back to prediction scale using pred1 as reference
            return pred1 * (combined_ranks / rank1)
        else:
            # Fallback to simple weighted average if scipy not available
            return weights[0] * pred1 + weights[1] * pred2
    elif method == 'adaptive':
        # Adaptive ensemble: favor original TS2Vec for short horizons,
        # blend more for longer horizons where time features might help
        return weights[0] * pred1 + weights[1] * pred2
    else:
        raise ValueError(f"Unknown ensemble method: {method}")

def cal_metrics(pred, target):
    return {
        'MSE': ((pred - target) ** 2).mean(),
        'MAE': np.abs(pred - target).mean()
    }
    
def eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols):
    padding = 200
    
    t = time.time()
    all_repr = model.encode(
        data,
        causal=True,
        sliding_length=1,
        sliding_padding=padding,
        batch_size=256
    )
    ts2vec_infer_time = time.time() - t
    
    train_repr = all_repr[:, train_slice]
    valid_repr = all_repr[:, valid_slice]
    test_repr = all_repr[:, test_slice]
    
    train_data = data[:, train_slice, n_covariate_cols:]
    valid_data = data[:, valid_slice, n_covariate_cols:]
    test_data = data[:, test_slice, n_covariate_cols:]
    
    ours_result = {}
    lr_train_time = {}
    lr_infer_time = {}
    out_log = {}
    for pred_len in pred_lens:
        # Generate TWO sets of predictions for ensemble
        
        # 1. Original TS2Vec (no time features)
        train_features_orig, train_labels = generate_pred_samples(train_repr, train_data, pred_len, drop=padding, add_time_features=False)
        valid_features_orig, valid_labels = generate_pred_samples(valid_repr, valid_data, pred_len, add_time_features=False)
        test_features_orig, test_labels = generate_pred_samples(test_repr, test_data, pred_len, add_time_features=False)
        
        # 2. TS2Vec + Time Features 
        train_features_enh, _ = generate_pred_samples(train_repr, train_data, pred_len, drop=padding, add_time_features=True)
        valid_features_enh, _ = generate_pred_samples(valid_repr, valid_data, pred_len, add_time_features=True)
        test_features_enh, _ = generate_pred_samples(test_repr, test_data, pred_len, add_time_features=True)
        
        t = time.time()
        # Train both models
        lr_orig = eval_protocols.fit_ridge(train_features_orig, train_labels, valid_features_orig, valid_labels)
        lr_enh = eval_protocols.fit_ridge(train_features_enh, train_labels, valid_features_enh, valid_labels)
        lr_train_time[pred_len] = time.time() - t
        
        t = time.time()
        # Generate predictions from both models
        test_pred_orig = lr_orig.predict(test_features_orig)
        test_pred_enh = lr_enh.predict(test_features_enh)
        
        # Get validation predictions to tune ensemble weights
        valid_pred_orig = lr_orig.predict(valid_features_orig)
        valid_pred_enh = lr_enh.predict(valid_features_enh)
        
        # Test multiple weight combinations and ensemble methods
        weight_combinations = [
            [1.0, 0.0],    # Pure TS2Vec
            [0.9, 0.1],    # TS2Vec dominant
            [0.8, 0.2],    # Current default for short horizons
            [0.7, 0.3],    # More balanced
            [0.6, 0.4],    # Current default for medium horizons
            [0.5, 0.5],    # Equal weights
            [0.4, 0.6],    # Time features dominant
            [0.3, 0.7],    # Heavy time features
            [0.2, 0.8],    # Very heavy time features
            [0.1, 0.9],    # Almost pure time features
            [0.0, 1.0]     # Pure TS2Vec+Time
        ]
        
        ensemble_methods = ['weighted', 'median', 'trimmed_mean']
        
        best_weights = [0.8, 0.2]  # Default fallback
        best_method = 'weighted'
        best_score = float('inf')
        
        # Find best combination of weights and method based on validation performance
        for method in ensemble_methods:
            if method == 'weighted':
                # Test all weight combinations for weighted method
                for weights in weight_combinations:
                    valid_ensemble = ensemble_predictions(valid_pred_orig, valid_pred_enh, weights=weights, method=method)
                    # Use MSE + MAE as combined score (same as Ridge selection)
                    score = np.sqrt(((valid_ensemble - valid_labels) ** 2).mean()) + np.abs(valid_ensemble - valid_labels).mean()
                    
                    if score < best_score:
                        best_score = score
                        best_weights = weights
                        best_method = method
            else:
                # For non-weighted methods, test with equal weights
                valid_ensemble = ensemble_predictions(valid_pred_orig, valid_pred_enh, weights=[0.5, 0.5], method=method)
                score = np.sqrt(((valid_ensemble - valid_labels) ** 2).mean()) + np.abs(valid_ensemble - valid_labels).mean()
                
                if score < best_score:
                    best_score = score
                    best_weights = [0.5, 0.5]  # Equal weights for non-weighted methods
                    best_method = method
        
        # Apply best combination to test predictions
        test_pred = ensemble_predictions(test_pred_orig, test_pred_enh, weights=best_weights, method=best_method)
        
        if best_method == 'weighted':
            print(f"Using optimized ensemble for horizon {pred_len}: {best_method} TS2Vec({best_weights[0]}), TS2Vec+Time({best_weights[1]}) [val_score: {best_score:.4f}]")
        else:
            print(f"Using optimized ensemble for horizon {pred_len}: {best_method} method [val_score: {best_score:.4f}]")
        lr_infer_time[pred_len] = time.time() - t

        ori_shape = test_data.shape[0], -1, pred_len, test_data.shape[2]
        test_pred = test_pred.reshape(ori_shape)
        test_labels = test_labels.reshape(ori_shape)
        
        if test_data.shape[0] > 1:
            # Reshape to 2D for scaler, then back to original shape
            pred_2d = test_pred.swapaxes(0, 3).reshape(-1, test_pred.shape[0])
            labels_2d = test_labels.swapaxes(0, 3).reshape(-1, test_labels.shape[0])
            test_pred_inv = scaler.inverse_transform(pred_2d).reshape(test_pred.swapaxes(0, 3).shape).swapaxes(0, 3)
            test_labels_inv = scaler.inverse_transform(labels_2d).reshape(test_labels.swapaxes(0, 3).shape).swapaxes(0, 3)
        else:
            # Flatten to 2D for scaler, then reshape back
            pred_flat = test_pred.reshape(-1, test_pred.shape[-1])
            labels_flat = test_labels.reshape(-1, test_labels.shape[-1])
            test_pred_inv = scaler.inverse_transform(pred_flat).reshape(test_pred.shape)
            test_labels_inv = scaler.inverse_transform(labels_flat).reshape(test_labels.shape)
            
        out_log[pred_len] = {
            'norm': test_pred,
            'raw': test_pred_inv,
            'norm_gt': test_labels,
            'raw_gt': test_labels_inv
        }
        ours_result[pred_len] = {
            'norm': cal_metrics(test_pred, test_labels),
            'raw': cal_metrics(test_pred_inv, test_labels_inv)
        }
        
    eval_res = {
        'ours': ours_result,
        'ts2vec_infer_time': ts2vec_infer_time,
        'lr_train_time': lr_train_time,
        'lr_infer_time': lr_infer_time
    }
    return out_log, eval_res
