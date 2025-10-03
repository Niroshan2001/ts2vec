import numpy as np
import time
from . import _eval_protocols as eval_protocols

# Import hybrid model for ensemble
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from boosted_hybrid_model import get_hybrid_predictions
    HAS_HYBRID = True
except ImportError:
    print("Boosted Hybrid Model not available, using TS2Vec only")
    HAS_HYBRID = False

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
        method: 'weighted', 'adaptive', or 'median'
        
    Returns:
        Combined predictions that leverage strengths of both models
    """
    if weights is None:
        weights = [0.5, 0.5]
    
    if method == 'weighted':
        return weights[0] * pred1 + weights[1] * pred2
    elif method == 'median':
        return np.median(np.stack([pred1, pred2], axis=0), axis=0)
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
    
    # Get hybrid model predictions for ensemble
    hybrid_predictions = None
    if HAS_HYBRID:
        try:
            print("Generating hybrid model predictions...")
            hybrid_predictions = get_hybrid_predictions(
                data[:, :, n_covariate_cols:], train_slice, valid_slice, test_slice, pred_lens, scaler
            )
        except Exception as e:
            print(f"Hybrid model failed: {e}")
            hybrid_predictions = None
    
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
        
        # Three-way ensemble: TS2Vec + TS2Vec+Time + Hybrid Model
        if hybrid_predictions and pred_len in hybrid_predictions and hybrid_predictions[pred_len] is not None:
            # Get hybrid predictions in the right shape
            hybrid_pred = hybrid_predictions[pred_len]['norm']
            
            # Debug shapes
            print(f"Shapes - Hybrid: {hybrid_pred.shape}, TS2Vec: {test_pred_orig.shape}")
            
            # Calculate expected TS2Vec shape (flattened)
            expected_ts2vec_samples = len(test_pred_orig) // pred_len
            expected_hybrid_samples = hybrid_pred.shape[0]
            
            # Find minimum sample count and align
            min_samples = min(expected_ts2vec_samples, expected_hybrid_samples)
            
            if min_samples > 0:
                # Truncate TS2Vec predictions to match sample count
                ts2vec_truncated = test_pred_orig[:min_samples * pred_len]
                enhanced_truncated = test_pred_enh[:min_samples * pred_len]
                
                # Truncate and flatten hybrid predictions
                hybrid_truncated = hybrid_pred[:min_samples].reshape(-1)
                
                # Verify all arrays have same size
                if len(ts2vec_truncated) == len(enhanced_truncated) == len(hybrid_truncated):
                    # Three-way ensemble with adaptive weights
                    if pred_len <= 48:
                        w1, w2, w3 = 0.7, 0.1, 0.2  # TS2Vec, TS2Vec+Time, Hybrid
                    elif pred_len <= 168:
                        w1, w2, w3 = 0.5, 0.2, 0.3
                    else:
                        w1, w2, w3 = 0.4, 0.2, 0.4
                        
                    test_pred = w1 * ts2vec_truncated + w2 * enhanced_truncated + w3 * hybrid_truncated
                    print(f"Using 3-way ensemble for horizon {pred_len}: TS2Vec({w1}), TS2Vec+Time({w2}), Hybrid({w3})")
                    print(f"Ensemble shapes - TS2Vec: {ts2vec_truncated.shape}, Enhanced: {enhanced_truncated.shape}, Hybrid: {hybrid_truncated.shape}")
                else:
                    print(f"Size mismatch after truncation for horizon {pred_len}: TS2Vec={len(ts2vec_truncated)}, Enhanced={len(enhanced_truncated)}, Hybrid={len(hybrid_truncated)}")
                    print("Falling back to 2-way ensemble")
                    # Two-way ensemble fallback
                    if pred_len <= 48:
                        weights = [0.8, 0.2]
                    elif pred_len <= 168:
                        weights = [0.6, 0.4]
                    else:
                        weights = [0.5, 0.5]
                    test_pred = ensemble_predictions(test_pred_orig, test_pred_enh, weights=weights, method='weighted')
            else:
                print(f"No valid samples for alignment at horizon {pred_len}, falling back to 2-way ensemble")
                # Two-way ensemble fallback
                if pred_len <= 48:
                    weights = [0.8, 0.2]
                elif pred_len <= 168:
                    weights = [0.6, 0.4]
                else:
                    weights = [0.5, 0.5]
                test_pred = ensemble_predictions(test_pred_orig, test_pred_enh, weights=weights, method='weighted')
        else:
            # Two-way ensemble: TS2Vec + TS2Vec+Time (fallback)
            if pred_len <= 48:
                weights = [0.8, 0.2]
            elif pred_len <= 168:
                weights = [0.6, 0.4]  
            else:
                weights = [0.5, 0.5]
            test_pred = ensemble_predictions(test_pred_orig, test_pred_enh, weights=weights, method='weighted')
            print(f"Using 2-way ensemble for horizon {pred_len}: TS2Vec({weights[0]}), TS2Vec+Time({weights[1]})")
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
