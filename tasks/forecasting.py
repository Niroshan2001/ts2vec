import numpy as np
import time
from . import _eval_protocols as eval_protocols

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
        # Generate features WITH explicit time features for better temporal modeling
        train_features, train_labels = generate_pred_samples(train_repr, train_data, pred_len, drop=padding, add_time_features=False)
        valid_features, valid_labels = generate_pred_samples(valid_repr, valid_data, pred_len, add_time_features=False)
        test_features, test_labels = generate_pred_samples(test_repr, test_data, pred_len, add_time_features=False)
        
        t = time.time()
        # Use Ridge regression with time features (simpler and more stable)
        lr = eval_protocols.fit_ridge(train_features, train_labels, valid_features, valid_labels)
        lr_train_time[pred_len] = time.time() - t
        
        t = time.time()
        test_pred = lr.predict(test_features)
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
