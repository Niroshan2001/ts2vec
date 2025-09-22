#!/usr/bin/env python3
"""
TS2Vec Anomaly Detection Testing Framework
Reproduce paper results and test MSM improvements
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import torch
import time
from ts2vec import TS2Vec
from ts2vec_msm import TS2VecMSM
import warnings
warnings.filterwarnings('ignore')

class TS2VecAnomalyDetector:
    """
    TS2Vec anomaly detection implementation following the paper methodology
    """
    def __init__(self, model, Z=21, beta=4):
        self.model = model
        self.Z = Z  # Window size for local average
        self.beta = beta  # Threshold multiplier
        self.mu = None
        self.sigma = None
        
    def compute_anomaly_score(self, data, normalize=True):
        """
        Compute anomaly scores using masked vs unmasked representation difference
        
        Args:
            data: Time series data [length, features]
            normalize: Whether to apply local averaging normalization
            
        Returns:
            anomaly_scores: Raw anomaly scores for each timestamp
            adjusted_scores: Normalized anomaly scores (if normalize=True)
        """
        if len(data.shape) == 2:
            data = data[np.newaxis, :, :]  # Add batch dimension
            
        length = data.shape[1]
        anomaly_scores = []
        
        # Set model to eval mode
        if hasattr(self.model, 'eval'):
            self.model.eval()
        elif hasattr(self.model, 'net'):
            self.model.net.eval()
        
        with torch.no_grad():
            for t in range(1, length):  # Start from 1 since we need at least one point
                # 1. Forward with last observation masked
                masked_data = data.copy()
                masked_data[:, t, :] = 0  # Mask the current timestamp
                
                # 2. Forward without masking
                unmasked_data = data.copy()
                
                # Get representations for timestamp t
                try:
                    # Encode masked and unmasked versions
                    masked_repr = self.model.encode(masked_data, encoding_window='full_series')
                    unmasked_repr = self.model.encode(unmasked_data, encoding_window='full_series')
                    
                    # Handle different representation shapes
                    if len(masked_repr.shape) == 3:
                        # If we get [batch, time, features], take the representation at time t
                        if masked_repr.shape[1] > t:
                            r_m_t = masked_repr[0, t, :]
                            r_u_t = unmasked_repr[0, t, :]
                        else:
                            # Use the last available representation
                            r_m_t = masked_repr[0, -1, :]
                            r_u_t = unmasked_repr[0, -1, :]
                    else:
                        # If we get [batch, features], use the full representation
                        r_m_t = masked_repr[0, :]
                        r_u_t = unmasked_repr[0, :]
                    
                    # Compute L1 distance (anomaly score)
                    score = np.sum(np.abs(r_u_t - r_m_t))
                    anomaly_scores.append(score)
                    
                except Exception as e:
                    print(f"Error at timestamp {t}: {e}")
                    anomaly_scores.append(0.0)
        
        anomaly_scores = np.array(anomaly_scores)
        
        if not normalize:
            return anomaly_scores, anomaly_scores
        
        # Apply local averaging normalization as in paper
        adjusted_scores = []
        for t in range(len(anomaly_scores)):
            if t < self.Z:
                # Not enough history, use raw score
                adjusted_scores.append(0.0)
            else:
                # Compute local average of preceding Z points
                local_avg = np.mean(anomaly_scores[max(0, t-self.Z):t])
                if local_avg > 0:
                    adj_score = (anomaly_scores[t] - local_avg) / local_avg
                else:
                    adj_score = 0.0
                adjusted_scores.append(adj_score)
        
        return anomaly_scores, np.array(adjusted_scores)
    
    def fit_threshold(self, train_scores):
        """
        Fit threshold parameters on training data
        """
        self.mu = np.mean(train_scores)
        self.sigma = np.std(train_scores)
        threshold = self.mu + self.beta * self.sigma
        return threshold
    
    def predict_anomalies(self, scores, threshold=None):
        """
        Predict anomalies based on threshold
        """
        if threshold is None:
            threshold = self.mu + self.beta * self.sigma
        return scores > threshold

def load_synthetic_anomaly_data():
    """
    Create synthetic anomaly data for testing
    """
    np.random.seed(42)
    length = 1000
    
    # Generate normal sinusoidal pattern with noise
    t = np.linspace(0, 4*np.pi, length)
    normal_data = np.sin(t) + 0.1 * np.sin(10*t) + 0.05 * np.random.randn(length)
    
    # Add anomalies
    anomaly_labels = np.zeros(length)
    
    # Point anomalies (outliers)
    anomaly_indices = [200, 450, 700, 850]
    for idx in anomaly_indices:
        normal_data[idx] += np.random.choice([-1, 1]) * np.random.uniform(2, 4)
        anomaly_labels[idx] = 1
    
    # Contextual anomalies (pattern breaks)
    for start in [300, 600]:
        end = start + 20
        normal_data[start:end] += 0.5 * np.sin(20 * t[start:end])
        anomaly_labels[start:end] = 1
    
    return normal_data.reshape(-1, 1), anomaly_labels

def evaluate_anomaly_detection(y_true, y_pred, y_scores=None):
    """
    Comprehensive evaluation of anomaly detection performance
    """
    # Basic metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    results = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'n_anomalies': np.sum(y_true),
        'n_detected': np.sum(y_pred),
        'true_positives': np.sum((y_true == 1) & (y_pred == 1))
    }
    
    if y_scores is not None:
        try:
            auc = roc_auc_score(y_true, y_scores)
            results['auc'] = auc
        except:
            results['auc'] = 0.0
    
    return results

def test_baseline_ts2vec():
    """
    Test baseline TS2Vec anomaly detection
    """
    print("🔍 TESTING BASELINE TS2VEC ANOMALY DETECTION")
    print("="*60)
    
    # Generate test data
    data, labels = load_synthetic_anomaly_data()
    print(f"📊 Data: {data.shape}, Anomalies: {np.sum(labels)}/{len(labels)} ({np.sum(labels)/len(labels)*100:.1f}%)")
    
    # Split into train/test
    split_point = len(data) // 2
    train_data = data[:split_point]
    test_data = data[split_point:]
    test_labels = labels[split_point:]
    
    # Initialize TS2Vec model
    try:
        from utils import init_dl_program
        device = init_dl_program(0, seed=42, max_threads=None, deterministic=True)
        
        model = TS2Vec(
            input_dims=1,
            output_dims=320,
            device=device,
            lr=0.001,
            batch_size=8,
            max_train_length=None
        )
        
        print("🚂 Training TS2Vec on normal data...")
        start_time = time.time()
        
        # Train on normal data only (unsupervised)
        train_data_3d = train_data[np.newaxis, :, :]
        model.fit(train_data_3d, n_iters=200, verbose=False)
        
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.1f}s")
        
        # Initialize detector
        detector = TS2VecAnomalyDetector(model, Z=21, beta=4)
        
        # Compute anomaly scores on training data to fit threshold
        print("📊 Computing training scores for threshold fitting...")
        train_scores_raw, train_scores_adj = detector.compute_anomaly_score(train_data_3d)
        threshold = detector.fit_threshold(train_scores_adj)
        print(f"🎯 Threshold: μ={detector.mu:.4f}, σ={detector.sigma:.4f}, threshold={threshold:.4f}")
        
        # Test on test data
        print("🔍 Testing on test data...")
        test_data_3d = test_data[np.newaxis, :, :]
        test_scores_raw, test_scores_adj = detector.compute_anomaly_score(test_data_3d)
        
        # Predict anomalies
        predictions = detector.predict_anomalies(test_scores_adj, threshold)
        
        # Evaluate
        # Align predictions with labels (account for window size)
        aligned_labels = test_labels[detector.Z:]
        aligned_predictions = predictions[detector.Z:]
        
        results = evaluate_anomaly_detection(aligned_labels, aligned_predictions, test_scores_adj[detector.Z:])
        
        print(f"\n📈 BASELINE TS2VEC RESULTS:")
        print(f"   🎯 F1 Score: {results['f1']:.4f}")
        print(f"   📊 Precision: {results['precision']:.4f}")
        print(f"   📊 Recall: {results['recall']:.4f}")
        print(f"   🔍 True Anomalies: {results['n_anomalies']}")
        print(f"   🔍 Detected: {results['n_detected']}")
        print(f"   ✅ True Positives: {results['true_positives']}")
        if 'auc' in results:
            print(f"   📈 AUC: {results['auc']:.4f}")
        
        return model, detector, results, test_scores_adj, aligned_labels
        
    except Exception as e:
        print(f"❌ Baseline test failed: {e}")
        return None, None, None, None, None

def test_msm_anomaly_detection(baseline_results=None):
    """
    Test TS2Vec-MSM for anomaly detection improvement
    """
    print("\n🚀 TESTING TS2VEC-MSM ANOMALY DETECTION")
    print("="*60)
    
    # Generate test data
    data, labels = load_synthetic_anomaly_data()
    split_point = len(data) // 2
    train_data = data[:split_point]
    test_data = data[split_point:]
    test_labels = labels[split_point:]
    
    # Test different MSM configurations
    msm_configs = [
        {'name': 'light_msm', 'lambda': 0.01, 'description': 'Very light MSM'},
        {'name': 'minimal_msm', 'lambda': 0.05, 'description': 'Minimal MSM'},
        {'name': 'standard_msm', 'lambda': 0.1, 'description': 'Standard MSM'}
    ]
    
    best_f1 = 0
    best_config = None
    all_results = []
    
    for config in msm_configs:
        print(f"\n🧪 Testing {config['name']} (λ={config['lambda']})...")
        
        try:
            from utils import init_dl_program
            device = init_dl_program(0, seed=42, max_threads=None, deterministic=True)
            
            model = TS2VecMSM(
                input_dims=1,
                output_dims=320,
                device=device,
                lr=0.001,
                batch_size=8,
                max_train_length=None,
                msm_weight=config['lambda'],
                msm_mask_rate=0.15,
                msm_decoder_depth=3,
                dynamic_lambda=False
            )
            
            # Train
            train_data_3d = train_data[np.newaxis, :, :]
            model.fit(train_data_3d, n_iters=200, verbose=False)
            
            # Test
            detector = TS2VecAnomalyDetector(model, Z=21, beta=4)
            train_scores_raw, train_scores_adj = detector.compute_anomaly_score(train_data_3d)
            threshold = detector.fit_threshold(train_scores_adj)
            
            test_data_3d = test_data[np.newaxis, :, :]
            test_scores_raw, test_scores_adj = detector.compute_anomaly_score(test_data_3d)
            predictions = detector.predict_anomalies(test_scores_adj, threshold)
            
            # Evaluate
            aligned_labels = test_labels[detector.Z:]
            aligned_predictions = predictions[detector.Z:]
            results = evaluate_anomaly_detection(aligned_labels, aligned_predictions, test_scores_adj[detector.Z:])
            
            results['config'] = config
            all_results.append(results)
            
            improvement = ""
            if baseline_results:
                f1_improvement = results['f1'] - baseline_results['f1']
                improvement = f" (Δ: {f1_improvement:+.4f})"
            
            print(f"   🎯 F1: {results['f1']:.4f}{improvement}")
            print(f"   📊 Precision: {results['precision']:.4f}, Recall: {results['recall']:.4f}")
            
            if results['f1'] > best_f1:
                best_f1 = results['f1']
                best_config = config
                
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    if best_config:
        print(f"\n🏆 BEST MSM CONFIG: {best_config['name']} (λ={best_config['lambda']})")
        print(f"📈 Best F1: {best_f1:.4f}")
        
    return all_results, best_config

def visualize_anomaly_scores(scores, labels, title="Anomaly Scores"):
    """
    Visualize anomaly scores and ground truth
    """
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(scores, label='Anomaly Scores', alpha=0.7)
    plt.ylabel('Score')
    plt.title(f'{title} - Anomaly Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(labels, 'r-', linewidth=2, label='True Anomalies')
    plt.ylabel('Anomaly')
    plt.xlabel('Time')
    plt.title('Ground Truth Anomalies')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main testing function
    """
    print("🔍 TS2VEC ANOMALY DETECTION TESTING FRAMEWORK")
    print("="*70)
    
    # Test baseline
    baseline_model, baseline_detector, baseline_results, baseline_scores, test_labels = test_baseline_ts2vec()
    
    if baseline_results is None:
        print("❌ Baseline test failed, cannot proceed")
        return
    
    # Test MSM variants
    msm_results, best_msm_config = test_msm_anomaly_detection(baseline_results)
    
    # Summary
    print("\n📊 FINAL COMPARISON")
    print("="*50)
    print(f"🎯 Baseline TS2Vec F1: {baseline_results['f1']:.4f}")
    
    if best_msm_config and msm_results:
        best_msm_result = max(msm_results, key=lambda x: x['f1'])
        improvement = best_msm_result['f1'] - baseline_results['f1']
        print(f"🚀 Best MSM F1: {best_msm_result['f1']:.4f}")
        print(f"📈 Improvement: {improvement:+.4f}")
        
        if improvement > 0.01:
            print("✅ MSM shows significant improvement for anomaly detection!")
        elif improvement > 0:
            print("🤔 MSM shows slight improvement")
        else:
            print("📉 MSM does not improve anomaly detection")
    
    # Visualize results
    if baseline_scores is not None and test_labels is not None:
        print("\n📈 Generating visualization...")
        visualize_anomaly_scores(baseline_scores, test_labels, "Baseline TS2Vec")

if __name__ == '__main__':
    main()