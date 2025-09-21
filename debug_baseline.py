#!/usr/bin/env python3
"""
Debug Baseline: Test if λ=0.0 reproduces paper results
This will help isolate if the problem is in MSM component or base implementation
"""

import numpy as np
import time
import datautils
from ts2vec_msm import TS2VecMSM
from ts2vec import TS2Vec
import torch
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pandas as pd

# Test datasets from our experiment
TEST_DATASETS = ["GunPoint", "CricketZ", "Wine", "Computers", "ProximalPhalanxTW"]

BASELINE_RESULTS = {
    "GunPoint": 0.980,
    "CricketZ": 0.792, 
    "Wine": 0.870,
    "Computers": 0.660,
    "ProximalPhalanxTW": 0.824
}

def test_baseline_reproduction(dataset_name):
    """Test if we can reproduce baseline results with both TS2Vec and TS2VecMSM(λ=0)"""
    print(f"\n🔍 Testing {dataset_name} (Paper baseline: {BASELINE_RESULTS[dataset_name]:.3f})")
    
    # Initialize
    from utils import init_dl_program
    device = init_dl_program(0, seed=42, max_threads=None, deterministic=True)
    
    # Load data
    train_data, train_labels, test_data, test_labels = datautils.load_UCR(dataset_name)
    input_dims = train_data.shape[-1]
    n_iters = 200 if train_data.size <= 100000 else 600
    
    results = {}
    
    # Test 1: Original TS2Vec
    print(f"   🧪 Testing original TS2Vec...")
    try:
        model = TS2Vec(
            input_dims=input_dims,
            output_dims=320,
            device=device,
            lr=0.001,
            batch_size=8,
            max_train_length=3000
        )
        
        model.fit(train_data, n_iters=n_iters, verbose=False)
        model.eval()
        
        train_repr = model.encode(train_data, encoding_window='full_series')
        test_repr = model.encode(test_data, encoding_window='full_series')
        
        if len(train_repr.shape) == 3:
            train_repr = train_repr.reshape(train_repr.shape[0], -1)
            test_repr = test_repr.reshape(test_repr.shape[0], -1)
        
        # Quick SVM eval
        clf = SVC(C=1.0, gamma='scale')
        scaler = StandardScaler()
        train_repr_scaled = scaler.fit_transform(train_repr)
        test_repr_scaled = scaler.transform(test_repr)
        clf.fit(train_repr_scaled, train_labels)
        pred = clf.predict(test_repr_scaled)
        acc = accuracy_score(test_labels, pred)
        
        results['original_ts2vec'] = acc
        print(f"      ✅ Original TS2Vec: {acc:.4f}")
        
    except Exception as e:
        results['original_ts2vec'] = 0.0
        print(f"      ❌ Original TS2Vec failed: {e}")
    
    # Test 2: TS2VecMSM with λ=0.0 (should be equivalent)
    print(f"   🧪 Testing TS2VecMSM with λ=0.0...")
    try:
        model = TS2VecMSM(
            input_dims=input_dims,
            output_dims=320,
            device=device,
            lr=0.001,
            batch_size=8,
            max_train_length=3000,
            msm_weight=0.0,  # Pure contrastive
            msm_mask_rate=0.15,
            msm_decoder_depth=3,
            dynamic_lambda=False
        )
        
        model.fit(train_data, n_iters=n_iters, verbose=False)
        model.eval()
        
        train_repr = model.encode(train_data, encoding_window='full_series')
        test_repr = model.encode(test_data, encoding_window='full_series')
        
        if len(train_repr.shape) == 3:
            train_repr = train_repr.reshape(train_repr.shape[0], -1)
            test_repr = test_repr.reshape(test_repr.shape[0], -1)
        
        # Quick SVM eval
        clf = SVC(C=1.0, gamma='scale')
        scaler = StandardScaler()
        train_repr_scaled = scaler.fit_transform(train_repr)
        test_repr_scaled = scaler.transform(test_repr)
        clf.fit(train_repr_scaled, train_labels)
        pred = clf.predict(test_repr_scaled)
        acc = accuracy_score(test_labels, pred)
        
        results['ts2vec_msm_lambda0'] = acc
        print(f"      ✅ TS2VecMSM(λ=0): {acc:.4f}")
        
    except Exception as e:
        results['ts2vec_msm_lambda0'] = 0.0
        print(f"      ❌ TS2VecMSM(λ=0) failed: {e}")
    
    # Test 3: Very small λ
    print(f"   🧪 Testing TS2VecMSM with λ=0.01...")
    try:
        model = TS2VecMSM(
            input_dims=input_dims,
            output_dims=320,
            device=device,
            lr=0.001,
            batch_size=8,
            max_train_length=3000,
            msm_weight=0.01,  # Very small MSM
            msm_mask_rate=0.15,
            msm_decoder_depth=3,
            dynamic_lambda=False
        )
        
        model.fit(train_data, n_iters=n_iters, verbose=False)
        model.eval()
        
        train_repr = model.encode(train_data, encoding_window='full_series')
        test_repr = model.encode(test_data, encoding_window='full_series')
        
        if len(train_repr.shape) == 3:
            train_repr = train_repr.reshape(train_repr.shape[0], -1)
            test_repr = test_repr.reshape(test_repr.shape[0], -1)
        
        # Quick SVM eval
        clf = SVC(C=1.0, gamma='scale')
        scaler = StandardScaler()
        train_repr_scaled = scaler.fit_transform(train_repr)
        test_repr_scaled = scaler.transform(test_repr)
        clf.fit(train_repr_scaled, train_labels)
        pred = clf.predict(test_repr_scaled)
        acc = accuracy_score(test_labels, pred)
        
        results['ts2vec_msm_lambda001'] = acc
        print(f"      ✅ TS2VecMSM(λ=0.01): {acc:.4f}")
        
    except Exception as e:
        results['ts2vec_msm_lambda001'] = 0.0
        print(f"      ❌ TS2VecMSM(λ=0.01) failed: {e}")
    
    # Analysis
    paper_baseline = BASELINE_RESULTS[dataset_name]
    print(f"   📊 Paper baseline: {paper_baseline:.4f}")
    
    for method, acc in results.items():
        if acc > 0:
            diff = acc - paper_baseline
            print(f"   📈 {method}: {acc:.4f} (Δ: {diff:+.4f})")
    
    return results

def main():
    print("🔍 BASELINE REPRODUCTION TEST")
    print("=" * 60)
    print("Goal: Verify if our implementation can reproduce paper results")
    print("=" * 60)
    
    all_results = []
    
    for dataset in TEST_DATASETS:
        results = test_baseline_reproduction(dataset)
        results['dataset'] = dataset
        results['paper_baseline'] = BASELINE_RESULTS[dataset]
        all_results.append(results)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    df = pd.DataFrame(all_results)
    
    methods = ['original_ts2vec', 'ts2vec_msm_lambda0', 'ts2vec_msm_lambda001']
    
    for method in methods:
        if method in df.columns:
            successful = df[df[method] > 0]
            if len(successful) > 0:
                avg_acc = successful[method].mean()
                avg_diff = successful[method].mean() - successful['paper_baseline'].mean()
                print(f"\n🎯 {method.upper()}:")
                print(f"   📊 Average accuracy: {avg_acc:.4f}")
                print(f"   📈 Average vs paper: {avg_diff:+.4f}")
                print(f"   ✅ Success rate: {len(successful)}/{len(df)}")
    
    # Check if any method consistently matches paper
    best_gaps = []
    for _, row in df.iterrows():
        dataset = row['dataset']
        paper = row['paper_baseline']
        best_acc = 0
        best_method = None
        
        for method in methods:
            if method in row and row[method] > best_acc:
                best_acc = row[method]
                best_method = method
        
        gap = abs(best_acc - paper) if best_acc > 0 else 1.0
        best_gaps.append(gap)
        
        print(f"\n📋 {dataset}: Best={best_method} ({best_acc:.4f}) vs Paper ({paper:.4f}) Gap={gap:.4f}")
    
    avg_gap = np.mean(best_gaps)
    print(f"\n🎯 CONCLUSION:")
    print(f"   📊 Average gap from paper: {avg_gap:.4f}")
    
    if avg_gap < 0.05:
        print(f"   ✅ Implementation looks good! MSM component likely causing issues.")
        print(f"   💡 Next: Try λ=0.01-0.05 range with curriculum learning")
    elif avg_gap < 0.1:
        print(f"   ⚠️  Some implementation differences, but reasonable.")
        print(f"   💡 Next: Focus on MSM tuning with very small λ values")
    else:
        print(f"   ❌ Significant implementation issues detected.")
        print(f"   💡 Next: Debug base TS2Vec implementation first")
    
    # Save results
    df.to_csv('baseline_reproduction_test.csv', index=False)
    print(f"\n💾 Results saved to baseline_reproduction_test.csv")

if __name__ == '__main__':
    main()