import os
import numpy as np
import argparse
import time
import datetime
import tasks
import datautils
from ts2vec_msm import TS2VecMSM
from ts2vec import TS2Vec
import torch
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.pipeline import make_pipeline
import json

def train_and_evaluate(dataset, lambda_val, repr_dims=320, epochs=25, batch_size=8, seed=42, device='cuda'):
    """Train and evaluate a single configuration"""
    
    # Set seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    # Load data
    train_data, train_labels, test_data, test_labels = datautils.load_UCR(dataset)
    input_dims = train_data.shape[-1]
    
    # Initialize model based on lambda
    if lambda_val == 0.0:
        model = TS2Vec(
            input_dims=input_dims,
            output_dims=repr_dims,
            device=device,
            lr=0.001,
            batch_size=batch_size,
            max_train_length=None
        )
        n_iters = 200 if train_data.size <= 100000 else 600
        model_type = "baseline_ts2vec"
    else:
        model = TS2VecMSM(
            input_dims=input_dims,
            output_dims=repr_dims,
            device=device,
            lr=0.001,
            batch_size=batch_size,
            max_train_length=None,
            msm_weight=lambda_val,
            msm_mask_rate=0.15,
            msm_decoder_depth=3,
            dynamic_lambda=False
        )
        n_iters = epochs * (len(train_data) // batch_size + 1)
        model_type = "ts2vec_msm"
    
    # Training
    print(f"🚂 Training with λ={lambda_val:.2f}...")
    start_time = time.time()
    
    try:
        loss_log = model.fit(train_data, n_iters=n_iters, verbose=False)
        training_time = time.time() - start_time
        
        # Set to eval mode if available
        if hasattr(model, 'eval'):
            model.eval()
        
        # Evaluation
        train_repr = model.encode(
            train_data,
            causal=False,
            sliding_length=None,
            sliding_padding=0,
            batch_size=batch_size
        )
        
        test_repr = model.encode(
            test_data,
            causal=False,
            sliding_length=None,
            sliding_padding=0,
            batch_size=batch_size
        )
        
        # Flatten representations
        if len(train_repr.shape) == 3:
            train_repr_flat = train_repr.mean(axis=1)
            test_repr_flat = test_repr.mean(axis=1)
        else:
            train_repr_flat = train_repr.reshape(train_repr.shape[0], -1)
            test_repr_flat = test_repr.reshape(test_repr.shape[0], -1)
        
        # Classification
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
        clf.fit(train_repr_flat, train_labels)
        
        test_pred = clf.predict(test_repr_flat)
        accuracy = accuracy_score(test_labels, test_pred)
        
        # Calculate AUPRC
        try:
            proba = clf.predict_proba(test_repr_flat)
            if proba.shape[1] == 2:
                auprc = average_precision_score(test_labels, proba[:, 1])
            else:
                test_labels_onehot = label_binarize(test_labels, classes=np.unique(train_labels))
                if test_labels_onehot.ndim == 1:
                    test_labels_onehot = test_labels_onehot.reshape(-1, 1)
                auprc = average_precision_score(test_labels_onehot, proba, average='weighted')
        except Exception:
            auprc = 0.0
        
        final_loss = float(loss_log[-1]) if len(loss_log) > 0 else None
        
        return {
            'lambda': lambda_val,
            'accuracy': accuracy,
            'auprc': auprc,
            'final_loss': final_loss,
            'training_time': training_time,
            'model_type': model_type,
            'n_iters': n_iters,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'lambda': lambda_val,
            'accuracy': 0.0,
            'auprc': 0.0,
            'final_loss': None,
            'training_time': 0.0,
            'model_type': model_type,
            'n_iters': n_iters,
            'success': False,
            'error': str(e)
        }

def hyperparameter_search(dataset, run_name, lambda_values=None, repr_dims_values=None, 
                         epochs=25, batch_size=8, seed=42):
    """Perform hyperparameter search"""
    
    if lambda_values is None:
        lambda_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    if repr_dims_values is None:
        repr_dims_values = [320]  # Default, can be expanded
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=== TS2Vec-MSM Hyperparameter Tuning ===")
    print(f"Dataset: {dataset}")
    print(f"Lambda values to test: {lambda_values}")
    print(f"Representation dimensions: {repr_dims_values}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Seed: {seed}")
    
    # Load data once to get info
    train_data, train_labels, test_data, test_labels = datautils.load_UCR(dataset)
    print(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}")
    print(f"Classes: {len(np.unique(train_labels))}")
    
    results = []
    total_experiments = len(lambda_values) * len(repr_dims_values)
    experiment_count = 0
    
    print(f"\n🔍 Starting {total_experiments} experiments...")
    
    for repr_dims in repr_dims_values:
        for lambda_val in lambda_values:
            experiment_count += 1
            print(f"\n📊 Experiment {experiment_count}/{total_experiments}: λ={lambda_val:.2f}, dims={repr_dims}")
            
            result = train_and_evaluate(
                dataset=dataset,
                lambda_val=lambda_val,
                repr_dims=repr_dims,
                epochs=epochs,
                batch_size=batch_size,
                seed=seed,
                device=device
            )
            
            result['repr_dims'] = repr_dims
            result['experiment_id'] = experiment_count
            results.append(result)
            
            if result['success']:
                print(f"   ✅ Accuracy: {result['accuracy']:.4f}, AUPRC: {result['auprc']:.4f}, "
                      f"Loss: {result['final_loss']:.6f}, Time: {result['training_time']:.1f}s")
            else:
                print(f"   ❌ Failed: {result['error']}")
    
    # Create results directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"hyperparameter_results/UCR_{dataset}_{run_name}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Convert to DataFrame for easier analysis
    df_results = pd.DataFrame(results)
    
    # Save detailed results
    df_results.to_csv(f"{results_dir}/detailed_results.csv", index=False)
    
    # Save as JSON for easy loading
    with open(f"{results_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Analysis
    successful_results = df_results[df_results['success'] == True]
    
    if len(successful_results) > 0:
        # Find best configurations
        best_accuracy = successful_results.loc[successful_results['accuracy'].idxmax()]
        best_auprc = successful_results.loc[successful_results['auprc'].idxmax()]
        
        print(f"\n=== HYPERPARAMETER TUNING RESULTS ===")
        print(f"📊 Dataset: {dataset}")
        print(f"📈 Total experiments: {total_experiments}")
        print(f"✅ Successful experiments: {len(successful_results)}")
        print(f"❌ Failed experiments: {total_experiments - len(successful_results)}")
        
        print(f"\n🏆 BEST ACCURACY: {best_accuracy['accuracy']:.4f}")
        print(f"   λ = {best_accuracy['lambda']:.2f}")
        print(f"   Repr dims = {best_accuracy['repr_dims']}")
        print(f"   AUPRC = {best_accuracy['auprc']:.4f}")
        print(f"   Final loss = {best_accuracy['final_loss']:.6f}")
        print(f"   Training time = {best_accuracy['training_time']:.1f}s")
        
        print(f"\n🎯 BEST AUPRC: {best_auprc['auprc']:.4f}")
        print(f"   λ = {best_auprc['lambda']:.2f}")
        print(f"   Repr dims = {best_auprc['repr_dims']}")
        print(f"   Accuracy = {best_auprc['accuracy']:.4f}")
        print(f"   Final loss = {best_auprc['final_loss']:.6f}")
        print(f"   Training time = {best_auprc['training_time']:.1f}s")
        
        # Lambda analysis
        lambda_analysis = successful_results.groupby('lambda').agg({
            'accuracy': ['mean', 'std', 'max'],
            'auprc': ['mean', 'std', 'max'],
            'training_time': 'mean'
        }).round(4)
        
        print(f"\n📊 LAMBDA ANALYSIS:")
        print("Lambda\tAcc_Mean\tAcc_Std\tAcc_Max\tAUPRC_Mean\tAUPRC_Std\tAUPRC_Max\tTime_Mean")
        for lambda_val in successful_results['lambda'].unique():
            subset = successful_results[successful_results['lambda'] == lambda_val]
            print(f"{lambda_val:.1f}\t{subset['accuracy'].mean():.4f}\t\t"
                  f"{subset['accuracy'].std():.4f}\t{subset['accuracy'].max():.4f}\t"
                  f"{subset['auprc'].mean():.4f}\t\t{subset['auprc'].std():.4f}\t"
                  f"{subset['auprc'].max():.4f}\t\t{subset['training_time'].mean():.1f}s")
        
        # Save summary
        summary = {
            'dataset': dataset,
            'timestamp': timestamp,
            'total_experiments': total_experiments,
            'successful_experiments': len(successful_results),
            'best_accuracy': {
                'lambda': float(best_accuracy['lambda']),
                'repr_dims': int(best_accuracy['repr_dims']),
                'accuracy': float(best_accuracy['accuracy']),
                'auprc': float(best_accuracy['auprc']),
                'final_loss': float(best_accuracy['final_loss']),
                'training_time': float(best_accuracy['training_time'])
            },
            'best_auprc': {
                'lambda': float(best_auprc['lambda']),
                'repr_dims': int(best_auprc['repr_dims']),
                'accuracy': float(best_auprc['accuracy']),
                'auprc': float(best_auprc['auprc']),
                'final_loss': float(best_auprc['final_loss']),
                'training_time': float(best_auprc['training_time'])
            }
        }
        
        with open(f"{results_dir}/summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_dir}")
        print(f"📁 Files: detailed_results.csv, results.json, summary.json")
        
        # Recommendation
        print(f"\n🎯 RECOMMENDATION:")
        print(f"   For best accuracy: Use λ = {best_accuracy['lambda']:.2f}")
        print(f"   Command: python train_ucr_msm.py {dataset} best_config --msm-weight {best_accuracy['lambda']:.2f} --repr-dims {best_accuracy['repr_dims']} --epochs {epochs}")
        
        return best_accuracy['lambda'], best_accuracy['accuracy'], results_dir
    
    else:
        print("❌ All experiments failed!")
        return None, None, results_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The UCR dataset name')
    parser.add_argument('run_name', help='The run name for saving results')
    parser.add_argument('--lambda-min', type=float, default=0.0, help='Minimum lambda value')
    parser.add_argument('--lambda-max', type=float, default=1.0, help='Maximum lambda value')
    parser.add_argument('--lambda-step', type=float, default=0.1, help='Lambda step size')
    parser.add_argument('--repr-dims', type=int, nargs='+', default=[320], help='Representation dimensions to test')
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer lambda values')
    
    args = parser.parse_args()
    
    # Generate lambda values
    if args.quick:
        lambda_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        print("🚀 Quick mode: Testing 5 lambda values")
    else:
        lambda_values = np.arange(args.lambda_min, args.lambda_max + args.lambda_step, args.lambda_step).round(2).tolist()
        print(f"🔍 Full mode: Testing {len(lambda_values)} lambda values")
    
    # Run hyperparameter search
    best_lambda, best_accuracy, results_dir = hyperparameter_search(
        dataset=args.dataset,
        run_name=args.run_name,
        lambda_values=lambda_values,
        repr_dims_values=args.repr_dims,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed
    )
    
    if best_lambda is not None:
        print(f"\n🎉 Hyperparameter tuning completed!")
        print(f"🏆 Best λ = {best_lambda:.2f} with accuracy = {best_accuracy:.4f}")
        print(f"💾 All results saved to: {results_dir}")
    else:
        print(f"\n❌ Hyperparameter tuning failed!")