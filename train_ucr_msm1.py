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
import json
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.preprocessing import label_binarize

def train_and_evaluate_combination(train_data, train_labels, test_data, test_labels, 
                                 input_dims, repr_dims, device, config, n_iters):
    """Train and evaluate a single hyperparameter combination"""
    print(f"\n🔄 Testing: lr={config['lr']}, batch_size={config['batch_size']}, "
          f"msm_weight={config['msm_weight']}, mask_rate={config['mask_rate']}, "
          f"decoder_depth={config['decoder_depth']}, dynamic_lambda={config['dynamic_lambda']}")
    
    try:
        # Create model with current configuration
        if config['msm_weight'] == 0.0:
            model = TS2Vec(
                input_dims=input_dims,
                output_dims=repr_dims,
                device=device,
                lr=config['lr'],
                batch_size=config['batch_size'],
                max_train_length=config['max_train_length']
            )
            model_type = "baseline_ts2vec"
        else:
            model = TS2VecMSM(
                input_dims=input_dims,
                output_dims=repr_dims,
                device=device,
                lr=config['lr'],
                batch_size=config['batch_size'],
                max_train_length=config['max_train_length'],
                msm_weight=config['msm_weight'],
                msm_mask_rate=config['mask_rate'],
                msm_decoder_depth=config['decoder_depth'],
                dynamic_lambda=config['dynamic_lambda']
            )
            model_type = "ts2vec_msm"
        
        # Training
        start_time = time.time()
        loss_log = model.fit(train_data, n_iters=n_iters, verbose=False)
        training_time = time.time() - start_time
        
        # Set to eval mode
        if hasattr(model, 'eval'):
            model.eval()
        
        # Evaluation
        eval_start = time.time()
        train_repr = model.encode(train_data, encoding_window='full_series')
        test_repr = model.encode(test_data, encoding_window='full_series')
        
        # Handle 3D representations
        if len(train_repr.shape) == 3:
            train_repr = train_repr.reshape(train_repr.shape[0], -1)
            test_repr = test_repr.reshape(test_repr.shape[0], -1)
        
        # SVM evaluation
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(probability=True))
        ])
        
        param_grid = {
            'svm__C': [0.01, 0.1, 1, 10, 100],
            'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
        }
        
        grid_search = GridSearchCV(pipe, param_grid, cv=3, scoring='accuracy', n_jobs=1)
        grid_search.fit(train_repr, train_labels)
        
        test_pred = grid_search.predict(test_repr)
        test_proba = grid_search.predict_proba(test_repr)
        
        accuracy = accuracy_score(test_labels, test_pred)
        
        # Calculate AUPRC
        try:
            test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max()+1))
            auprc = average_precision_score(test_labels_onehot, test_proba)
        except Exception:
            auprc = 0.0
        
        eval_time = time.time() - eval_start
        final_loss = float(loss_log[-1]) if len(loss_log) > 0 else None
        
        result = {
            'lr': config['lr'],
            'batch_size': config['batch_size'],
            'msm_weight': config['msm_weight'],
            'mask_rate': config['mask_rate'],
            'decoder_depth': config['decoder_depth'],
            'dynamic_lambda': config['dynamic_lambda'],
            'max_train_length': config['max_train_length'],
            'accuracy': accuracy,
            'auprc': auprc,
            'final_loss': final_loss,
            'training_time': training_time,
            'eval_time': eval_time,
            'model_type': model_type,
            'success': True,
            'error': None
        }
        
        print(f"   ✅ Accuracy: {accuracy:.4f}, AUPRC: {auprc:.4f}, "
              f"Loss: {final_loss:.6f}, Time: {training_time:.1f}s")
        
        return result
        
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}")
        result = {
            'lr': config['lr'],
            'batch_size': config['batch_size'],
            'msm_weight': config['msm_weight'],
            'mask_rate': config['mask_rate'],
            'decoder_depth': config['decoder_depth'],
            'dynamic_lambda': config['dynamic_lambda'],
            'max_train_length': config['max_train_length'],
            'accuracy': 0.0,
            'auprc': 0.0,
            'final_loss': None,
            'training_time': 0.0,
            'eval_time': 0.0,
            'model_type': 'failed',
            'success': False,
            'error': str(e)
        }
        return result

def run_hyperparameter_search(dataset_name):
    """Run comprehensive hyperparameter search"""
    print(f"🚀 HYPERPARAMETER SEARCH FOR {dataset_name}")
    print("=" * 80)
    
    # Initialize device and random seeds
    from utils import init_dl_program
    device = init_dl_program(0, seed=42, max_threads=None, deterministic=True)
    
    # Load data
    print("Loading UCR data... ", end="")
    train_data, train_labels, test_data, test_labels = datautils.load_UCR(dataset_name)
    print("done")
    
    input_dims = train_data.shape[-1]
    repr_dims = 320
    n_iters = 200 if train_data.size <= 100000 else 600
    
    print(f"Dataset: {dataset_name}")
    print(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}")
    print(f"Classes: {len(np.unique(train_labels))}")
    print(f"Training iterations: {n_iters}")
    
    # Define hyperparameter combinations to test
    hyperparameter_combinations = [
        # Baseline (λ=0.0) with different settings
        {'lr': 0.001, 'batch_size': 8, 'msm_weight': 0.0, 'mask_rate': 0.15, 'decoder_depth': 3, 'dynamic_lambda': False, 'max_train_length': 3000},
        {'lr': 0.0005, 'batch_size': 8, 'msm_weight': 0.0, 'mask_rate': 0.15, 'decoder_depth': 3, 'dynamic_lambda': False, 'max_train_length': None},
        {'lr': 0.001, 'batch_size': 16, 'msm_weight': 0.0, 'mask_rate': 0.15, 'decoder_depth': 3, 'dynamic_lambda': False, 'max_train_length': 3000},
        
        # Light MSM (λ=0.1-0.3)
        {'lr': 0.001, 'batch_size': 8, 'msm_weight': 0.1, 'mask_rate': 0.15, 'decoder_depth': 3, 'dynamic_lambda': False, 'max_train_length': 3000},
        {'lr': 0.0005, 'batch_size': 8, 'msm_weight': 0.1, 'mask_rate': 0.20, 'decoder_depth': 4, 'dynamic_lambda': True, 'max_train_length': None},
        {'lr': 0.001, 'batch_size': 8, 'msm_weight': 0.3, 'mask_rate': 0.15, 'decoder_depth': 3, 'dynamic_lambda': False, 'max_train_length': 3000},
        {'lr': 0.0005, 'batch_size': 4, 'msm_weight': 0.3, 'mask_rate': 0.25, 'decoder_depth': 4, 'dynamic_lambda': True, 'max_train_length': None},
        
        # Moderate MSM (λ=0.5)
        {'lr': 0.001, 'batch_size': 8, 'msm_weight': 0.5, 'mask_rate': 0.15, 'decoder_depth': 3, 'dynamic_lambda': False, 'max_train_length': 3000},
        {'lr': 0.0005, 'batch_size': 8, 'msm_weight': 0.5, 'mask_rate': 0.15, 'decoder_depth': 3, 'dynamic_lambda': True, 'max_train_length': 3000},
        {'lr': 0.0005, 'batch_size': 4, 'msm_weight': 0.5, 'mask_rate': 0.25, 'decoder_depth': 4, 'dynamic_lambda': True, 'max_train_length': None},
        {'lr': 0.001, 'batch_size': 16, 'msm_weight': 0.5, 'mask_rate': 0.20, 'decoder_depth': 3, 'dynamic_lambda': False, 'max_train_length': 3000},
        
        # Heavy MSM (λ=0.7)
        {'lr': 0.0005, 'batch_size': 8, 'msm_weight': 0.7, 'mask_rate': 0.20, 'decoder_depth': 4, 'dynamic_lambda': True, 'max_train_length': None},
        {'lr': 0.001, 'batch_size': 8, 'msm_weight': 0.7, 'mask_rate': 0.15, 'decoder_depth': 3, 'dynamic_lambda': False, 'max_train_length': 3000},
        
        # Advanced configurations
        {'lr': 0.0005, 'batch_size': 4, 'msm_weight': 0.3, 'mask_rate': 0.30, 'decoder_depth': 5, 'dynamic_lambda': True, 'max_train_length': None},
        {'lr': 0.0005, 'batch_size': 4, 'msm_weight': 0.5, 'mask_rate': 0.35, 'decoder_depth': 4, 'dynamic_lambda': True, 'max_train_length': None},
        {'lr': 0.005, 'batch_size': 8, 'msm_weight': 0.1, 'mask_rate': 0.10, 'decoder_depth': 2, 'dynamic_lambda': False, 'max_train_length': 3000},
        
        # Exploration configurations
        {'lr': 0.0001, 'batch_size': 8, 'msm_weight': 0.3, 'mask_rate': 0.20, 'decoder_depth': 3, 'dynamic_lambda': True, 'max_train_length': None},
        {'lr': 0.002, 'batch_size': 16, 'msm_weight': 0.2, 'mask_rate': 0.15, 'decoder_depth': 3, 'dynamic_lambda': False, 'max_train_length': 3000},
    ]
    
    all_results = []
    
    print(f"\n🧪 Testing {len(hyperparameter_combinations)} hyperparameter combinations...")
    
    for i, config in enumerate(hyperparameter_combinations, 1):
        print(f"\n[{i}/{len(hyperparameter_combinations)}] Configuration {i}:")
        
        result = train_and_evaluate_combination(
            train_data, train_labels, test_data, test_labels,
            input_dims, repr_dims, device, config, n_iters
        )
        
        result['config_id'] = i
        result['dataset'] = dataset_name
        all_results.append(result)
        
        # Save intermediate results after each combination
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"hyperparameter_results_{dataset_name}_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                'dataset': dataset_name,
                'timestamp': timestamp,
                'total_combinations': len(hyperparameter_combinations),
                'completed_combinations': i,
                'results': all_results
            }, f, indent=2)
        
        # Print running best
        successful_results = [r for r in all_results if r['success']]
        if successful_results:
            best_result = max(successful_results, key=lambda x: x['accuracy'])
            print(f"   📊 Current best: Config {best_result['config_id']} with accuracy {best_result['accuracy']:.4f}")
    
    # Final analysis
    print("\n" + "=" * 80)
    print("📊 HYPERPARAMETER SEARCH RESULTS")
    print("=" * 80)
    
    successful_results = [r for r in all_results if r['success']]
    failed_results = [r for r in all_results if not r['success']]
    
    print(f"✅ Successful configurations: {len(successful_results)}/{len(all_results)}")
    print(f"❌ Failed configurations: {len(failed_results)}")
    
    if successful_results:
        # Sort by accuracy
        successful_results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        print(f"\n🏆 TOP 5 CONFIGURATIONS:")
        for i, result in enumerate(successful_results[:5], 1):
            print(f"   {i}. Config {result['config_id']}: Accuracy = {result['accuracy']:.4f}, AUPRC = {result['auprc']:.4f}")
            print(f"      λ={result['msm_weight']}, lr={result['lr']}, batch={result['batch_size']}, "
                  f"mask={result['mask_rate']}, depth={result['decoder_depth']}, dynamic={result['dynamic_lambda']}")
        
        # Best configuration details
        best = successful_results[0]
        print(f"\n🎯 BEST CONFIGURATION (Config {best['config_id']}):")
        print(f"   Accuracy: {best['accuracy']:.4f}")
        print(f"   AUPRC: {best['auprc']:.4f}")
        print(f"   MSM Weight (λ): {best['msm_weight']}")
        print(f"   Learning Rate: {best['lr']}")
        print(f"   Batch Size: {best['batch_size']}")
        print(f"   Mask Rate: {best['mask_rate']}")
        print(f"   Decoder Depth: {best['decoder_depth']}")
        print(f"   Dynamic Lambda: {best['dynamic_lambda']}")
        print(f"   Max Train Length: {best['max_train_length']}")
        print(f"   Training Time: {best['training_time']:.1f}s")
        print(f"   Final Loss: {best['final_loss']:.6f}")
        
        # Analysis by MSM weight
        lambda_analysis = {}
        for result in successful_results:
            lambda_val = result['msm_weight']
            if lambda_val not in lambda_analysis:
                lambda_analysis[lambda_val] = []
            lambda_analysis[lambda_val].append(result['accuracy'])
        
        print(f"\n📈 ANALYSIS BY MSM WEIGHT (λ):")
        for lambda_val in sorted(lambda_analysis.keys()):
            accuracies = lambda_analysis[lambda_val]
            avg_acc = np.mean(accuracies)
            max_acc = np.max(accuracies)
            print(f"   λ={lambda_val}: Avg={avg_acc:.4f}, Max={max_acc:.4f} ({len(accuracies)} configs)")
    
    # Save final results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    final_results_file = f"final_hyperparameter_results_{dataset_name}_{timestamp}.json"
    
    final_data = {
        'dataset': dataset_name,
        'timestamp': timestamp,
        'total_combinations': len(hyperparameter_combinations),
        'successful_combinations': len(successful_results),
        'failed_combinations': len(failed_results),
        'all_results': all_results,
        'best_result': successful_results[0] if successful_results else None,
        'lambda_analysis': lambda_analysis if successful_results else None
    }
    
    with open(final_results_file, 'w') as f:
        json.dump(final_data, f, indent=2)
    
    # Save as CSV for easy analysis
    df = pd.DataFrame(all_results)
    csv_file = f"hyperparameter_results_{dataset_name}_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    
    print(f"\n💾 Results saved to:")
    print(f"   JSON: {final_results_file}")
    print(f"   CSV: {csv_file}")
    
    return all_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The UCR dataset name')
    args = parser.parse_args()
    
    results = run_hyperparameter_search(args.dataset)
    print("\n🎉 Hyperparameter search completed!")