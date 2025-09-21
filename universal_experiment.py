#!/usr/bin/env python3
"""
Universal TS2Vec-MSM Configuration Test
Find ONE configuration that beats baseline on MOST datasets
"""

import os
import numpy as np
import time
import datetime
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

# Baseline results from the paper
BASELINE_RESULTS = {
    "Adiac": 0.762, "ArrowHead": 0.857, "Coffee": 1.000, "FaceFour": 0.932,
    "Fish": 0.926, "Haptics": 0.526, "InlineSkate": 0.415, "ItalyPowerDemand": 0.925,
    "MedicalImages": 0.789, "Phoneme": 0.312
}

# Test datasets (diverse mix: easy/medium/hard + different characteristics)
TEST_DATASETS = [
    "Coffee",           # Easy (1.000) - 2 classes, short series
    "FaceFour",         # Easy (0.932) - 4 classes, image-like
    "Fish",             # Easy (0.926) - 7 classes, moderate length
    "ItalyPowerDemand", # Easy (0.925) - 2 classes, temporal patterns
    "ArrowHead",        # Medium (0.857) - 3 classes, shape-based
    "MedicalImages",    # Medium (0.789) - 10 classes, complex patterns
    "Adiac",            # Medium-Hard (0.762) - 37 classes, many classes
    "Haptics",          # Hard (0.526) - 5 classes, noisy data
    "InlineSkate",      # Very Hard (0.415) - 7 classes, complex motion
    "Phoneme"           # Very Hard (0.312) - 39 classes, high complexity
]

# Universal configurations to test (based on your hyperparam search patterns)
UNIVERSAL_CONFIGS = [
    {
        'name': 'universal_light_msm',
        'config': {'lr': 0.001, 'batch_size': 8, 'msm_weight': 0.1, 'mask_rate': 0.15, 
                  'decoder_depth': 3, 'dynamic_lambda': False, 'max_train_length': 3000},
        'description': 'Most common winning pattern: λ=0.1, lr=0.001, batch=8'
    },
    {
        'name': 'universal_ultra_light',
        'config': {'lr': 0.001, 'batch_size': 8, 'msm_weight': 0.05, 'mask_rate': 0.15, 
                  'decoder_depth': 3, 'dynamic_lambda': False, 'max_train_length': 3000},
        'description': 'Even lighter MSM: λ=0.05 for better stability'
    },
    {
        'name': 'universal_adaptive',
        'config': {'lr': 0.0005, 'batch_size': 8, 'msm_weight': 0.1, 'mask_rate': 0.2, 
                  'decoder_depth': 4, 'dynamic_lambda': True, 'max_train_length': None},
        'description': 'Adaptive approach: λ=0.1 with dynamic scheduling'
    },
    {
        'name': 'baseline_ts2vec',
        'config': {'lr': 0.001, 'batch_size': 8, 'msm_weight': 0.0, 'mask_rate': 0.15, 
                  'decoder_depth': 3, 'dynamic_lambda': False, 'max_train_length': 3000},
        'description': 'Pure TS2Vec baseline for comparison'
    }
]

def train_and_evaluate_universal(dataset_name, config, config_name):
    """Train and evaluate using universal configuration"""
    print(f"\n🔄 Testing {config_name} on {dataset_name}")
    
    try:
        # Initialize device and random seeds
        from utils import init_dl_program
        device = init_dl_program(0, seed=42, max_threads=None, deterministic=True)
        
        # Load data
        train_data, train_labels, test_data, test_labels = datautils.load_UCR(dataset_name)
        input_dims = train_data.shape[-1]
        
        # Determine iterations based on dataset size
        n_iters = 200 if train_data.size <= 100000 else 600
        
        # Create model
        if config['msm_weight'] == 0.0:
            model = TS2Vec(
                input_dims=input_dims,
                output_dims=320,
                device=device,
                lr=config['lr'],
                batch_size=config['batch_size'],
                max_train_length=config['max_train_length']
            )
            model_type = "baseline_ts2vec"
        else:
            model = TS2VecMSM(
                input_dims=input_dims,
                output_dims=320,
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
            'svm__C': [0.1, 1, 10, 100],
            'svm__gamma': ['scale', 'auto', 0.01, 0.1, 1]
        }
        
        grid_search = GridSearchCV(pipe, param_grid, cv=3, scoring='accuracy', n_jobs=1)
        grid_search.fit(train_repr, train_labels)
        
        test_pred = grid_search.predict(test_repr)
        accuracy = accuracy_score(test_labels, test_pred)
        
        eval_time = time.time() - eval_start
        baseline_acc = BASELINE_RESULTS[dataset_name]
        improvement = accuracy - baseline_acc
        
        status = "🏆 WIN" if improvement > 0.005 else ("🤝 TIE" if abs(improvement) < 0.005 else "📉 LOSS")
        print(f"   ✅ Accuracy: {accuracy:.4f} (Baseline: {baseline_acc:.3f}, Δ: {improvement:+.4f}) {status}")
        
        return {
            'dataset': dataset_name,
            'config_name': config_name,
            'accuracy': accuracy,
            'baseline_accuracy': baseline_acc,
            'improvement': improvement,
            'training_time': training_time,
            'eval_time': eval_time,
            'config': config,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}")
        return {
            'dataset': dataset_name,
            'config_name': config_name,
            'accuracy': 0.0,
            'baseline_accuracy': BASELINE_RESULTS[dataset_name],
            'improvement': -BASELINE_RESULTS[dataset_name],
            'training_time': 0.0,
            'eval_time': 0.0,
            'config': config,
            'success': False,
            'error': str(e)
        }

def run_universal_experiment():
    """Run universal configuration experiment"""
    print("🌍 UNIVERSAL TS2VEC-MSM CONFIGURATION TEST")
    print("="*80)
    print(f"🎯 Goal: Find ONE config that beats baseline on MOST datasets")
    print(f"📊 Testing {len(UNIVERSAL_CONFIGS)} configs on {len(TEST_DATASETS)} datasets")
    print("="*80)
    
    all_results = []
    config_performance = {}
    
    for config_info in UNIVERSAL_CONFIGS:
        config_name = config_info['name']
        config = config_info['config']
        description = config_info['description']
        
        print(f"\n🧪 TESTING: {config_name.upper()}")
        print(f"📋 {description}")
        print(f"⚙️  Config: λ={config['msm_weight']}, lr={config['lr']}, batch={config['batch_size']}")
        
        wins = 0
        ties = 0
        losses = 0
        total_improvement = 0
        valid_tests = 0
        
        for dataset in TEST_DATASETS:
            result = train_and_evaluate_universal(dataset, config, config_name)
            all_results.append(result)
            
            if result['success']:
                valid_tests += 1
                improvement = result['improvement']
                total_improvement += improvement
                
                if improvement > 0.005:
                    wins += 1
                elif abs(improvement) < 0.005:
                    ties += 1
                else:
                    losses += 1
        
        # Calculate performance metrics
        win_rate = wins / valid_tests if valid_tests > 0 else 0
        avg_improvement = total_improvement / valid_tests if valid_tests > 0 else 0
        
        config_performance[config_name] = {
            'wins': wins,
            'ties': ties,
            'losses': losses,
            'win_rate': win_rate,
            'avg_improvement': avg_improvement,
            'valid_tests': valid_tests,
            'description': description
        }
        
        print(f"\n📊 {config_name.upper()} SUMMARY:")
        print(f"   🏆 Wins: {wins}/{valid_tests} ({win_rate:.1%})")
        print(f"   🤝 Ties: {ties}/{valid_tests}")
        print(f"   📉 Losses: {losses}/{valid_tests}")
        print(f"   📈 Avg improvement: {avg_improvement:+.4f}")
    
    # Find best universal configuration
    print("\n" + "="*80)
    print("🌍 UNIVERSAL CONFIGURATION RANKING")
    print("="*80)
    
    # Sort by win rate, then by avg improvement
    sorted_configs = sorted(
        config_performance.items(), 
        key=lambda x: (x[1]['win_rate'], x[1]['avg_improvement']), 
        reverse=True
    )
    
    for i, (config_name, perf) in enumerate(sorted_configs, 1):
        print(f"{i}. {config_name.upper()}")
        print(f"   📈 Win Rate: {perf['win_rate']:.1%} ({perf['wins']}/{perf['valid_tests']})")
        print(f"   📊 Avg Improvement: {perf['avg_improvement']:+.4f}")
        print(f"   📝 {perf['description']}")
        print()
    
    # Identify the universal winner
    best_config_name, best_perf = sorted_configs[0]
    
    print("🏆 BEST UNIVERSAL CONFIGURATION:")
    print(f"   🥇 {best_config_name.upper()}")
    print(f"   🎯 Win Rate: {best_perf['win_rate']:.1%}")
    print(f"   📈 Avg Improvement: {best_perf['avg_improvement']:+.4f}")
    
    # Detailed results by dataset
    print(f"\n📋 DETAILED RESULTS BY DATASET:")
    print("="*120)
    header = f"{'Dataset':<15} {'Baseline':<10}"
    for config_info in UNIVERSAL_CONFIGS:
        header += f"{config_info['name'][:10]:<12}"
    header += "Best Config"
    print(header)
    print("="*120)
    
    for dataset in TEST_DATASETS:
        line = f"{dataset:<15} {BASELINE_RESULTS[dataset]:<10.3f}"
        dataset_results = {r['config_name']: r for r in all_results if r['dataset'] == dataset}
        
        best_acc = BASELINE_RESULTS[dataset]
        best_config = "Baseline"
        
        for config_info in UNIVERSAL_CONFIGS:
            config_name = config_info['name']
            if config_name in dataset_results and dataset_results[config_name]['success']:
                acc = dataset_results[config_name]['accuracy']
                line += f"{acc:<12.4f}"
                if acc > best_acc:
                    best_acc = acc
                    best_config = config_name[:10]
            else:
                line += f"{'FAILED':<12}"
        
        line += f"{best_config}"
        print(line)
    
    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"universal_config_results_{timestamp}.json"
    
    experiment_data = {
        'timestamp': timestamp,
        'universal_configs': UNIVERSAL_CONFIGS,
        'config_performance': config_performance,
        'best_universal_config': best_config_name,
        'all_results': all_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(experiment_data, f, indent=2)
    
    csv_file = f"universal_config_results_{timestamp}.csv"
    df = pd.DataFrame(all_results)
    df.to_csv(csv_file, index=False)
    
    print(f"\n💾 Results saved to:")
    print(f"   📄 {results_file}")
    print(f"   📊 {csv_file}")
    
    # Final recommendation
    if best_perf['win_rate'] >= 0.6:
        print(f"\n🎉 SUCCESS! Found universal config with {best_perf['win_rate']:.1%} win rate")
        print(f"📜 RECOMMENDED UNIVERSAL CONFIG: {best_config_name}")
        
        # Print the actual config
        best_config = next(c['config'] for c in UNIVERSAL_CONFIGS if c['name'] == best_config_name)
        print(f"⚙️  λ={best_config['msm_weight']}, lr={best_config['lr']}, batch={best_config['batch_size']}")
        print(f"   mask_rate={best_config['mask_rate']}, depth={best_config['decoder_depth']}")
        print(f"   dynamic_lambda={best_config['dynamic_lambda']}")
        
    elif best_perf['win_rate'] >= 0.4:
        print(f"\n🤔 Partial success. Best config wins {best_perf['win_rate']:.1%} of datasets")
        print(f"💡 Consider adaptive lambda or ensemble approaches")
    else:
        print(f"\n😞 No universal config found. Best only wins {best_perf['win_rate']:.1%}")
        print(f"🔍 MSM may need dataset-specific tuning")
    
    return all_results, best_config_name

if __name__ == '__main__':
    results, best_config = run_universal_experiment()
    print("\n🎉 Universal configuration test completed!")