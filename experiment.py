#!/usr/bin/env python3
"""
TS2Vec-MSM vs Baseline Experiment
Tests optimized MSM configurations on 10 selected UCR datasets
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
    "Adiac": 0.762, "ArrowHead": 0.857, "Beef": 0.767, "BeetleFly": 0.900,
    "BirdChicken": 0.800, "Car": 0.833, "CBF": 1.000, "ChlorineConcentration": 0.832,
    "CinCECGTorso": 0.827, "Coffee": 1.000, "Computers": 0.660, "CricketX": 0.782,
    "CricketY": 0.749, "CricketZ": 0.792, "DiatomSizeReduction": 0.984,
    "DistalPhalanxOutlineCorrect": 0.761, "DistalPhalanxOutlineAgeGroup": 0.727,
    "DistalPhalanxTW": 0.698, "Earthquakes": 0.748, "ECG200": 0.920,
    "ECG5000": 0.935, "ECGFiveDays": 1.000, "ElectricDevices": 0.721,
    "FaceAll": 0.771, "FaceFour": 0.932, "FacesUCR": 0.924, "FiftyWords": 0.771,
    "Fish": 0.926, "FordA": 0.936, "FordB": 0.794, "GunPoint": 0.980,
    "Ham": 0.714, "HandOutlines": 0.922, "Haptics": 0.526, "Herring": 0.641,
    "InlineSkate": 0.415, "InsectWingbeatSound": 0.630, "ItalyPowerDemand": 0.925,
    "LargeKitchenAppliances": 0.845, "Lightning2": 0.869, "Lightning7": 0.863,
    "Mallat": 0.914, "Meat": 0.950, "MedicalImages": 0.789,
    "MiddlePhalanxOutlineCorrect": 0.838, "MiddlePhalanxOutlineAgeGroup": 0.636,
    "MiddlePhalanxTW": 0.584, "MoteStrain": 0.861, "NonInvasiveFetalECGThorax1": 0.930,
    "NonInvasiveFetalECGThorax2": 0.938, "OliveOil": 0.900, "OSULeaf": 0.851,
    "PhalangesOutlinesCorrect": 0.809, "Phoneme": 0.312, "Plane": 1.000,
    "ProximalPhalanxOutlineCorrect": 0.887, "ProximalPhalanxOutlineAgeGroup": 0.834,
    "ProximalPhalanxTW": 0.824, "RefrigerationDevices": 0.589, "ScreenType": 0.411,
    "ShapeletSim": 1.000, "ShapesAll": 0.902, "SmallKitchenAppliances": 0.731,
    "SonyAIBORobotSurface1": 0.903, "SonyAIBORobotSurface2": 0.871,
    "StarLightCurves": 0.969, "Strawberry": 0.962, "SwedishLeaf": 0.941,
    "Symbols": 0.976, "SyntheticControl": 0.997, "ToeSegmentation1": 0.917,
    "ToeSegmentation2": 0.892, "Trace": 1.000, "TwoLeadECG": 0.986,
    "TwoPatterns": 1.000, "UWaveGestureLibraryX": 0.795, "UWaveGestureLibraryY": 0.719,
    "UWaveGestureLibraryZ": 0.770, "UWaveGestureLibraryAll": 0.930, "Wafer": 0.998,
    "Wine": 0.870, "WordSynonyms": 0.676, "Worms": 0.701, "WormsTwoClass": 0.805,
    "Yoga": 0.887
}

# Best configurations from hyperparameter search
BEST_CONFIGS = {
    'light_msm': {
        'lr': 0.001, 'batch_size': 8, 'msm_weight': 0.1,
        'mask_rate': 0.15, 'decoder_depth': 3, 'dynamic_lambda': False,
        'max_train_length': 3000
    },
    'adaptive_msm': {
        'lr': 0.0005, 'batch_size': 4, 'msm_weight': 0.3,
        'mask_rate': 0.25, 'decoder_depth': 4, 'dynamic_lambda': True,
        'max_train_length': None
    }
}

# Selected 10 datasets for testing (diverse difficulty levels)
SELECTED_DATASETS = [
    "ProximalPhalanxTW",         
    "Computers",      
    "CricketZ",      
    "OSULeaf",         
    "GunPoint",       
    "Lightning2",     
    "FiftyWords",           
    "ShapesAll",      
    "Wine",          
    "Phoneme" 
]

def train_and_evaluate(dataset_name, config_name, config, baseline_acc):
    """Train and evaluate a single configuration"""
    print(f"\n🔄 Testing {config_name} on {dataset_name} (Baseline: {baseline_acc:.3f})")
    
    try:
        # Initialize device and random seeds
        from utils import init_dl_program
        device = init_dl_program(0, seed=42, max_threads=None, deterministic=True)
        
        # Load data
        train_data, train_labels, test_data, test_labels = datautils.load_UCR(dataset_name)
        input_dims = train_data.shape[-1]
        
        # Determine iterations
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
            model_type = "baseline"
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
            model_type = "msm"
        
        # Training
        print(f"   🚂 Training {model_type} for {n_iters} iterations...")
        start_time = time.time()
        loss_log = model.fit(train_data, n_iters=n_iters, verbose=False)
        training_time = time.time() - start_time
        
        # Set to eval mode
        if hasattr(model, 'eval'):
            model.eval()
        
        # Evaluation
        print(f"   📊 Evaluating...")
        eval_start = time.time()
        train_repr = model.encode(train_data, encoding_window='full_series')
        test_repr = model.encode(test_data, encoding_window='full_series')
        
        # Handle 3D representations
        if len(train_repr.shape) == 3:
            train_repr = train_repr.reshape(train_repr.shape[0], -1)
            test_repr = test_repr.reshape(test_repr.shape[0], -1)
        
        # SVM evaluation with reduced parameter grid for speed
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
        improvement = accuracy - baseline_acc
        
        print(f"   ✅ Accuracy: {accuracy:.4f} (Baseline: {baseline_acc:.3f}, Δ: {improvement:+.4f})")
        print(f"   📈 AUPRC: {auprc:.4f}")
        print(f"   ⏱️  Training: {training_time:.1f}s, Eval: {eval_time:.1f}s")
        
        return {
            'dataset': dataset_name,
            'config_name': config_name,
            'accuracy': accuracy,
            'auprc': auprc,
            'baseline_accuracy': baseline_acc,
            'improvement': improvement,
            'final_loss': final_loss,
            'training_time': training_time,
            'eval_time': eval_time,
            'model_type': model_type,
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
            'auprc': 0.0,
            'baseline_accuracy': baseline_acc,
            'improvement': -baseline_acc,
            'final_loss': None,
            'training_time': 0.0,
            'eval_time': 0.0,
            'model_type': 'failed',
            'config': config,
            'success': False,
            'error': str(e)
        }

def run_experiment():
    """Run the complete experiment"""
    print("🚀 TS2VEC-MSM vs BASELINE EXPERIMENT")
    print("=" * 80)
    print(f"📊 Testing {len(SELECTED_DATASETS)} datasets with {len(BEST_CONFIGS)} configurations")
    print(f"🎯 Goal: See how many datasets MSM outperforms baseline")
    print("=" * 80)
    
    all_results = []
    wins_by_config = {config_name: 0 for config_name in BEST_CONFIGS.keys()}
    
    for i, dataset in enumerate(SELECTED_DATASETS, 1):
        baseline_acc = BASELINE_RESULTS[dataset]
        print(f"\n📋 [{i}/{len(SELECTED_DATASETS)}] Dataset: {dataset}")
        
        # Test each configuration
        for config_name, config in BEST_CONFIGS.items():
            result = train_and_evaluate(dataset, config_name, config, baseline_acc)
            all_results.append(result)
            
            # Count wins
            if result['success'] and result['improvement'] > 0:
                wins_by_config[config_name] += 1
                print(f"   🏆 {config_name} WINS! (+{result['improvement']:.4f})")
            elif result['success']:
                print(f"   📉 {config_name} loses ({result['improvement']:.4f})")
    
    # Analysis
    print("\n" + "=" * 80)
    print("📊 EXPERIMENT RESULTS")
    print("=" * 80)
    
    successful_results = [r for r in all_results if r['success']]
    failed_results = [r for r in all_results if not r['success']]
    
    print(f"✅ Successful experiments: {len(successful_results)}/{len(all_results)}")
    print(f"❌ Failed experiments: {len(failed_results)}")
    
    if successful_results:
        # Results by configuration
        for config_name in BEST_CONFIGS.keys():
            config_results = [r for r in successful_results if r['config_name'] == config_name]
            if config_results:
                wins = wins_by_config[config_name]
                win_rate = wins / len(config_results)
                avg_improvement = np.mean([r['improvement'] for r in config_results])
                avg_accuracy = np.mean([r['accuracy'] for r in config_results])
                
                print(f"\n🎯 {config_name.upper()}:")
                print(f"   🏆 Wins: {wins}/{len(config_results)} ({win_rate:.1%})")
                print(f"   📈 Average improvement: {avg_improvement:+.4f}")
                print(f"   📊 Average accuracy: {avg_accuracy:.4f}")
        
        # Best performing datasets for MSM
        msm_improvements = []
        for r in successful_results:
            if r['config_name'] != 'baseline' and r['improvement'] > 0:
                msm_improvements.append(r)
        
        if msm_improvements:
            msm_improvements.sort(key=lambda x: x['improvement'], reverse=True)
            print(f"\n🏆 TOP MSM IMPROVEMENTS:")
            for i, r in enumerate(msm_improvements[:5], 1):
                print(f"   {i}. {r['dataset']} ({r['config_name']}): +{r['improvement']:.4f} "
                      f"({r['accuracy']:.4f} vs {r['baseline_accuracy']:.4f})")
        
        # Detailed comparison table
        print(f"\n📋 DETAILED RESULTS:")
        print("=" * 100)
        print(f"{'Dataset':<20} {'Baseline':<10} {'Light MSM':<12} {'Adaptive MSM':<15} {'Best Method':<15}")
        print("=" * 100)
        
        for dataset in SELECTED_DATASETS:
            baseline_acc = BASELINE_RESULTS[dataset]
            dataset_results = {r['config_name']: r for r in successful_results if r['dataset'] == dataset}
            
            light_msm_acc = dataset_results.get('light_msm', {}).get('accuracy', 0.0)
            adaptive_msm_acc = dataset_results.get('adaptive_msm', {}).get('accuracy', 0.0)
            
            best_acc = max(baseline_acc, light_msm_acc, adaptive_msm_acc)
            if best_acc == baseline_acc:
                best_method = "Baseline"
            elif best_acc == light_msm_acc:
                best_method = "Light MSM"
            else:
                best_method = "Adaptive MSM"
            
            print(f"{dataset:<20} {baseline_acc:<10.4f} {light_msm_acc:<12.4f} "
                  f"{adaptive_msm_acc:<15.4f} {best_method:<15}")
    
    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"experiment_results_{timestamp}.json"
    
    experiment_data = {
        'timestamp': timestamp,
        'datasets': SELECTED_DATASETS,
        'configurations': BEST_CONFIGS,
        'wins_by_config': wins_by_config,
        'all_results': all_results,
        'summary': {
            'total_experiments': len(all_results),
            'successful_experiments': len(successful_results),
            'failed_experiments': len(failed_results)
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(experiment_data, f, indent=2)
    
    # Save as CSV
    csv_file = f"experiment_results_{timestamp}.csv"
    df = pd.DataFrame(all_results)
    df.to_csv(csv_file, index=False)
    
    print(f"\n💾 Results saved to:")
    print(f"   📄 {results_file}")
    print(f"   📊 {csv_file}")
    
    # Final recommendation
    print(f"\n🎯 CONCLUSION:")
    total_wins = sum(wins_by_config.values())
    total_tested = len(SELECTED_DATASETS) * len(BEST_CONFIGS)
    
    if total_wins > len(SELECTED_DATASETS):
        print(f"🎉 MSM approach shows promise! {total_wins} wins out of {total_tested} experiments")
        print(f"📈 MSM outperformed baseline on {total_wins}/{len(SELECTED_DATASETS)} datasets")
    else:
        print(f"🤔 MSM needs more tuning. Only {total_wins} wins out of {total_tested} experiments")
        print(f"💡 Consider trying λ=0.05 or ensemble approaches")
    
    return all_results

if __name__ == '__main__':
    results = run_experiment()
    print("\n🎉 Experiment completed!")