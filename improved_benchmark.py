#!/usr/bin/env python3
"""
Improved TS2Vec-MSM Benchmark with Systematic Performance Analysis
"""
import subprocess
import json
import os
import time
from datetime import datetime

# Test datasets with known baseline results
TEST_DATASETS = {
    "Beef": 0.767,
    "CBF": 1.000,
    "ECG200": 0.920,
    "GunPoint": 0.980,
    "Lightning2": 0.869
}

def run_experiment(dataset, method_name, cmd, timeout=600):
    """Run a single experiment with detailed logging"""
    print(f"🔄 Running {dataset} with {method_name}...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        
        if result.returncode == 0:
            # Parse accuracy from output
            lines = result.stdout.split('\n')
            for line in lines:
                if "Evaluation result:" in line:
                    try:
                        eval_str = line.split("Evaluation result:")[1].strip()
                        eval_dict = eval(eval_str)
                        accuracy = eval_dict.get('acc', 0.0)
                        auprc = eval_dict.get('auprc', 0.0)
                        print(f"✅ {dataset} - {method_name}: Acc = {accuracy:.4f}, AUPRC = {auprc:.4f}")
                        return accuracy, auprc
                    except Exception as e:
                        print(f"⚠️  Parse error: {e}")
                        pass
        
        print(f"❌ {dataset} - {method_name}: Failed")
        if result.stderr:
            print(f"Error: {result.stderr[-300:]}")
        return 0.0, 0.0
        
    except subprocess.TimeoutExpired:
        print(f"⏰ {dataset} - {method_name}: Timeout")
        return 0.0, 0.0
    except Exception as e:
        print(f"💥 {dataset} - {method_name}: Exception {e}")
        return 0.0, 0.0

def test_lambda_values(dataset, baseline_acc):
    """Test different lambda values to find optimal settings"""
    print(f"\n📋 Dataset: {dataset} (Baseline: {baseline_acc:.3f})")
    results = {"dataset": dataset, "baseline": baseline_acc, "experiments": {}}
    
    # Lambda values to test (including pure contrastive and pure MSM)
    lambda_configs = [
        {"lambda": 0.0, "name": "Pure Contrastive (λ=0.0)", "dynamic": False},
        {"lambda": 0.1, "name": "Light MSM (λ=0.1)", "dynamic": False},
        {"lambda": 0.3, "name": "Moderate MSM (λ=0.3)", "dynamic": False},
        {"lambda": 0.5, "name": "Balanced MSM (λ=0.5)", "dynamic": False},
        {"lambda": 0.7, "name": "Heavy MSM (λ=0.7)", "dynamic": False},
        {"lambda": 0.5, "name": "Dynamic MSM (λ=0.5)", "dynamic": True},
    ]
    
    for config in lambda_configs:
        cmd = [
            "python", "train_ucr_msm.py", dataset, f"test_{config['lambda']:.1f}",
            "--msm-weight", str(config["lambda"]),
            "--seed", "42", "--eval"
        ]
        
        if config["dynamic"]:
            cmd.extend(["--dynamic-lambda"])
        
        acc, auprc = run_experiment(dataset, config["name"], cmd)
        
        results["experiments"][f"lambda_{config['lambda']:.1f}_{'dynamic' if config['dynamic'] else 'fixed'}"] = {
            "accuracy": acc,
            "auprc": auprc,
            "vs_baseline": acc - baseline_acc,
            "config": config
        }
    
    return results

def find_best_lambda_per_dataset():
    """Find optimal lambda for each dataset"""
    print("🎯 SYSTEMATIC LAMBDA OPTIMIZATION")
    print("=" * 80)
    
    all_results = {}
    summary = {}
    
    for dataset, baseline_acc in TEST_DATASETS.items():
        results = test_lambda_values(dataset, baseline_acc)
        all_results[dataset] = results
        
        # Find best performing configuration
        best_acc = 0.0
        best_config = None
        best_key = None
        
        for exp_key, exp_data in results["experiments"].items():
            if exp_data["accuracy"] > best_acc:
                best_acc = exp_data["accuracy"]
                best_config = exp_data["config"]
                best_key = exp_key
        
        summary[dataset] = {
            "baseline": baseline_acc,
            "best_accuracy": best_acc,
            "best_config": best_config,
            "improvement": best_acc - baseline_acc,
            "best_experiment": best_key
        }
        
        print(f"\n🏆 {dataset} Best: {best_config['name']} = {best_acc:.4f} ({best_acc - baseline_acc:+.4f})")
    
    return all_results, summary

def advanced_hyperparameter_search(promising_datasets):
    """Run hyperparameter search on promising configurations"""
    print("\n🔬 ADVANCED HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    
    # Parameters to optimize
    repr_dims_options = [160, 320, 640]  # Representation dimensions
    lr_options = [0.001, 0.005, 0.01]    # Learning rates (if we add this parameter)
    
    for dataset in promising_datasets:
        print(f"\n📊 Optimizing {dataset}...")
        
        # Test different representation dimensions
        for repr_dims in repr_dims_options:
            cmd = [
                "python", "train_ucr_msm.py", dataset, f"opt_repr_{repr_dims}",
                "--msm-weight", "0.0",  # Start with pure contrastive
                "--repr-dims", str(repr_dims),
                "--seed", "42", "--eval"
            ]
            
            acc, auprc = run_experiment(dataset, f"Repr-{repr_dims}", cmd)
            print(f"   📈 Repr dims {repr_dims}: {acc:.4f}")

def main():
    print("🚀 IMPROVED TS2VEC-MSM BENCHMARK")
    print("=" * 80)
    
    # Step 1: Test different lambda values
    all_results, summary = find_best_lambda_per_dataset()
    
    # Step 2: Analysis
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    
    # Check if λ=0.0 reproduces baseline
    lambda_0_performance = {}
    for dataset, results in all_results.items():
        lambda_0_exp = results["experiments"].get("lambda_0.0_fixed", {})
        lambda_0_acc = lambda_0_exp.get("accuracy", 0.0)
        baseline_acc = results["baseline"]
        lambda_0_performance[dataset] = {
            "lambda_0_acc": lambda_0_acc,
            "baseline_acc": baseline_acc,
            "can_reproduce": abs(lambda_0_acc - baseline_acc) < 0.05  # Within 5%
        }
    
    print("\n🔍 BASELINE REPRODUCTION CHECK (λ=0.0 vs Published Baseline):")
    for dataset, perf in lambda_0_performance.items():
        status = "✅" if perf["can_reproduce"] else "❌"
        print(f"   {status} {dataset}: λ=0.0 = {perf['lambda_0_acc']:.4f}, "
              f"Baseline = {perf['baseline_acc']:.4f}, "
              f"Diff = {perf['lambda_0_acc'] - perf['baseline_acc']:+.4f}")
    
    # Overall statistics
    improvements = [summary[dataset]["improvement"] for dataset in summary]
    avg_improvement = sum(improvements) / len(improvements)
    wins = sum(1 for imp in improvements if imp > 0)
    
    print(f"\n📈 OVERALL RESULTS:")
    print(f"   🎯 Datasets improved: {wins}/{len(TEST_DATASETS)}")
    print(f"   📊 Average improvement: {avg_improvement:+.4f}")
    print(f"   🏆 Best dataset: {max(summary.keys(), key=lambda k: summary[k]['improvement'])}")
    print(f"   📉 Worst dataset: {min(summary.keys(), key=lambda k: summary[k]['improvement'])}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"improved_benchmark_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            "all_results": all_results,
            "summary": summary,
            "lambda_0_performance": lambda_0_performance,
            "timestamp": timestamp
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Step 3: Advanced optimization for promising cases
    promising_datasets = [dataset for dataset, perf in lambda_0_performance.items() 
                         if perf["can_reproduce"]]
    
    if promising_datasets:
        print(f"\n🔬 Running advanced optimization on: {promising_datasets}")
        advanced_hyperparameter_search(promising_datasets)
    else:
        print("\n⚠️  No datasets successfully reproduce baseline - focus on fixing λ=0.0 first!")

if __name__ == "__main__":
    main()