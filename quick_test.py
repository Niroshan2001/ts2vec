#!/usr/bin/env python3
"""
Quick UCR Test Script - Test TS2Vec-MSM variants vs published baseline results
"""

import subprocess
import json
import time

# Test datasets with their published TS2Vec baseline accuracies
TEST_DATASETS = {
    # Small datasets (fast testing)
    "Coffee": 1.000,
    "OliveOil": 0.900, 
    "Meat": 0.950,
    "Plane": 1.000,
    "Wine": 0.870,
    
    # Medium datasets  
    # "Haptics": 0.526,
    # "CricketX": 0.782,
    # "ECG200": 0.920,
    # "GunPoint": 0.980,
    # "Beef": 0.767
}

def run_experiment(dataset, method_name, cmd, timeout=600):  # Increased timeout to 10 minutes
    """Run a single experiment"""
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
                    except:
                        pass
        
        print(f"❌ Failed to get result for {dataset} - {method_name}")
        print(f"Error output: {result.stderr}")
        return 0.0, 0.0
        
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout for {dataset} - {method_name} (>{timeout}s)")
        return 0.0, 0.0

def main():
    print("🧪 Running TS2Vec-MSM vs Published Baseline Comparison")
    print("=" * 60)
    
    results = {}
    
    for dataset, baseline_acc in TEST_DATASETS.items():
        print(f"\n📋 Testing Dataset: {dataset} (Baseline: {baseline_acc:.3f})")
        results[dataset] = {
            "baseline_published": baseline_acc,
            "experiments": {}
        }
        
        # 1. TS2Vec-MSM (λ=0.5 fixed)
        cmd_msm_5 = [
            "python", "train_ucr_msm.py", dataset, "test_lambda5",
            "--msm-weight", "0.5", "--seed", "42", "--eval"
        ]
        acc, auprc = run_experiment(dataset, "TS2Vec-MSM (λ=0.5)", cmd_msm_5)
        results[dataset]["experiments"]["msm_lambda_5"] = {
            "accuracy": acc, "auprc": auprc,
            "vs_baseline": acc - baseline_acc
        }
        
        # 2. TS2Vec-MSM Dynamic (λ=0.5)
        cmd_dynamic = [
            "python", "train_ucr_msm.py", dataset, "test_dynamic",
            "--msm-weight", "0.5", "--dynamic-lambda", "--seed", "42", "--eval"
        ]
        acc, auprc = run_experiment(dataset, "TS2Vec-MSM Dynamic (λ=0.5)", cmd_dynamic)
        results[dataset]["experiments"]["msm_dynamic_5"] = {
            "accuracy": acc, "auprc": auprc, 
            "vs_baseline": acc - baseline_acc
        }
    
    # Print comprehensive summary
    print("\n" + "=" * 100)
    print("📊 TS2Vec-MSM PERFORMANCE COMPARISON")
    print("=" * 100)
    
    print(f"{'Dataset':<15} {'Baseline':<10} {'λ=0.5':<10} {'Dynamic':<10} {'λ=0.5 Δ':<10} {'Dyn Δ':<10}")
    print("-" * 100)
    
    for dataset, baseline_acc in TEST_DATASETS.items():
        exp = results[dataset]["experiments"]
        
        msm_acc = exp["msm_lambda_5"]["accuracy"]
        dynamic_acc = exp["msm_dynamic_5"]["accuracy"]
        msm_delta = exp["msm_lambda_5"]["vs_baseline"]
        dynamic_delta = exp["msm_dynamic_5"]["vs_baseline"]
        
        print(f"{dataset:<15} {baseline_acc:<10.3f} {msm_acc:<10.3f} {dynamic_acc:<10.3f} " +
              f"{msm_delta:+10.3f} {dynamic_delta:+10.3f}")
    
    # Calculate statistics
    valid_datasets = [d for d in TEST_DATASETS.keys() 
                     if results[d]["experiments"]["msm_lambda_5"]["accuracy"] > 0]
    
    if valid_datasets:
        avg_baseline = sum(TEST_DATASETS[d] for d in valid_datasets) / len(valid_datasets)
        avg_msm = sum(results[d]["experiments"]["msm_lambda_5"]["accuracy"] for d in valid_datasets) / len(valid_datasets)
        avg_dynamic = sum(results[d]["experiments"]["msm_dynamic_5"]["accuracy"] for d in valid_datasets) / len(valid_datasets)
        
        print("-" * 100)
        print(f"{'AVERAGE':<15} {avg_baseline:<10.3f} {avg_msm:<10.3f} {avg_dynamic:<10.3f} " +
              f"{avg_msm-avg_baseline:+10.3f} {avg_dynamic-avg_baseline:+10.3f}")
        
        # Count wins/losses
        msm_wins = sum(1 for d in valid_datasets if results[d]["experiments"]["msm_lambda_5"]["vs_baseline"] > 0.001)
        dynamic_wins = sum(1 for d in valid_datasets if results[d]["experiments"]["msm_dynamic_5"]["vs_baseline"] > 0.001)
        
        print("\n📈 PERFORMANCE ANALYSIS:")
        print(f"  • TS2Vec-MSM (λ=0.5): {msm_wins}/{len(valid_datasets)} wins vs baseline")
        print(f"  • Dynamic λ=0.5: {dynamic_wins}/{len(valid_datasets)} wins vs baseline")
        print(f"  • Average improvement: λ=0.5: {avg_msm-avg_baseline:+.3f}, Dynamic: {avg_dynamic-avg_baseline:+.3f}")
    
    # Save results
    with open("msm_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: msm_benchmark_results.json")
    print("🎉 TS2Vec-MSM benchmark completed!")

if __name__ == "__main__":
    main()