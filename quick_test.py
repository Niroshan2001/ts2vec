#!/usr/bin/env python3
"""
Quick UCR Test Script - Test a few datasets quickly
"""

import subprocess
import json
import time

# Test datasets (small, medium, large)
TEST_DATASETS = [
    "Coffee",      # Small, easy (perfect baseline)
    "OliveOil",    # Small, medium difficulty  
    "Meat",        # Small, harder
    "Haptics",     # Medium size, harder
    "CricketX"     # Medium size, medium difficulty
]

def run_experiment(dataset, method_name, cmd):
    """Run a single experiment"""
    print(f"🔄 Running {dataset} with {method_name}...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
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
        return 0.0, 0.0
        
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout for {dataset} - {method_name}")
        return 0.0, 0.0

def main():
    print("🧪 Running Quick UCR Benchmark Test")
    print("=" * 50)
    
    results = {}
    
    for dataset in TEST_DATASETS:
        print(f"\n📋 Testing Dataset: {dataset}")
        results[dataset] = {}
        
        # 1. Baseline TS2Vec
        cmd_baseline = [
            "python", "train.py", dataset, "UCR",
            "--loader", "UCR", "--batch-size", "8", "--repr-dims", "320",
            "--seed", "42", "--eval"
        ]
        acc, auprc = run_experiment(dataset, "Baseline TS2Vec", cmd_baseline)
        results[dataset]["baseline"] = {"accuracy": acc, "auprc": auprc}
        
        # 2. TS2Vec-MSM (λ=0.0 - should match baseline)
        cmd_msm_0 = [
            "python", "train_ucr_msm.py", dataset, "test_lambda0",
            "--msm-weight", "0.0", "--seed", "42", "--eval"
        ]
        acc, auprc = run_experiment(dataset, "TS2Vec-MSM (λ=0.0)", cmd_msm_0)
        results[dataset]["msm_lambda_0"] = {"accuracy": acc, "auprc": auprc}
        
        # 3. TS2Vec-MSM (λ=0.3)
        cmd_msm_3 = [
            "python", "train_ucr_msm.py", dataset, "test_lambda3",
            "--msm-weight", "0.3", "--seed", "42", "--eval"
        ]
        acc, auprc = run_experiment(dataset, "TS2Vec-MSM (λ=0.3)", cmd_msm_3)
        results[dataset]["msm_lambda_3"] = {"accuracy": acc, "auprc": auprc}
        
        # 4. TS2Vec-MSM Dynamic (λ=0.4)
        cmd_dynamic = [
            "python", "train_ucr_msm.py", dataset, "test_dynamic",
            "--msm-weight", "0.4", "--dynamic-lambda", "--seed", "42", "--eval"
        ]
        acc, auprc = run_experiment(dataset, "TS2Vec-MSM Dynamic (λ=0.4)", cmd_dynamic)
        results[dataset]["msm_dynamic"] = {"accuracy": acc, "auprc": auprc}
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 QUICK BENCHMARK RESULTS")
    print("=" * 80)
    
    print(f"{'Dataset':<15} {'Baseline':<10} {'MSM λ=0.0':<12} {'MSM λ=0.3':<12} {'Dynamic':<10}")
    print("-" * 80)
    
    for dataset in TEST_DATASETS:
        baseline_acc = results[dataset]["baseline"]["accuracy"]
        msm0_acc = results[dataset]["msm_lambda_0"]["accuracy"]
        msm3_acc = results[dataset]["msm_lambda_3"]["accuracy"]
        dynamic_acc = results[dataset]["msm_dynamic"]["accuracy"]
        
        print(f"{dataset:<15} {baseline_acc:<10.4f} {msm0_acc:<12.4f} {msm3_acc:<12.4f} {dynamic_acc:<10.4f}")
    
    # Calculate averages
    avg_baseline = sum(results[d]["baseline"]["accuracy"] for d in TEST_DATASETS) / len(TEST_DATASETS)
    avg_msm0 = sum(results[d]["msm_lambda_0"]["accuracy"] for d in TEST_DATASETS) / len(TEST_DATASETS)
    avg_msm3 = sum(results[d]["msm_lambda_3"]["accuracy"] for d in TEST_DATASETS) / len(TEST_DATASETS)
    avg_dynamic = sum(results[d]["msm_dynamic"]["accuracy"] for d in TEST_DATASETS) / len(TEST_DATASETS)
    
    print("-" * 80)
    print(f"{'AVERAGE':<15} {avg_baseline:<10.4f} {avg_msm0:<12.4f} {avg_msm3:<12.4f} {avg_dynamic:<10.4f}")
    
    # Save results
    with open("quick_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: quick_benchmark_results.json")
    print("🎉 Quick benchmark completed!")

if __name__ == "__main__":
    main()