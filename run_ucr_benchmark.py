#!/usr/bin/env python3
"""
UCR Benchmark Script for TS2Vec-MSM
Runs all 85 UCR datasets and compares results with baseline methods
"""

import subprocess
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

# List of all 85 UCR datasets from the paper
UCR_DATASETS = [
    "Adiac", "ArrowHead", "Beef", "BeetleFly", "BirdChicken", "Car", "CBF", 
    "ChlorineConcentration", "CinCECGTorso", "Coffee", "Computers", "CricketX", 
    "CricketY", "CricketZ", "DiatomSizeReduction", "DistalPhalanxOutlineCorrect", 
    "DistalPhalanxOutlineAgeGroup", "DistalPhalanxTW", "Earthquakes", "ECG200", 
    "ECG5000", "ECGFiveDays", "ElectricDevices", "FaceAll", "FaceFour", "FacesUCR", 
    "FiftyWords", "Fish", "FordA", "FordB", "GunPoint", "Ham", "HandOutlines", 
    "Haptics", "Herring", "InlineSkate", "InsectWingbeatSound", "ItalyPowerDemand", 
    "LargeKitchenAppliances", "Lightning2", "Lightning7", "Mallat", "Meat", 
    "MedicalImages", "MiddlePhalanxOutlineCorrect", "MiddlePhalanxOutlineAgeGroup", 
    "MiddlePhalanxTW", "MoteStrain", "NonInvasiveFetalECGThorax1", 
    "NonInvasiveFetalECGThorax2", "OliveOil", "OSULeaf", "PhalangesOutlinesCorrect", 
    "Phoneme", "Plane", "ProximalPhalanxOutlineCorrect", "ProximalPhalanxOutlineAgeGroup", 
    "ProximalPhalanxTW", "RefrigerationDevices", "ScreenType", "ShapeletSim", 
    "ShapesAll", "SmallKitchenAppliances", "SonyAIBORobotSurface1", 
    "SonyAIBORobotSurface2", "StarLightCurves", "Strawberry", "SwedishLeaf", 
    "Symbols", "SyntheticControl", "ToeSegmentation1", "ToeSegmentation2", 
    "Trace", "TwoLeadECG", "TwoPatterns", "UWaveGestureLibraryX", 
    "UWaveGestureLibraryY", "UWaveGestureLibraryZ", "UWaveGestureLibraryAll", 
    "Wafer", "Wine", "WordSynonyms", "Worms", "WormsTwoClass", "Yoga"
]

# Baseline results from the paper table
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

def run_single_experiment(dataset, method_config, max_retries=3):
    """Run a single experiment with retry logic"""
    
    for attempt in range(max_retries):
        try:
            if method_config["type"] == "baseline":
                # Run original TS2Vec baseline
                cmd = [
                    "python", "train.py", dataset, "UCR",
                    "--loader", "UCR", 
                    "--batch-size", "8",
                    "--repr-dims", "320",
                    "--seed", str(method_config["seed"]),
                    "--eval"
                ]
            else:
                # Run TS2Vec-MSM
                cmd = [
                    "python", "train_ucr_msm.py", dataset, f"exp_{method_config['name']}",
                    "--msm-weight", str(method_config["lambda"]),
                    "--seed", str(method_config["seed"]),
                    "--eval"
                ]
                
                if method_config.get("dynamic_lambda", False):
                    cmd.append("--dynamic-lambda")
            
            print(f"🔄 Running {dataset} with {method_config['name']} (attempt {attempt + 1}/{max_retries})")
            print(f"Command: {' '.join(cmd)}")
            
            # Run the command
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                # Parse the output to extract accuracy
                output = result.stdout
                
                # Look for evaluation result
                lines = output.split('\n')
                for line in lines:
                    if "Evaluation result:" in line:
                        # Extract accuracy from the result
                        try:
                            eval_str = line.split("Evaluation result:")[1].strip()
                            eval_dict = eval(eval_str)  # Parse the dictionary
                            accuracy = eval_dict.get('acc', 0.0)
                            auprc = eval_dict.get('auprc', 0.0)
                            
                            print(f"✅ {dataset} - {method_config['name']}: Accuracy = {accuracy:.4f}, AUPRC = {auprc:.4f}")
                            return {
                                "dataset": dataset,
                                "method": method_config["name"],
                                "accuracy": accuracy,
                                "auprc": auprc,
                                "success": True,
                                "attempt": attempt + 1
                            }
                        except Exception as e:
                            print(f"⚠️  Failed to parse result for {dataset}: {e}")
                            continue
                
                print(f"⚠️  No evaluation result found in output for {dataset}")
                print("Output:", output[-500:])  # Print last 500 chars for debugging
                
            else:
                print(f"❌ Command failed for {dataset} (attempt {attempt + 1})")
                print("STDERR:", result.stderr[-500:])
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout for {dataset} (attempt {attempt + 1})")
        except Exception as e:
            print(f"💥 Error for {dataset} (attempt {attempt + 1}): {e}")
    
    # All attempts failed
    print(f"❌ Failed all {max_retries} attempts for {dataset}")
    return {
        "dataset": dataset,
        "method": method_config["name"],
        "accuracy": 0.0,
        "auprc": 0.0,
        "success": False,
        "attempt": max_retries
    }

def main():
    parser = argparse.ArgumentParser(description="Run UCR benchmark for TS2Vec-MSM")
    parser.add_argument("--methods", nargs="+", default=["baseline", "msm_static", "msm_dynamic"], 
                        help="Methods to run: baseline, msm_static, msm_dynamic")
    parser.add_argument("--datasets", nargs="+", default=UCR_DATASETS[:10],  # First 10 for quick test
                        help="Datasets to run (default: first 10)")
    parser.add_argument("--lambda", type=float, default=0.4, help="Lambda value for MSM methods")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="ucr_benchmark_results.json", help="Output file")
    parser.add_argument("--full", action="store_true", help="Run all 85 datasets")
    
    args = parser.parse_args()
    
    if args.full:
        datasets = UCR_DATASETS
        print(f"🚀 Running FULL UCR benchmark on all {len(datasets)} datasets")
    else:
        datasets = args.datasets
        print(f"🧪 Running test benchmark on {len(datasets)} datasets")
    
    # Define method configurations
    method_configs = []
    
    if "baseline" in args.methods:
        method_configs.append({
            "name": "baseline_ts2vec",
            "type": "baseline",
            "seed": args.seed
        })
    
    if "msm_static" in args.methods:
        method_configs.append({
            "name": f"ts2vec_msm_lambda_{args.lambda}",
            "type": "msm",
            "lambda": args.lambda,
            "dynamic_lambda": False,
            "seed": args.seed
        })
    
    if "msm_dynamic" in args.methods:
        method_configs.append({
            "name": f"ts2vec_msm_dynamic_{args.lambda}",
            "type": "msm",
            "lambda": args.lambda,
            "dynamic_lambda": True,
            "seed": args.seed
        })
    
    print(f"📊 Methods: {[config['name'] for config in method_configs]}")
    print(f"📁 Datasets: {datasets}")
    print(f"🎯 Lambda: {args.lambda}")
    print(f"🌱 Seed: {args.seed}")
    print("=" * 80)
    
    # Run experiments
    all_results = []
    start_time = time.time()
    
    for i, dataset in enumerate(datasets):
        print(f"\n📋 Dataset {i+1}/{len(datasets)}: {dataset}")
        print(f"📊 Baseline accuracy from paper: {BASELINE_RESULTS.get(dataset, 'N/A')}")
        
        for config in method_configs:
            result = run_single_experiment(dataset, config)
            all_results.append(result)
            
            # Save intermediate results
            with open(args.output, 'w') as f:
                json.dump(all_results, f, indent=2)
    
    # Calculate summary statistics
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 80)
    
    # Group results by method
    results_by_method = {}
    for result in all_results:
        method = result["method"]
        if method not in results_by_method:
            results_by_method[method] = []
        results_by_method[method].append(result)
    
    # Calculate averages and comparisons
    summary = {}
    for method, results in results_by_method.items():
        successful_results = [r for r in results if r["success"]]
        
        if successful_results:
            avg_accuracy = np.mean([r["accuracy"] for r in successful_results])
            avg_auprc = np.mean([r["auprc"] for r in successful_results])
            success_rate = len(successful_results) / len(results)
            
            # Compare with baselines
            baseline_comparison = []
            for r in successful_results:
                baseline_acc = BASELINE_RESULTS.get(r["dataset"])
                if baseline_acc:
                    improvement = r["accuracy"] - baseline_acc
                    baseline_comparison.append(improvement)
            
            avg_improvement = np.mean(baseline_comparison) if baseline_comparison else 0.0
            
            summary[method] = {
                "avg_accuracy": avg_accuracy,
                "avg_auprc": avg_auprc,
                "success_rate": success_rate,
                "avg_improvement": avg_improvement,
                "successful_datasets": len(successful_results),
                "total_datasets": len(results)
            }
            
            print(f"\n🎯 {method}:")
            print(f"   📈 Average Accuracy: {avg_accuracy:.4f}")
            print(f"   📈 Average AUPRC: {avg_auprc:.4f}")
            print(f"   ✅ Success Rate: {success_rate:.2%} ({len(successful_results)}/{len(results)})")
            print(f"   📊 Avg Improvement vs Baseline: {avg_improvement:+.4f}")
        else:
            print(f"\n❌ {method}: No successful results")
    
    # Save final results
    final_output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time_seconds": total_time,
        "config": {
            "datasets": datasets,
            "methods": args.methods,
            "lambda": args.lambda,
            "seed": args.seed
        },
        "results": all_results,
        "summary": summary
    }
    
    with open(args.output, 'w') as f:
        json.dump(final_output, f, indent=2)
    
    # Create CSV summary
    csv_file = args.output.replace('.json', '.csv')
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(csv_file, index=False)
    
    print(f"\n💾 Results saved to:")
    print(f"   📄 {args.output}")
    print(f"   📊 {csv_file}")
    print(f"\n⏱️  Total runtime: {datetime.timedelta(seconds=total_time)}")
    print("\n🎉 Benchmark completed!")

if __name__ == "__main__":
    import datetime
    main()