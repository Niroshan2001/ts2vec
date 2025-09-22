#!/usr/bin/env python3
"""
KPI Dataset Setup Guide

This script provides step-by-step instructions for setting up the KPI dataset
for anomaly detection experiments with TS2Vec and TS2Vec-MSM.
"""

import os
import sys
from pathlib import Path

def check_dataset_files():
    """Check if KPI dataset files are available"""
    print("🔍 Checking for KPI dataset files...")
    
    datasets_dir = Path("datasets")
    
    # Check for raw data files
    train_csv = datasets_dir / "phase2_train.csv"
    test_hdf = datasets_dir / "phase2_ground_truth.hdf"
    processed_pkl = Path("kpi.pkl")
    
    print(f"\n📁 Dataset Directory: {datasets_dir.absolute()}")
    print(f"   📄 phase2_train.csv: {'✅ Found' if train_csv.exists() else '❌ Missing'}")
    print(f"   📄 phase2_ground_truth.hdf: {'✅ Found' if test_hdf.exists() else '❌ Missing'}")
    print(f"   📄 kpi.pkl (processed): {'✅ Found' if processed_pkl.exists() else '❌ Missing'}")
    
    return {
        'train_csv': train_csv.exists(),
        'test_hdf': test_hdf.exists(),
        'processed_pkl': processed_pkl.exists()
    }

def provide_download_instructions():
    """Provide instructions for downloading the KPI dataset"""
    print("\n📋 KPI DATASET DOWNLOAD INSTRUCTIONS")
    print("=" * 50)
    print("""
The KPI dataset is from the AIOPS Challenge for anomaly detection in Key Performance Indicators.

🔗 Dataset Sources:
1. Original Paper: "Unsupervised Anomaly Detection via Variational Auto-Encoder for Seasonal KPIs in Web Applications"
2. AIOPS Challenge: https://competition.aiops-challenge.com/
3. Alternative: Search for "KPI anomaly detection dataset AIOPS" or "phase2_train.csv phase2_ground_truth.hdf"

📥 Required Files:
- phase2_train.csv    (Training data with KPI values and labels)
- phase2_ground_truth.hdf  (Ground truth anomaly labels for test data)

📂 File Placement:
1. Download both files
2. Place them in the 'datasets/' directory
3. File structure should be:
   datasets/
   ├── phase2_train.csv
   ├── phase2_ground_truth.hdf
   └── preprocess_kpi.py
    """)

def check_preprocessing_script():
    """Check if preprocessing script is available"""
    preprocess_script = Path("datasets/preprocess_kpi.py")
    
    if preprocess_script.exists():
        print(f"✅ Preprocessing script found: {preprocess_script}")
        return True
    else:
        print(f"❌ Preprocessing script missing: {preprocess_script}")
        return False

def provide_preprocessing_instructions():
    """Provide instructions for preprocessing the dataset"""
    print("\n📋 DATASET PREPROCESSING INSTRUCTIONS")
    print("=" * 50)
    print("""
Once you have the raw KPI dataset files, follow these steps:

1️⃣ Navigate to the project directory:
   cd "d:\\Sem 7\\Adavanced ML\\Research Code\\ts2vec"

2️⃣ Run the preprocessing script:
   python datasets/preprocess_kpi.py

3️⃣ This will generate 'kpi.pkl' containing:
   - Preprocessed time series data
   - Normalized values (mean=0, std=1)
   - Train/test splits with proper timestamps
   - Anomaly labels

4️⃣ The script performs:
   - Stationarity testing with Augmented Dickey-Fuller test
   - Z-score normalization per KPI
   - Proper time series formatting for TS2Vec
    """)

def check_training_readiness():
    """Check if ready for model training"""
    print("\n📋 MODEL TRAINING READINESS CHECK")
    print("=" * 40)
    
    # Check for processed data
    if Path("kpi.pkl").exists():
        print("✅ Processed dataset (kpi.pkl) available")
        ready_for_training = True
    else:
        print("❌ Processed dataset (kpi.pkl) missing")
        ready_for_training = False
    
    # Check for training scripts
    train_script = Path("train.py")
    if train_script.exists():
        print("✅ Training script (train.py) available")
    else:
        print("❌ Training script (train.py) missing")
        ready_for_training = False
    
    # Check for model files
    ts2vec_script = Path("ts2vec.py")
    msm_script = Path("ts2vec_msm.py")
    
    if ts2vec_script.exists():
        print("✅ TS2Vec model (ts2vec.py) available")
    else:
        print("❌ TS2Vec model (ts2vec.py) missing")
        ready_for_training = False
        
    if msm_script.exists():
        print("✅ TS2Vec-MSM model (ts2vec_msm.py) available")
    else:
        print("❌ TS2Vec-MSM model (ts2vec_msm.py) missing")
        ready_for_training = False
    
    return ready_for_training

def provide_training_instructions():
    """Provide model training instructions"""
    print("\n📋 MODEL TRAINING INSTRUCTIONS")
    print("=" * 40)
    print("""
Train both baseline and MSM models for comparison:

🤖 BASELINE TS2Vec:
   python train.py kpi anomaly_0 --loader anomaly --repr-dims 320 --max-threads 8 --seed 1 --eval

🤖 TS2Vec-MSM HYBRID:
   python train.py kpi anomaly_0 --loader anomaly --repr-dims 320 --max-threads 8 --seed 1 --eval --msm --mask-rate 0.1 --msm-weight 0.1

⚙️ Training Parameters:
   - Dataset: kpi
   - Task: anomaly_0 (anomaly detection)
   - Representation dims: 320
   - Evaluation: included (--eval)
   - MSM config: mask_rate=0.1, msm_weight=0.1

📁 Model Output:
   Models will be saved in training_results/ directory
    """)

def provide_evaluation_instructions():
    """Provide evaluation instructions"""
    print("\n📋 EVALUATION INSTRUCTIONS")
    print("=" * 40)
    print("""
After training both models, run the comparison:

🎯 RUN COMPARISON:
   python kpi_anomaly_comparison.py \\
     --data-path kpi.pkl \\
     --baseline-model training_results/kpi_anomaly_0/model.pkl \\
     --msm-model training_results/kpi_anomaly_0_msm/model.pkl \\
     --output kpi_results.pkl

📊 EXPECTED OUTPUT:
   - F1 Score comparison
   - Precision/Recall analysis  
   - Performance improvement metrics
   - Time overhead assessment
   - Overall recommendation

🎯 SUCCESS CRITERIA:
   - MSM F1 score > Baseline F1 score
   - Statistically significant improvement
   - Reasonable computational overhead
    """)

def main():
    print("🚀 KPI ANOMALY DETECTION SETUP GUIDE")
    print("=" * 50)
    
    # Check current status
    status = check_dataset_files()
    
    # Provide appropriate instructions based on status
    if not (status['train_csv'] and status['test_hdf']):
        provide_download_instructions()
    
    if status['train_csv'] and status['test_hdf'] and not status['processed_pkl']:
        if check_preprocessing_script():
            provide_preprocessing_instructions()
    
    if status['processed_pkl']:
        if check_training_readiness():
            provide_training_instructions()
            provide_evaluation_instructions()
        
        print("\n🎯 NEXT STEPS SUMMARY:")
        print("1. Download KPI dataset files (if not done)")
        print("2. Run preprocessing: python datasets/preprocess_kpi.py")
        print("3. Train baseline model")
        print("4. Train MSM model")
        print("5. Run comparison: python kpi_anomaly_comparison.py")
    
    print(f"\n✨ Setup guide completed!")

if __name__ == "__main__":
    main()