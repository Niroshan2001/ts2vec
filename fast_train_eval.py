import os
import numpy as np
import argparse
import time
import datetime
import tasks
import datautils
from ts2vec_msm import TS2VecMSM
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The dataset name')
    parser.add_argument('run_name', help='The folder name used to save model, output and evaluation metrics')
    parser.add_argument('--loader', type=str, required=True, help='The data loader')
    parser.add_argument('--msm-weight', type=float, default=0.5, help='Weight for MSM loss')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension')
    parser.add_argument('--seed', type=int, default=42, help='The random seed')
    parser.add_argument('--sample-size', type=int, default=5000, help='Max points per dataset for fast evaluation')
    
    args = parser.parse_args()
    
    print("=== TS2Vec-MSM Fast Training & Evaluation ===")
    print(f"Dataset: {args.dataset}")
    print(f"MSM Weight (λ): {args.msm_weight}")
    print(f"Sample size for evaluation: {args.sample_size}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    print("Loading data... ", end="")
    
    # Load anomaly data
    all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(args.dataset)
    
    # Get dimensions and prepare training data using the same method as train_msm.py
    input_dims = 1  # Anomaly data is typically univariate
    
    # Generate training data using the same method as original TS2Vec
    from datautils import gen_ano_train_data
    training_data = gen_ano_train_data(all_train_data)
    
    print("done")
    print(f"Input dimensions: {input_dims}")
    print(f"Training data shape: {training_data.shape}")
    print(f"Total datasets: {len(all_train_data)}")
    
    # Create model with memory-efficient settings
    model = TS2VecMSM(
        input_dims=input_dims,
        output_dims=args.repr_dims,
        device=device,
        lr=0.001,
        batch_size=4,  # Reduced for memory
        max_train_length=1000,  # Reduced for memory
        msm_weight=args.msm_weight,
        msm_mask_rate=0.15,
        msm_decoder_depth=2,  # Reduced for memory
        dynamic_lambda=False
    )
    
    print("🚂 Training...")
    t = time.time()
    loss_log = model.fit(
        training_data,
        n_iters=100,  # Quick training
        verbose=True
    )
    training_time = time.time() - t
    print(f"✅ Training completed in: {datetime.timedelta(seconds=training_time)}")
    
    # Set model to evaluation mode
    model.eval()
    
    # Sample data for fast evaluation
    print(f"📊 Sampling data for fast evaluation (max {args.sample_size} points per dataset)...")
    
    sampled_train_data = {}
    sampled_test_data = {}
    sampled_train_labels = {}
    sampled_test_labels = {}
    sampled_train_timestamps = {}
    sampled_test_timestamps = {}
    
    for key in all_train_data.keys():
        # Sample training data
        train_size = min(len(all_train_data[key]), args.sample_size)
        if train_size < len(all_train_data[key]):
            indices = np.random.choice(len(all_train_data[key]), train_size, replace=False)
            sampled_train_data[key] = all_train_data[key][indices]
            sampled_train_labels[key] = all_train_labels[key][indices]
            sampled_train_timestamps[key] = all_train_timestamps[key][indices]
        else:
            sampled_train_data[key] = all_train_data[key]
            sampled_train_labels[key] = all_train_labels[key]
            sampled_train_timestamps[key] = all_train_timestamps[key]
        
        # Sample test data
        test_size = min(len(all_test_data[key]), args.sample_size)
        if test_size < len(all_test_data[key]):
            indices = np.random.choice(len(all_test_data[key]), test_size, replace=False)
            sampled_test_data[key] = all_test_data[key][indices]
            sampled_test_labels[key] = all_test_labels[key][indices]
            sampled_test_timestamps[key] = all_test_timestamps[key][indices]
        else:
            sampled_test_data[key] = all_test_data[key]
            sampled_test_labels[key] = all_test_labels[key]
            sampled_test_timestamps[key] = all_test_timestamps[key]
    
    print(f"📈 Evaluating on sampled data...")
    
    # Clear CUDA cache before evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    eval_start = time.time()
    try:
        out, eval_res = tasks.eval_anomaly_detection(
            model, 
            sampled_train_data, sampled_train_labels, sampled_train_timestamps, 
            sampled_test_data, sampled_test_labels, sampled_test_timestamps, 
            delay
        )
        
        eval_time = time.time() - eval_start
        print(f"✅ Evaluation completed in: {datetime.timedelta(seconds=eval_time)}")
        print(f"📊 Results: {eval_res}")
        
        # Save results
        run_dir = f'training/{args.dataset}__{args.run_name}_lambda_{args.msm_weight}_fast'
        os.makedirs(run_dir, exist_ok=True)
        
        model.save(f'{run_dir}/model.pkl')
        np.save(f'{run_dir}/loss_log.npy', loss_log)
        np.save(f'{run_dir}/eval_res.npy', eval_res)
        np.save(f'{run_dir}/out.npy', out)
        
        # Summary
        summary = {
            'dataset': args.dataset,
            'lambda': args.msm_weight,
            'configuration': 'contrastive' if args.msm_weight == 0 else 'msm' if args.msm_weight == 1 else 'hybrid',
            'final_loss': float(loss_log[-1]),
            'training_time': training_time,
            'eval_time': eval_time,
            'eval_results': eval_res,
            'sample_size': args.sample_size
        }
        
        np.save(f'{run_dir}/summary.npy', summary)
        print(f"💾 Results saved to: {run_dir}")
        
        # Analysis
        print(f"\n=== TS2Vec-MSM RESULTS ===")
        print(f"🎯 Configuration: λ={args.msm_weight} ({'Contrastive' if args.msm_weight == 0 else 'MSM' if args.msm_weight == 1 else 'Hybrid'})")
        print(f"📊 Final training loss: {loss_log[-1]:.6f}")
        print(f"⏱️  Training time: {datetime.timedelta(seconds=training_time)}")
        print(f"⏱️  Evaluation time: {datetime.timedelta(seconds=eval_time)}")
        
        if 'ours' in eval_res:
            for metric, value in eval_res['ours'].items():
                print(f"📈 {metric}: {value:.4f}")
        
        print("🎉 TS2Vec-MSM experiment completed successfully!")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        print("🎯 Training was successful - issue is with evaluation only")
        
        # Save training results anyway
        run_dir = f'training/{args.dataset}__{args.run_name}_lambda_{args.msm_weight}_train_only'
        os.makedirs(run_dir, exist_ok=True)
        
        model.save(f'{run_dir}/model.pkl')
        np.save(f'{run_dir}/loss_log.npy', loss_log)
        
        summary = {
            'dataset': args.dataset,
            'lambda': args.msm_weight,
            'configuration': 'contrastive' if args.msm_weight == 0 else 'msm' if args.msm_weight == 1 else 'hybrid',
            'final_loss': float(loss_log[-1]),
            'training_time': training_time,
            'evaluation': 'failed'
        }
        
        np.save(f'{run_dir}/summary.npy', summary)
        print(f"💾 Training results saved to: {run_dir}")
