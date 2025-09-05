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
    parser.add_argument('dataset', help='The UCR dataset name')
    parser.add_argument('run_name', help='The folder name used to save model, output and evaluation metrics')
    parser.add_argument('--loader', type=str, default='UCR', help='The data loader')
    parser.add_argument('--msm-weight', type=float, default=0.5, help='Weight for MSM loss (λ)')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension')
    parser.add_argument('--seed', type=int, default=42, help='The random seed')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    
    args = parser.parse_args()
    
    print("=== TS2Vec-MSM UCR Classification ===")
    print(f"Dataset: {args.dataset}")
    print(f"MSM Weight (λ): {args.msm_weight}")
    print(f"Training epochs: {args.epochs}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    print("Loading UCR data... ", end="")
    
    # Load UCR classification data
    train_data, train_labels, test_data, test_labels = datautils.load_UCR(args.dataset)
    
    print("done")
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Test labels shape: {test_labels.shape}")
    print(f"Number of classes: {len(np.unique(train_labels))}")
    
    # Get input dimensions
    input_dims = train_data.shape[-1]
    print(f"Input dimensions: {input_dims}")
    
    # Create TS2Vec-MSM model
    model = TS2VecMSM(
        input_dims=input_dims,
        output_dims=args.repr_dims,
        device=device,
        lr=0.001,
        batch_size=args.batch_size,
        max_train_length=None,  # Use full sequences for UCR
        msm_weight=args.msm_weight,
        msm_mask_rate=0.15,
        msm_decoder_depth=3,
        dynamic_lambda=False
    )
    
    print(f"🚂 Training TS2Vec-MSM (λ={args.msm_weight})...")
    t = time.time()
    
    # Calculate iterations based on epochs
    n_iters = args.epochs * (len(train_data) // args.batch_size + 1)
    
    loss_log = model.fit(
        train_data,
        n_iters=n_iters,
        verbose=True
    )
    
    training_time = time.time() - t
    print(f"✅ Training completed in: {datetime.timedelta(seconds=training_time)}")
    
    # Set model to evaluation mode
    model.eval()
    
    print("📊 Evaluating classification performance...")
    eval_start = time.time()
    
    try:
        # Evaluate classification performance
        y_score, eval_res = tasks.eval_classification(
            model, 
            train_data, train_labels, 
            test_data, test_labels, 
            eval_protocol='linear'
        )
        
        eval_time = time.time() - eval_start
        print(f"✅ Evaluation completed in: {datetime.timedelta(seconds=eval_time)}")
        
        # Save results
        config_name = 'contrastive' if args.msm_weight == 0 else 'msm' if args.msm_weight == 1 else 'hybrid'
        run_dir = f'training/UCR_{args.dataset}__{args.run_name}_lambda_{args.msm_weight}_{config_name}'
        os.makedirs(run_dir, exist_ok=True)
        
        model.save(f'{run_dir}/model.pkl')
        np.save(f'{run_dir}/loss_log.npy', loss_log)
        np.save(f'{run_dir}/eval_res.npy', eval_res)
        np.save(f'{run_dir}/y_score.npy', y_score)
        
        # Summary
        summary = {
            'dataset': args.dataset,
            'lambda': args.msm_weight,
            'configuration': config_name,
            'final_loss': float(loss_log[-1]) if len(loss_log) > 0 else None,
            'training_time': training_time,
            'eval_time': eval_time,
            'accuracy': eval_res['acc'],
            'auprc': eval_res['auprc'],
            'epochs': args.epochs,
            'n_iters': n_iters
        }
        
        np.save(f'{run_dir}/summary.npy', summary)
        print(f"💾 Results saved to: {run_dir}")
        
        # Results Analysis
        print(f"\n=== TS2Vec-MSM UCR RESULTS ===")
        print(f"📊 Dataset: {args.dataset}")
        print(f"🎯 Configuration: λ={args.msm_weight} ({config_name.title()})")
        print(f"📈 Final training loss: {loss_log[-1]:.6f}")
        print(f"⏱️  Training time: {datetime.timedelta(seconds=training_time)}")
        print(f"⏱️  Evaluation time: {datetime.timedelta(seconds=eval_time)}")
        print(f"🎯 Classification Accuracy: {eval_res['acc']:.4f}")
        print(f"📈 AUPRC: {eval_res['auprc']:.4f}")
        
        print("🎉 TS2Vec-MSM UCR experiment completed successfully!")
        
        return eval_res
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        print("🎯 Training was successful - issue is with evaluation only")
        
        # Save training results anyway
        run_dir = f'training/UCR_{args.dataset}__{args.run_name}_lambda_{args.msm_weight}_train_only'
        os.makedirs(run_dir, exist_ok=True)
        
        model.save(f'{run_dir}/model.pkl')
        np.save(f'{run_dir}/loss_log.npy', loss_log)
        
        summary = {
            'dataset': args.dataset,
            'lambda': args.msm_weight,
            'configuration': 'contrastive' if args.msm_weight == 0 else 'msm' if args.msm_weight == 1 else 'hybrid',
            'final_loss': float(loss_log[-1]) if len(loss_log) > 0 else None,
            'training_time': training_time,
            'evaluation': 'failed',
            'epochs': args.epochs
        }
        
        np.save(f'{run_dir}/summary.npy', summary)
        print(f"💾 Training results saved to: {run_dir}")
        
        return None
