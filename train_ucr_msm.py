import os
import numpy as np
import argparse
import time
import datetime
import tasks
import datautils
from ts2vec_msm import TS2VecMSM
from ts2vec import TS2Vec  # Import baseline TS2Vec
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The UCR dataset name')
    parser.add_argument('run_name', help='The folder name used to save model, output and evaluation metrics')
    parser.add_argument('--loader', type=str, default='UCR', help='The data loader')
    parser.add_argument('--msm-weight', type=float, default=0.5, help='Weight for MSM loss (λ)')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs (for epoch-based training)')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--max-train-length', type=int, default=3000, help='Maximum training sequence length (use None for full sequences)')
    parser.add_argument('--use-epochs', action='store_true', help='Use epoch-based training instead of TS2Vec-style iterations')
    parser.add_argument('--eval', action='store_true', help='Whether to perform evaluation after training')
    
    args = parser.parse_args()
    
    print("=== TS2Vec-MSM UCR Classification ===")
    print(f"Dataset: {args.dataset}")
    print(f"MSM Weight (λ): {args.msm_weight}")  # BUG: This should show the actual parsed value
    
    # Debug: Print the actual arguments to verify parsing
    print(f"DEBUG - Parsed msm_weight: {args.msm_weight}")
    print(f"DEBUG - Type: {type(args.msm_weight)}")
    
    # Determine training approach
    use_iterations = not args.use_epochs
    if use_iterations:
        print("Training approach: TS2Vec-style iterations (200/600 based on dataset size)")
    else:
        print(f"Training approach: Epoch-based ({args.epochs} epochs)")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Max train length: {args.max_train_length}")
    
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
    
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
    
    # UPDATED: Use same iteration logic for both models
    if use_iterations:
        # Original TS2Vec approach: iterations based on dataset size
        n_iters = 200 if train_data.size <= 100000 else 600
        training_method = f"iterations_{n_iters}"
        print(f"📊 Dataset size: {train_data.size} → Using {n_iters} iterations")
    else:
        # Epoch-based approach
        n_iters = args.epochs * (len(train_data) // args.batch_size + 1)
        training_method = f"epochs_{args.epochs}"
        print(f"📊 Using {args.epochs} epochs → {n_iters} iterations")
    
    # FIXED: Better condition checking for lambda == 0.0
    print(f"DEBUG - Checking lambda: {args.msm_weight} == 0.0 ? {args.msm_weight == 0.0}")
    print(f"DEBUG - abs(lambda) < 1e-8 ? {abs(args.msm_weight) < 1e-8}")
    
    # Initialize model based on lambda (both use same training iterations and parameters now)
    if abs(args.msm_weight) < 1e-8:  # Use tolerance for float comparison
        print("🎯 Using baseline TS2Vec (λ≈0.0 - pure contrastive learning)")
        model = TS2Vec(
            input_dims=input_dims,
            output_dims=args.repr_dims,
            device=device,
            lr=0.001,  # Same as original
            batch_size=args.batch_size,
            max_train_length=args.max_train_length  # UPDATED: Use same as original (3000)
        )
        model_type = "baseline_ts2vec"
        
    else:
        print(f"🚂 Using TS2Vec-MSM (λ={args.msm_weight} - hybrid learning)")
        model = TS2VecMSM(
            input_dims=input_dims,
            output_dims=args.repr_dims,
            device=device,
            lr=0.001,  # Same as original
            batch_size=args.batch_size,
            max_train_length=args.max_train_length,  # UPDATED: Use same as original (3000)
            msm_weight=args.msm_weight,
            msm_mask_rate=0.15,
            msm_decoder_depth=3,
            dynamic_lambda=False
        )
        model_type = "ts2vec_msm"
    
    print(f"🚂 Training {model_type} (λ={args.msm_weight}) with {training_method}...")
    t = time.time()
    
    loss_log = model.fit(
        train_data,
        n_iters=n_iters,
        verbose=True
    )
    
    training_time = time.time() - t
    print(f"✅ Training completed in: {datetime.timedelta(seconds=training_time)}")
    print(f"Training time: {datetime.timedelta(seconds=training_time)}")
    
    # Evaluation (only if --eval flag is provided, like original)
    if args.eval:
        # Set to eval mode if available
        if hasattr(model, 'eval'):
            model.eval()
            print("📊 Model set to evaluation mode")
        
        print("📊 Evaluating classification performance...")
        eval_start = time.time()
        
        try:
            # Generate representations using same approach as original
            print("   🔄 Encoding training data...")
            train_repr = model.encode(
                train_data,
                causal=False,
                sliding_length=None,
                sliding_padding=0,
                batch_size=args.batch_size
            )
            
            print("   🔄 Encoding test data...")
            test_repr = model.encode(
                test_data,
                causal=False,
                sliding_length=None,
                sliding_padding=0,
                batch_size=args.batch_size
            )
            
            # FIXED: Use same evaluation method as original TS2Vec
            print("   🔄 Using original TS2Vec evaluation (SVM-based)...")
            
            # Use the same evaluation approach as original TS2Vec with SVM
            _, eval_res = tasks.eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='svm')
            
            eval_time = time.time() - eval_start
            print(f"✅ Evaluation completed in: {datetime.timedelta(seconds=eval_time)}")
            
            # Save results with training method info
            config_name = 'contrastive' if abs(args.msm_weight) < 1e-8 else 'msm' if abs(args.msm_weight - 1.0) < 1e-8 else 'hybrid'
            approach = "iterations" if use_iterations else f"epochs_{args.epochs}"
            run_dir = f'training/UCR_{args.dataset}__{args.run_name}_lambda_{args.msm_weight}_{config_name}_{model_type}_{approach}'
            os.makedirs(run_dir, exist_ok=True)
            
            model.save(f'{run_dir}/model.pkl')
            np.save(f'{run_dir}/loss_log.npy', loss_log)
            np.save(f'{run_dir}/eval_res.npy', eval_res)
            
            # Enhanced summary with training method info
            summary = {
                'dataset': args.dataset,
                'lambda': args.msm_weight,
                'configuration': config_name,
                'model_type': model_type,
                'training_method': training_method,
                'use_iterations': use_iterations,
                'final_loss': float(loss_log[-1]) if len(loss_log) > 0 else None,
                'training_time': training_time,
                'eval_time': eval_time,
                'accuracy': eval_res['acc'],
                'auprc': eval_res['auprc'],
                'n_iters': n_iters,
                'epochs_specified': args.epochs,
                'batch_size': args.batch_size,
                'seed': args.seed,
                'dataset_size': train_data.size,
                'max_train_length': args.max_train_length
            }
            
            np.save(f'{run_dir}/summary.npy', summary)
            print(f"💾 Results saved to: {run_dir}")
            
            # Results Analysis (same format as original)
            print(f"\nEvaluation result: {eval_res}")
            print("Finished.")
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            print("🎯 Training was successful - issue is with evaluation only")
            import traceback
            traceback.print_exc()
            
            # Save training results anyway
            config_name = 'contrastive' if abs(args.msm_weight) < 1e-8 else 'msm' if abs(args.msm_weight - 1.0) < 1e-8 else 'hybrid'
            model_type_fallback = "baseline_ts2vec" if abs(args.msm_weight) < 1e-8 else "ts2vec_msm"
            approach = "iterations" if use_iterations else f"epochs_{args.epochs}"
            run_dir = f'training/UCR_{args.dataset}__{args.run_name}_lambda_{args.msm_weight}_{config_name}_{model_type_fallback}_{approach}_train_only'
            os.makedirs(run_dir, exist_ok=True)
            
            model.save(f'{run_dir}/model.pkl')
            np.save(f'{run_dir}/loss_log.npy', loss_log)
            
            summary = {
                'dataset': args.dataset,
                'lambda': args.msm_weight,
                'configuration': config_name,
                'model_type': model_type_fallback,
                'training_method': training_method,
                'use_iterations': use_iterations,
                'final_loss': float(loss_log[-1]) if len(loss_log) > 0 else None,
                'training_time': training_time,
                'evaluation': 'failed',
                'n_iters': n_iters,
                'epochs_specified': args.epochs,
                'batch_size': args.batch_size,
                'seed': args.seed,
                'dataset_size': train_data.size,
                'max_train_length': args.max_train_length
            }
            
            np.save(f'{run_dir}/summary.npy', summary)
            print(f"💾 Training results saved to: {run_dir}")
    else:
        print("Skipping evaluation (use --eval flag to enable)")
        
        # Save training results only
        config_name = 'contrastive' if abs(args.msm_weight) < 1e-8 else 'msm' if abs(args.msm_weight - 1.0) < 1e-8 else 'hybrid'
        approach = "iterations" if use_iterations else f"epochs_{args.epochs}"
        run_dir = f'training/UCR_{args.dataset}__{args.run_name}_lambda_{args.msm_weight}_{config_name}_{model_type}_{approach}_train_only'
        os.makedirs(run_dir, exist_ok=True)
        
        model.save(f'{run_dir}/model.pkl')
        np.save(f'{run_dir}/loss_log.npy', loss_log)
        
        summary = {
            'dataset': args.dataset,
            'lambda': args.msm_weight,
            'configuration': config_name,
            'model_type': model_type,
            'training_method': training_method,
            'use_iterations': use_iterations,
            'final_loss': float(loss_log[-1]) if len(loss_log) > 0 else None,
            'training_time': training_time,
            'evaluation': 'skipped',
            'n_iters': n_iters,
            'epochs_specified': args.epochs,
            'batch_size': args.batch_size,
            'seed': args.seed,
            'dataset_size': train_data.size,
            'max_train_length': args.max_train_length
        }
        
        np.save(f'{run_dir}/summary.npy', summary)
        print(f"💾 Training results saved to: {run_dir}")
        print("Finished.")