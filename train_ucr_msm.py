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
    parser.add_argument('--use-epochs', action='store_true', help='Use epoch-based training instead of TS2Vec-style iterations')
    
    args = parser.parse_args()
    
    print("=== TS2Vec-MSM UCR Classification ===")
    print(f"Dataset: {args.dataset}")
    print(f"MSM Weight (λ): {args.msm_weight}")
    
    # Determine training approach
    use_iterations = not args.use_epochs
    if use_iterations:
        print("Training approach: TS2Vec-style iterations (200/600 based on dataset size)")
    else:
        print(f"Training approach: Epoch-based ({args.epochs} epochs)")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
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
    
    # Initialize model based on lambda (both use same training iterations now)
    if args.msm_weight == 0.0:
        print("🎯 Using baseline TS2Vec (λ=0.0 - pure contrastive learning)")
        model = TS2Vec(
            input_dims=input_dims,
            output_dims=args.repr_dims,
            device=device,
            lr=0.001,
            batch_size=args.batch_size,
            max_train_length=None  # Use full sequences for UCR
        )
        model_type = "baseline_ts2vec"
        
    else:
        print(f"🚂 Using TS2Vec-MSM (λ={args.msm_weight} - hybrid learning)")
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
    
    # Set to eval mode if available
    if hasattr(model, 'eval'):
        model.eval()
        print("📊 Model set to evaluation mode")
    
    print("📊 Evaluating classification performance...")
    eval_start = time.time()
    
    try:
        # Generate representations using correct parameters
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
        
        # Handle different output shapes properly
        if len(train_repr.shape) == 3:
            # If output is 3D (batch, time, features), take mean over time
            train_repr_flat = train_repr.mean(axis=1)
            test_repr_flat = test_repr.mean(axis=1)
        else:
            # If output is 2D (batch, features), use directly
            train_repr_flat = train_repr.reshape(train_repr.shape[0], -1)
            test_repr_flat = test_repr.reshape(test_repr.shape[0], -1)
        
        print(f"   📊 Train representations: {train_repr.shape} → {train_repr_flat.shape}")
        print(f"   📊 Test representations: {test_repr.shape} → {test_repr_flat.shape}")
        
        # Use sklearn directly for classification
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.metrics import accuracy_score, classification_report
        
        # Create pipeline with scaling and logistic regression
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=args.seed))
        clf.fit(train_repr_flat, train_labels)
        
        # Predictions
        test_pred = clf.predict(test_repr_flat)
        accuracy = accuracy_score(test_labels, test_pred)
        
        # Calculate AUPRC for binary/multiclass
        try:
            from sklearn.metrics import average_precision_score
            from sklearn.preprocessing import label_binarize
            proba = clf.predict_proba(test_repr_flat)
            if proba.shape[1] == 2:  # Binary classification
                auprc = average_precision_score(test_labels, proba[:, 1])
            else:  # Multiclass
                test_labels_onehot = label_binarize(test_labels, classes=np.unique(train_labels))
                if test_labels_onehot.ndim == 1:
                    test_labels_onehot = test_labels_onehot.reshape(-1, 1)
                auprc = average_precision_score(test_labels_onehot, proba, average='weighted')
        except Exception as auprc_error:
            print(f"   ⚠️  AUPRC calculation warning: {auprc_error}")
            auprc = 0.0  # Fallback if AUPRC calculation fails
        
        eval_res = {'acc': accuracy, 'auprc': auprc}
        y_score = test_pred
        
        eval_time = time.time() - eval_start
        print(f"✅ Evaluation completed in: {datetime.timedelta(seconds=eval_time)}")
        
        # Save results with training method info
        config_name = 'contrastive' if args.msm_weight == 0 else 'msm' if args.msm_weight == 1 else 'hybrid'
        approach = "iterations" if use_iterations else f"epochs_{args.epochs}"
        run_dir = f'training/UCR_{args.dataset}__{args.run_name}_lambda_{args.msm_weight}_{config_name}_{model_type}_{approach}'
        os.makedirs(run_dir, exist_ok=True)
        
        model.save(f'{run_dir}/model.pkl')
        np.save(f'{run_dir}/loss_log.npy', loss_log)
        np.save(f'{run_dir}/eval_res.npy', eval_res)
        np.save(f'{run_dir}/y_score.npy', y_score)
        
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
            'dataset_size': train_data.size
        }
        
        np.save(f'{run_dir}/summary.npy', summary)
        print(f"💾 Results saved to: {run_dir}")
        
        # Results Analysis
        print(f"\n=== TS2Vec-MSM UCR RESULTS ===")
        print(f"📊 Dataset: {args.dataset}")
        print(f"🎯 Configuration: λ={args.msm_weight} ({config_name.title()}) using {model_type}")
        print(f"🔧 Training method: {training_method}")
        print(f"📈 Final training loss: {loss_log[-1]:.6f}")
        print(f"⏱️  Training time: {datetime.timedelta(seconds=training_time)}")
        print(f"⏱️  Evaluation time: {datetime.timedelta(seconds=eval_time)}")
        print(f"🎯 Classification Accuracy: {eval_res['acc']:.4f}")
        print(f"📈 AUPRC: {eval_res['auprc']:.4f}")
        print(f"🔧 Training iterations: {n_iters}")
        
        if use_iterations:
            equivalent_epochs = n_iters * args.batch_size / len(train_data)
            print(f"📊 Equivalent epochs: ~{equivalent_epochs:.2f}")
        
        print("🎉 TS2Vec-MSM UCR experiment completed successfully!")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        print("🎯 Training was successful - issue is with evaluation only")
        import traceback
        traceback.print_exc()
        
        # Save training results anyway
        config_name = 'contrastive' if args.msm_weight == 0 else 'msm' if args.msm_weight == 1 else 'hybrid'
        model_type_fallback = "baseline_ts2vec" if args.msm_weight == 0 else "ts2vec_msm"
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
            'dataset_size': train_data.size
        }
        
        np.save(f'{run_dir}/summary.npy', summary)
        print(f"💾 Training results saved to: {run_dir}")