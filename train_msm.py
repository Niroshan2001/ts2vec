import os
import numpy as np
import argparse
import time
import datetime
import tasks
import datautils
from ts2vec_msm import TS2VecMSM
import torch

def save_checkpoint_callback(model, loss):
    """Callback function to save model checkpoint"""
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The dataset name')
    parser.add_argument('run_name', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--loader', type=str, required=True, help='The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--max-train-length', type=int, default=3000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=42, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=8, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--eval', action='store_true', help='Whether to perform evaluation after training')
    parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')
    
    # MSM-specific arguments
    parser.add_argument('--msm-weight', type=float, default=0.5, help='Weight for MSM loss (λ parameter, 0=contrastive only, 1=MSM only)')
    parser.add_argument('--msm-mask-rate', type=float, default=0.15, help='Percentage of timestamps to mask for MSM')
    parser.add_argument('--msm-decoder-depth', type=int, default=3, help='Number of layers in the MSM decoder')
    parser.add_argument('--dynamic-lambda', action='store_true', help='Whether to use dynamic λ scheduling during training')
    
    args = parser.parse_args()
    
    print("Dataset:", args.dataset)
    print("Arguments:", args)
    
    device = 'cuda' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    
    if args.max_threads is not None:
        torch.set_num_threads(args.max_threads)
    
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    print("Loading data... ", end="")
    
    # Data loading logic (same as original train.py)
    if args.loader == 'UCR':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UCR(args.dataset)
    elif args.loader == 'UEA':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UEA(args.dataset)
    elif args.loader == 'forecast_csv':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset)
    elif args.loader == 'forecast_csv_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset, univar=True)
    elif args.loader == 'forecast_npy':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(args.dataset)
    elif args.loader == 'forecast_npy_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(args.dataset, univar=True)
    elif args.loader == 'anomaly':
        task_type = 'anomaly_detection'
        all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(args.dataset)
    elif args.loader == 'anomaly_coldstart':
        task_type = 'anomaly_detection_coldstart'
        all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(args.dataset)
    else:
        raise ValueError(f"Unknown loader {args.loader}")
    
    print("done")
    
    # Determine input dimensions
    if task_type == 'classification':
        input_dims = train_data.shape[-1]
        training_data = train_data
    elif task_type == 'forecasting':
        input_dims = data.shape[-1]
        training_data = data
    else:  # anomaly detection
        # all_train_data is a dictionary, get first key's data to determine dimensions
        first_key = list(all_train_data.keys())[0]
        input_dims = 1  # Anomaly data is typically univariate
        
        # Generate training data using the same method as original TS2Vec
        from datautils import gen_ano_train_data
        training_data = gen_ano_train_data(all_train_data)
    
    # Create TS2Vec-MSM model
    model = TS2VecMSM(
        input_dims=input_dims,
        output_dims=args.repr_dims,
        device=device,
        lr=args.lr,
        batch_size=args.batch_size,
        max_train_length=args.max_train_length,
        after_epoch_callback=save_checkpoint_callback,
        # MSM parameters
        msm_weight=args.msm_weight,
        msm_mask_rate=args.msm_mask_rate,
        msm_decoder_depth=args.msm_decoder_depth,
        dynamic_lambda=args.dynamic_lambda
    )
    
    loss_log = []
    
    # Add irregular missing values if specified
    if args.irregular > 0:
        if task_type == 'classification':
            raise ValueError("Irregular missing values not supported for classification")
        elif task_type == 'forecasting':
            missing_ratio = args.irregular
            n_missing = int(data.size * missing_ratio)
            indices = np.random.choice(data.size, n_missing, replace=False)
            data.flat[indices] = np.nan
            print(f"Added {missing_ratio*100}% irregular missing values")
        else:  # anomaly detection
            missing_ratio = args.irregular
            for key in all_train_data:
                train_data_i = all_train_data[key]
                n_missing = int(train_data_i.size * missing_ratio)
                indices = np.random.choice(train_data_i.size, n_missing, replace=False)
                train_data_i.flat[indices] = np.nan
            print(f"Added {missing_ratio*100}% irregular missing values to anomaly data")
    
    # Training
    print("Training...")
    t = time.time()
    loss_log = model.fit(
        training_data,
        n_epochs=args.epochs,
        n_iters=args.iters,
        verbose=True
    )
    training_time = time.time() - t
    print(f"Training time: {datetime.timedelta(seconds=training_time)}")
    
    # Set model to evaluation mode
    model.eval()
    
    # Create output directory
    run_dir = 'training/' + args.dataset + '__' + args.run_name + '_' + str(datetime.datetime.now()).replace(' ', '_').replace(':', '_').replace('.', '_')
    os.makedirs(run_dir, exist_ok=True)
    
    # Save model
    model.save(f'{run_dir}/model.pkl')
    
    # Evaluation
    if args.eval:
        if task_type == 'classification':
            out, eval_res = tasks.eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='svm')
        elif task_type == 'forecasting':
            out, eval_res = tasks.eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols)
        elif task_type == 'anomaly_detection':
            out, eval_res = tasks.eval_anomaly_detection(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay)
        elif task_type == 'anomaly_detection_coldstart':
            out, eval_res = tasks.eval_anomaly_detection_coldstart(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay)
        else:
            assert False, 'unknown task type'
        
        # Save results
        print("Evaluation completed")
        print("Results:", eval_res)
        
        # Save evaluation results
        np.save(f'{run_dir}/out.npy', out)
        np.save(f'{run_dir}/eval_res.npy', eval_res)
        
        # Save training log
        np.save(f'{run_dir}/loss_log.npy', loss_log)
        
        print(f"Model and results saved to {run_dir}")
