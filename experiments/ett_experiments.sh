# TS2Vec-MSM Experiments for ETT Forecasting Dataset

echo "=== ETT FORECASTING EXPERIMENTS ==="

# ETTh1 Experiments
echo "Running ETTh1 experiments..."

# Contrastive only (λ=0)
python -u train_msm.py ETTh1 forecast_contrastive --loader forecast_csv --msm-weight 0.0 --repr-dims 320 --max-threads 8 --seed 42 --eval

# MSM only (λ=1) 
python -u train_msm.py ETTh1 forecast_msm --loader forecast_csv --msm-weight 1.0 --repr-dims 320 --max-threads 8 --seed 42 --eval

# Hybrid static (λ=0.5)
python -u train_msm.py ETTh1 forecast_hybrid_static --loader forecast_csv --msm-weight 0.5 --repr-dims 320 --max-threads 8 --seed 42 --eval

# Hybrid dynamic (λ changes over time)
python -u train_msm.py ETTh1 forecast_hybrid_dynamic --loader forecast_csv --msm-weight 0.5 --dynamic-lambda --repr-dims 320 --max-threads 8 --seed 42 --eval

# ETTh2 Experiments  
echo "Running ETTh2 experiments..."
python -u train_msm.py ETTh2 forecast_contrastive --loader forecast_csv --msm-weight 0.0 --repr-dims 320 --max-threads 8 --seed 42 --eval
python -u train_msm.py ETTh2 forecast_hybrid_dynamic --loader forecast_csv --msm-weight 0.5 --dynamic-lambda --repr-dims 320 --max-threads 8 --seed 42 --eval

# ETTm1 Experiments
echo "Running ETTm1 experiments..."
python -u train_msm.py ETTm1 forecast_contrastive --loader forecast_csv --msm-weight 0.0 --repr-dims 320 --max-threads 8 --seed 42 --eval
python -u train_msm.py ETTm1 forecast_hybrid_dynamic --loader forecast_csv --msm-weight 0.5 --dynamic-lambda --repr-dims 320 --max-threads 8 --seed 42 --eval
