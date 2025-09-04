# Ablation Study Scripts for TS2Vec-MSM
# Run these experiments to validate the hybrid approach

# ===== CONTRASTIVE ONLY (λ=0, equivalent to original TS2Vec) =====
echo "Running Contrastive-Only Experiments..."

# Classification
python -u train_msm.py UCR ECG200_contrastive --loader UCR --msm-weight 0.0 --repr-dims 320 --seed 42 --eval
python -u train_msm.py UEA Epilepsy_contrastive --loader UEA --msm-weight 0.0 --repr-dims 320 --seed 42 --eval

# Forecasting
python -u train_msm.py ETTh1 forecast_contrastive --loader forecast_csv --msm-weight 0.0 --repr-dims 320 --seed 42 --eval

# Anomaly Detection
python -u train_msm.py kpi anomaly_contrastive --loader anomaly --msm-weight 0.0 --repr-dims 320 --seed 42 --eval

# ===== MSM ONLY (λ=1, generative only) =====
echo "Running MSM-Only Experiments..."

# Classification
python -u train_msm.py UCR ECG200_msm --loader UCR --msm-weight 1.0 --repr-dims 320 --seed 42 --eval
python -u train_msm.py UEA Epilepsy_msm --loader UEA --msm-weight 1.0 --repr-dims 320 --seed 42 --eval

# Forecasting
python -u train_msm.py ETTh1 forecast_msm --loader forecast_csv --msm-weight 1.0 --repr-dims 320 --seed 42 --eval

# Anomaly Detection
python -u train_msm.py kpi anomaly_msm --loader anomaly --msm-weight 1.0 --repr-dims 320 --seed 42 --eval

# ===== HYBRID STATIC (λ=0.5, balanced hybrid) =====
echo "Running Hybrid Static Experiments..."

# Classification
python -u train_msm.py UCR ECG200_hybrid_static --loader UCR --msm-weight 0.5 --repr-dims 320 --seed 42 --eval
python -u train_msm.py UEA Epilepsy_hybrid_static --loader UEA --msm-weight 0.5 --repr-dims 320 --seed 42 --eval

# Forecasting
python -u train_msm.py ETTh1 forecast_hybrid_static --loader forecast_csv --msm-weight 0.5 --repr-dims 320 --seed 42 --eval

# Anomaly Detection
python -u train_msm.py kpi anomaly_hybrid_static --loader anomaly --msm-weight 0.5 --repr-dims 320 --seed 42 --eval

# ===== HYBRID DYNAMIC (λ changes over time) =====
echo "Running Hybrid Dynamic Experiments..."

# Classification
python -u train_msm.py UCR ECG200_hybrid_dynamic --loader UCR --msm-weight 0.5 --dynamic-lambda --repr-dims 320 --seed 42 --eval
python -u train_msm.py UEA Epilepsy_hybrid_dynamic --loader UEA --msm-weight 0.5 --dynamic-lambda --repr-dims 320 --seed 42 --eval

# Forecasting
python -u train_msm.py ETTh1 forecast_hybrid_dynamic --loader forecast_csv --msm-weight 0.5 --dynamic-lambda --repr-dims 320 --seed 42 --eval

# Anomaly Detection
python -u train_msm.py kpi anomaly_hybrid_dynamic --loader anomaly --msm-weight 0.5 --dynamic-lambda --repr-dims 320 --seed 42 --eval
