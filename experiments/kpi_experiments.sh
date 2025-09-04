# TS2Vec-MSM Experiments for KPI Anomaly Detection

echo "=== KPI ANOMALY DETECTION EXPERIMENTS ==="

# Multiple seeds for statistical significance
seeds=(1 2 3 42 100)

for seed in "${seeds[@]}"; do
    echo "Running experiments with seed: $seed"
    
    # Contrastive only (λ=0)
    python -u train_msm.py kpi anomaly_contrastive_seed_$seed --loader anomaly --msm-weight 0.0 --repr-dims 320 --max-threads 8 --seed $seed --eval
    
    # MSM only (λ=1)
    python -u train_msm.py kpi anomaly_msm_seed_$seed --loader anomaly --msm-weight 1.0 --repr-dims 320 --max-threads 8 --seed $seed --eval
    
    # Hybrid static (λ=0.5)
    python -u train_msm.py kpi anomaly_hybrid_static_seed_$seed --loader anomaly --msm-weight 0.5 --repr-dims 320 --max-threads 8 --seed $seed --eval
    
    # Hybrid dynamic (λ changes over time)
    python -u train_msm.py kpi anomaly_hybrid_dynamic_seed_$seed --loader anomaly --msm-weight 0.5 --dynamic-lambda --repr-dims 320 --max-threads 8 --seed $seed --eval
done

echo "KPI anomaly detection experiments completed!"
