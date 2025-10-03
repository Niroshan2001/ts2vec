import numpy as np
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, train_test_split

# Add XGBoost import with fallback
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

def fit_svm(features, y, MAX_SAMPLES=10000):
    nb_classes = np.unique(y, return_counts=True)[1].shape[0]
    train_size = features.shape[0]

    svm = SVC(C=np.inf, gamma='scale')
    if train_size // nb_classes < 5 or train_size < 50:
        return svm.fit(features, y)
    else:
        grid_search = GridSearchCV(
            svm, {
                'C': [
                    0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000,
                    np.inf
                ],
                'kernel': ['rbf'],
                'degree': [3],
                'gamma': ['scale'],
                'coef0': [0],
                'shrinking': [True],
                'probability': [False],
                'tol': [0.001],
                'cache_size': [200],
                'class_weight': [None],
                'verbose': [False],
                'max_iter': [10000000],
                'decision_function_shape': ['ovr'],
                'random_state': [None]
            },
            cv=5, n_jobs=5
        )
        # If the training set is too large, subsample MAX_SAMPLES examples
        if train_size > MAX_SAMPLES:
            split = train_test_split(
                features, y,
                train_size=MAX_SAMPLES, random_state=0, stratify=y
            )
            features = split[0]
            y = split[2]
            
        grid_search.fit(features, y)
        return grid_search.best_estimator_

def fit_lr(features, y, MAX_SAMPLES=100000):
    # If the training set is too large, subsample MAX_SAMPLES examples
    if features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            features, y,
            train_size=MAX_SAMPLES, random_state=0, stratify=y
        )
        features = split[0]
        y = split[2]
        
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            random_state=0,
            max_iter=1000000,
            multi_class='ovr'
        )
    )
    pipe.fit(features, y)
    return pipe

def fit_knn(features, y):
    pipe = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=1)
    )
    pipe.fit(features, y)
    return pipe

def fit_ridge(train_features, train_y, valid_features, valid_y, MAX_SAMPLES=100000):
    # If the training set is too large, subsample MAX_SAMPLES examples
    if train_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            train_features, train_y,
            train_size=MAX_SAMPLES, random_state=0
        )
        train_features = split[0]
        train_y = split[2]
    if valid_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            valid_features, valid_y,
            train_size=MAX_SAMPLES, random_state=0
        )
        valid_features = split[0]
        valid_y = split[2]
    
    # Use wider range of alpha values with higher regularization to avoid ill-conditioning
    alphas = [0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    valid_results = []
    for alpha in alphas:
        lr = Ridge(alpha=alpha, solver='cholesky').fit(train_features, train_y)  # More stable solver
        valid_pred = lr.predict(valid_features)
        score = np.sqrt(((valid_pred - valid_y) ** 2).mean()) + np.abs(valid_pred - valid_y).mean()
        valid_results.append(score)
    best_alpha = alphas[np.argmin(valid_results)]
    
    lr = Ridge(alpha=best_alpha, solver='cholesky')  # Use stable solver
    lr.fit(train_features, train_y)
    return lr

def fit_xgboost(train_features, train_y, valid_features, valid_y, MAX_SAMPLES=100000):
    """Fit XGBoost regressor with validation-based early stopping
    
    This provides a more powerful non-linear regression head compared to Ridge,
    especially beneficial for long-horizon forecasting where complex temporal
    relationships need to be captured.
    
    Args:
        train_features: Training feature matrix
        train_y: Training targets
        valid_features: Validation feature matrix  
        valid_y: Validation targets
        MAX_SAMPLES: Maximum samples to use for training (for efficiency)
        
    Returns:
        Trained XGBoost model (or Ridge as fallback if XGBoost unavailable)
    """
    if not HAS_XGB:
        print("XGBoost not available, falling back to Ridge regression")
        return fit_ridge(train_features, train_y, valid_features, valid_y, MAX_SAMPLES)
    
    # Subsample if dataset too large for efficient training
    if train_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            train_features, train_y,
            train_size=MAX_SAMPLES, random_state=0
        )
        train_features = split[0]
        train_y = split[2]
    
    # XGBoost configuration optimized for time series forecasting
    model = xgb.XGBRegressor(
        n_estimators=100,        # Moderate number to avoid overfitting
        max_depth=6,             # Reasonable depth for capturing interactions
        learning_rate=0.1,       # Conservative learning rate
        subsample=0.8,           # Row subsampling for regularization
        colsample_bytree=0.8,    # Feature subsampling
        random_state=42,
        n_jobs=-1,               # Use all CPU cores
        verbosity=0              # Suppress output
    )
    
    # Fit with early stopping based on validation performance
    model.fit(
        train_features, train_y,
        eval_set=[(valid_features, valid_y)],
        early_stopping_rounds=10,
        verbose=False
    )
    
    return model
