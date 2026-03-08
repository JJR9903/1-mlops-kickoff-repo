"""
Module: Model Training
----------------------
Role: Bundle preprocessing and algorithms into a single Pipeline and fit
    on training data.
Input: pandas.DataFrame (Processed) + ColumnTransformer (Recipe).
Output: Serialized scikit-learn Pipeline in `models/`.
"""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from xgboost import XGBClassifier, XGBRegressor


def _validate_and_fill_param_grid(
    param_grid: dict,
    estimator,                      # XGBClassifier or XGBRegressor instance
    default_grid: dict,
) -> dict:
    """
    1. Checks all keys in param_grid use the "model__" prefix and map to
       a real hyperparameter of the given estimator.
    2. For any key present in the default grid but missing from the caller's
       param_grid, fills it in with the default values and warns the user.

    Raises ValueError on unrecognized keys.
    Returns the validated + filled param_grid.
    """
    # Build the set of valid param names from the actual estimator
    valid_params = set(estimator.get_params().keys())   # e.g. {"max_depth", "n_estimators", ...}

    # ---- 1. Check for unrecognized keys --------------------------------
    unrecognized = []
    for key in param_grid:
        if not key.startswith("model__"):
            unrecognized.append((key, "missing 'model__' prefix"))
            continue
        bare = key.replace("model__", "", 1)
        if bare not in valid_params:
            unrecognized.append((key, f"'{bare}' is not a valid hyperparameter"))

    if unrecognized:
        details = "\n  ".join(f"{k} → {reason}" for k, reason in unrecognized)
        raise ValueError(
            f"[train] Invalid param_grid keys:\n  {details}\n"
            f"Valid hyperparameter names: {sorted(valid_params)}"
        )

    # ---- 2. Fill missing keys from default grid ------------------------
    filled_grid = dict(param_grid)  # copy — don't mutate caller's dict
    missing_keys = [k for k in default_grid if k not in filled_grid]

    if missing_keys:
        for key in missing_keys:
            filled_grid[key] = default_grid[key]
            print(
                f"[train] WARNING: '{key}' not in param_grid — "
                f"using default values: {default_grid[key]}"
            )

    return filled_grid




def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor,
    problem_type: str,
    param_grid: dict = None,          # ← caller decides search space
) -> Pipeline:
    """
    Inputs:
    - X_train:      Feature DataFrame (training split only — NO test data!)
    - y_train:      Target Series (training split only)
    - preprocessor: Unfitted ColumnTransformer from get_feature_preprocessor()
    - problem_type: "classification" or "regression"
    - param_grid:   Optional dict of hyperparameters to search over.
                    Keys must use "model__" prefix (e.g. "model__max_depth").
                    If None, a sensible default grid is used.

    Outputs:
    - Fitted sklearn Pipeline [("preprocess", preprocessor), 
        ("model", estimator)]
      Already refitted on the full X_train with best hyperparameters found.
    """
    print(f"[train] Starting model training | problem_type='{problem_type}'")

    # ------------------------------------------------------------------ #
    # 1. Build pipeline + set CV strategy and scoring by problem type
    # ------------------------------------------------------------------ #
    if problem_type == "classification":
        estimator = XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring = "f1"

    elif problem_type == "regression":
        estimator = XGBRegressor(
            eval_metric="rmse",
            random_state=42,
        )
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scoring = "neg_root_mean_squared_error"

    else:
        raise ValueError(f"[train] Unknown problem_type='{problem_type}'. Use 'classification' or 'regression'.")

    # ------------------------------------------------------------------ #
    # 2. Validate + fill param_grid
    # ------------------------------------------------------------------ #
    default_param_grid = {
        "model__max_depth":        [3, 4, 5, 6],
        "model__learning_rate":    [0.01, 0.05, 0.1, 0.15],
        "model__n_estimators":     [100, 200, 300],
        "model__subsample":        [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "model__gamma":            [0, 0.1, 0.2],
    }
    
    param_grid = _validate_and_fill_param_grid(
        param_grid   = param_grid or {},
        estimator    = estimator,
        default_grid = default_param_grid,
    )
    
    # ------------------------------------------------------------------ #
    # 3. Build pipeline and run grid search
    # ------------------------------------------------------------------ #
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model",      estimator),
    ])

    # ------------------------------------------------------------------ #
    # 4. Grid search — refit=True (default) refits on full X_train
    # ------------------------------------------------------------------ #
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,   # explicit — best_estimator_ is already the final model
    )
    grid_search.fit(X_train, y_train)

    # ------------------------------------------------------------------ #
    # 4. Report results
    # ------------------------------------------------------------------ #
    best_score = (
        grid_search.best_score_
        if problem_type == "classification"
        else -grid_search.best_score_   # flip sign: neg_rmse → rmse
    )
    metric_label = "F1" if problem_type == "classification" else "RMSE"

    print(f"[train] Best Parameters: {grid_search.best_params_}")
    print(f"[train] Best CV {metric_label}: {best_score:.4f}")

    # grid_search.best_estimator_ is the refitted on X_train
    return grid_search.best_estimator_