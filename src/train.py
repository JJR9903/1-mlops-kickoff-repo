"""
Module: Model Training
----------------------
Role: Bundle preprocessing and algorithms into a single Pipeline and fit
    on training data.
Input: pandas.DataFrame (Processed) + ColumnTransformer (Recipe).
Output: Serialized scikit-learn Pipeline in `models/`.
"""
import pandas as pd
import wandb
from src.logger import get_logger

logger = get_logger(__name__)
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
    valid_params = set(estimator.get_params().keys())  # e.g.{"max_depth", ...}

    # ---- 1. Check for unrecognized keys --------------------------------
    unrecognized = []
    for key in param_grid:
        if not key.startswith("model__"):
            unrecognized.append((key, "missing 'model__' prefix"))
            continue
        bare = key.replace("model__", "", 1)
        if bare not in valid_params:
            msg = f"'{bare}' is not a valid hyperparameter"
            unrecognized.append((key, msg))

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
            logger.warning(
                f"'{key}' not in param_grid — "
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
    logger.info(f"Starting model training | problem_type='{problem_type}'")

    # ------------------------------------------------------------------ #
    # 1. Build pipeline + set CV strategy and scoring by problem type
    # ------------------------------------------------------------------ #
    if problem_type == "classification":
        estimator = XGBClassifier(
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
        msg = (
            f"[train] Unknown problem_type='{problem_type}'. "
            f"Use 'classification' or 'regression'."
        )
        raise ValueError(msg)

    # ------------------------------------------------------------------ #
    # 2. Validate + fill param_grid
    # ------------------------------------------------------------------ #
    default_param_grid = {
        "model__max_depth":        [3, 6],
        "model__learning_rate":    [0.01, 0.15],
        "model__n_estimators":     [100, 300],
        "model__subsample":        [0.7, 0.9],
        "model__colsample_bytree": [0.7, 0.9],
        "model__gamma":            [0, 0.1],
    }

    param_grid = _validate_and_fill_param_grid(
        param_grid=param_grid or {},
        estimator=estimator,
        default_grid=default_param_grid,
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

    logger.info(f"Best Parameters: {grid_search.best_params_}")
    logger.info(f"Best CV {metric_label}: {best_score:.4f}")

    # ------------------------------------------------------------------ #
    # 5. W&B: Log training hyperparameters and CV score
    # ------------------------------------------------------------------ #
    if wandb.run is not None:
        # Strip "model__" prefix for cleaner display in W&B
        clean_params = {
            k.replace("model__", ""): v
            for k, v in grid_search.best_params_.items()
        }
        wandb.config.update({"best_params": clean_params})
        wandb.log({
            f"train/best_cv_{metric_label.lower()}": best_score,
            "train/n_cv_folds": cv.get_n_splits(),
        })
        logger.info("Logged training metrics to W&B")

    # grid_search.best_estimator_ is the refitted on X_train
    return grid_search.best_estimator_
