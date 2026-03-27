from contextlib import asynccontextmanager
from pathlib import Path
from typing import List
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
import numpy as np
import wandb
from src.clean_data import clean_dataframe
from src.validate import validate_dataframe
from src.infer import run_inference
from src.config import load_config

CFG = load_config()

# Check if maybe we should switch settings to Telco (similar to main.py logic)


from src.schemas import TelcoChurnInput, PredictResponse

# ==========================================
# 2. Lifespan & Global State
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    import os
    # 1. Path preparation
    model_dir = Path("models")
    model_path = model_dir / "model.joblib"
    model_source = os.environ.get("MODEL_SOURCE", "local")
    
    # Ensure directory exists
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Ensured directory exists: {model_dir.absolute()}")
    
    app.state.model = None
    app.state.model_source = "local"
    
    # 2. Optional: Download from W&B
    if model_source == "wandb":
        print("MODEL_SOURCE is 'wandb'. "
              "Attempting to download production model...")
        wandb_entity = os.environ.get("WANDB_ENTITY", "rom-gelin-ie-university")
        wandb_project = os.environ.get("WANDB_PROJECT", "mlops-churn-prediction")        
        model_name = os.environ.get("WANDB_MODEL_NAME", "churn-model")
        model_alias = os.environ.get("WANDB_MODEL_ALIAS", "prod")        
        
        artifact_n = f"{wandb_entity}/{wandb_project}/{model_name}:{model_alias}"
        if not wandb_entity:
            artifact_n = f"{wandb_project}/{model_name}:{model_alias}"
            
        try:
            api = wandb.Api()
            print(f"Connecting to W&B to fetch artifact: {artifact_n}")
            artifact = api.artifact(artifact_n, type="model")
            artifact.download(root=str(model_dir))
            
            app.state.model_source = f"wandb ({artifact_n})"
            app.state.model_version = f"wandb version ({artifact.version})"
            app.state.model_aliases = f"wandb aliases ({artifact.aliases})"
            print(f"Successfully downloaded {artifact_n} "
                  f"to {model_path.absolute()}")
        except Exception as e:
            print(f"Failed to download model from W&B: {e}")
            app.state.model_source = "local (fallback)"

    # 3. Final Load
    print(f"Searching for model file at: {model_path.absolute()}")
    if not model_path.exists():
        print(f"CRITICAL: Model file NOT found at {model_path.absolute()}. "
              "Health check will fail.")
        app.state.model = None
    else:
        try:
            app.state.model = joblib.load(model_path)
            print("SUCCESS: Model loaded into app.state "
                  f"from {model_path.absolute()}")
            print(f"Source: {app.state.model_source}")
        except Exception as e:
            print(f"ERROR: Failed to load model file: {e}")
            app.state.model = None

    yield
    # Cleanup on shutdown
    app.state.model = None

# Create the FastAPI application
app = FastAPI(
    title="MLOps Prediction API",
    description="Live prediction service using the trained model pipeline.",
    lifespan=lifespan
)

# ==========================================
# 3. The Endpoints
# ==========================================
@app.get("/health")
def health_check():
    """
    Heartbeat endpoint.
    Checks if the model was loaded successfully.
    """
    if getattr(app.state, "model", None) is None:
        return {"status": "unhealthy", "reason": "Model is not loaded."}
    return {
        "status": "ok", 
        "model_version": getattr(app.state, "model_version", "local"),
        "model_source": getattr(app.state, "model_source", "local"),
        "model_aliases": getattr(app.state, "model_aliases", "local")
    }

@app.post("/predict", response_model=List[PredictResponse])
def predict(payload: List[TelcoChurnInput]):
    """
    On-demand prediction endpoint.
    Passes data through the exact same preprocessing pipeline used in training.
    """
    if getattr(app.state, "model", None) is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    
    try:
        # Convert Request to DataFrame
        # dict() creates dictionaries representing each row.
        df_raw = pd.DataFrame([item.model_dump() for item in payload])
        
        # In to clean_dataframe
        # clean_dataframe operates with the assumption that target_column is in df,
        # but safely handles if it is not (target mapping checks if in cols).
        target_column = CFG.get("target_config", {}).get("column", "target")
        df_clean = clean_dataframe(df_raw, target_column=target_column)
        
        # Validation checks
        schema = CFG.get("schema", {})
        target_config = CFG.get("target_config", {})
        
        # validate_dataframe expects the target column for some checks 
        # For inference we temporarily add a dummy target column so validation doesn't fail Check 4
        # or we just temporarily bypass validate_dataframe if it strictly demands target.
        # Looking at validate.py it aggressively checks if target_column is not in df.columns:
        if target_column not in df_clean.columns:
            df_clean[target_column] = target_config.get("allowed_classes", [0])[0]  # Dummy target
            
        validate_dataframe(df_clean, schema=schema, target_config=target_config)
        
        # Drop dummy target for inference
        X_infer = df_clean.drop(columns=[target_column])
        
        # Run inference
        df_preds = run_inference(app.state.model, X_infer, include_proba=True)
        
        # Format response
        results = []
        for _, row in df_preds.iterrows():
            results.append(
                PredictResponse(
                    prediction=int(row["prediction"]),
                    proba=float(row["proba"]) if pd.notna(row.get("proba")) else None
                )
            )
        
        return results

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


from src.ui import init as init_ui


init_ui(app)

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8050))
    # Production mode: reload=False
    uvicorn.run("src.api:app", host="0.0.0.0", port=port, reload=False)
