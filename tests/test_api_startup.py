import pytest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI
import os

@pytest.mark.asyncio
async def test_lifespan_local_success():
    """Test lifespan when MODEL_SOURCE is local and model exists."""
    app = FastAPI()
    
    with patch("src.api.Path.exists", return_value=True), \
         patch("src.api.joblib.load", return_value="mock_model"), \
         patch.dict(os.environ, {"MODEL_SOURCE": "local"}):
        
        from src.api import lifespan
        async with lifespan(app):
            assert app.state.model == "mock_model"
            assert app.state.model_source == "local"

@pytest.mark.asyncio
async def test_lifespan_local_missing():
    """Test lifespan when MODEL_SOURCE is local but model is missing."""
    app = FastAPI()
    
    with patch("src.api.Path.exists", return_value=False), \
         patch.dict(os.environ, {"MODEL_SOURCE": "local"}):
        
        from src.api import lifespan
        async with lifespan(app):
            assert app.state.model is None
            assert app.state.model_source == "local"

@pytest.mark.asyncio
async def test_lifespan_wandb_success():
    """Test lifespan when MODEL_SOURCE is wandb and download succeeds."""
    app = FastAPI()
    
    mock_artifact = MagicMock()
    mock_artifact.version = "v1"
    mock_artifact.aliases = ["prod"]
    
    # We patch the Api class *where it is imported from*
    with patch("wandb.Api") as mock_api_class, \
         patch("src.api.Path.exists", return_value=True), \
         patch("src.api.joblib.load", return_value="wandb_model"), \
         patch.dict(os.environ, {
             "MODEL_SOURCE": "wandb",
             "WANDB_API_KEY": "test-key"
         }):
        
        mock_api_instance = MagicMock()
        mock_api_class.return_value = mock_api_instance
        mock_api_instance.artifact.return_value = mock_artifact
        
        from src.api import lifespan
        async with lifespan(app):
            assert app.state.model == "wandb_model"
            assert "wandb" in app.state.model_source
            assert app.state.model_version == "wandb version (v1)"
