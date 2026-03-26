from pathlib import Path
from dotenv import load_dotenv
import os

ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)


def get_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def get_wandb_api_key() -> str:
    return get_env_var("WANDB_API_KEY")


def get_wandb_project() -> str:
    return get_env_var("WANDB_PROJECT")


def get_wandb_entity() -> str:
    return get_env_var("WANDB_ENTITY")