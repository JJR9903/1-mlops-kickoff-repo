import logging
import yaml
from pathlib import Path
from typing import Any

def load_config() -> dict[str, Any]:
    """Load configuration from config.yaml."""
    config_path = Path("config.yaml")
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_logger(name: str) -> logging.Logger:
    """
    Create a robust logger for modules based on config.yaml.
    - Logs to both console and a file.
    - Level and format are configurable.
    """
    config = load_config()
    logging_cfg = config.get("logging", {})
    
    log_level_str = logging_cfg.get("level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_file = logging_cfg.get("log_file", "logs/main.log")
    log_format_type = logging_cfg.get("format", "text")

    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if logger is already configured
    if logger.handlers:
        return logger

    logger.setLevel(log_level)

    # Standard format for text
    if log_format_type == "text":
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        # Fallback to simple format if json or other is requested but not implemented
        formatter = logging.Formatter(
            fmt='{"time": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}',
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Ensure log directory exists
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # File Handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(log_level)
    fh.setFormatter(formatter)

    # Console Handler
    sh = logging.StreamHandler()
    sh.setLevel(log_level)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    
    # Prevent propagation to root logger to avoid duplicate logs in some environments
    logger.propagate = False

    return logger
