import os
import logging
from pathlib import Path
from src.logger import get_logger

def test_logger_initialization():
    logger = get_logger("test_module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_module"
    assert logger.level == logging.INFO

def test_logger_file_creation():
    log_file = Path("logs/test.log")
    if log_file.exists():
        log_file.unlink()
    
    # Mocking config or just using default which should create logs/main.log
    # But specifically for this test, we want to see if a file is created.
    logger = get_logger("file_test")
    logger.info("Test message")
    
    # Check if the default log file is created
    default_log = Path("logs/main.log")
    assert default_log.exists()
    
    with open(default_log, "r") as f:
        content = f.read()
        assert "Test message" in content

if __name__ == "__main__":
    # Simple manual verification script
    try:
        test_logger_initialization()
        print("Logger initialization test PASSED")
        test_logger_file_creation()
        print("Logger file creation test PASSED")
    except Exception as e:
        print(f"Tests FAILED: {e}")
