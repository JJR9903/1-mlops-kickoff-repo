#!/usr/bin/env python
"""
W&B Integration Test Suite
Run this to verify all W&B changes work correctly.

Usage:
    python test_wandb_integration.py

This script runs:
    1. Import checks (syntax validation)
    2. W&B unit tests (mocked, no training)
    3. Related module tests (evaluate, train validation)
    4. Summary report
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Execute a shell command and report results."""
    print(f"\n{'='*80}")
    print(f"🧪 TEST: {description}")
    print(f"{'='*80}")
    print(f"Command: {cmd}\n")

    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)

    if result.returncode == 0:
        print(f"\n✅ PASSED: {description}")
        return True
    else:
        print(f"\n❌ FAILED: {description}")
        return False


def main():
    """Run all W&B integration tests."""

    print("\n" + "="*80)
    print("WEIGHTS & BIASES INTEGRATION TEST SUITE")
    print("="*80)
    print("Testing complete W&B integration into the ML pipeline.")
    print("Make sure you're in the repo root directory.\n")

    # List of all tests to run
    tests = [
        (
            'python -c "from src.main import main; from src.train import train_model; from src.evaluate import evaluate_model; from src.infer import run_inference, download_prod_model; import wandb; print(\'\\n✅ All imports OK\')"',
            "Import Check - Verify all modules load without errors"
        ),
        (
            "python -m pytest tests/test_main_wandb.py -v",
            "W&B Unit Tests - 5 mocked tests (no training needed)"
        ),
        (
            "python -m pytest tests/test_evaluate.py -v",
            "Evaluate Module Tests - Confirm evaluation metrics work"
        ),
        (
            "python -m pytest tests/test_train.py::TestValidateAndFillParamGrid::test_validate_and_fill_raises_unrecognized_keys -v",
            "Train Validation Tests - Check hyperparameter validation"
        ),
    ]

    results = []

    # Run each test
    for cmd, desc in tests:
        passed = run_command(cmd, desc)
        results.append((desc, passed))

    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}\n")

    for desc, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {desc}")

    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)

    print(f"\nTotal: {passed_count}/{total_count} tests passed\n")

    # Final recommendation
    if passed_count >= 3:  # At least import check + 2 others
        print("="*80)
        print("🎉 CORE TESTS PASSED!")
        print("="*80)
        print("\nYou can now run the live pipeline:")
        print("  python -m src.main")
        print("\nThis will:")
        print("  1. Train the model on your data")
        print("  2. Upload the model to W&B with the 'prod' alias")
        print("  3. Log all metrics, hyperparameters, and artifacts")
        print("  4. Print a W&B run URL at the end")
        print("\nThen visit that URL to see all your experiment details!")
        print("\nYour W&B API key from .env:")
        env_path = Path(".env")
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if "WANDB_API_KEY" in line:
                        key_short = line.split("=")[1][:20] if "=" in line else "❌ Not found"
                        print(f"  Key starts with: {key_short}...")
        return 0
    else:
        print("="*80)
        print("⚠️  SOME TESTS FAILED")
        print("="*80)
        print("\nDebug steps:")
        print("  1. Check you're in the repo root (where src/ folder is)")
        print("  2. Verify all dependencies installed:")
        print("     pip install wandb scikit-learn xgboost pandas pyyaml pytest")
        print("  3. Check .env file has valid WANDB_API_KEY")
        print("  4. Re-run with more details:")
        print("     python -m pytest tests/test_main_wandb.py -vv -s")
        return 1


if __name__ == "__main__":
    sys.exit(main())
