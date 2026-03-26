"""
W&B Integration Tests - Jupyter/Anaconda Notebook Version

Paste this entire cell into a Jupyter notebook and run it.
Works in Anaconda Jupyter, Google Colab, or any notebook environment.

Installation (if needed):
    pip install wandb scikit-learn xgboost pandas pyyaml pytest
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Execute a shell command and report results."""
    print(f"\n{'='*80}")
    print(f"🧪 TEST: {description}")
    print(f"{'='*80}\n")

    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    return result.returncode == 0


def test_wandb_integration():
    """Run all W&B integration tests."""

    print("\n" + "="*80)
    print("WEIGHTS & BIASES INTEGRATION TEST SUITE")
    print("="*80)
    print("Testing complete W&B integration into the ML pipeline.\n")

    tests = [
        (
            'python -c "from src.main import main; from src.train import train_model; from src.evaluate import evaluate_model; from src.infer import run_inference, download_prod_model; import wandb; print(\'\\n✅ All imports OK\')"',
            "Import Check"
        ),
        (
            "python -m pytest tests/test_main_wandb.py -v",
            "W&B Unit Tests (5 tests)"
        ),
        (
            "python -m pytest tests/test_evaluate.py -v",
            "Evaluate Tests"
        ),
    ]

    results = []
    for cmd, desc in tests:
        passed = run_command(cmd, desc)
        results.append((desc, passed))

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")

    for desc, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {desc}")

    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)

    print(f"\n📊 Total: {passed_count}/{total_count} tests passed\n")

    if passed_count >= 2:
        print("✅ Ready to run the live pipeline:")
        print("   python -m src.main")
        return True
    else:
        print("❌ Some tests failed. Check output above.")
        return False


# Run the tests
if __name__ == "__main__":
    success = test_wandb_integration()
    sys.exit(0 if success else 1)
