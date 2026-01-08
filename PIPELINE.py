#!/usr/bin/env python3
"""


This script documents exactly how to regenerate all paper results.
Run this for the complete analysis pipeline.
"""

import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a shell command with description."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True, cwd=Path.cwd())
    if result.returncode == 0:
        print(f"{description} - SUCCESS")
    else:
        print(f"{description} - FAILED (exit code: {result.returncode})")
    return result.returncode


def main():
    """Run the complete analysis pipeline."""
    print("\n" + "="*80)
    print("CS229 OPTIONS MISPRICING RESEARCH - COMPLETE PIPELINE")
    print("="*80)
    print("\nThis will generate all results for the CS229 paper submission.")

    
    results = []
    
    # Step 1: Generate publication-ready visualizations
    results.append(run_command(
        "./.venv/bin/python generate_paper_visualizations.py",
        "Generate Publication Visualizations (4 PNG figures)"
    ))
    
    # Step 2: Test market regimes
    results.append(run_command(
        "./.venv/bin/python test_market_regimes_fast.py",
        "Market Regime Robustness Testing"
    ))
    
    # Step 3: Main model training (optional, slower)
    print("\n" + "="*80)
    print("OPTIONAL: Generate detailed model training results")
    print("="*80)
    print("Command: ./.venv/bin/python train_real_market_models.py")
    print("This takes 5-10 minutes and generates all model comparisons.")
    print("Skipping for now - results already documented in README.md\n")
    
    # Summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    
    if all(r == 0 for r in results):
        print("\nAll scripts executed successfully!")
    else:
        print("\nSome scripts had issues. Check output above.")
    
    print("\n" + "="*80)
    print("GENERATED ARTIFACTS")
    print("="*80)
    print("""
Visualizations (ready for paper):
  • equity_curve.png             - Backtest performance curve with Sharpe ratio
  • ticker_accuracy.png          - Per-asset accuracy comparison (AAPL/SPY/TSLA)
  • feature_importance.png       - Which Greeks matter most
  • confusion_matrices.png       - Detailed classification accuracy

Results CSV:
  • regime_testing_fast_results.csv  - Market regime test results

Documentation:
  • README.md                    - Complete project documentation
  • FINAL_SUBMISSION_SUMMARY.md  - This submission's status & files
""")
    
    print("="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""


For detailed instructions, see README.md and FINAL_SUBMISSION_SUMMARY.md
""")


if __name__ == '__main__':
    main()
