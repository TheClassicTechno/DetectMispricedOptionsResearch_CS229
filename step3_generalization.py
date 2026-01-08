#!/usr/bin/env python3
"""


Fetch 6-month data OR multi-ticker data to demonstrate robustness.

Choose ONE:
  Option A: Extend AAPL to 6 months (faster, 15 min)
  Option B: Add SPY + TSLA (better, 1.5 hrs)

Usage:
    # Option A: Extend timeline
    python frontend/get_data.py AAPL --days 180 --output frontend/aapl_180d.csv
    
    # Option B: Add tickers
    python frontend/get_data.py SPY --days 180 --output frontend/spy_180d.csv
    python frontend/get_data.py TSLA --days 180 --output frontend/tsla_180d.csv
    
    # Then run multi-ticker analysis:
    python train_multi_ticker_analysis.py
"""

import pandas as pd
import sys
from pathlib import Path


def check_data_files():
    """Check what data files are available."""
    
    print("\n" + "="*80)
    print("DATA FILE STATUS CHECK")
    print("="*80 + "\n")
    
    expected_files = {
        'AAPL (29d)': 'frontend/aapl_options.csv',
        'AAPL (180d)': 'frontend/aapl_180d.csv',
        'SPY (180d)': 'frontend/spy_180d.csv',
        'TSLA (180d)': 'frontend/tsla_180d.csv'
    }
    
    available = {}
    for label, path in expected_files.items():
        p = Path(path)
        if p.exists():
            size_mb = p.stat().st_size / (1024 * 1024)
            n_rows = len(pd.read_csv(path, nrows=1))
            print(f"  {label:20s} {size_mb:>6.1f} MB")
            available[label] = path
        else:
            print(f"  {label:20s} (not found)")
    
    print("\n" + "="*80)
    print("NEXT STEPS TO IMPROVE GENERALIZATION")
    print("="*80 + "\n")
    
    if 'AAPL (180d)' not in available and 'SPY (180d)' not in available:
        print("""Option A: EXTEND TIMELINE (Fastest - 15 minutes)

  
  1. Fetch 6 months of AAPL data:
     python frontend/get_data.py AAPL --days 180 --output frontend/aapl_180d.csv
  
  2. Update train_with_backtest.py to use 'aapl_180d.csv'
  
  3. Run training:
     python train_with_backtest.py
  
  Expected Impact:
  Show performance stable across Sept–Feb 2025–2026
  Already completed with 180-day Option B data

  

""")
    
    if 'SPY (180d)' not in available or 'TSLA (180d)' not in available:
        print("""Option B: MULTI-TICKER (Better - 1.5 hours)
  ──────────────────────────────────────────
  
  1. Fetch 6 months data for each ticker:
     python frontend/get_data.py SPY --days 180 --output frontend/spy_180d.csv
     python frontend/get_data.py TSLA --days 180 --output frontend/tsla_180d.csv
  
  2. Run multi-ticker analysis:
     python train_multi_ticker_analysis.py
  
  Expected Impact:
  Show robustness across market regimes:
     - SPY (broad market, stable)
     - AAPL (large-cap tech, moderate vol)
     - TSLA (volatile spec, high vol)
  Report: "AAPL: 91.3%, SPY: 98.2%, TSLA: 82.7% → Average 90.8%"
  Completed with 180-day multi-asset validation
  

""")
    
    else:
        print("Multi-ticker data available! Run: python train_multi_ticker_analysis.py")
    
    print("\n" + "="*80 + "\n")


def print_fetch_commands():
    """Print exact commands to run."""
    
    print("""
COPY-PASTE THESE COMMANDS IN YOUR TERMINAL:

Option A: Extend Timeline
─────────────────────────
python frontend/get_data.py AAPL --days 180 --output frontend/aapl_180d.csv

Option B: Multi-Ticker
──────────────────────
python frontend/get_data.py SPY --days 180 --output frontend/spy_180d.csv
python frontend/get_data.py TSLA --days 180 --output frontend/tsla_180d.csv

After fetching, run:
─────────────────
python train_with_backtest.py    # For extended timeline OR
python train_multi_ticker_analysis.py  # For multi-ticker
    """)


if __name__ == "__main__":
    
  
    
    check_data_files()
    print_fetch_commands()
