# aapl_options_dump.py
# Quick export of Apple options (calls & puts) from Yahoo Finance.
# Output: aapl_options.csv

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from dateutil import parser
from py_vollib.black_scholes_merton import black_scholes_merton
from py_vollib.black_scholes_merton.greeks import analytical as greeks
import argparse
import yfinance as yf

def mid_price(bid, ask, last):
    if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > 0:
        return 0.5 * (bid + ask)
    if pd.notna(last) and last > 0:
        return last
    if pd.notna(bid) and bid > 0:
        return bid
    if pd.notna(ask) and ask > 0:
        return ask
    return np.nan

def get_risk_free_rate():
    """Fetches the 13-week US Treasury bill yield as a proxy for the risk-free rate."""
    try:
        rate = yf.Ticker("^IRX").history(period="5d")['Close'].iloc[-1] / 100
        return rate if pd.notna(rate) else 0.05 # Fallback rate
    except (IndexError, KeyError):
        return 0.05 # Fallback rate if fetch fails

def classify_price(row):
    """Classifies an option as overpriced, underpriced, or fair."""
    mkt_price = row['mkt_price']
    bs_price = row['bs_price']
    if pd.isna(mkt_price) or pd.isna(bs_price) or bs_price <= 0:
        return 'N/A'
    if mkt_price > bs_price * 1.1:
        return 'overpriced'
    if mkt_price < bs_price * 0.9:
        return 'underpriced'
    return 'fair'

def main(ticker="AAPL", out_csv="aapl_options.csv", max_expiries=None, days=5):
    tkr = yf.Ticker(ticker)

    # latest close and previous close
    hist_period = f"{days+5}d" # Fetch a bit more history to be safe
    hist = tkr.history(period=hist_period, interval="1d")["Close"].ffill()
    if len(hist) < 1:
        raise SystemExit(f"Could not fetch historical data for {ticker}")
    
    # Use the last `days` worth of data
    hist = hist.tail(days)
    if len(hist) < days:
        print(f"Warning: Could only fetch {len(hist)} days of data, not the requested {days}.")

    # VIX, risk-free rate, and dividend yield.
    vix = yf.Ticker("^VIX").history(period="1d")["Close"].iloc[0]
    risk_free_rate = get_risk_free_rate()
    dividend_yield = tkr.info.get("dividendYield") or 0.0

    # expirations
    expiries = tkr.options or []
    if max_expiries:
        expiries = expiries[:max_expiries]
    if not expiries:
        raise SystemExit("No option expirations available from Yahoo.")

    rows = []

    for date_index in range(len(hist)):
        if date_index == 0:
            continue # Skip first day as we need a previous close

        today = hist.index[date_index].date()
        latest_close = hist.iloc[date_index]
        previous_close = hist.iloc[date_index - 1]

        # We'll just use the same set of expiries for each historical day for simplicity
        for exp in expiries:
            exp_dt = parser.parse(exp).date()
            tau_days = (exp_dt - today).days
            if tau_days <= 0:
                continue
            T_yrs = tau_days / 365.0

            chain = tkr.option_chain(exp)
            for df, opt_type in [(chain.calls, "C"), (chain.puts, "P")]:
                if df is None or df.empty:
                    continue
                d = df.rename(columns={"impliedVolatility": "implied_vol"})
                d["market_price"] = d.apply(lambda r: mid_price(r.get("bid"), r.get("ask"), r.get("lastPrice")), axis=1)
                d = d.dropna(subset=["market_price", "implied_vol"])

                sub = d[[
                    "strike", "market_price", "implied_vol", "lastPrice"
                ]].copy()

                sub["date"] = today.isoformat()
                sub["ticker"] = ticker # stock name
                sub["option_type"] = opt_type
                sub["S"] = latest_close
                sub["previous_close"] = previous_close
                sub["K"] = sub["strike"].astype(float)
                sub["tau_days"] = tau_days
                sub["iv"] = sub["implied_vol"]
                sub["moneyness"] = sub["K"] / latest_close
                sub["vix"] = vix

                # Calculate BS price and Greeks
                flag = opt_type.lower()
                sub["bs_price"] = sub.apply(lambda r: black_scholes_merton(flag, S=latest_close, K=r["K"], t=T_yrs, r=risk_free_rate, sigma=r["iv"], q=dividend_yield), axis=1)
                sub["delta"] = sub.apply(lambda r: greeks.delta(flag, S=latest_close, K=r["K"], t=T_yrs, r=risk_free_rate, sigma=r["iv"], q=dividend_yield), axis=1)
                sub["gamma"] = sub.apply(lambda r: greeks.gamma(flag, S=latest_close, K=r["K"], t=T_yrs, r=risk_free_rate, sigma=r["iv"], q=dividend_yield), axis=1)
                sub["theta"] = sub.apply(lambda r: greeks.theta(flag, S=latest_close, K=r["K"], t=T_yrs, r=risk_free_rate, sigma=r["iv"], q=dividend_yield), axis=1)
                sub["vega"] = sub.apply(lambda r: greeks.vega(flag, S=latest_close, K=r["K"], t=T_yrs, r=risk_free_rate, sigma=r["iv"], q=dividend_yield), axis=1)

                # Rename and add placeholder columns
                sub = sub.rename(columns={"market_price": "mkt_price", "lastPrice": "last_trade_price"})
                sub["label_uf_over"] = sub.apply(classify_price, axis=1)
                sub["residual"] = sub["mkt_price"] - sub["bs_price"]
                sub["fwd_option_return"] = np.nan

                rows.append(sub)

    if not rows:
        raise SystemExit("No option rows collected (empty chains).")

    # Define final column order based on your request
    final_cols = [
        "date", "ticker", "option_type", "S", "previous_close", "strike", "K",
        "tau_days", "iv", "delta", "gamma", "theta", "vega", "vix", "bs_price",
        "mkt_price", "last_trade_price", "residual", "label_uf_over", "fwd_option_return", "moneyness"
    ]

    out = pd.concat(rows, ignore_index=True)
    out = out[final_cols].sort_values(["tau_days", "option_type", "K"])

    out.to_csv(out_csv, index=False)
    print(f"Wrote {len(out):,} rows to {out_csv}")

if __name__ == "__main__":
    # Set max_expiries to a small number (e.g., 5) if you want a quick test
    arg_parser = argparse.ArgumentParser(
        description="Fetch options data for a given stock ticker from Yahoo Finance."
    )
    arg_parser.add_argument(
        "ticker",
        help="The stock ticker symbol to fetch data for (e.g., AAPL, MSFT)."
    )
    arg_parser.add_argument(
        "--output", "-o",
        help="Path to the output CSV file. Defaults to 'ticker_options.csv'.",
        default=None
    )
    arg_parser.add_argument(
        "--max-expiries", "-m",
        type=int,
        help="Limit the number of expiration dates to fetch (for quicker testing).",
        default=None
    )
    arg_parser.add_argument(
        "--days", "-d",
        type=int,
        help="Number of historical days to fetch data for.",
        default=5
    )
    args = arg_parser.parse_args()
    output_file = args.output or f"{args.ticker.lower()}_options.csv"

    main(ticker=args.ticker, out_csv=output_file, max_expiries=args.max_expiries, days=args.days)
