"""
Phase 0 Discovery Script — verify Finnhub SDK methods, Tiingo endpoint, and yfinance quoteType.
Run: poetry run python scripts/verify_finnhub.py
"""
import os
import json
import requests
import datetime
from dotenv import load_dotenv

load_dotenv()

FINNHUB_KEY = os.environ.get("FINNHUB_API_KEY", "")
TIINGO_KEY = os.environ.get("TIINGO_API_KEY", "")

print("=" * 60)
print("FINNHUB_API_KEY present:", bool(FINNHUB_KEY))
print("TIINGO_API_KEY present:", bool(TIINGO_KEY))
print("=" * 60)

# ---------------------------------------------------------------------------
# 0b — Finnhub SDK method discovery
# ---------------------------------------------------------------------------
import finnhub

client = finnhub.Client(api_key=FINNHUB_KEY)
client._session.verify = False

print("\n--- ETF-related methods on finnhub.Client ---")
etf_methods = [m for m in dir(client) if "etf" in m.lower()]
print(etf_methods)

print("\n--- Fund-related methods on finnhub.Client ---")
fund_methods = [m for m in dir(client) if "fund" in m.lower()]
print(fund_methods)

# Test each ETF method
ETF_TICKER = "SPY"
print(f"\n--- Testing ETF methods on {ETF_TICKER} ---")

for method_name in etf_methods:
    method = getattr(client, method_name)
    print(f"\n>>> client.{method_name}(symbol='{ETF_TICKER}')")
    try:
        result = method(symbol=ETF_TICKER)
        print(f"  Type: {type(result)}")
        if isinstance(result, dict):
            print(f"  Top-level keys: {list(result.keys())}")
            # Print nested keys for important sub-dicts
            for k, v in result.items():
                if isinstance(v, dict):
                    print(f"    {k} keys: {list(v.keys())}")
                elif isinstance(v, list) and v:
                    print(f"    {k}: list of {len(v)} items, first item keys: {list(v[0].keys()) if isinstance(v[0], dict) else type(v[0])}")
                else:
                    print(f"    {k}: {v}")
        elif isinstance(result, list):
            print(f"  List of {len(result)} items")
            if result and isinstance(result[0], dict):
                print(f"  First item keys: {list(result[0].keys())}")
        else:
            print(f"  Value: {result}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")

# Test fund methods
print(f"\n--- Testing fund methods on {ETF_TICKER} ---")
for method_name in fund_methods:
    method = getattr(client, method_name)
    print(f"\n>>> client.{method_name}(symbol='{ETF_TICKER}')")
    try:
        result = method(symbol=ETF_TICKER)
        print(f"  Type: {type(result)}")
        if isinstance(result, dict):
            print(f"  Top-level keys: {list(result.keys())}")
        elif isinstance(result, list):
            print(f"  List of {len(result)} items")
            if result and isinstance(result[0], dict):
                print(f"  First item keys: {list(result[0].keys())}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")

# Also test with a mutual fund ticker
MF_TICKER = "VFIAX"
print(f"\n--- Testing fund methods on {MF_TICKER} (mutual fund) ---")
for method_name in fund_methods:
    method = getattr(client, method_name)
    print(f"\n>>> client.{method_name}(symbol='{MF_TICKER}')")
    try:
        result = method(symbol=MF_TICKER)
        print(f"  Type: {type(result)}")
        if isinstance(result, dict):
            print(f"  Top-level keys: {list(result.keys())}")
            for k, v in result.items():
                if isinstance(v, (dict, list)):
                    print(f"    {k}: {type(v).__name__} len={len(v) if isinstance(v, (list, dict)) else '?'}")
                else:
                    print(f"    {k}: {v}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")

# ---------------------------------------------------------------------------
# 0c — Tiingo endpoint
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("0c — Tiingo price history endpoint")
print("=" * 60)

if TIINGO_KEY:
    url = f"https://api.tiingo.com/tiingo/daily/SPY/prices"
    params = {"startDate": "2024-01-01", "token": TIINGO_KEY}
    try:
        resp = requests.get(url, params=params, timeout=10)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"Total records: {len(data)}")
            if data:
                print(f"First record: {data[0]}")
                print(f"Last record: {data[-1]}")
                print(f"Field names (keys): {list(data[0].keys())}")
        else:
            print(f"Response body: {resp.text[:500]}")
    except Exception as e:
        print(f"ERROR: {e}")
else:
    print("TIINGO_API_KEY not set — skipping Tiingo test")

# ---------------------------------------------------------------------------
# 0d — yfinance quoteType
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("0d — yfinance quoteType values")
print("=" * 60)

import yfinance as yf

for ticker in ["SPY", "VFIAX", "AAPL", "UNKNOWN_TICKER_XYZ"]:
    try:
        info = yf.Ticker(ticker).info or {}
        qt = info.get("quoteType", "MISSING")
        name = info.get("longName", info.get("shortName", ""))
        print(f"{ticker:25s} quoteType={qt:15s} name={name[:50]}")
    except Exception as e:
        print(f"{ticker:25s} ERROR: {e}")

print("\nDone.")
