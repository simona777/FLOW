# app.py — Flow (Streamlit + Binance Testnet)
# Autor: Simona x ChatGPT — educațional. Rulează pe TESTNET.

import os
import json
import math
import time
from datetime import datetime
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from binance.client import Client

# =============== Config & Secrets ===============
def _get_secret(name: str, default: str = "") -> str:
    # 1) Streamlit Cloud Secrets  2) Env vars  3) default
    return str(st.secrets.get(name, os.getenv(name, default)))

API_KEY = _get_secret("BINANCE_API_KEY")
API_SECRET = _get_secret("BINANCE_API_SECRET")
USE_TESTNET = _get_secret("USE_TESTNET", "true").lower() == "true"
TESTNET_URL = "https://testnet.binance.vision"

# =============== Client ===============
@st.cache_resource(show_spinner=False)
def get_client() -> Client:
    base_url = TESTNET_URL if USE_TESTNET else "https://api.binance.com"
    return Client(api_key=API_KEY, api_secret=API_SECRET)

# =============== Data helpers ===============
@st.cache_data(show_spinner=False, ttl=60)
def fetch_klines(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    """
    Citește kline-uri din Binance. Cache 60s ca să nu abuzăm API-ul.
    """
    client = get_client()
    raw = client.klines(symbol=symbol, interval=interval, limit=limit)
    cols = ["open_time","open","high","low","close","volume","close_time",
            "qav","num_trades","taker_buy_base","taker_buy_quote","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    return df[["open_time","open","high","low","close","volume","close_time"]]

def add_indicators(df: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    out = df.copy()
    out["sma_fast"] = out["close"].rolling(fast).mean()
    out["sma_slow"] = out["close"].rolling(slow).mean()
    out["signal"] = 0
    out.loc[out["sma_fast"] > out["sma_slow"], "signal"] = 1
    out.loc[out["sma_fast"] < out["sma_slow"], "signal"] = -1
    out["cross"] = out["signal"].diff().fillna(0)
    return out

# =============== Backtest ===============
def backtest_sma(df: pd.DataFrame, base_alloc_usdt: float = 100.0, fee: float = 0.001) -> Tuple[Dict[str, Any], pd.Series]:
    usdt = 10_000.0
    coin = 0.0
    trades = 0
    equity_curve = []

    for _, r in df.iterrows():
        px = r["close"]
        cross = r["cross"]

        if cross == 1 and usdt > 10:
            size = min(base_alloc_usdt, usdt)
            qty = (size / px) * (1 - fee)
            coin += qty
            usdt -= size
            trades += 1
        elif cross == -1 and coin > 0:
            proceeds = coin * px * (1 - fee)
            usdt += proceeds
            coin = 0.0
            trades += 1

        equity_curve.append(usdt + coin * px)

    final_equity = equity_curve[-1] if equity_curve else 10_000.0
    pnl_pct = (final_equity / 10_000.0 - 1) * 100.0
    res = {
        "trades": trades,
        "final_equity": round(final_equity, 2),
        "pnl_pct": round(pnl_pct, 2)
    }
    return res, pd.Series(equity_curve, index=df["close_time"])

# =============== Live helpers (Testnet) ===============
def _get_balances() -> Dict[str, float]:
    acc = get_client().account()
    out = {}
    for b in acc.get("balances", []):
        out[b["asset"]] = float(b["free"])
    return out

def _round_step(qty: float, step: float) -> float:
    if step <= 0:
        return qty
    precision = max(0, int(round(-math.log10(step))))
    return float(f"{qty:.{precision}f}")

def market_buy_usdt(symbol: str, usdt_amount: float) -> Any:
    c = get_client()
    info = c.exchange_info(symbol=symbol)
    lot = [f for f in info["symbols"][0]["filters"] if f["filterType"] == "LOT_SIZE"][0]
    step = float(lot["stepSize"])
    price = float(c.ticker_price(symbol=symbol)["price"])
    qty = _round_step(usdt_amount / price, step)
    if qty <= 0:
        raise ValueError("Cantitate prea mică pentru a respecta pasul minim.")
    return c.new_order(symbol=symbol, side="BUY", type="MARKET", quantity=qty)

def market_sell_all(symbol: str) -> Any:
    c = get_client()
    base = symbol.replace("USDT", "")
    balances = _get_balances()
    free = balances.get(base, 0.0)
    if free <= 0:
        raise ValueError(f"Nu există {base} disponibil.")
    info = c.exchange_info(symbol=symbol)
    lot = [f for f in info["symbols"][0]["filters"] if f["filterType"] == "LOT_SIZE"][0]
    step = float(lot["stepSize"])
    qty = _round_step(free, step)
    if qty <= 0:
        raise ValueError("Cantitate zero după rotunjire.")
    return c.new_order(symbol=symbol, side="SELL", type="MARKET", quantity=qty)

# =============== UI ===============
st.set_page_config(page_title="Flow – Binance Testnet", page_icon="💧", layout="wide")
st.title("💧 Flow — Binance SMA Bot (Testnet)")

with st.sidebar:
    st.markdown("### ⚙️ Parametri")
    symbol = st.text_input("Symbol", "BTCUSDT")
    timeframe = st.selectbox("Timeframe", ["1m", "3m", "5m", "15m", "1h", "4h", "1d"], index=0)
    fast = st.number_input("SMA rapidă (FAST)", min_value=5, max_value=200, value=20, step=1)
    slow = st.number_input("SMA lentă (SLOW)", min_value=10, max_value=400, value=50, step=1)
    base_alloc = st.number_input("Alocare/ordin (USDT)", min_value=10, max_value=50_000, value=100, step=10)
    run_bt = st.button("Rulează Backtest")

    st.markdown("---")
    st.markdown("### 🔐 Conexiune")
    st.write("**USE_TESTNET:**", "ON ✅" if USE_TESTNET else "OFF ❌")
    has_keys = bool(API_KEY and API_SECRET)
    st.write("**Chei API prezente:**", "DA ✅" if has_keys else "NU ❌")

tab_bt, tab_live = st.tabs(["🔍 Backtest", "🟢 Live (Testnet)"])

with tab_bt:
    st.subheader("Backtest SMA Crossover")
    if run_bt:
        try:
            df = fetch_klines(symbol, timeframe, limit=max(slow + 5, 200))
            df = add_indicators(df, fast, slow)
            res, curve = backtest_sma(df, base_alloc_usdt=float(base_alloc))
            st.json({
                "symbol": symbol, "timeframe": timeframe,
                "fast": fast, "slow": slow,
                "trades": res["trades"], "final_equity": res["final_equity"], "pnl_pct": res["pnl_pct"]
            })
            st.line_chart(
                df.set_index("close_time")[["close", "sma_fast", "sma_slow"]].dropna(),
                height=320
            )
            st.caption("Linia prețului și mediile mobile. Backtestul folosește închiderile lumânărilor.")
        except Exception as e:
            st.error(f"Eroare backtest: {e}")

with tab_live:
    st.subheader("Ordine manuale pe Testnet (pentru siguranță)")
    st.caption("Nu rulează non-stop. Trimite manual ordine de test.")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📥 BUY (folosesc alocarea)"):
            if not has_keys:
                st.warning("Adaugă cheile în Secrets înainte de a trimite ordine.")
            else:
                try:
                    out = market_buy_usdt(symbol, float(base_alloc))
                    st.success("BUY OK")
                    st.code(json.dumps(out, indent=2))
                except Exception as e:
                    st.error(f"Eroare BUY: {e}")

    with col2:
        if st.button("📤 SELL (vând tot asset-ul)"):
            if not has_keys:
                st.warning("Adaugă cheile în Secrets înainte de a trimite ordine.")
            else:
                try:
                    out = market_sell_all(symbol)
                    st.success("SELL OK")
                    st.code(json.dumps(out, indent=2))
                except Exception as e:
                    st.error(f"Eroare SELL: {e}")

    with col3:
        if st.button("💼 Balans"):
            try:
                b = _get_balances()
                base = symbol.replace("USDT", "")
                st.info(f'USDT: {b.get("USDT", 0.0)} | {base}: {b.get(base, 0.0)}')
            except Exception as e:
                st.error(f"Eroare balans: {e}")

st.markdown("---")
st.caption("Educațional. Rulează pe Binance TESTNET. Nu reprezintă sfat financiar. Gestionează riscul cu atenție.")
