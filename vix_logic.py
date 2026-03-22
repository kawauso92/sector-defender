from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import math
import pandas as pd
import yfinance as yf

VIX_TICKER = "^VIX"
TARGET_TICKER = "2558.T"
SIGNAL_VIX_CHANGE_PCT = 10.0
SIGNAL_VIX_LEVEL = 25.0
LIMIT_MULTIPLIER = 0.999
KELLY_FRACTION = 0.39
BACKTEST_ANNUAL_RETURN = 0.1589
BACKTEST_WIN_RATE = 0.938
BACKTEST_MAX_DD = -0.0059
RECENT_DAYS = 10


@dataclass
class VixBacktestSummary:
    annual_return: float
    win_rate: float
    max_drawdown: float
    kelly_fraction: float


@dataclass
class VixSignalPackage:
    current_vix: float
    vix_change_pct: float
    vix_date: pd.Timestamp
    signal_active: bool
    latest_2558_close: float
    limit_price: float
    recent_vix: pd.DataFrame
    backtest_summary: VixBacktestSummary


def _normalize_history(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("No market data returned from yfinance.")
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    frame = frame.dropna(axis=0, how="all")
    return frame.sort_index()


def _fetch_history(ticker: str, period: str = "1mo") -> pd.DataFrame:
    history = yf.Ticker(ticker).history(period=period, auto_adjust=False)
    return _normalize_history(history)


def load_vix_signal_package() -> VixSignalPackage:
    vix_history = _fetch_history(VIX_TICKER, period="1mo")
    asset_history = _fetch_history(TARGET_TICKER, period="1mo")

    if len(vix_history) < 2:
        raise ValueError("Not enough VIX data to calculate day-over-day change.")
    if asset_history.empty:
        raise ValueError("Not enough 2558.T data to calculate limit price.")

    current_vix = float(vix_history["Close"].iloc[-1])
    previous_vix = float(vix_history["Close"].iloc[-2])
    vix_change_pct = ((current_vix / previous_vix) - 1.0) * 100.0
    vix_date = pd.Timestamp(vix_history.index[-1])

    latest_2558_close = float(asset_history["Close"].iloc[-1])
    limit_price = latest_2558_close * LIMIT_MULTIPLIER

    recent_vix = vix_history[["Close"]].tail(5).reset_index()
    recent_vix.columns = ["日付", "VIX"]
    recent_vix["日付"] = pd.to_datetime(recent_vix["日付"], utc=True).dt.tz_convert("Asia/Tokyo").dt.strftime("%Y-%m-%d")
    recent_vix["VIX"] = recent_vix["VIX"].map(lambda value: f"{value:.2f}")

    return VixSignalPackage(
        current_vix=current_vix,
        vix_change_pct=vix_change_pct,
        vix_date=vix_date,
        signal_active=vix_change_pct >= SIGNAL_VIX_CHANGE_PCT and current_vix >= SIGNAL_VIX_LEVEL,
        latest_2558_close=latest_2558_close,
        limit_price=limit_price,
        recent_vix=recent_vix,
        backtest_summary=VixBacktestSummary(
            annual_return=BACKTEST_ANNUAL_RETURN,
            win_rate=BACKTEST_WIN_RATE,
            max_drawdown=BACKTEST_MAX_DD,
            kelly_fraction=KELLY_FRACTION,
        ),
    )


def calculate_vix_recommended_position(collateral_jpy: float, is_holding: bool) -> dict[str, float]:
    if is_holding:
        fraction = 0.0
    else:
        fraction = KELLY_FRACTION
    amount_jpy = float(collateral_jpy) * fraction
    return {
        "fraction": fraction,
        "amount_jpy": amount_jpy,
    }


def calculate_vix_units(amount_jpy: float, limit_price_jpy: float) -> int:
    if amount_jpy <= 0 or limit_price_jpy <= 0:
        return 0
    return max(0, math.floor(amount_jpy / limit_price_jpy))
