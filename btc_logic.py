from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

BTC_TICKER = "BTC-USDT"
USDJPY_TICKER = "USDJPY=X"
BTC_H1_CACHE = Path("btc_1h_cache.csv")
BTC_DROP_RESULTS = Path("btc_drop_results.csv")
TRADE_HISTORY_CSV = Path("trade_history.csv")

SIGNAL_DROP_THRESHOLD = -0.03
ENTRY_MULTIPLIER = 0.997
TAKE_PROFIT_MULTIPLIER = 1.014
STOP_LOSS_MULTIPLIER = 0.986
ENTRY_EXPIRY_HOURS = 2
DEFAULT_USDJPY = 150.0

TRADE_HISTORY_COLUMNS = [
    "signal_date",
    "signal_price",
    "entry_type",
    "entry_price",
    "entry_size_usdt",
    "leverage",
    "entry_jpy",
    "take_profit",
    "stop_loss",
    "exit_price",
    "exit_reason",
    "pnl_usdt",
    "pnl_jpy",
    "pnl_pct",
    "status",
]


@dataclass
class BtcBacktestSummary:
    pattern: str
    annual_return_net: float
    win_rate: float
    max_losing_streak: int


@dataclass
class BtcSignalPackage:
    current_price: float
    latest_change_pct: float
    latest_timestamp: pd.Timestamp
    signal_active: bool
    signal_drop_pct: float
    limit_price: float
    take_profit_price: float
    stop_loss_price: float
    expiry_time: pd.Timestamp
    recent_drops: pd.DataFrame
    backtest_summary: BtcBacktestSummary
    cache_updated: bool
    usd_jpy_rate: float
    usd_jpy_timestamp: pd.Timestamp
    usd_jpy_fallback_used: bool


def _normalize_history(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("BTC history is empty.")
    history = frame.copy()
    if isinstance(history.columns, pd.MultiIndex):
        history.columns = history.columns.get_level_values(0)
    history.index = pd.to_datetime(history.index, utc=True, errors="coerce")
    history = history.dropna(axis=0, how="all")
    history = history.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    for column in ["open", "high", "low", "close", "volume"]:
        if column in history.columns:
            history[column] = pd.to_numeric(history[column], errors="coerce")
    history = history.dropna(subset=["open", "high", "low", "close"])
    history = history.sort_index()
    history = history.reset_index().rename(columns={"Datetime": "open_time", "Date": "open_time"})
    history["close_time"] = history["open_time"] + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)
    return history[["open_time", "open", "high", "low", "close", "volume", "close_time"]].reset_index(drop=True)


def _download_btc_hourly_history() -> pd.DataFrame:
    history = yf.download(
        BTC_TICKER,
        period="60d",
        interval="1h",
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    return _normalize_history(history)


def _read_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if frame.empty:
        return None
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True, errors="coerce")
    frame["close_time"] = pd.to_datetime(frame["close_time"], utc=True, errors="coerce")
    for column in ["open", "high", "low", "close", "volume"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open_time", "close_time", "open", "high", "low", "close"])
    frame = frame.sort_values("open_time").drop_duplicates(subset=["open_time"]).reset_index(drop=True)
    return frame


def load_or_refresh_btc_hourly_cache(cache_path: Path = BTC_H1_CACHE) -> tuple[pd.DataFrame, bool]:
    cached = _read_cache(cache_path)
    now_complete = pd.Timestamp.now(tz="UTC").floor("1h") - pd.Timedelta(hours=1)

    if cached is not None and not cached.empty:
        last_cached = pd.Timestamp(cached["open_time"].max())
        if last_cached >= now_complete - pd.Timedelta(hours=1):
            return cached, False

    fresh = _download_btc_hourly_history()
    fresh.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return fresh, True


def _select_backtest_row(frame: pd.DataFrame) -> pd.Series:
    qualified = frame[
        (frame["first_half_return_net"] > 0.0) & (frame["second_half_return_net"] > 0.0)
    ].copy()
    if qualified.empty:
        qualified = frame.copy()
    qualified = qualified.sort_values("annual_return_net", ascending=False)
    return qualified.iloc[0]


def load_backtest_summary(path: Path = BTC_DROP_RESULTS) -> BtcBacktestSummary:
    if path.exists():
        frame = pd.read_csv(path)
        if not frame.empty:
            selected = _select_backtest_row(frame)
            return BtcBacktestSummary(
                pattern=str(selected["pattern"]),
                annual_return_net=float(selected["annual_return_net"]),
                win_rate=float(selected["win_rate"]),
                max_losing_streak=int(selected["max_losing_streak"]),
            )
    return BtcBacktestSummary(
        pattern="M1xL10",
        annual_return_net=0.2381,
        win_rate=0.5586,
        max_losing_streak=4,
    )


def _extract_fast_last_price(ticker: yf.Ticker) -> float | None:
    try:
        fast_info = ticker.fast_info
        if hasattr(fast_info, "get"):
            for key in ["lastPrice", "last_price", "regularMarketPrice"]:
                value = fast_info.get(key)
                if value is not None:
                    return float(value)
        for key in ["lastPrice", "last_price", "regularMarketPrice"]:
            if hasattr(fast_info, key):
                value = getattr(fast_info, key)
                if value is not None:
                    return float(value)
    except Exception:
        return None
    return None


def load_usd_jpy_rate() -> tuple[float, pd.Timestamp, bool]:
    fallback_timestamp = pd.Timestamp.now(tz="UTC")
    try:
        history = yf.Ticker(USDJPY_TICKER).history(period="5d", interval="1h", auto_adjust=False)
        history = history.dropna(subset=["Close"])
        if history.empty:
            raise ValueError("USDJPY history is empty.")
        latest_rate = float(history["Close"].iloc[-1])
        latest_timestamp = pd.to_datetime(history.index[-1], utc=True)
        return latest_rate, latest_timestamp, False
    except Exception:
        return DEFAULT_USDJPY, fallback_timestamp, True


def build_btc_signal_package() -> BtcSignalPackage:
    hourly, cache_updated = load_or_refresh_btc_hourly_cache()
    frame = hourly.copy()
    frame["prev_close"] = frame["close"].shift(1)
    frame["hourly_return"] = frame["close"] / frame["prev_close"] - 1.0
    frame = frame.dropna(subset=["prev_close", "hourly_return"]).reset_index(drop=True)
    latest = frame.iloc[-1]

    ticker = yf.Ticker(BTC_TICKER)
    current_price = _extract_fast_last_price(ticker)
    if current_price is None or current_price <= 0:
        current_price = float(latest["close"])

    latest_timestamp = pd.Timestamp(latest["close_time"])
    latest_change_pct = float(latest["hourly_return"] * 100.0)
    limit_price = current_price * ENTRY_MULTIPLIER
    take_profit_price = limit_price * TAKE_PROFIT_MULTIPLIER
    stop_loss_price = limit_price * STOP_LOSS_MULTIPLIER

    drops = frame[frame["hourly_return"] <= SIGNAL_DROP_THRESHOLD].copy()
    recent_drops = drops[["close_time", "close", "hourly_return"]].tail(5).copy()
    recent_drops["hourly_return_pct"] = recent_drops["hourly_return"] * 100.0
    recent_drops = recent_drops.rename(
        columns={
            "close_time": "急落時刻",
            "close": "終値(USDT)",
            "hourly_return_pct": "1時間変化率",
        }
    )

    usd_jpy_rate, usd_jpy_timestamp, usd_jpy_fallback_used = load_usd_jpy_rate()

    return BtcSignalPackage(
        current_price=float(current_price),
        latest_change_pct=latest_change_pct,
        latest_timestamp=latest_timestamp,
        signal_active=latest_change_pct <= SIGNAL_DROP_THRESHOLD * 100.0,
        signal_drop_pct=latest_change_pct,
        limit_price=limit_price,
        take_profit_price=take_profit_price,
        stop_loss_price=stop_loss_price,
        expiry_time=latest_timestamp + pd.Timedelta(hours=ENTRY_EXPIRY_HOURS),
        recent_drops=recent_drops.sort_values("急落時刻", ascending=False).reset_index(drop=True),
        backtest_summary=load_backtest_summary(),
        cache_updated=cache_updated,
        usd_jpy_rate=usd_jpy_rate,
        usd_jpy_timestamp=usd_jpy_timestamp,
        usd_jpy_fallback_used=usd_jpy_fallback_used,
    )


def normalize_monte_carlo_sequence(sequence: list[int] | tuple[int, ...] | None) -> list[int]:
    if not sequence:
        return [1, 2, 3]
    normalized = [max(1, int(value)) for value in sequence]
    return normalized or [1, 2, 3]


def monte_carlo_fraction(sequence: list[int] | tuple[int, ...] | None) -> float:
    seq = normalize_monte_carlo_sequence(sequence)
    units = seq[0] if len(seq) == 1 else seq[0] + seq[-1]
    return float(np.clip(units / 10.0, 0.05, 0.60))


def apply_monte_carlo_result(sequence: list[int] | tuple[int, ...] | None, won: bool) -> list[int]:
    seq = normalize_monte_carlo_sequence(sequence)
    units = seq[0] if len(seq) == 1 else seq[0] + seq[-1]
    if won:
        seq = [] if len(seq) == 1 else seq[1:-1]
        return seq or [1, 2, 3]
    seq.append(units)
    return seq


def format_monte_carlo_sequence(sequence: list[int] | tuple[int, ...] | None) -> str:
    seq = normalize_monte_carlo_sequence(sequence)
    return "[" + ", ".join(str(value) for value in seq) + "]"


def calculate_recommended_position(
    collateral_jpy: float,
    leverage: int,
    management: str,
    monte_carlo_sequence: list[int] | tuple[int, ...] | None,
    usd_jpy_rate: float = DEFAULT_USDJPY,
) -> dict[str, float]:
    fraction = 0.20 if management == "fixed" else monte_carlo_fraction(monte_carlo_sequence)
    margin_jpy = float(collateral_jpy) * fraction
    margin_usdt = margin_jpy / max(float(usd_jpy_rate), 1e-9)
    position_usdt = margin_usdt * int(leverage)
    return {
        "fraction": fraction,
        "margin_jpy": margin_jpy,
        "margin_usdt": margin_usdt,
        "position_usdt": position_usdt,
    }


def _coerce_trade_history_types(frame: pd.DataFrame) -> pd.DataFrame:
    if "signal_date" in frame.columns:
        frame["signal_date"] = pd.to_datetime(frame["signal_date"], utc=True, errors="coerce")
    for column in [
        "signal_price",
        "entry_price",
        "entry_size_usdt",
        "leverage",
        "entry_jpy",
        "take_profit",
        "stop_loss",
        "exit_price",
        "pnl_usdt",
        "pnl_jpy",
        "pnl_pct",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_trade_history(path: Path = TRADE_HISTORY_CSV) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=TRADE_HISTORY_COLUMNS)
    frame = pd.read_csv(path)
    for column in TRADE_HISTORY_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    frame = frame[TRADE_HISTORY_COLUMNS].copy()
    return _coerce_trade_history_types(frame)


def save_trade_history(frame: pd.DataFrame, path: Path = TRADE_HISTORY_CSV) -> None:
    output = frame.copy()
    for column in TRADE_HISTORY_COLUMNS:
        if column not in output.columns:
            output[column] = np.nan
    output = output[TRADE_HISTORY_COLUMNS]
    output.to_csv(path, index=False, encoding="utf-8-sig")


def append_trade_history_row(row: dict[str, Any], path: Path = TRADE_HISTORY_CSV) -> pd.DataFrame:
    history = load_trade_history(path)
    payload = {column: row.get(column, np.nan) for column in TRADE_HISTORY_COLUMNS}
    updated = pd.concat([history, pd.DataFrame([payload])], ignore_index=True)
    save_trade_history(updated, path)
    return updated


def close_open_trade(
    exit_price: float,
    exit_reason: str,
    usd_jpy_rate: float,
    path: Path = TRADE_HISTORY_CSV,
) -> pd.DataFrame:
    history = load_trade_history(path)
    open_positions = history[history["status"] == "open"]
    if open_positions.empty:
        raise ValueError("No open BTC trade found.")

    target_index = open_positions.index[-1]
    entry_price = float(history.at[target_index, "entry_price"])
    entry_size_usdt = float(history.at[target_index, "entry_size_usdt"])
    leverage = float(history.at[target_index, "leverage"])
    entry_jpy = float(history.at[target_index, "entry_jpy"])
    if entry_price <= 0 or entry_size_usdt <= 0 or leverage <= 0:
        raise ValueError("Open trade is missing entry data.")

    margin_usdt = entry_size_usdt / leverage
    pnl_usdt = margin_usdt * ((float(exit_price) / entry_price) - 1.0) * leverage
    pnl_jpy = pnl_usdt * float(usd_jpy_rate)
    pnl_pct = pnl_jpy / entry_jpy if entry_jpy else np.nan

    history.at[target_index, "exit_price"] = float(exit_price)
    history.at[target_index, "exit_reason"] = exit_reason
    history.at[target_index, "pnl_usdt"] = pnl_usdt
    history.at[target_index, "pnl_jpy"] = pnl_jpy
    history.at[target_index, "pnl_pct"] = pnl_pct
    history.at[target_index, "status"] = "closed"
    save_trade_history(history, path)
    return history


def summarize_trade_history(frame: pd.DataFrame) -> dict[str, float]:
    closed = frame[frame["status"] == "closed"].copy()
    count = int(len(closed))
    wins = int((closed["pnl_jpy"] > 0).sum()) if count else 0
    return {
        "closed_count": count,
        "total_pnl_usdt": float(closed["pnl_usdt"].sum()) if count else 0.0,
        "total_pnl_jpy": float(closed["pnl_jpy"].sum()) if count else 0.0,
        "win_rate": wins / count if count else 0.0,
        "avg_pnl_usdt": float(closed["pnl_usdt"].mean()) if count else 0.0,
        "avg_pnl_jpy": float(closed["pnl_jpy"].mean()) if count else 0.0,
    }
