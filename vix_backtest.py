from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf
except ImportError:
    yf = None

BACKTEST_START = pd.Timestamp("2010-01-01")
BACKTEST_END = pd.Timestamp("2025-12-31")
FETCH_END = BACKTEST_END + pd.Timedelta(days=1)
LONGTERM_START = pd.Timestamp("1990-01-01")
SPY_LONGTERM_START = pd.Timestamp("1993-01-01")
ROUND_TRIP_COST = 0.0005
ONE_WAY_COST = ROUND_TRIP_COST / 2
VIX_TICKER = "^VIX"
ETF_TICKER = "1570.T"
COMPARISON_TICKERS = [ETF_TICKER, "2558.T", "2568.T", "2631.T"]
US_PROXY_TICKERS = ["SPY", "QQQ", "USDJPY=X"]
LONGTERM_TICKERS = [ETF_TICKER, "^N225", "SPY"]
RESULTS_CSV = Path("vix_results.csv")
RESULTS_PNG = Path("vix_results.png")
CONDITIONS_CSV = Path("vix_conditions.csv")
CONDITIONS_PNG = Path("vix_conditions.png")
NANPIN_CSV = Path("vix_nanpin.csv")
NANPIN_PNG = Path("vix_nanpin.png")
FIXED_ALLOCATION_RATIO = 0.20
KELLY_ALLOCATION_CAP = 0.50
KELLY_WARMUP_TRADE_COUNT = 5
KELLY_CSV = Path("vix_kelly.csv")
KELLY_PNG = Path("vix_kelly.png")
TICKERS_CSV = Path("vix_tickers.csv")
TICKERS_PNG = Path("vix_tickers.png")
EXTENDED_CSV = Path("vix_extended.csv")
EXTENDED_PNG = Path("vix_extended.png")
LONGTERM_CSV = Path("vix_longterm.csv")
LONGTERM_PNG = Path("vix_longterm.png")
TRADES_CSV = Path("vix_trades.csv")
MONTE_CSV = Path("vix_monte.csv")
MONTE_PNG = Path("vix_monte.png")
KELLY_SINGLE_CAP = 0.60
KELLY_SINGLE_FLOOR = 0.10
KELLY_SINGLE_WARMUP_RATIO = 0.20
KELLY_FIXED_39_RATIO = 0.39
FX_SWITCH_DOWN_THRESHOLD = -0.01
FX_SWITCH_UP_THRESHOLD = 0.01
FX_SWITCH_LOW_2558_WEIGHT = 0.30
FX_SWITCH_HIGH_2558_WEIGHT = 0.70
MONTE_INITIAL_SEQUENCE = [1.0, 2.0, 3.0]
MONTE_MIN_RATIO = 0.05
MONTE_MAX_RATIO = 0.60


@dataclass(frozen=True)
class ExitPattern:
    profit_code: str
    stop_code: str
    take_profit: float | None
    max_hold_days: int | None
    vix_reversion: bool
    stop_loss: float | None

    @property
    def label(self) -> str:
        return f"{self.profit_code}x{self.stop_code}"


@dataclass
class BacktestResult:
    name: str
    pattern_label: str
    averaging: bool
    trades: pd.DataFrame
    daily_results: pd.DataFrame
    annual_return_gross: float
    annual_return_net: float
    max_drawdown: float
    win_rate: float
    total_trade_count: int
    avg_holding_days: float
    avg_return_per_trade: float
    avg_cost_per_trade: float
    avg_net_per_trade: float
    risk_return_ratio: float
    yearly_returns: pd.DataFrame
    period_stats: pd.DataFrame
    both_periods_positive: bool
    max_drawdown_within_limit: bool
    trade_count_meets_minimum: bool
    ruin_probability: float = 0.0
    ruin_days: int = 0
    win_rate_ci_lower: float = 0.0
    win_rate_ci_upper: float = 0.0
    avg_trades_per_year: float = 0.0
    avg_allocation_ratio: float = 0.0


@dataclass(frozen=True)
class ConditionPattern:
    code: str
    pct_change_threshold: float | None
    absolute_vix_threshold: float
    description: str


@dataclass(frozen=True)
class NanpinPattern:
    code: str
    description: str
    max_add_count: int


@dataclass(frozen=True)
class CapitalMode:
    code: str
    description: str


@dataclass(frozen=True)
class KellyPattern:
    code: str
    description: str


@dataclass(frozen=True)
class TickerComparisonPattern:
    code: str
    tickers: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class SwitchPattern:
    code: str
    description: str


@dataclass(frozen=True)
class LongTermPattern:
    code: str
    ticker: str
    description: str
    requested_start: pd.Timestamp


@dataclass(frozen=True)
class MontePattern:
    code: str
    description: str


PATTERNS = [
    ExitPattern("L1", "S0", None, 1, False, None),
    ExitPattern("L2", "S0", 0.03, 3, False, None),
    ExitPattern("L2", "S1", 0.03, 3, False, 0.05),
    ExitPattern("L2", "S2", 0.03, 3, False, 0.10),
    ExitPattern("L3", "S0", 0.05, 5, False, None),
    ExitPattern("L3", "S1", 0.05, 5, False, 0.05),
    ExitPattern("L3", "S2", 0.05, 5, False, 0.10),
    ExitPattern("L4", "S0", 0.10, 10, False, None),
    ExitPattern("L4", "S1", 0.10, 10, False, 0.05),
    ExitPattern("L4", "S2", 0.10, 10, False, 0.10),
    ExitPattern("L5", "S0", None, None, True, None),
]

CONDITION_PATTERNS = [
    ConditionPattern("C1", 0.20, 25.0, "VIX daily change >= +20% and VIX close >= 25"),
    ConditionPattern("C2", 0.10, 20.0, "VIX daily change >= +10% and VIX close >= 20"),
    ConditionPattern("C3", None, 20.0, "VIX close >= 20 only"),
    ConditionPattern("C4", 0.10, 25.0, "VIX daily change >= +10% and VIX close >= 25"),
]

NANPIN_PATTERNS = [
    NanpinPattern("N0", "No averaging", 0),
    NanpinPattern("N1", "VIX-linked averaging", 2),
    NanpinPattern("N2", "Time-based averaging", 2),
]

CAPITAL_MODES = [
    CapitalMode("Fixed", "Fixed 20% allocation"),
    CapitalMode("Kelly", "Half Kelly allocation"),
]

KELLY_PATTERNS = [
    KellyPattern("K0", "Fixed 20%"),
    KellyPattern("K1", "Half Kelly"),
    KellyPattern("K2", "Full Kelly"),
    KellyPattern("K3", "Fixed 39%"),
]

TICKER_COMPARISON_PATTERNS = [
    TickerComparisonPattern("T1", ("1570.T",), "1570.T"),
    TickerComparisonPattern("T2", ("2558.T",), "2558.T"),
    TickerComparisonPattern("T3", ("2568.T",), "2568.T"),
    TickerComparisonPattern("T4", ("2631.T",), "2631.T"),
    TickerComparisonPattern("T5", ("1570.T", "2558.T"), "1570.T + 2558.T"),
    TickerComparisonPattern("T6", ("1570.T", "2568.T"), "1570.T + 2568.T"),
    TickerComparisonPattern("T7", ("SPY_JPY",), "SPY x USDJPY"),
    TickerComparisonPattern("T8", ("QQQ_JPY",), "QQQ x USDJPY"),
    TickerComparisonPattern("T9", ("1570.T", "SPY_JPY"), "1570.T + SPY x USDJPY"),
]

SWITCH_PATTERNS = [
    SwitchPattern("SW0", "Always 50/50 in 1570 + 2558"),
    SwitchPattern("SW1", "FX-based switching between 1570 and 2558"),
    SwitchPattern("SW2", "Always 1570 only"),
]

LONGTERM_PATTERNS = [
    LongTermPattern("LT1", ETF_TICKER, "1570.T fixed-20%", BACKTEST_START),
    LongTermPattern("LT2", "^N225", "^N225 fixed-20%", LONGTERM_START),
    LongTermPattern("LT3", "SPY", "SPY fixed-20%", SPY_LONGTERM_START),
]

MONTE_PATTERNS = [
    MontePattern("MC0", "Fixed 20%"),
    MontePattern("MC1", "Decomposition Monte Carlo"),
    MontePattern("MC2", "Kelly basis"),
]


def _normalize_download_frame(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw.empty:
        raise ValueError("yfinance returned no data.")

    if isinstance(raw.columns, pd.MultiIndex):
        return raw

    if len(tickers) != 1:
        raise ValueError("Unexpected single-level columns for multi-ticker download.")

    raw.columns = pd.MultiIndex.from_product([tickers, raw.columns])
    return raw


def _build_field_frame(history: pd.DataFrame, tickers: list[str], field: str) -> pd.DataFrame:
    values: dict[str, pd.Series] = {}

    for ticker in tickers:
        if ticker not in history.columns.get_level_values(0):
            continue
        if field not in history[ticker].columns:
            continue
        values[ticker] = pd.to_numeric(history[ticker][field], errors="coerce")

    if not values:
        raise ValueError(f"{field} data is missing.")

    return pd.DataFrame(values).sort_index()


def _download_with_yahoo_chart_api(
    ticker: str,
    start_date: pd.Timestamp = BACKTEST_START,
    end_date: pd.Timestamp = BACKTEST_END,
) -> pd.DataFrame:
    period1 = int(pd.Timestamp(start_date).timestamp())
    period2 = int((pd.Timestamp(end_date) + pd.Timedelta(days=1)).timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    response = requests.get(
        url,
        params={
            "period1": period1,
            "period2": period2,
            "interval": "1d",
            "includePrePost": "false",
            "events": "div,splits,capitalGains",
        },
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("chart", {}).get("result", [])
    if not results:
        raise ValueError(f"Yahoo chart API returned no data for {ticker}.")

    result = results[0]
    timestamps = result.get("timestamp", [])
    quote = result.get("indicators", {}).get("quote", [{}])[0]
    frame = pd.DataFrame(
        {
            "Open": quote.get("open", []),
            "High": quote.get("high", []),
            "Low": quote.get("low", []),
            "Close": quote.get("close", []),
            "Volume": quote.get("volume", []),
        },
        index=pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("Asia/Tokyo").tz_localize(None),
    )
    frame.index.name = "Date"
    return frame.sort_index()


def _download_market_data_fallback(
    tickers: list[str],
    start_date: pd.Timestamp = BACKTEST_START,
    end_date: pd.Timestamp = BACKTEST_END,
) -> dict[str, pd.DataFrame]:
    history_by_ticker = {
        ticker: _download_with_yahoo_chart_api(ticker, start_date=start_date, end_date=end_date)
        for ticker in tickers
    }
    fields = ["Open", "High", "Low", "Close"]
    market_data: dict[str, pd.DataFrame] = {}

    for field in fields:
        market_data[field.lower()] = pd.DataFrame(
            {
                ticker: pd.to_numeric(history[field], errors="coerce")
                for ticker, history in history_by_ticker.items()
                if field in history.columns
            }
        ).sort_index()

    return market_data


def _normalize_frame_index(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.index = pd.to_datetime(normalized.index).normalize()
    normalized = normalized.groupby(normalized.index).last()
    normalized.index.name = frame.index.name
    return normalized.sort_index()


def _map_series_to_next_jp_trade_date(series: pd.Series, jp_trade_dates: pd.DatetimeIndex) -> pd.Series:
    mapped_values: dict[pd.Timestamp, float] = {}

    for raw_date, value in series.dropna().items():
        mapped_date = _next_trade_date(pd.Timestamp(raw_date), jp_trade_dates)
        if mapped_date is None:
            continue
        mapped_values[pd.Timestamp(mapped_date)] = float(value)

    if not mapped_values:
        return pd.Series(index=jp_trade_dates, dtype=float)

    mapped = pd.Series(mapped_values).sort_index()
    return mapped.reindex(jp_trade_dates).ffill()


def _build_jpy_proxy_series(
    us_open: pd.Series,
    us_close: pd.Series,
    fx_open: pd.Series,
    fx_close: pd.Series,
    jp_trade_dates: pd.DatetimeIndex,
) -> tuple[pd.Series, pd.Series]:
    raw_open = (us_open * fx_open).dropna().sort_index()
    raw_close = (us_close * fx_close).dropna().sort_index()
    aligned_open = _map_series_to_next_jp_trade_date(raw_open, jp_trade_dates)
    aligned_close = _map_series_to_next_jp_trade_date(raw_close, jp_trade_dates)
    return aligned_open, aligned_close


def _missing_ohlc_tickers(
    open_frame: pd.DataFrame,
    high_frame: pd.DataFrame,
    low_frame: pd.DataFrame,
    close_frame: pd.DataFrame,
    tickers: list[str],
) -> list[str]:
    missing: list[str] = []

    for ticker in tickers:
        if ticker not in open_frame.columns or open_frame[ticker].dropna().empty:
            missing.append(ticker)
            continue
        if ticker not in high_frame.columns or high_frame[ticker].dropna().empty:
            missing.append(ticker)
            continue
        if ticker not in low_frame.columns or low_frame[ticker].dropna().empty:
            missing.append(ticker)
            continue
        if ticker not in close_frame.columns or close_frame[ticker].dropna().empty:
            missing.append(ticker)

    return missing


def _download_ohlc_frames(
    tickers: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    use_fallback = yf is None

    if not use_fallback:
        try:
            raw = yf.download(
                tickers=tickers,
                start=pd.Timestamp(start_date).strftime("%Y-%m-%d"),
                end=(pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=True,
            )
            history = _normalize_download_frame(raw, tickers)
            open_frame = _build_field_frame(history, tickers, "Open")
            high_frame = _build_field_frame(history, tickers, "High")
            low_frame = _build_field_frame(history, tickers, "Low")
            close_frame = _build_field_frame(history, tickers, "Close")
            use_fallback = bool(_missing_ohlc_tickers(open_frame, high_frame, low_frame, close_frame, tickers))
        except Exception:
            use_fallback = True

    if use_fallback:
        fallback_data = _download_market_data_fallback(tickers, start_date=start_date, end_date=end_date)
        open_frame = fallback_data["open"]
        high_frame = fallback_data["high"]
        low_frame = fallback_data["low"]
        close_frame = fallback_data["close"]

    return {
        "open": open_frame,
        "high": high_frame,
        "low": low_frame,
        "close": close_frame,
    }


def download_market_data() -> dict[str, pd.DataFrame | pd.Series]:
    tickers = [VIX_TICKER] + COMPARISON_TICKERS + US_PROXY_TICKERS
    ohlc_frames = _download_ohlc_frames(tickers, start_date=BACKTEST_START, end_date=BACKTEST_END)
    open_frame = ohlc_frames["open"]
    high_frame = ohlc_frames["high"]
    low_frame = ohlc_frames["low"]
    close_frame = ohlc_frames["close"]

    open_frame = _normalize_frame_index(open_frame)
    high_frame = _normalize_frame_index(high_frame)
    low_frame = _normalize_frame_index(low_frame)
    close_frame = _normalize_frame_index(close_frame)

    # Use a Japan-business-day proxy calendar so SPY/QQQ JPY series can backfill 2010-2025
    # even before 1570.T was listed. Actual TSE-listed products still use their own valid dates.
    jp_trade_dates = pd.bdate_range(BACKTEST_START, BACKTEST_END)

    spy_jpy_open, spy_jpy_close = _build_jpy_proxy_series(
        open_frame["SPY"],
        close_frame["SPY"],
        open_frame["USDJPY=X"],
        close_frame["USDJPY=X"],
        jp_trade_dates,
    )
    qqq_jpy_open, qqq_jpy_close = _build_jpy_proxy_series(
        open_frame["QQQ"],
        close_frame["QQQ"],
        open_frame["USDJPY=X"],
        close_frame["USDJPY=X"],
        jp_trade_dates,
    )
    open_frame["SPY_JPY"] = spy_jpy_open
    close_frame["SPY_JPY"] = spy_jpy_close
    open_frame["QQQ_JPY"] = qqq_jpy_open
    close_frame["QQQ_JPY"] = qqq_jpy_close

    return {
        "open": open_frame,
        "high": high_frame,
        "low": low_frame,
        "close": close_frame,
        "vix_close": close_frame[VIX_TICKER].dropna().sort_index(),
        "usdjpy_close": close_frame["USDJPY=X"].dropna().sort_index(),
    }


def download_longterm_market_data() -> dict[str, pd.DataFrame | pd.Series]:
    tickers = [VIX_TICKER] + LONGTERM_TICKERS
    ohlc_frames = _download_ohlc_frames(tickers, start_date=LONGTERM_START, end_date=BACKTEST_END)
    open_frame = _normalize_frame_index(ohlc_frames["open"])
    high_frame = _normalize_frame_index(ohlc_frames["high"])
    low_frame = _normalize_frame_index(ohlc_frames["low"])
    close_frame = _normalize_frame_index(ohlc_frames["close"])

    return {
        "open": open_frame,
        "high": high_frame,
        "low": low_frame,
        "close": close_frame,
        "vix_close": close_frame[VIX_TICKER].dropna().sort_index(),
    }


def _next_trade_date(reference_date: pd.Timestamp, trade_dates: pd.DatetimeIndex) -> pd.Timestamp | None:
    position = trade_dates.searchsorted(reference_date + pd.Timedelta(days=1))
    if position >= len(trade_dates):
        return None
    return trade_dates[position]


def build_entry_signals(
    vix_close: pd.Series,
    trade_dates: pd.DatetimeIndex,
    pct_change_threshold: float | None = 0.20,
    absolute_vix_threshold: float = 25.0,
) -> pd.DataFrame:
    if len(trade_dates) == 0:
        return pd.DataFrame()

    vix_pct_change = vix_close.pct_change()
    signal_mask = vix_close >= absolute_vix_threshold
    if pct_change_threshold is not None:
        signal_mask = signal_mask & (vix_pct_change >= pct_change_threshold)
    first_trade_date = pd.Timestamp(trade_dates[0])
    signal_dates = vix_close.index[signal_mask.fillna(False) & (vix_close.index >= first_trade_date)]

    rows: list[dict[str, Any]] = []
    for signal_date in signal_dates:
        entry_date = _next_trade_date(signal_date, trade_dates)
        if entry_date is None or entry_date > BACKTEST_END:
            continue
        rows.append(
            {
                "signal_date": signal_date,
                "entry_date": entry_date,
                "vix_close": float(vix_close.loc[signal_date]),
                "vix_pct_change": float(vix_pct_change.loc[signal_date]),
            }
        )

    signals = pd.DataFrame(rows)
    if signals.empty:
        return signals

    signals = signals.sort_values(["entry_date", "signal_date"]).drop_duplicates(subset=["entry_date"], keep="last")
    return signals.reset_index(drop=True)


def _period_net_return(period_stats: pd.DataFrame, period_label: str) -> float:
    matched = period_stats.loc[period_stats["period"] == period_label, "net_annual_return"]
    if matched.empty:
        return 0.0
    return float(matched.iloc[0])


def _latest_vix_close_before(trade_date: pd.Timestamp, vix_close: pd.Series) -> float | None:
    position = vix_close.index.searchsorted(trade_date) - 1
    if position < 0:
        return None
    value = vix_close.iloc[position]
    if pd.isna(value):
        return None
    return float(value)


def _latest_series_value_on_or_before(series: pd.Series, target_date: pd.Timestamp) -> float | None:
    subset = series.loc[series.index <= target_date]
    if subset.empty:
        return None
    value = subset.iloc[-1]
    if pd.isna(value):
        return None
    return float(value)


def _latest_series_pct_change_on_or_before(series: pd.Series, target_date: pd.Timestamp) -> float | None:
    pct_series = series.pct_change()
    subset = pct_series.loc[pct_series.index <= target_date].dropna()
    if subset.empty:
        return None
    return float(subset.iloc[-1])


def _compute_half_kelly_fraction(history_returns: list[float]) -> float:
    if len(history_returns) < KELLY_WARMUP_TRADE_COUNT:
        return FIXED_ALLOCATION_RATIO

    returns = pd.Series(history_returns, dtype=float)
    wins = returns[returns > 0]
    losses = returns[returns < 0]

    if wins.empty:
        return 0.0
    if losses.empty:
        return KELLY_ALLOCATION_CAP

    win_rate = float((returns > 0).mean())
    average_win = float(wins.mean())
    average_loss = abs(float(losses.mean()))
    if average_loss <= 0:
        return KELLY_ALLOCATION_CAP

    payoff_ratio = average_win / average_loss
    if payoff_ratio <= 0:
        return 0.0

    kelly_fraction = win_rate - (1.0 - win_rate) / payoff_ratio
    half_kelly_fraction = max(0.0, kelly_fraction / 2.0)
    return min(KELLY_ALLOCATION_CAP, half_kelly_fraction)


def _compute_full_kelly_fraction(history_returns: list[float]) -> float:
    if len(history_returns) < KELLY_WARMUP_TRADE_COUNT:
        return KELLY_SINGLE_WARMUP_RATIO

    returns = pd.Series(history_returns, dtype=float)
    wins = returns[returns > 0]
    losses = returns[returns < 0]

    if wins.empty:
        return KELLY_SINGLE_FLOOR
    if losses.empty:
        return KELLY_SINGLE_CAP

    win_rate = float((returns > 0).mean())
    loss_rate = 1.0 - win_rate
    average_win = float(wins.mean())
    average_loss = abs(float(losses.mean()))
    if average_loss <= 0:
        return KELLY_SINGLE_CAP

    payoff_ratio = average_win / average_loss
    if payoff_ratio <= 0:
        return KELLY_SINGLE_FLOOR

    full_kelly_fraction = win_rate - (loss_rate / payoff_ratio)
    clipped_fraction = min(KELLY_SINGLE_CAP, max(KELLY_SINGLE_FLOOR, full_kelly_fraction))
    return clipped_fraction


def _resolve_kelly_allocation_ratio(kelly_pattern: KellyPattern, history_returns: list[float]) -> float:
    if kelly_pattern.code == "K0":
        return FIXED_ALLOCATION_RATIO
    if kelly_pattern.code == "K1":
        if len(history_returns) < KELLY_WARMUP_TRADE_COUNT:
            return KELLY_SINGLE_WARMUP_RATIO
        full_kelly = _compute_full_kelly_fraction(history_returns)
        return min(KELLY_SINGLE_CAP, max(KELLY_SINGLE_FLOOR, full_kelly / 2.0))
    if kelly_pattern.code == "K2":
        return _compute_full_kelly_fraction(history_returns)
    if kelly_pattern.code == "K3":
        return KELLY_FIXED_39_RATIO
    return FIXED_ALLOCATION_RATIO


def _clone_monte_sequence(sequence: list[float] | None = None) -> list[float]:
    base = MONTE_INITIAL_SEQUENCE if sequence is None else sequence
    return [float(value) for value in base]


def _resolve_monte_allocation_ratio(sequence: list[float]) -> tuple[float, float]:
    active_sequence = _clone_monte_sequence(sequence) if sequence else _clone_monte_sequence()
    if len(active_sequence) == 1:
        wager_units = float(active_sequence[0])
    else:
        wager_units = float(active_sequence[0] + active_sequence[-1])

    base_units = float(sum(active_sequence))
    raw_ratio = FIXED_ALLOCATION_RATIO * (wager_units / base_units) if base_units > 0 else FIXED_ALLOCATION_RATIO
    clipped_ratio = min(MONTE_MAX_RATIO, max(MONTE_MIN_RATIO, raw_ratio))
    return clipped_ratio, wager_units


def _update_monte_sequence(sequence: list[float], won_trade: bool, wager_units: float) -> list[float]:
    active_sequence = _clone_monte_sequence(sequence) if sequence else _clone_monte_sequence()

    if won_trade:
        if len(active_sequence) == 1:
            active_sequence = []
        else:
            active_sequence = active_sequence[1:-1]
    else:
        active_sequence.append(float(wager_units))

    if not active_sequence:
        return _clone_monte_sequence()
    return active_sequence


def _nanpin_target_offsets(nanpin_pattern: NanpinPattern) -> list[int]:
    if nanpin_pattern.code == "N0":
        return []
    return [3, 7]


def _should_add_position(
    nanpin_pattern: NanpinPattern,
    offset: int,
    trade_date: pd.Timestamp,
    vix_close: pd.Series,
) -> bool:
    latest_vix = _latest_vix_close_before(trade_date, vix_close)
    if latest_vix is None:
        return False

    if nanpin_pattern.code == "N1":
        if offset == 3:
            return latest_vix > 30.0
        if offset == 7:
            return latest_vix > 40.0
        return False

    if nanpin_pattern.code == "N2":
        return latest_vix >= 25.0

    return False


def _annualized_return(return_series: pd.Series, trade_dates: pd.Series) -> float:
    if return_series.empty:
        return 0.0
    # CAGR based on compounded daily returns, not a simple sum of returns.
    equity_curve = (1.0 + return_series).cumprod()
    span_days = max((trade_dates.iloc[-1] - trade_dates.iloc[0]).days, 1)
    years = span_days / 365.25
    return float(equity_curve.iloc[-1] ** (1 / years) - 1) if years > 0 else 0.0


def _build_yearly_returns(daily_results: pd.DataFrame) -> pd.DataFrame:
    grouped = daily_results.groupby(daily_results["trade_date"].dt.year)
    gross = grouped["gross_return"].apply(lambda s: (1.0 + s).prod() - 1.0).reindex(range(2010, 2026), fill_value=0.0)
    net = grouped["net_return"].apply(lambda s: (1.0 + s).prod() - 1.0).reindex(range(2010, 2026), fill_value=0.0)
    return pd.DataFrame({"year": gross.index, "gross_return": gross.values, "net_return": net.values})


def _build_yearly_returns_for_range(
    daily_results: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    grouped = daily_results.groupby(daily_results["trade_date"].dt.year)
    gross = grouped["gross_return"].apply(lambda s: (1.0 + s).prod() - 1.0).reindex(range(start_year, end_year + 1), fill_value=0.0)
    net = grouped["net_return"].apply(lambda s: (1.0 + s).prod() - 1.0).reindex(range(start_year, end_year + 1), fill_value=0.0)
    return pd.DataFrame({"year": gross.index, "gross_return": gross.values, "net_return": net.values})


def _build_period_stats(daily_results: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    periods = [
        ("2010-2017", pd.Timestamp("2010-01-01"), pd.Timestamp("2017-12-31")),
        ("2018-2025", pd.Timestamp("2018-01-01"), pd.Timestamp("2025-12-31")),
    ]
    rows: list[dict[str, Any]] = []

    for label, start, end in periods:
        period_daily = daily_results[(daily_results["trade_date"] >= start) & (daily_results["trade_date"] <= end)].copy()
        period_trades = trades[(trades["exit_date"] >= start) & (trades["exit_date"] <= end)].copy()

        gross_annual = _annualized_return(period_daily["gross_return"], period_daily["trade_date"]) if not period_daily.empty else 0.0
        net_annual = _annualized_return(period_daily["net_return"], period_daily["trade_date"]) if not period_daily.empty else 0.0
        win_rate = float((period_trades["net_return"] > 0).mean()) if not period_trades.empty else 0.0
        avg_holding = float(period_trades["holding_days"].mean()) if not period_trades.empty else 0.0

        rows.append(
            {
                "period": label,
                "gross_annual_return": gross_annual,
                "net_annual_return": net_annual,
                "win_rate": win_rate,
                "trade_count": int(len(period_trades)),
                "avg_holding_days": avg_holding,
            }
        )

    return pd.DataFrame(rows)


def _build_equal_split_period_stats(
    daily_results: pd.DataFrame,
    trades: pd.DataFrame,
) -> pd.DataFrame:
    if daily_results.empty:
        return pd.DataFrame(
            columns=["period", "gross_annual_return", "net_annual_return", "win_rate", "trade_count", "avg_holding_days"]
        )

    start_date = pd.Timestamp(daily_results["trade_date"].min())
    end_date = pd.Timestamp(daily_results["trade_date"].max())
    midpoint = start_date + ((end_date - start_date) / 2)
    second_start = midpoint + pd.Timedelta(days=1)

    periods = [
        (f"{start_date.strftime('%Y-%m-%d')} to {midpoint.strftime('%Y-%m-%d')}", start_date, midpoint),
        (f"{second_start.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}", second_start, end_date),
    ]
    rows: list[dict[str, Any]] = []

    for label, start, end in periods:
        period_daily = daily_results[(daily_results["trade_date"] >= start) & (daily_results["trade_date"] <= end)].copy()
        period_trades = trades[(trades["exit_date"] >= start) & (trades["exit_date"] <= end)].copy()
        gross_annual = _annualized_return(period_daily["gross_return"], period_daily["trade_date"]) if not period_daily.empty else 0.0
        net_annual = _annualized_return(period_daily["net_return"], period_daily["trade_date"]) if not period_daily.empty else 0.0
        win_rate = float((period_trades["net_return"] > 0).mean()) if not period_trades.empty else 0.0
        avg_holding = float(period_trades["holding_days"].mean()) if not period_trades.empty else 0.0

        rows.append(
            {
                "period": label,
                "gross_annual_return": gross_annual,
                "net_annual_return": net_annual,
                "win_rate": win_rate,
                "trade_count": int(len(period_trades)),
                "avg_holding_days": avg_holding,
            }
        )

    return pd.DataFrame(rows)


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0

    phat = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    centre = (phat + z2 / (2.0 * total)) / denominator
    margin = z * np.sqrt(((phat * (1.0 - phat)) + z2 / (4.0 * total)) / total) / denominator
    return max(0.0, float(centre - margin)), min(1.0, float(centre + margin))


def _years_between(start_date: pd.Timestamp, end_date: pd.Timestamp) -> float:
    span_days = max((pd.Timestamp(end_date) - pd.Timestamp(start_date)).days, 1)
    return span_days / 365.25


def _resolve_stop_exit(day_open: float, day_low: float, stop_price: float) -> float:
    if day_open <= stop_price:
        return day_open
    return stop_price if day_low <= stop_price else np.nan


def _resolve_take_profit_exit(day_open: float, day_high: float, take_profit_price: float) -> float:
    if day_open >= take_profit_price:
        return day_open
    return take_profit_price if day_high >= take_profit_price else np.nan


def _first_vix_reversion_exit_date(
    signal_date: pd.Timestamp,
    vix_close: pd.Series,
    trade_dates: pd.DatetimeIndex,
) -> pd.Timestamp | None:
    reversion_dates = vix_close.index[(vix_close.index > signal_date) & (vix_close <= 20.0)]
    if len(reversion_dates) == 0:
        return None
    return _next_trade_date(reversion_dates[0], trade_dates)


def simulate_trade(
    signal_date: pd.Timestamp,
    entry_date: pd.Timestamp,
    pattern: ExitPattern,
    averaging: bool,
    trade_dates: pd.DatetimeIndex,
    opens: pd.Series,
    highs: pd.Series,
    lows: pd.Series,
    closes: pd.Series,
    vix_close: pd.Series,
) -> dict[str, Any] | None:
    if entry_date not in trade_dates:
        return None

    entry_idx = trade_dates.get_loc(entry_date)
    entry_open = opens.loc[entry_date]
    if pd.isna(entry_open) or entry_open <= 0:
        return None

    shares = 1.0 / float(entry_open)
    total_entry_notional = 1.0
    initial_entry_price = float(entry_open)
    averaging_used = False
    add_entry_date: pd.Timestamp | None = None
    add_entry_price: float | None = None
    scheduled_add_date: pd.Timestamp | None = None
    vix_exit_date = _first_vix_reversion_exit_date(signal_date, vix_close, trade_dates) if pattern.vix_reversion else None
    exit_date: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str | None = None

    for idx in range(entry_idx, len(trade_dates)):
        trade_date = trade_dates[idx]
        day_open = opens.loc[trade_date]
        day_high = highs.loc[trade_date]
        day_low = lows.loc[trade_date]
        day_close = closes.loc[trade_date]

        if pd.isna(day_open) or pd.isna(day_high) or pd.isna(day_low) or pd.isna(day_close):
            continue

        if scheduled_add_date is not None and trade_date == scheduled_add_date:
            if day_open > 0:
                shares += 1.0 / float(day_open)
                total_entry_notional += 1.0
                averaging_used = True
                add_entry_date = trade_date
                add_entry_price = float(day_open)
            scheduled_add_date = None

        average_entry_price = total_entry_notional / shares
        days_since_entry = idx - entry_idx

        stop_triggered = False
        stop_price = np.nan
        if pattern.stop_loss is not None:
            stop_level = average_entry_price * (1.0 - pattern.stop_loss)
            stop_price = _resolve_stop_exit(float(day_open), float(day_low), float(stop_level))
            stop_triggered = not np.isnan(stop_price)

        take_profit_triggered = False
        take_profit_price = np.nan
        if pattern.take_profit is not None:
            take_profit_level = average_entry_price * (1.0 + pattern.take_profit)
            take_profit_price = _resolve_take_profit_exit(float(day_open), float(day_high), float(take_profit_level))
            take_profit_triggered = not np.isnan(take_profit_price)

        if stop_triggered and take_profit_triggered:
            exit_date = trade_date
            exit_price = float(stop_price)
            exit_reason = pattern.stop_code
            break

        if stop_triggered:
            exit_date = trade_date
            exit_price = float(stop_price)
            exit_reason = pattern.stop_code
            break

        if take_profit_triggered:
            exit_date = trade_date
            exit_price = float(take_profit_price)
            exit_reason = pattern.profit_code
            break

        if pattern.max_hold_days is not None and days_since_entry >= pattern.max_hold_days:
            exit_date = trade_date
            exit_price = float(day_close)
            exit_reason = f"{pattern.profit_code}_timeout"
            break

        if pattern.vix_reversion and vix_exit_date is not None and trade_date >= vix_exit_date:
            exit_date = trade_date
            exit_price = float(day_close)
            exit_reason = "L5_vix_reversion"
            break

        if averaging and (not averaging_used) and scheduled_add_date is None and day_close <= initial_entry_price * 0.97:
            if idx + 1 < len(trade_dates):
                scheduled_add_date = trade_dates[idx + 1]

    if exit_date is None or exit_price is None:
        last_trade_date = trade_dates[-1]
        last_close = closes.loc[last_trade_date]
        if pd.isna(last_close):
            return None
        exit_date = last_trade_date
        exit_price = float(last_close)
        exit_reason = "forced_exit_end_of_sample"

    exit_idx = trade_dates.get_loc(exit_date)
    gross_return = (exit_price * shares - total_entry_notional) / total_entry_notional
    net_return = gross_return - ROUND_TRIP_COST

    return {
        "signal_date": signal_date,
        "entry_date": entry_date,
        "exit_date": exit_date,
        "entry_price": initial_entry_price,
        "add_entry_date": add_entry_date,
        "add_entry_price": add_entry_price,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "averaging_used": averaging_used,
        "deployed_units": total_entry_notional,
        "gross_return": float(gross_return),
        "cost": ROUND_TRIP_COST,
        "net_return": float(net_return),
        "holding_days": int(exit_idx - entry_idx),
    }


def _build_tranche_allocations(
    nanpin_pattern: NanpinPattern,
    capital_mode: CapitalMode,
    capital_before_trade: float,
    history_net_returns: list[float],
) -> list[float]:
    tranche_count = 1 + nanpin_pattern.max_add_count

    if capital_before_trade <= 0:
        return [0.0] * tranche_count

    if capital_mode.code == "Fixed":
        return [capital_before_trade * FIXED_ALLOCATION_RATIO] * tranche_count

    kelly_fraction = _compute_half_kelly_fraction(history_net_returns)
    total_budget = capital_before_trade * kelly_fraction
    if total_budget <= 0:
        return [0.0] * tranche_count

    tranche_amount = total_budget / tranche_count
    return [tranche_amount] * tranche_count


def simulate_nanpin_trade(
    signal_date: pd.Timestamp,
    entry_date: pd.Timestamp,
    nanpin_pattern: NanpinPattern,
    capital_mode: CapitalMode,
    trade_dates: pd.DatetimeIndex,
    opens: pd.Series,
    closes: pd.Series,
    vix_close: pd.Series,
    gross_capital_before: float,
    net_capital_before: float,
    history_net_returns: list[float],
) -> dict[str, Any] | None:
    if entry_date not in trade_dates:
        return None

    exit_date = _first_vix_reversion_exit_date(signal_date, vix_close, trade_dates)
    if exit_date is None:
        exit_date = trade_dates[-1]

    entry_idx = trade_dates.get_loc(entry_date)
    exit_idx = trade_dates.get_loc(exit_date)
    if exit_idx < entry_idx:
        return None

    entry_open = opens.loc[entry_date]
    exit_close = closes.loc[exit_date]
    if pd.isna(entry_open) or pd.isna(exit_close) or entry_open <= 0:
        return None

    gross_tranches = _build_tranche_allocations(nanpin_pattern, capital_mode, gross_capital_before, history_net_returns)
    net_tranches = _build_tranche_allocations(nanpin_pattern, capital_mode, net_capital_before, history_net_returns)
    if not gross_tranches or gross_tranches[0] <= 0 or not net_tranches or net_tranches[0] <= 0:
        return None

    deployed_gross = gross_tranches[0]
    deployed_net = net_tranches[0]
    shares_gross = deployed_gross / float(entry_open)
    shares_net = deployed_net / float(entry_open)
    add_dates: list[pd.Timestamp] = []

    for add_index, offset in enumerate(_nanpin_target_offsets(nanpin_pattern), start=1):
        if entry_idx + offset > exit_idx or entry_idx + offset >= len(trade_dates):
            continue

        add_date = trade_dates[entry_idx + offset]
        if not _should_add_position(nanpin_pattern, offset, add_date, vix_close):
            continue

        add_open = opens.loc[add_date]
        if pd.isna(add_open) or add_open <= 0:
            continue

        tranche_gross = gross_tranches[add_index] if add_index < len(gross_tranches) else 0.0
        tranche_net = net_tranches[add_index] if add_index < len(net_tranches) else 0.0
        if tranche_gross <= 0 or tranche_net <= 0:
            continue

        deployed_gross += tranche_gross
        deployed_net += tranche_net
        shares_gross += tranche_gross / float(add_open)
        shares_net += tranche_net / float(add_open)
        add_dates.append(add_date)

    gross_exit_value = shares_gross * float(exit_close)
    net_exit_value = shares_net * float(exit_close)
    gross_pnl = gross_exit_value - deployed_gross
    cost_amount = deployed_net * ROUND_TRIP_COST
    net_pnl = net_exit_value - deployed_net - cost_amount
    gross_return_on_deployed = gross_pnl / deployed_gross if deployed_gross > 0 else 0.0
    net_return_on_deployed = net_pnl / deployed_net if deployed_net > 0 else 0.0
    gross_return_on_equity = gross_pnl / gross_capital_before if gross_capital_before > 0 else 0.0
    net_return_on_equity = net_pnl / net_capital_before if net_capital_before > 0 else 0.0

    return {
        "signal_date": signal_date,
        "entry_date": entry_date,
        "exit_date": exit_date,
        "entry_price": float(entry_open),
        "exit_price": float(exit_close),
        "add_count": len(add_dates),
        "add_dates": ",".join(date.strftime("%Y-%m-%d") for date in add_dates),
        "deployed_gross": float(deployed_gross),
        "deployed_net": float(deployed_net),
        "gross_pnl": float(gross_pnl),
        "net_pnl": float(net_pnl),
        "cost_amount": float(cost_amount),
        "gross_return": float(gross_return_on_deployed),
        "net_return": float(net_return_on_deployed),
        "gross_portfolio_return": float(gross_return_on_equity),
        "net_portfolio_return": float(net_return_on_equity),
        "holding_days": int(exit_idx - entry_idx),
    }


def run_nanpin_backtest(
    nanpin_pattern: NanpinPattern,
    capital_mode: CapitalMode,
    signals: pd.DataFrame,
    market_data: dict[str, pd.DataFrame | pd.Series],
) -> BacktestResult:
    open_frame = market_data["open"]
    close_frame = market_data["close"]
    vix_close = market_data["vix_close"]

    opens = open_frame[ETF_TICKER].dropna().sort_index()
    closes = close_frame[ETF_TICKER].reindex(opens.index)
    trade_dates = opens.index.intersection(closes.dropna().index)

    trade_rows: list[dict[str, Any]] = []
    history_net_returns: list[float] = []
    last_exit_date: pd.Timestamp | None = None
    gross_capital = 1.0
    net_capital = 1.0

    full_trade_dates = pd.DataFrame({"trade_date": trade_dates[(trade_dates >= BACKTEST_START) & (trade_dates <= BACKTEST_END)]})
    full_trade_dates["gross_return"] = 0.0
    full_trade_dates["net_return"] = 0.0
    full_trade_dates["trade_count"] = 0

    for signal in signals.itertuples(index=False):
        signal_date = pd.Timestamp(signal.signal_date)
        entry_date = pd.Timestamp(signal.entry_date)

        if last_exit_date is not None and entry_date <= last_exit_date:
            continue

        trade = simulate_nanpin_trade(
            signal_date=signal_date,
            entry_date=entry_date,
            nanpin_pattern=nanpin_pattern,
            capital_mode=capital_mode,
            trade_dates=trade_dates,
            opens=opens,
            closes=closes,
            vix_close=vix_close,
            gross_capital_before=gross_capital,
            net_capital_before=net_capital,
            history_net_returns=history_net_returns,
        )
        if trade is None:
            continue

        gross_capital *= 1.0 + float(trade["gross_portfolio_return"])
        net_capital *= 1.0 + float(trade["net_portfolio_return"])
        history_net_returns.append(float(trade["net_return"]))
        trade_rows.append(trade)
        last_exit_date = pd.Timestamp(trade["exit_date"])

        mask = full_trade_dates["trade_date"] == trade["exit_date"]
        full_trade_dates.loc[mask, "gross_return"] += float(trade["gross_portfolio_return"])
        full_trade_dates.loc[mask, "net_return"] += float(trade["net_portfolio_return"])
        full_trade_dates.loc[mask, "trade_count"] += 1

    trades = pd.DataFrame(trade_rows)
    if trades.empty:
        raise ValueError(f"No trades were generated for {nanpin_pattern.code} x {capital_mode.code}.")

    daily_results = full_trade_dates.copy()
    daily_results["gross_equity_curve"] = (1.0 + daily_results["gross_return"]).cumprod()
    daily_results["net_equity_curve"] = (1.0 + daily_results["net_return"]).cumprod()
    daily_results["drawdown"] = daily_results["net_equity_curve"] / daily_results["net_equity_curve"].cummax() - 1.0

    annual_return_gross = _annualized_return(daily_results["gross_return"], daily_results["trade_date"])
    annual_return_net = _annualized_return(daily_results["net_return"], daily_results["trade_date"])
    max_drawdown = float(daily_results["drawdown"].min()) if not daily_results.empty else 0.0
    win_rate = float((trades["net_return"] > 0).mean()) if not trades.empty else 0.0
    total_trade_count = int(len(trades))
    avg_holding_days = float(trades["holding_days"].mean()) if not trades.empty else 0.0
    avg_return_per_trade = float(trades["gross_return"].mean()) if not trades.empty else 0.0
    avg_cost_per_trade = float((trades["cost_amount"] / trades["deployed_net"]).mean()) if not trades.empty else 0.0
    avg_net_per_trade = float(trades["net_return"].mean()) if not trades.empty else 0.0
    risk_return_ratio = annual_return_net / abs(max_drawdown) if max_drawdown < 0 else 0.0
    yearly_returns = _build_yearly_returns(daily_results)
    period_stats = _build_period_stats(daily_results, trades)
    both_periods_positive = bool((period_stats["net_annual_return"] > 0).all())
    max_drawdown_within_limit = max_drawdown >= -0.30
    trade_count_meets_minimum = total_trade_count >= 10

    strategy_name = f"{nanpin_pattern.code} x {capital_mode.code}"
    trades.insert(0, "strategy", strategy_name)
    trades.insert(1, "nanpin", nanpin_pattern.code)
    trades.insert(2, "capital_mode", capital_mode.code)
    daily_results.insert(0, "strategy", strategy_name)
    daily_results.insert(1, "nanpin", nanpin_pattern.code)
    daily_results.insert(2, "capital_mode", capital_mode.code)

    return BacktestResult(
        name=strategy_name,
        pattern_label=strategy_name,
        averaging=nanpin_pattern.code != "N0",
        trades=trades,
        daily_results=daily_results,
        annual_return_gross=annual_return_gross,
        annual_return_net=annual_return_net,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        total_trade_count=total_trade_count,
        avg_holding_days=avg_holding_days,
        avg_return_per_trade=avg_return_per_trade,
        avg_cost_per_trade=avg_cost_per_trade,
        avg_net_per_trade=avg_net_per_trade,
        risk_return_ratio=risk_return_ratio,
        yearly_returns=yearly_returns,
        period_stats=period_stats,
        both_periods_positive=both_periods_positive,
        max_drawdown_within_limit=max_drawdown_within_limit,
        trade_count_meets_minimum=trade_count_meets_minimum,
    )


def run_kelly_backtest(
    kelly_pattern: KellyPattern,
    signals: pd.DataFrame,
    market_data: dict[str, pd.DataFrame | pd.Series],
) -> BacktestResult:
    open_frame = market_data["open"]
    close_frame = market_data["close"]
    vix_close = market_data["vix_close"]

    opens = open_frame[ETF_TICKER].dropna().sort_index()
    closes = close_frame[ETF_TICKER].reindex(opens.index)
    trade_dates = opens.index.intersection(closes.dropna().index)

    trade_rows: list[dict[str, Any]] = []
    history_net_returns: list[float] = []
    last_exit_date: pd.Timestamp | None = None

    full_trade_dates = pd.DataFrame({"trade_date": trade_dates[(trade_dates >= BACKTEST_START) & (trade_dates <= BACKTEST_END)]})
    full_trade_dates["gross_return"] = 0.0
    full_trade_dates["net_return"] = 0.0
    full_trade_dates["trade_count"] = 0

    for signal in signals.itertuples(index=False):
        signal_date = pd.Timestamp(signal.signal_date)
        entry_date = pd.Timestamp(signal.entry_date)

        if last_exit_date is not None and entry_date <= last_exit_date:
            continue

        exit_date = _first_vix_reversion_exit_date(signal_date, vix_close, trade_dates)
        if exit_date is None:
            exit_date = trade_dates[-1]

        if entry_date not in trade_dates or exit_date not in trade_dates:
            continue

        entry_open = opens.loc[entry_date]
        exit_close = closes.loc[exit_date]
        if pd.isna(entry_open) or pd.isna(exit_close) or entry_open <= 0:
            continue

        allocation_ratio = _resolve_kelly_allocation_ratio(kelly_pattern, history_net_returns)
        exposure_multiplier = allocation_ratio / FIXED_ALLOCATION_RATIO
        gross_trade_return = (float(exit_close) / float(entry_open)) - 1.0
        net_trade_return = gross_trade_return - ROUND_TRIP_COST
        gross_portfolio_return = gross_trade_return * exposure_multiplier
        net_portfolio_return = net_trade_return * exposure_multiplier

        history_net_returns.append(float(net_trade_return))
        last_exit_date = exit_date

        trade_rows.append(
            {
                "signal_date": signal_date,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": float(entry_open),
                "exit_price": float(exit_close),
                "allocation_ratio": float(allocation_ratio),
                "exposure_multiplier": float(exposure_multiplier),
                "deployed_gross": float(allocation_ratio),
                "deployed_net": float(allocation_ratio),
                "gross_pnl": float(gross_portfolio_return),
                "net_pnl": float(net_portfolio_return),
                "cost_amount": float(ROUND_TRIP_COST * exposure_multiplier),
                "cost_rate": float(ROUND_TRIP_COST * exposure_multiplier),
                "gross_return": float(gross_trade_return),
                "net_return": float(net_trade_return),
                "gross_portfolio_return": float(gross_portfolio_return),
                "net_portfolio_return": float(net_portfolio_return),
                "holding_days": int(trade_dates.get_loc(exit_date) - trade_dates.get_loc(entry_date)),
            }
        )

        mask = full_trade_dates["trade_date"] == exit_date
        full_trade_dates.loc[mask, "gross_return"] += float(gross_portfolio_return)
        full_trade_dates.loc[mask, "net_return"] += float(net_portfolio_return)
        full_trade_dates.loc[mask, "trade_count"] += 1

    trades = pd.DataFrame(trade_rows)
    if trades.empty:
        raise ValueError(f"No trades were generated for {kelly_pattern.code}.")

    daily_results = full_trade_dates.copy()
    daily_results["gross_equity_curve"] = (1.0 + daily_results["gross_return"]).cumprod()
    daily_results["net_equity_curve"] = (1.0 + daily_results["net_return"]).cumprod()
    daily_results["drawdown"] = daily_results["net_equity_curve"] / daily_results["net_equity_curve"].cummax() - 1.0

    annual_return_gross = _annualized_return(daily_results["gross_return"], daily_results["trade_date"])
    annual_return_net = _annualized_return(daily_results["net_return"], daily_results["trade_date"])
    max_drawdown = float(daily_results["drawdown"].min()) if not daily_results.empty else 0.0
    win_rate = float((trades["net_return"] > 0).mean()) if not trades.empty else 0.0
    total_trade_count = int(len(trades))
    avg_holding_days = float(trades["holding_days"].mean()) if not trades.empty else 0.0
    avg_return_per_trade = float(trades["gross_return"].mean()) if not trades.empty else 0.0
    avg_cost_per_trade = float(trades["cost_rate"].mean()) if not trades.empty else 0.0
    avg_net_per_trade = float(trades["net_return"].mean()) if not trades.empty else 0.0
    risk_return_ratio = annual_return_net / abs(max_drawdown) if max_drawdown < 0 else 0.0
    yearly_returns = _build_yearly_returns(daily_results)
    period_stats = _build_period_stats(daily_results, trades)
    both_periods_positive = bool((period_stats["net_annual_return"] > 0).all())
    max_drawdown_within_limit = max_drawdown >= -0.30
    trade_count_meets_minimum = total_trade_count >= 10
    ruin_days = int((daily_results["net_equity_curve"] <= 0.5).sum())
    ruin_probability = ruin_days / len(daily_results) if len(daily_results) > 0 else 0.0

    strategy_name = kelly_pattern.code
    trades.insert(0, "strategy", strategy_name)
    daily_results.insert(0, "strategy", strategy_name)

    return BacktestResult(
        name=strategy_name,
        pattern_label=strategy_name,
        averaging=False,
        trades=trades,
        daily_results=daily_results,
        annual_return_gross=annual_return_gross,
        annual_return_net=annual_return_net,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        total_trade_count=total_trade_count,
        avg_holding_days=avg_holding_days,
        avg_return_per_trade=avg_return_per_trade,
        avg_cost_per_trade=avg_cost_per_trade,
        avg_net_per_trade=avg_net_per_trade,
        risk_return_ratio=risk_return_ratio,
        yearly_returns=yearly_returns,
        period_stats=period_stats,
        both_periods_positive=both_periods_positive,
        max_drawdown_within_limit=max_drawdown_within_limit,
        trade_count_meets_minimum=trade_count_meets_minimum,
        ruin_probability=ruin_probability,
        ruin_days=ruin_days,
    )


def _common_trade_dates_for_tickers(
    tickers: tuple[str, ...],
    open_frame: pd.DataFrame,
    close_frame: pd.DataFrame,
) -> pd.DatetimeIndex:
    common_dates: pd.DatetimeIndex | None = None

    for ticker in tickers:
        if ticker not in open_frame.columns or ticker not in close_frame.columns:
            return pd.DatetimeIndex([])
        ticker_dates = open_frame[ticker].dropna().index.intersection(close_frame[ticker].dropna().index)
        common_dates = ticker_dates if common_dates is None else common_dates.intersection(ticker_dates)

    if common_dates is None:
        return pd.DatetimeIndex([])

    return common_dates.sort_values()


def _first_available_trade_date(
    tickers: tuple[str, ...],
    open_frame: pd.DataFrame,
    close_frame: pd.DataFrame,
) -> pd.Timestamp | None:
    common_dates = _common_trade_dates_for_tickers(tickers, open_frame, close_frame)
    if len(common_dates) == 0:
        return None
    return pd.Timestamp(common_dates[0])


def run_ticker_comparison_backtest(
    ticker_pattern: TickerComparisonPattern,
    signals: pd.DataFrame,
    market_data: dict[str, pd.DataFrame | pd.Series],
) -> BacktestResult:
    open_frame = market_data["open"]
    close_frame = market_data["close"]
    vix_close = market_data["vix_close"]

    trade_dates = _common_trade_dates_for_tickers(ticker_pattern.tickers, open_frame, close_frame)
    if len(trade_dates) == 0:
        raise ValueError(f"No overlapping trade dates were found for {ticker_pattern.code}.")

    ticker_opens = {ticker: open_frame[ticker].reindex(trade_dates) for ticker in ticker_pattern.tickers}
    ticker_closes = {ticker: close_frame[ticker].reindex(trade_dates) for ticker in ticker_pattern.tickers}

    trade_rows: list[dict[str, Any]] = []
    history_net_returns: list[float] = []
    last_exit_date: pd.Timestamp | None = None

    full_trade_dates = pd.DataFrame({"trade_date": trade_dates[(trade_dates >= BACKTEST_START) & (trade_dates <= BACKTEST_END)]})
    full_trade_dates["gross_return"] = 0.0
    full_trade_dates["net_return"] = 0.0
    full_trade_dates["trade_count"] = 0

    for signal in signals.itertuples(index=False):
        signal_date = pd.Timestamp(signal.signal_date)
        entry_date = pd.Timestamp(signal.entry_date)

        if entry_date not in trade_dates:
            continue
        if last_exit_date is not None and entry_date <= last_exit_date:
            continue

        exit_date = _first_vix_reversion_exit_date(signal_date, vix_close, trade_dates)
        if exit_date is None:
            exit_date = trade_dates[-1]

        gross_returns: list[float] = []
        for ticker in ticker_pattern.tickers:
            entry_open = ticker_opens[ticker].loc[entry_date]
            exit_close = ticker_closes[ticker].loc[exit_date]
            if pd.isna(entry_open) or pd.isna(exit_close) or entry_open <= 0:
                gross_returns = []
                break
            gross_returns.append((float(exit_close) / float(entry_open)) - 1.0)

        if not gross_returns:
            continue

        allocation_ratio = _resolve_kelly_allocation_ratio(KellyPattern("K2", "Full Kelly"), history_net_returns)
        exposure_multiplier = allocation_ratio / FIXED_ALLOCATION_RATIO
        gross_trade_return = float(np.mean(gross_returns))
        net_trade_return = gross_trade_return - ROUND_TRIP_COST
        gross_portfolio_return = gross_trade_return * exposure_multiplier
        net_portfolio_return = net_trade_return * exposure_multiplier

        history_net_returns.append(float(net_trade_return))
        last_exit_date = exit_date

        trade_rows.append(
            {
                "signal_date": signal_date,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "allocation_ratio": float(allocation_ratio),
                "exposure_multiplier": float(exposure_multiplier),
                "gross_return": gross_trade_return,
                "net_return": net_trade_return,
                "gross_portfolio_return": gross_portfolio_return,
                "net_portfolio_return": net_portfolio_return,
                "holding_days": int(trade_dates.get_loc(exit_date) - trade_dates.get_loc(entry_date)),
            }
        )

        mask = full_trade_dates["trade_date"] == exit_date
        full_trade_dates.loc[mask, "gross_return"] += gross_portfolio_return
        full_trade_dates.loc[mask, "net_return"] += net_portfolio_return
        full_trade_dates.loc[mask, "trade_count"] += 1

    trades = pd.DataFrame(trade_rows)
    if trades.empty:
        raise ValueError(f"No trades were generated for {ticker_pattern.code}.")

    daily_results = full_trade_dates.copy()
    daily_results["gross_equity_curve"] = (1.0 + daily_results["gross_return"]).cumprod()
    daily_results["net_equity_curve"] = (1.0 + daily_results["net_return"]).cumprod()
    daily_results["drawdown"] = daily_results["net_equity_curve"] / daily_results["net_equity_curve"].cummax() - 1.0

    annual_return_gross = _annualized_return(daily_results["gross_return"], daily_results["trade_date"])
    annual_return_net = _annualized_return(daily_results["net_return"], daily_results["trade_date"])
    max_drawdown = float(daily_results["drawdown"].min()) if not daily_results.empty else 0.0
    win_rate = float((trades["net_return"] > 0).mean()) if not trades.empty else 0.0
    total_trade_count = int(len(trades))
    avg_holding_days = float(trades["holding_days"].mean()) if not trades.empty else 0.0
    avg_return_per_trade = float(trades["gross_return"].mean()) if not trades.empty else 0.0
    avg_cost_per_trade = float((ROUND_TRIP_COST * trades["exposure_multiplier"]).mean()) if not trades.empty else 0.0
    avg_net_per_trade = float(trades["net_return"].mean()) if not trades.empty else 0.0
    risk_return_ratio = annual_return_net / abs(max_drawdown) if max_drawdown < 0 else 0.0
    yearly_returns = _build_yearly_returns(daily_results)
    period_stats = _build_period_stats(daily_results, trades)
    both_periods_positive = bool((period_stats["net_annual_return"] > 0).all())
    max_drawdown_within_limit = max_drawdown >= -0.30
    trade_count_meets_minimum = total_trade_count >= 10

    data_start_date = _first_available_trade_date(ticker_pattern.tickers, open_frame, close_frame)
    strategy_name = f"{ticker_pattern.code} | {ticker_pattern.description}"
    trades.insert(0, "strategy", strategy_name)
    trades.insert(1, "data_start_date", data_start_date.strftime("%Y-%m-%d") if data_start_date is not None else "")
    daily_results.insert(0, "strategy", strategy_name)

    result = BacktestResult(
        name=strategy_name,
        pattern_label=ticker_pattern.code,
        averaging=False,
        trades=trades,
        daily_results=daily_results,
        annual_return_gross=annual_return_gross,
        annual_return_net=annual_return_net,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        total_trade_count=total_trade_count,
        avg_holding_days=avg_holding_days,
        avg_return_per_trade=avg_return_per_trade,
        avg_cost_per_trade=avg_cost_per_trade,
        avg_net_per_trade=avg_net_per_trade,
        risk_return_ratio=risk_return_ratio,
        yearly_returns=yearly_returns,
        period_stats=period_stats,
        both_periods_positive=both_periods_positive,
        max_drawdown_within_limit=max_drawdown_within_limit,
        trade_count_meets_minimum=trade_count_meets_minimum,
    )
    return result


def _switch_weights(pattern: SwitchPattern, signal_date: pd.Timestamp, usdjpy_close: pd.Series) -> tuple[float, float]:
    if pattern.code == "SW0":
        return 0.5, 0.5
    if pattern.code == "SW2":
        return 1.0, 0.0

    fx_change = _latest_series_pct_change_on_or_before(usdjpy_close, signal_date)
    if fx_change is None:
        return 0.5, 0.5
    if fx_change <= FX_SWITCH_DOWN_THRESHOLD:
        return 1.0 - FX_SWITCH_LOW_2558_WEIGHT, FX_SWITCH_LOW_2558_WEIGHT
    if fx_change >= FX_SWITCH_UP_THRESHOLD:
        return 1.0 - FX_SWITCH_HIGH_2558_WEIGHT, FX_SWITCH_HIGH_2558_WEIGHT
    return 0.5, 0.5


def run_switch_backtest(
    switch_pattern: SwitchPattern,
    signals: pd.DataFrame,
    market_data: dict[str, pd.DataFrame | pd.Series],
) -> BacktestResult:
    open_frame = market_data["open"]
    close_frame = market_data["close"]
    vix_close = market_data["vix_close"]
    usdjpy_close = market_data["usdjpy_close"]

    trade_dates = _common_trade_dates_for_tickers(("1570.T", "2558.T"), open_frame, close_frame)
    if len(trade_dates) == 0:
        raise ValueError(f"No overlapping trade dates were found for {switch_pattern.code}.")

    opens_1570 = open_frame["1570.T"].reindex(trade_dates)
    closes_1570 = close_frame["1570.T"].reindex(trade_dates)
    opens_2558 = open_frame["2558.T"].reindex(trade_dates)
    closes_2558 = close_frame["2558.T"].reindex(trade_dates)

    trade_rows: list[dict[str, Any]] = []
    history_net_returns: list[float] = []
    last_exit_date: pd.Timestamp | None = None

    full_trade_dates = pd.DataFrame({"trade_date": trade_dates[(trade_dates >= BACKTEST_START) & (trade_dates <= BACKTEST_END)]})
    full_trade_dates["gross_return"] = 0.0
    full_trade_dates["net_return"] = 0.0
    full_trade_dates["trade_count"] = 0

    for signal in signals.itertuples(index=False):
        signal_date = pd.Timestamp(signal.signal_date)
        entry_date = pd.Timestamp(signal.entry_date)

        if entry_date not in trade_dates:
            continue
        if last_exit_date is not None and entry_date <= last_exit_date:
            continue

        exit_date = _first_vix_reversion_exit_date(signal_date, vix_close, trade_dates)
        if exit_date is None:
            exit_date = trade_dates[-1]

        weight_1570, weight_2558 = _switch_weights(switch_pattern, signal_date, usdjpy_close)
        weighted_returns: list[float] = []

        if weight_1570 > 0:
            entry_open = opens_1570.loc[entry_date]
            exit_close = closes_1570.loc[exit_date]
            if pd.isna(entry_open) or pd.isna(exit_close) or entry_open <= 0:
                continue
            weighted_returns.append(weight_1570 * ((float(exit_close) / float(entry_open)) - 1.0))

        if weight_2558 > 0:
            entry_open = opens_2558.loc[entry_date]
            exit_close = closes_2558.loc[exit_date]
            if pd.isna(entry_open) or pd.isna(exit_close) or entry_open <= 0:
                continue
            weighted_returns.append(weight_2558 * ((float(exit_close) / float(entry_open)) - 1.0))

        if not weighted_returns:
            continue

        allocation_ratio = _resolve_kelly_allocation_ratio(KellyPattern("K2", "Full Kelly"), history_net_returns)
        exposure_multiplier = allocation_ratio / FIXED_ALLOCATION_RATIO
        gross_trade_return = float(np.sum(weighted_returns))
        net_trade_return = gross_trade_return - ROUND_TRIP_COST
        gross_portfolio_return = gross_trade_return * exposure_multiplier
        net_portfolio_return = net_trade_return * exposure_multiplier

        history_net_returns.append(float(net_trade_return))
        last_exit_date = exit_date

        trade_rows.append(
            {
                "signal_date": signal_date,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "weight_1570": weight_1570,
                "weight_2558": weight_2558,
                "allocation_ratio": float(allocation_ratio),
                "exposure_multiplier": float(exposure_multiplier),
                "gross_return": gross_trade_return,
                "net_return": net_trade_return,
                "gross_portfolio_return": gross_portfolio_return,
                "net_portfolio_return": net_portfolio_return,
                "holding_days": int(trade_dates.get_loc(exit_date) - trade_dates.get_loc(entry_date)),
            }
        )

        mask = full_trade_dates["trade_date"] == exit_date
        full_trade_dates.loc[mask, "gross_return"] += gross_portfolio_return
        full_trade_dates.loc[mask, "net_return"] += net_portfolio_return
        full_trade_dates.loc[mask, "trade_count"] += 1

    trades = pd.DataFrame(trade_rows)
    if trades.empty:
        raise ValueError(f"No trades were generated for {switch_pattern.code}.")

    daily_results = full_trade_dates.copy()
    daily_results["gross_equity_curve"] = (1.0 + daily_results["gross_return"]).cumprod()
    daily_results["net_equity_curve"] = (1.0 + daily_results["net_return"]).cumprod()
    daily_results["drawdown"] = daily_results["net_equity_curve"] / daily_results["net_equity_curve"].cummax() - 1.0

    annual_return_gross = _annualized_return(daily_results["gross_return"], daily_results["trade_date"])
    annual_return_net = _annualized_return(daily_results["net_return"], daily_results["trade_date"])
    max_drawdown = float(daily_results["drawdown"].min()) if not daily_results.empty else 0.0
    win_rate = float((trades["net_return"] > 0).mean()) if not trades.empty else 0.0
    total_trade_count = int(len(trades))
    avg_holding_days = float(trades["holding_days"].mean()) if not trades.empty else 0.0
    avg_return_per_trade = float(trades["gross_return"].mean()) if not trades.empty else 0.0
    avg_cost_per_trade = float((ROUND_TRIP_COST * trades["exposure_multiplier"]).mean()) if not trades.empty else 0.0
    avg_net_per_trade = float(trades["net_return"].mean()) if not trades.empty else 0.0
    risk_return_ratio = annual_return_net / abs(max_drawdown) if max_drawdown < 0 else 0.0
    yearly_returns = _build_yearly_returns(daily_results)
    period_stats = _build_period_stats(daily_results, trades)
    both_periods_positive = bool((period_stats["net_annual_return"] > 0).all())
    max_drawdown_within_limit = max_drawdown >= -0.30
    trade_count_meets_minimum = total_trade_count >= 10

    strategy_name = f"{switch_pattern.code} | {switch_pattern.description}"
    data_start_date = _first_available_trade_date(("1570.T", "2558.T"), open_frame, close_frame)
    trades.insert(0, "strategy", strategy_name)
    trades.insert(1, "data_start_date", data_start_date.strftime("%Y-%m-%d") if data_start_date is not None else "")
    daily_results.insert(0, "strategy", strategy_name)

    return BacktestResult(
        name=strategy_name,
        pattern_label=switch_pattern.code,
        averaging=False,
        trades=trades,
        daily_results=daily_results,
        annual_return_gross=annual_return_gross,
        annual_return_net=annual_return_net,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        total_trade_count=total_trade_count,
        avg_holding_days=avg_holding_days,
        avg_return_per_trade=avg_return_per_trade,
        avg_cost_per_trade=avg_cost_per_trade,
        avg_net_per_trade=avg_net_per_trade,
        risk_return_ratio=risk_return_ratio,
        yearly_returns=yearly_returns,
        period_stats=period_stats,
        both_periods_positive=both_periods_positive,
        max_drawdown_within_limit=max_drawdown_within_limit,
        trade_count_meets_minimum=trade_count_meets_minimum,
    )


def build_signal_window_correlation_analysis(
    signals: pd.DataFrame,
    market_data: dict[str, pd.DataFrame | pd.Series],
) -> pd.DataFrame:
    close_frame = market_data["close"]
    vix_close = market_data["vix_close"]
    usdjpy_close = market_data["usdjpy_close"]

    common_trade_dates = _common_trade_dates_for_tickers(("1570.T", "2558.T"), close_frame, close_frame)
    if len(common_trade_dates) == 0 or signals.empty:
        return pd.DataFrame()

    close_1570 = close_frame["1570.T"].reindex(common_trade_dates)
    close_2558 = close_frame["2558.T"].reindex(common_trade_dates)
    ret_1570 = close_1570.pct_change()
    ret_2558 = close_2558.pct_change()

    window_pairs: list[pd.DataFrame] = []
    fx_vix_pairs: list[pd.DataFrame] = []

    for signal in signals.itertuples(index=False):
        signal_date = pd.Timestamp(signal.signal_date)
        entry_date = pd.Timestamp(signal.entry_date)
        exit_date = _first_vix_reversion_exit_date(signal_date, vix_close, common_trade_dates)
        if exit_date is None:
            exit_date = common_trade_dates[-1]
        if entry_date not in common_trade_dates:
            continue

        window_index = common_trade_dates[(common_trade_dates >= entry_date) & (common_trade_dates <= exit_date)]
        if len(window_index) < 2:
            continue

        window_df = pd.DataFrame(
            {
                "ret_1570": ret_1570.reindex(window_index),
                "ret_2558": ret_2558.reindex(window_index),
            }
        ).dropna()
        if not window_df.empty:
            window_pairs.append(window_df)

        vix_window = vix_close.reindex(window_index).pct_change()
        fx_window = usdjpy_close.reindex(window_index).pct_change()
        fx_vix_df = pd.DataFrame({"usd_jpy_return": fx_window, "vix_return": vix_window}).dropna()
        if not fx_vix_df.empty:
            fx_vix_pairs.append(fx_vix_df)

    corr_rows: list[dict[str, Any]] = []
    if window_pairs:
        merged = pd.concat(window_pairs, ignore_index=True)
        corr_rows.append(
            {
                "section": "correlation",
                "metric": "1570_vs_2558_signal_window_return_corr",
                "value": float(merged["ret_1570"].corr(merged["ret_2558"])),
                "observations": int(len(merged)),
            }
        )
    if fx_vix_pairs:
        merged_fx = pd.concat(fx_vix_pairs, ignore_index=True)
        corr_rows.append(
            {
                "section": "correlation",
                "metric": "usd_jpy_vs_vix_signal_window_return_corr",
                "value": float(merged_fx["usd_jpy_return"].corr(merged_fx["vix_return"])),
                "observations": int(len(merged_fx)),
            }
        )

    return pd.DataFrame(corr_rows)


def run_backtest(
    pattern: ExitPattern,
    averaging: bool,
    signals: pd.DataFrame,
    market_data: dict[str, pd.DataFrame | pd.Series],
) -> BacktestResult:
    open_frame = market_data["open"]
    high_frame = market_data["high"]
    low_frame = market_data["low"]
    close_frame = market_data["close"]
    vix_close = market_data["vix_close"]

    opens = open_frame[ETF_TICKER].dropna().sort_index()
    highs = high_frame[ETF_TICKER].reindex(opens.index)
    lows = low_frame[ETF_TICKER].reindex(opens.index)
    closes = close_frame[ETF_TICKER].reindex(opens.index)
    trade_dates = opens.index.intersection(closes.dropna().index)

    trade_rows: list[dict[str, Any]] = []
    last_exit_date: pd.Timestamp | None = None

    for signal in signals.itertuples(index=False):
        signal_date = pd.Timestamp(signal.signal_date)
        entry_date = pd.Timestamp(signal.entry_date)

        if last_exit_date is not None and entry_date <= last_exit_date:
            continue

        trade = simulate_trade(
            signal_date=signal_date,
            entry_date=entry_date,
            pattern=pattern,
            averaging=averaging,
            trade_dates=trade_dates,
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            vix_close=vix_close,
        )
        if trade is None:
            continue

        trade_rows.append(trade)
        last_exit_date = pd.Timestamp(trade["exit_date"])

    trades = pd.DataFrame(trade_rows)
    if trades.empty:
        raise ValueError(f"No trades were generated for {pattern.label}.")

    full_trade_dates = pd.DataFrame({"trade_date": trade_dates[(trade_dates >= BACKTEST_START) & (trade_dates <= BACKTEST_END)]})
    full_trade_dates["gross_return"] = 0.0
    full_trade_dates["net_return"] = 0.0
    full_trade_dates["trade_count"] = 0

    for trade in trades.itertuples(index=False):
        mask = full_trade_dates["trade_date"] == trade.exit_date
        full_trade_dates.loc[mask, "gross_return"] += float(trade.gross_return)
        full_trade_dates.loc[mask, "net_return"] += float(trade.net_return)
        full_trade_dates.loc[mask, "trade_count"] += 1

    daily_results = full_trade_dates.copy()
    daily_results["gross_equity_curve"] = (1.0 + daily_results["gross_return"]).cumprod()
    daily_results["net_equity_curve"] = (1.0 + daily_results["net_return"]).cumprod()
    daily_results["drawdown"] = daily_results["net_equity_curve"] / daily_results["net_equity_curve"].cummax() - 1.0

    annual_return_gross = _annualized_return(daily_results["gross_return"], daily_results["trade_date"])
    annual_return_net = _annualized_return(daily_results["net_return"], daily_results["trade_date"])
    max_drawdown = float(daily_results["drawdown"].min()) if not daily_results.empty else 0.0
    win_rate = float((trades["net_return"] > 0).mean()) if not trades.empty else 0.0
    total_trade_count = int(len(trades))
    avg_holding_days = float(trades["holding_days"].mean()) if not trades.empty else 0.0
    avg_return_per_trade = float(trades["gross_return"].mean()) if not trades.empty else 0.0
    avg_cost_per_trade = float(trades["cost"].mean()) if not trades.empty else 0.0
    avg_net_per_trade = float(trades["net_return"].mean()) if not trades.empty else 0.0
    risk_return_ratio = annual_return_net / abs(max_drawdown) if max_drawdown < 0 else 0.0
    yearly_returns = _build_yearly_returns(daily_results)
    period_stats = _build_period_stats(daily_results, trades)
    both_periods_positive = bool((period_stats["net_annual_return"] > 0).all())
    max_drawdown_within_limit = max_drawdown >= -0.30
    trade_count_meets_minimum = total_trade_count >= 10

    averaging_label = "On" if averaging else "Off"
    strategy_name = f"{pattern.label} | Averaging {averaging_label}"

    trades.insert(0, "strategy", strategy_name)
    trades.insert(1, "pattern", pattern.label)
    trades.insert(2, "averaging", averaging_label)
    daily_results.insert(0, "strategy", strategy_name)
    daily_results.insert(1, "pattern", pattern.label)
    daily_results.insert(2, "averaging", averaging_label)

    return BacktestResult(
        name=strategy_name,
        pattern_label=pattern.label,
        averaging=averaging,
        trades=trades,
        daily_results=daily_results,
        annual_return_gross=annual_return_gross,
        annual_return_net=annual_return_net,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        total_trade_count=total_trade_count,
        avg_holding_days=avg_holding_days,
        avg_return_per_trade=avg_return_per_trade,
        avg_cost_per_trade=avg_cost_per_trade,
        avg_net_per_trade=avg_net_per_trade,
        risk_return_ratio=risk_return_ratio,
        yearly_returns=yearly_returns,
        period_stats=period_stats,
        both_periods_positive=both_periods_positive,
        max_drawdown_within_limit=max_drawdown_within_limit,
        trade_count_meets_minimum=trade_count_meets_minimum,
    )


def run_longterm_backtest(
    longterm_pattern: LongTermPattern,
    market_data: dict[str, pd.DataFrame | pd.Series],
    condition: ConditionPattern,
) -> BacktestResult:
    open_frame = market_data["open"]
    close_frame = market_data["close"]
    vix_close = market_data["vix_close"]

    if longterm_pattern.ticker not in open_frame.columns or longterm_pattern.ticker not in close_frame.columns:
        raise ValueError(f"{longterm_pattern.ticker} is missing from long-term market data.")

    opens = open_frame[longterm_pattern.ticker].dropna().sort_index()
    closes = close_frame[longterm_pattern.ticker].reindex(opens.index)
    trade_dates = opens.index.intersection(closes.dropna().index)
    trade_dates = trade_dates[
        (trade_dates >= longterm_pattern.requested_start) & (trade_dates <= BACKTEST_END)
    ]
    if len(trade_dates) == 0:
        raise ValueError(f"No trade dates were found for {longterm_pattern.ticker}.")

    signals = build_entry_signals(
        vix_close,
        trade_dates,
        pct_change_threshold=condition.pct_change_threshold,
        absolute_vix_threshold=condition.absolute_vix_threshold,
    )
    if signals.empty:
        raise ValueError(f"No signals were generated for {longterm_pattern.ticker}.")

    trade_rows: list[dict[str, Any]] = []
    last_exit_date: pd.Timestamp | None = None
    full_trade_dates = pd.DataFrame({"trade_date": trade_dates})
    full_trade_dates["gross_return"] = 0.0
    full_trade_dates["net_return"] = 0.0
    full_trade_dates["trade_count"] = 0

    for signal in signals.itertuples(index=False):
        signal_date = pd.Timestamp(signal.signal_date)
        entry_date = pd.Timestamp(signal.entry_date)

        if last_exit_date is not None and entry_date <= last_exit_date:
            continue

        exit_date = _first_vix_reversion_exit_date(signal_date, vix_close, trade_dates)
        if exit_date is None:
            exit_date = trade_dates[-1]
        if entry_date not in trade_dates or exit_date not in trade_dates:
            continue

        entry_open = opens.loc[entry_date]
        exit_close = closes.loc[exit_date]
        if pd.isna(entry_open) or pd.isna(exit_close) or entry_open <= 0:
            continue

        gross_trade_return = (float(exit_close) / float(entry_open)) - 1.0
        net_trade_return = gross_trade_return - ROUND_TRIP_COST
        trade_rows.append(
            {
                "signal_date": signal_date,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": float(entry_open),
                "exit_price": float(exit_close),
                "gross_return": float(gross_trade_return),
                "net_return": float(net_trade_return),
                "cost": ROUND_TRIP_COST,
                "holding_days": int(trade_dates.get_loc(exit_date) - trade_dates.get_loc(entry_date)),
            }
        )
        last_exit_date = exit_date

        mask = full_trade_dates["trade_date"] == exit_date
        full_trade_dates.loc[mask, "gross_return"] += float(gross_trade_return)
        full_trade_dates.loc[mask, "net_return"] += float(net_trade_return)
        full_trade_dates.loc[mask, "trade_count"] += 1

    trades = pd.DataFrame(trade_rows)
    if trades.empty:
        raise ValueError(f"No long-term trades were generated for {longterm_pattern.ticker}.")

    daily_results = full_trade_dates.copy()
    daily_results["gross_equity_curve"] = (1.0 + daily_results["gross_return"]).cumprod()
    daily_results["net_equity_curve"] = (1.0 + daily_results["net_return"]).cumprod()
    daily_results["drawdown"] = daily_results["net_equity_curve"] / daily_results["net_equity_curve"].cummax() - 1.0

    data_start_date = pd.Timestamp(trade_dates[0])
    data_end_date = pd.Timestamp(trade_dates[-1])
    annual_return_gross = _annualized_return(daily_results["gross_return"], daily_results["trade_date"])
    annual_return_net = _annualized_return(daily_results["net_return"], daily_results["trade_date"])
    max_drawdown = float(daily_results["drawdown"].min()) if not daily_results.empty else 0.0
    win_count = int((trades["net_return"] > 0).sum()) if not trades.empty else 0
    total_trade_count = int(len(trades))
    win_rate = float(win_count / total_trade_count) if total_trade_count > 0 else 0.0
    win_rate_ci_lower, win_rate_ci_upper = _wilson_interval(win_count, total_trade_count)
    avg_holding_days = float(trades["holding_days"].mean()) if not trades.empty else 0.0
    avg_return_per_trade = float(trades["gross_return"].mean()) if not trades.empty else 0.0
    avg_cost_per_trade = float(trades["cost"].mean()) if not trades.empty else 0.0
    avg_net_per_trade = float(trades["net_return"].mean()) if not trades.empty else 0.0
    risk_return_ratio = annual_return_net / abs(max_drawdown) if max_drawdown < 0 else 0.0
    yearly_returns = _build_yearly_returns_for_range(daily_results, data_start_date.year, data_end_date.year)
    period_stats = _build_equal_split_period_stats(daily_results, trades)
    both_periods_positive = bool((period_stats["net_annual_return"] > 0).all()) if not period_stats.empty else False
    max_drawdown_within_limit = max_drawdown >= -0.30
    trade_count_meets_minimum = total_trade_count >= 10
    avg_trades_per_year = total_trade_count / _years_between(data_start_date, data_end_date)

    strategy_name = f"{longterm_pattern.code} | {longterm_pattern.description}"
    trades.insert(0, "strategy", strategy_name)
    trades.insert(1, "ticker", longterm_pattern.ticker)
    trades.insert(2, "data_start_date", data_start_date.strftime("%Y-%m-%d"))
    trades.insert(3, "data_end_date", data_end_date.strftime("%Y-%m-%d"))
    daily_results.insert(0, "strategy", strategy_name)
    daily_results.insert(1, "ticker", longterm_pattern.ticker)

    return BacktestResult(
        name=strategy_name,
        pattern_label=longterm_pattern.code,
        averaging=False,
        trades=trades,
        daily_results=daily_results,
        annual_return_gross=annual_return_gross,
        annual_return_net=annual_return_net,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        total_trade_count=total_trade_count,
        avg_holding_days=avg_holding_days,
        avg_return_per_trade=avg_return_per_trade,
        avg_cost_per_trade=avg_cost_per_trade,
        avg_net_per_trade=avg_net_per_trade,
        risk_return_ratio=risk_return_ratio,
        yearly_returns=yearly_returns,
        period_stats=period_stats,
        both_periods_positive=both_periods_positive,
        max_drawdown_within_limit=max_drawdown_within_limit,
        trade_count_meets_minimum=trade_count_meets_minimum,
        win_rate_ci_lower=win_rate_ci_lower,
        win_rate_ci_upper=win_rate_ci_upper,
        avg_trades_per_year=avg_trades_per_year,
    )


def run_monte_backtest(
    monte_pattern: MontePattern,
    signals: pd.DataFrame,
    market_data: dict[str, pd.DataFrame | pd.Series],
    ticker: str = "2558.T",
) -> BacktestResult:
    open_frame = market_data["open"]
    close_frame = market_data["close"]
    vix_close = market_data["vix_close"]

    opens = open_frame[ticker].dropna().sort_index()
    closes = close_frame[ticker].reindex(opens.index)
    trade_dates = opens.index.intersection(closes.dropna().index)

    trade_rows: list[dict[str, Any]] = []
    history_net_returns: list[float] = []
    last_exit_date: pd.Timestamp | None = None
    monte_sequence = _clone_monte_sequence()

    full_trade_dates = pd.DataFrame({"trade_date": trade_dates[(trade_dates >= BACKTEST_START) & (trade_dates <= BACKTEST_END)]})
    full_trade_dates["gross_return"] = 0.0
    full_trade_dates["net_return"] = 0.0
    full_trade_dates["trade_count"] = 0

    for signal in signals.itertuples(index=False):
        signal_date = pd.Timestamp(signal.signal_date)
        entry_date = pd.Timestamp(signal.entry_date)

        if last_exit_date is not None and entry_date <= last_exit_date:
            continue

        exit_date = _first_vix_reversion_exit_date(signal_date, vix_close, trade_dates)
        if exit_date is None:
            exit_date = trade_dates[-1]
        if entry_date not in trade_dates or exit_date not in trade_dates:
            continue

        entry_open = opens.loc[entry_date]
        exit_close = closes.loc[exit_date]
        if pd.isna(entry_open) or pd.isna(exit_close) or entry_open <= 0:
            continue

        if monte_pattern.code == "MC0":
            allocation_ratio = FIXED_ALLOCATION_RATIO
            monte_units = np.nan
            monte_before = ""
        elif monte_pattern.code == "MC1":
            allocation_ratio, monte_units = _resolve_monte_allocation_ratio(monte_sequence)
            monte_before = ",".join(f"{value:.4f}" for value in monte_sequence)
        else:
            allocation_ratio = _resolve_kelly_allocation_ratio(KellyPattern("K1", "Half Kelly"), history_net_returns)
            monte_units = np.nan
            monte_before = ""

        exposure_multiplier = allocation_ratio / FIXED_ALLOCATION_RATIO
        gross_trade_return = (float(exit_close) / float(entry_open)) - 1.0
        net_trade_return = gross_trade_return - ROUND_TRIP_COST
        gross_portfolio_return = gross_trade_return * exposure_multiplier
        net_portfolio_return = net_trade_return * exposure_multiplier
        won_trade = net_trade_return > 0

        if monte_pattern.code == "MC1":
            monte_sequence = _update_monte_sequence(monte_sequence, won_trade, float(monte_units))
            monte_after = ",".join(f"{value:.4f}" for value in monte_sequence)
        else:
            monte_after = ""

        history_net_returns.append(float(net_trade_return))
        last_exit_date = exit_date

        trade_rows.append(
            {
                "signal_date": signal_date,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": float(entry_open),
                "exit_price": float(exit_close),
                "allocation_ratio": float(allocation_ratio),
                "exposure_multiplier": float(exposure_multiplier),
                "gross_return": float(gross_trade_return),
                "net_return": float(net_trade_return),
                "gross_portfolio_return": float(gross_portfolio_return),
                "net_portfolio_return": float(net_portfolio_return),
                "holding_days": int(trade_dates.get_loc(exit_date) - trade_dates.get_loc(entry_date)),
                "monte_units": float(monte_units) if not pd.isna(monte_units) else np.nan,
                "monte_sequence_before": monte_before,
                "monte_sequence_after": monte_after,
            }
        )

        mask = full_trade_dates["trade_date"] == exit_date
        full_trade_dates.loc[mask, "gross_return"] += float(gross_portfolio_return)
        full_trade_dates.loc[mask, "net_return"] += float(net_portfolio_return)
        full_trade_dates.loc[mask, "trade_count"] += 1

    trades = pd.DataFrame(trade_rows)
    if trades.empty:
        raise ValueError(f"No trades were generated for {monte_pattern.code}.")

    daily_results = full_trade_dates.copy()
    daily_results["gross_equity_curve"] = (1.0 + daily_results["gross_return"]).cumprod()
    daily_results["net_equity_curve"] = (1.0 + daily_results["net_return"]).cumprod()
    daily_results["drawdown"] = daily_results["net_equity_curve"] / daily_results["net_equity_curve"].cummax() - 1.0

    annual_return_gross = _annualized_return(daily_results["gross_return"], daily_results["trade_date"])
    annual_return_net = _annualized_return(daily_results["net_return"], daily_results["trade_date"])
    max_drawdown = float(daily_results["drawdown"].min()) if not daily_results.empty else 0.0
    win_rate = float((trades["net_return"] > 0).mean()) if not trades.empty else 0.0
    total_trade_count = int(len(trades))
    avg_holding_days = float(trades["holding_days"].mean()) if not trades.empty else 0.0
    avg_return_per_trade = float(trades["gross_return"].mean()) if not trades.empty else 0.0
    avg_cost_per_trade = float((ROUND_TRIP_COST * trades["exposure_multiplier"]).mean()) if not trades.empty else 0.0
    avg_net_per_trade = float(trades["net_return"].mean()) if not trades.empty else 0.0
    avg_allocation_ratio = float(trades["allocation_ratio"].mean()) if not trades.empty else 0.0
    risk_return_ratio = annual_return_net / abs(max_drawdown) if max_drawdown < 0 else 0.0
    yearly_returns = _build_yearly_returns(daily_results)
    period_stats = _build_equal_split_period_stats(daily_results, trades)
    both_periods_positive = bool((period_stats["net_annual_return"] > 0).all()) if not period_stats.empty else False
    max_drawdown_within_limit = max_drawdown >= -0.30
    trade_count_meets_minimum = total_trade_count >= 10

    strategy_name = f"{monte_pattern.code} | {monte_pattern.description}"
    trades.insert(0, "strategy", strategy_name)
    trades.insert(1, "ticker", ticker)
    daily_results.insert(0, "strategy", strategy_name)

    return BacktestResult(
        name=strategy_name,
        pattern_label=monte_pattern.code,
        averaging=False,
        trades=trades,
        daily_results=daily_results,
        annual_return_gross=annual_return_gross,
        annual_return_net=annual_return_net,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        total_trade_count=total_trade_count,
        avg_holding_days=avg_holding_days,
        avg_return_per_trade=avg_return_per_trade,
        avg_cost_per_trade=avg_cost_per_trade,
        avg_net_per_trade=avg_net_per_trade,
        risk_return_ratio=risk_return_ratio,
        yearly_returns=yearly_returns,
        period_stats=period_stats,
        both_periods_positive=both_periods_positive,
        max_drawdown_within_limit=max_drawdown_within_limit,
        trade_count_meets_minimum=trade_count_meets_minimum,
        avg_allocation_ratio=avg_allocation_ratio,
    )


def select_best_result(results: list[BacktestResult]) -> BacktestResult:
    ranked = sorted(
        results,
        key=lambda result: (
            result.annual_return_net,
            int(result.both_periods_positive),
            int(result.max_drawdown_within_limit),
            int(result.trade_count_meets_minimum),
            result.max_drawdown,
            result.total_trade_count,
        ),
        reverse=True,
    )
    return ranked[0]


def build_results_csv(results: list[BacktestResult], best_result: BacktestResult) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for result in results:
        row: dict[str, Any] = {
            "strategy": result.name,
            "pattern": result.pattern_label,
            "averaging": "On" if result.averaging else "Off",
            "is_best": result.name == best_result.name,
            "annual_return_no_cost": result.annual_return_gross,
            "annual_return_with_cost": result.annual_return_net,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "total_trade_count": result.total_trade_count,
            "avg_holding_days": result.avg_holding_days,
            "avg_return_per_trade": result.avg_return_per_trade,
            "avg_cost_per_trade": result.avg_cost_per_trade,
            "avg_net_per_trade": result.avg_net_per_trade,
            "risk_return_ratio": result.risk_return_ratio,
            "both_periods_positive": result.both_periods_positive,
            "max_drawdown_within_limit": result.max_drawdown_within_limit,
            "trade_count_meets_minimum": result.trade_count_meets_minimum,
        }

        for yearly in result.yearly_returns.itertuples(index=False):
            row[f"gross_{yearly.year}"] = yearly.gross_return
            row[f"net_{yearly.year}"] = yearly.net_return

        for period in result.period_stats.itertuples(index=False):
            period_key = str(period.period).replace("-", "_")
            row[f"gross_{period_key}"] = period.gross_annual_return
            row[f"net_{period_key}"] = period.net_annual_return
            row[f"win_{period_key}"] = period.win_rate
            row[f"trades_{period_key}"] = period.trade_count
            row[f"avg_hold_{period_key}"] = period.avg_holding_days

        rows.append(row)

    summary = pd.DataFrame(rows)
    return summary.sort_values(
        [
            "annual_return_with_cost",
            "both_periods_positive",
            "max_drawdown_within_limit",
            "trade_count_meets_minimum",
            "max_drawdown",
            "total_trade_count",
        ],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)


def save_plot(results: list[BacktestResult]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(16, 11), sharex=True)

    for result in results:
        axes[0].plot(result.daily_results["trade_date"], result.daily_results["gross_equity_curve"], label=result.name, linewidth=1.2)
        axes[1].plot(result.daily_results["trade_date"], result.daily_results["net_equity_curve"], label=result.name, linewidth=1.2)

    axes[0].set_title("Cumulative Return Without Cost")
    axes[1].set_title("Cumulative Return With Cost")
    axes[0].set_ylabel("Gross Equity")
    axes[1].set_ylabel("Net Equity")
    axes[1].set_xlabel("Trade Date")

    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(RESULTS_PNG, dpi=150)
    plt.close(fig)


def build_condition_results_csv(results: list[BacktestResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for result in results:
        condition_code = result.name.split(" | ")[0]
        row: dict[str, Any] = {
            "condition": condition_code,
            "strategy": result.name,
            "annual_return_no_cost": result.annual_return_gross,
            "annual_return_with_cost": result.annual_return_net,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "total_trade_count": result.total_trade_count,
            "avg_holding_days": result.avg_holding_days,
            "avg_return_per_trade": result.avg_return_per_trade,
            "avg_cost_per_trade": result.avg_cost_per_trade,
            "avg_net_per_trade": result.avg_net_per_trade,
            "risk_return_ratio": result.risk_return_ratio,
            "net_2010_2017": _period_net_return(result.period_stats, "2010-2017"),
            "net_2018_2025": _period_net_return(result.period_stats, "2018-2025"),
        }

        for yearly in result.yearly_returns.itertuples(index=False):
            row[f"gross_{yearly.year}"] = yearly.gross_return
            row[f"net_{yearly.year}"] = yearly.net_return

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["annual_return_with_cost", "total_trade_count"],
        ascending=[False, False],
    ).reset_index(drop=True)


def save_condition_plot(results: list[BacktestResult]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    for result in results:
        label = result.name.split(" | ")[0]
        axes[0].plot(result.daily_results["trade_date"], result.daily_results["gross_equity_curve"], label=label, linewidth=1.5)
        axes[1].plot(result.daily_results["trade_date"], result.daily_results["net_equity_curve"], label=label, linewidth=1.5)

    axes[0].set_title("Condition Comparison Without Cost")
    axes[1].set_title("Condition Comparison With Cost")
    axes[0].set_ylabel("Gross Equity")
    axes[1].set_ylabel("Net Equity")
    axes[1].set_xlabel("Trade Date")

    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()

    fig.tight_layout()
    fig.savefig(CONDITIONS_PNG, dpi=150)
    plt.close(fig)


def build_nanpin_results_csv(results: list[BacktestResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for result in results:
        row: dict[str, Any] = {
            "strategy": result.name,
            "annual_return_no_cost": result.annual_return_gross,
            "annual_return_with_cost": result.annual_return_net,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "total_trade_count": result.total_trade_count,
            "avg_holding_days": result.avg_holding_days,
            "avg_return_per_trade": result.avg_return_per_trade,
            "avg_cost_per_trade": result.avg_cost_per_trade,
            "avg_net_per_trade": result.avg_net_per_trade,
            "risk_return_ratio": result.risk_return_ratio,
            "net_2010_2017": _period_net_return(result.period_stats, "2010-2017"),
            "net_2018_2025": _period_net_return(result.period_stats, "2018-2025"),
        }
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["annual_return_with_cost", "max_drawdown", "total_trade_count"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def save_nanpin_plot(results: list[BacktestResult]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    for result in results:
        axes[0].plot(result.daily_results["trade_date"], result.daily_results["gross_equity_curve"], label=result.name, linewidth=1.4)
        axes[1].plot(result.daily_results["trade_date"], result.daily_results["net_equity_curve"], label=result.name, linewidth=1.4)

    axes[0].set_title("Nanpin and Capital Management Without Cost")
    axes[1].set_title("Nanpin and Capital Management With Cost")
    axes[0].set_ylabel("Gross Equity")
    axes[1].set_ylabel("Net Equity")
    axes[1].set_xlabel("Trade Date")

    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()

    fig.tight_layout()
    fig.savefig(NANPIN_PNG, dpi=150)
    plt.close(fig)


def build_kelly_results_csv(results: list[BacktestResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for result in results:
        row: dict[str, Any] = {
            "strategy": result.name,
            "annual_return_no_cost": result.annual_return_gross,
            "annual_return_with_cost": result.annual_return_net,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "total_trade_count": result.total_trade_count,
            "avg_holding_days": result.avg_holding_days,
            "avg_return_per_trade": result.avg_return_per_trade,
            "avg_cost_per_trade": result.avg_cost_per_trade,
            "avg_net_per_trade": result.avg_net_per_trade,
            "risk_return_ratio": result.risk_return_ratio,
            "ruin_probability": result.ruin_probability,
            "ruin_days": result.ruin_days,
            "net_2010_2017": _period_net_return(result.period_stats, "2010-2017"),
            "net_2018_2025": _period_net_return(result.period_stats, "2018-2025"),
        }

        for yearly in result.yearly_returns.itertuples(index=False):
            row[f"gross_{yearly.year}"] = yearly.gross_return
            row[f"net_{yearly.year}"] = yearly.net_return

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["annual_return_with_cost", "max_drawdown", "total_trade_count"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def save_kelly_plot(results: list[BacktestResult]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    for result in results:
        axes[0].plot(result.daily_results["trade_date"], result.daily_results["gross_equity_curve"], label=result.name, linewidth=1.5)
        axes[1].plot(result.daily_results["trade_date"], result.daily_results["net_equity_curve"], label=result.name, linewidth=1.5)

    axes[0].set_title("Kelly Comparison Without Cost")
    axes[1].set_title("Kelly Comparison With Cost")
    axes[0].set_ylabel("Gross Equity")
    axes[1].set_ylabel("Net Equity")
    axes[1].set_xlabel("Trade Date")

    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()

    fig.tight_layout()
    fig.savefig(KELLY_PNG, dpi=150)
    plt.close(fig)


def _result_data_start_date(result: BacktestResult) -> str:
    if "data_start_date" in result.trades.columns and not result.trades.empty:
        value = result.trades["data_start_date"].iloc[0]
        return str(value)
    return ""


def _result_data_end_date(result: BacktestResult) -> str:
    if result.daily_results.empty:
        return ""
    return pd.Timestamp(result.daily_results["trade_date"].max()).strftime("%Y-%m-%d")


def _result_data_period(result: BacktestResult) -> str:
    start_date = _result_data_start_date(result)
    end_date = _result_data_end_date(result)
    if start_date and end_date:
        return f"{start_date} to {end_date}"
    return start_date or end_date


def build_ticker_results_csv(results: list[BacktestResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for result in results:
        row: dict[str, Any] = {
            "strategy": result.name,
            "ticker_code": result.pattern_label,
            "data_start_date": _result_data_start_date(result),
            "annual_return_no_cost": result.annual_return_gross,
            "annual_return_with_cost": result.annual_return_net,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "total_trade_count": result.total_trade_count,
            "avg_holding_days": result.avg_holding_days,
            "risk_return_ratio": result.risk_return_ratio,
            "net_2010_2017": _period_net_return(result.period_stats, "2010-2017"),
            "net_2018_2025": _period_net_return(result.period_stats, "2018-2025"),
        }

        for yearly in result.yearly_returns.itertuples(index=False):
            row[f"gross_{yearly.year}"] = yearly.gross_return
            row[f"net_{yearly.year}"] = yearly.net_return

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["annual_return_with_cost", "risk_return_ratio", "total_trade_count"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def save_ticker_plot(results: list[BacktestResult]) -> None:
    fig, ax = plt.subplots(figsize=(15, 8))

    for result in results:
        ax.plot(result.daily_results["trade_date"], result.daily_results["net_equity_curve"], label=result.name, linewidth=1.6)

    ax.set_title("Ticker Comparison With Cost")
    ax.set_ylabel("Net Equity")
    ax.set_xlabel("Trade Date")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(TICKERS_PNG, dpi=150)
    plt.close(fig)


def build_extended_results_csv(results: list[BacktestResult], correlation_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for result in results:
        row: dict[str, Any] = {
            "section": "pattern",
            "strategy": result.name,
            "pattern_code": result.pattern_label,
            "data_start_date": _result_data_start_date(result),
            "data_end_date": _result_data_end_date(result),
            "data_period": _result_data_period(result),
            "annual_return_no_cost": result.annual_return_gross,
            "annual_return_with_cost": result.annual_return_net,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "total_trade_count": result.total_trade_count,
            "avg_holding_days": result.avg_holding_days,
            "risk_return_ratio": result.risk_return_ratio,
            "net_2010_2017": _period_net_return(result.period_stats, "2010-2017"),
            "net_2018_2025": _period_net_return(result.period_stats, "2018-2025"),
        }

        for yearly in result.yearly_returns.itertuples(index=False):
            row[f"gross_{yearly.year}"] = yearly.gross_return
            row[f"net_{yearly.year}"] = yearly.net_return

        rows.append(row)

    patterns_df = pd.DataFrame(rows)
    if not patterns_df.empty:
        patterns_df = patterns_df.sort_values(
            ["annual_return_with_cost", "risk_return_ratio", "total_trade_count"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    if correlation_df.empty:
        return patterns_df

    return pd.concat([patterns_df, correlation_df], ignore_index=True, sort=False)


def save_extended_plot(results: list[BacktestResult]) -> None:
    fig, ax = plt.subplots(figsize=(15, 8))

    for result in results:
        ax.plot(result.daily_results["trade_date"], result.daily_results["net_equity_curve"], label=result.name, linewidth=1.5)

    ax.set_title("Extended Comparison With Cost")
    ax.set_ylabel("Net Equity")
    ax.set_xlabel("Trade Date")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(EXTENDED_PNG, dpi=150)
    plt.close(fig)


def build_longterm_results_csv(results: list[BacktestResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for result in results:
        periods = result.period_stats.to_dict("records")
        first_period = periods[0] if len(periods) > 0 else {}
        second_period = periods[1] if len(periods) > 1 else {}
        row: dict[str, Any] = {
            "strategy": result.name,
            "ticker_code": result.pattern_label,
            "ticker": result.trades["ticker"].iloc[0] if "ticker" in result.trades.columns and not result.trades.empty else "",
            "data_start_date": _result_data_start_date(result),
            "data_end_date": _result_data_end_date(result),
            "data_period": _result_data_period(result),
            "annual_return_no_cost": result.annual_return_gross,
            "annual_return_with_cost": result.annual_return_net,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "win_rate_ci_lower": result.win_rate_ci_lower,
            "win_rate_ci_upper": result.win_rate_ci_upper,
            "total_trade_count": result.total_trade_count,
            "avg_trades_per_year": result.avg_trades_per_year,
            "avg_holding_days": result.avg_holding_days,
            "avg_return_per_trade": result.avg_return_per_trade,
            "avg_cost_per_trade": result.avg_cost_per_trade,
            "avg_net_per_trade": result.avg_net_per_trade,
            "risk_return_ratio": result.risk_return_ratio,
            "first_half_label": first_period.get("period", ""),
            "first_half_net_annual_return": first_period.get("net_annual_return", 0.0),
            "second_half_label": second_period.get("period", ""),
            "second_half_net_annual_return": second_period.get("net_annual_return", 0.0),
        }

        for yearly in result.yearly_returns.itertuples(index=False):
            row[f"gross_{yearly.year}"] = yearly.gross_return
            row[f"net_{yearly.year}"] = yearly.net_return

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["annual_return_with_cost", "risk_return_ratio", "total_trade_count"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_longterm_trades_csv(results: list[BacktestResult]) -> pd.DataFrame:
    trade_frames: list[pd.DataFrame] = []

    for result in results:
        if result.trades.empty:
            continue
        columns = [
            "strategy",
            "ticker",
            "data_start_date",
            "data_end_date",
            "signal_date",
            "entry_date",
            "exit_date",
            "entry_price",
            "exit_price",
            "gross_return",
            "net_return",
            "holding_days",
        ]
        available_columns = [column for column in columns if column in result.trades.columns]
        trade_frames.append(result.trades[available_columns].copy())

    if not trade_frames:
        return pd.DataFrame()

    trades_df = pd.concat(trade_frames, ignore_index=True)
    return trades_df.sort_values(["strategy", "entry_date", "exit_date"]).reset_index(drop=True)


def save_longterm_plot(results: list[BacktestResult]) -> None:
    fig, ax = plt.subplots(figsize=(15, 8))

    for result in results:
        ax.plot(result.daily_results["trade_date"], result.daily_results["net_equity_curve"], label=result.name, linewidth=1.6)

    ax.set_title("Long-term Comparison With Cost")
    ax.set_ylabel("Net Equity")
    ax.set_xlabel("Trade Date")
    ax.set_xlim(LONGTERM_START, BACKTEST_END)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(LONGTERM_PNG, dpi=150)
    plt.close(fig)


def build_monte_results_csv(results: list[BacktestResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for result in results:
        periods = result.period_stats.to_dict("records")
        first_period = periods[0] if len(periods) > 0 else {}
        second_period = periods[1] if len(periods) > 1 else {}
        row: dict[str, Any] = {
            "strategy": result.name,
            "pattern_code": result.pattern_label,
            "annual_return_no_cost": result.annual_return_gross,
            "annual_return_with_cost": result.annual_return_net,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "total_trade_count": result.total_trade_count,
            "risk_return_ratio": result.risk_return_ratio,
            "avg_allocation_ratio": result.avg_allocation_ratio,
            "first_half_label": first_period.get("period", ""),
            "first_half_net_annual_return": first_period.get("net_annual_return", 0.0),
            "second_half_label": second_period.get("period", ""),
            "second_half_net_annual_return": second_period.get("net_annual_return", 0.0),
        }

        for yearly in result.yearly_returns.itertuples(index=False):
            row[f"gross_{yearly.year}"] = yearly.gross_return
            row[f"net_{yearly.year}"] = yearly.net_return

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["annual_return_with_cost", "risk_return_ratio", "avg_allocation_ratio"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def save_monte_plot(results: list[BacktestResult]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    for result in results:
        axes[0].plot(result.daily_results["trade_date"], result.daily_results["gross_equity_curve"], label=result.name, linewidth=1.6)
        axes[1].plot(result.daily_results["trade_date"], result.daily_results["net_equity_curve"], label=result.name, linewidth=1.6)

    axes[0].set_title("2558 Allocation Comparison No Cost")
    axes[0].set_ylabel("Gross Equity")
    axes[1].set_title("2558 Allocation Comparison With Cost")
    axes[1].set_ylabel("Net Equity")
    axes[1].set_xlabel("Trade Date")

    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(MONTE_PNG, dpi=150)
    plt.close(fig)


def print_best_result(result: BacktestResult) -> None:
    print("\n=== Best Pattern ===")
    print(f"Strategy: {result.name}")
    print(f"Annual Return (No Cost): {result.annual_return_gross:.2%}")
    print(f"Annual Return (With Cost): {result.annual_return_net:.2%}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")
    print(f"Win Rate: {result.win_rate:.2%}")
    print(f"Total Trade Count: {result.total_trade_count}")
    print(f"Average Holding Days: {result.avg_holding_days:.2f}")
    print(f"Average Return / Trade: {result.avg_return_per_trade:.2%}")
    print(f"Average Cost / Trade: {result.avg_cost_per_trade:.2%}")
    print(f"Average Net / Trade: {result.avg_net_per_trade:.2%}")
    print(f"Risk Return Ratio: {result.risk_return_ratio:.2f}")
    print(f"Both 2010-2017 and 2018-2025 positive: {result.both_periods_positive}")
    print(f"Max drawdown within -30%: {result.max_drawdown_within_limit}")
    print(f"Trade count >= 10: {result.trade_count_meets_minimum}")


def print_condition_comparison_table(results: list[BacktestResult]) -> None:
    rows: list[dict[str, str]] = []

    for result in results:
        condition_code = result.name.split(" | ")[0]
        rows.append(
            {
                "condition": condition_code,
                "annual_no_cost": f"{result.annual_return_gross:.2%}",
                "annual_with_cost": f"{result.annual_return_net:.2%}",
                "max_drawdown": f"{result.max_drawdown:.2%}",
                "win_rate": f"{result.win_rate:.2%}",
                "trade_count": str(result.total_trade_count),
                "avg_hold": f"{result.avg_holding_days:.2f}",
                "net_2010_2017": f"{_period_net_return(result.period_stats, '2010-2017'):.2%}",
                "net_2018_2025": f"{_period_net_return(result.period_stats, '2018-2025'):.2%}",
            }
        )

    comparison = pd.DataFrame(rows)
    print("\n=== Condition Variations (L5xS0, Averaging On) ===")
    print(comparison.to_string(index=False))


def print_condition_tradeoff_table(results: list[BacktestResult]) -> None:
    rows: list[dict[str, str]] = []

    for result in sorted(results, key=lambda item: item.total_trade_count):
        condition_code = result.name.split(" | ")[0]
        rows.append(
            {
                "condition": condition_code,
                "trade_count": str(result.total_trade_count),
                "annual_with_cost": f"{result.annual_return_net:.2%}",
                "annual_no_cost": f"{result.annual_return_gross:.2%}",
                "max_drawdown": f"{result.max_drawdown:.2%}",
            }
        )

    tradeoff = pd.DataFrame(rows)
    print("\n=== Trade Count vs Annual Return ===")
    print(tradeoff.to_string(index=False))


def print_nanpin_comparison_table(results: list[BacktestResult]) -> None:
    rows: list[dict[str, str]] = []

    for result in results:
        rows.append(
            {
                "strategy": result.name,
                "annual_with_cost": f"{result.annual_return_net:.2%}",
                "annual_no_cost": f"{result.annual_return_gross:.2%}",
                "max_drawdown": f"{result.max_drawdown:.2%}",
                "win_rate": f"{result.win_rate:.2%}",
                "trade_count": str(result.total_trade_count),
                "avg_hold": f"{result.avg_holding_days:.2f}",
                "net_2010_2017": f"{_period_net_return(result.period_stats, '2010-2017'):.2%}",
                "net_2018_2025": f"{_period_net_return(result.period_stats, '2018-2025'):.2%}",
            }
        )

    comparison = pd.DataFrame(rows)
    print("\n=== Nanpin + Capital Management Comparison (C4, L5xS0) ===")
    print(comparison.to_string(index=False))


def print_kelly_comparison_table(results: list[BacktestResult]) -> None:
    rows: list[dict[str, str]] = []

    for result in results:
        rows.append(
            {
                "strategy": result.name,
                "annual_no_cost": f"{result.annual_return_gross:.2%}",
                "annual_with_cost": f"{result.annual_return_net:.2%}",
                "max_drawdown": f"{result.max_drawdown:.2%}",
                "win_rate": f"{result.win_rate:.2%}",
                "trade_count": str(result.total_trade_count),
                "avg_hold": f"{result.avg_holding_days:.2f}",
                "avg_return": f"{result.avg_return_per_trade:.2%}",
                "avg_cost": f"{result.avg_cost_per_trade:.2%}",
                "avg_net": f"{result.avg_net_per_trade:.2%}",
                "risk_return": f"{result.risk_return_ratio:.2f}",
                "net_2010_2017": f"{_period_net_return(result.period_stats, '2010-2017'):.2%}",
                "net_2018_2025": f"{_period_net_return(result.period_stats, '2018-2025'):.2%}",
                "ruin_prob": f"{result.ruin_probability:.2%}",
            }
        )

    comparison = pd.DataFrame(rows)
    print("\n=== Kelly Allocation Comparison (C4, L5xS0, No Averaging) ===")
    print(comparison.to_string(index=False))


def print_kelly_detail(result: BacktestResult) -> None:
    print(f"\n=== {result.name} ===")
    print(f"Annual Return (No Cost): {result.annual_return_gross:.2%}")
    print(f"Annual Return (With Cost): {result.annual_return_net:.2%}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")
    print(f"Win Rate: {result.win_rate:.2%}")
    print(f"Total Trade Count: {result.total_trade_count}")
    print(f"Average Holding Days: {result.avg_holding_days:.2f}")
    print(f"Average Return / Trade: {result.avg_return_per_trade:.2%}")
    print(f"Average Cost / Trade: {result.avg_cost_per_trade:.2%}")
    print(f"Average Net / Trade: {result.avg_net_per_trade:.2%}")
    print(f"Risk Return Ratio: {result.risk_return_ratio:.2f}")
    print(f"Ruin Probability: {result.ruin_probability:.2%}")
    print_yearly_table(result)
    print_period_table(result)


def print_ticker_comparison_table(results: list[BacktestResult]) -> None:
    rows: list[dict[str, str]] = []

    for result in results:
        rows.append(
            {
                "strategy": result.name,
                "start_date": _result_data_start_date(result),
                "annual_no_cost": f"{result.annual_return_gross:.2%}",
                "annual_with_cost": f"{result.annual_return_net:.2%}",
                "max_drawdown": f"{result.max_drawdown:.2%}",
                "win_rate": f"{result.win_rate:.2%}",
                "trade_count": str(result.total_trade_count),
                "avg_hold": f"{result.avg_holding_days:.2f}",
                "risk_return": f"{result.risk_return_ratio:.2f}",
                "net_2010_2017": f"{_period_net_return(result.period_stats, '2010-2017'):.2%}",
                "net_2018_2025": f"{_period_net_return(result.period_stats, '2018-2025'):.2%}",
            }
        )

    comparison = pd.DataFrame(rows)
    print("\n=== Ticker Comparison (C4, L5xS0, K2, No Averaging) ===")
    print(comparison.to_string(index=False))


def print_ticker_detail(result: BacktestResult) -> None:
    print(f"\n=== {result.name} ===")
    print(f"Data Start Date: {_result_data_start_date(result)}")
    print(f"Annual Return (No Cost): {result.annual_return_gross:.2%}")
    print(f"Annual Return (With Cost): {result.annual_return_net:.2%}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")
    print(f"Win Rate: {result.win_rate:.2%}")
    print(f"Total Trade Count: {result.total_trade_count}")
    print(f"Average Holding Days: {result.avg_holding_days:.2f}")
    print(f"Risk Return Ratio: {result.risk_return_ratio:.2f}")
    print_yearly_table(result)
    print_period_table(result)


def print_extended_comparison_table(results: list[BacktestResult]) -> None:
    rows: list[dict[str, str]] = []

    for result in results:
        rows.append(
            {
                "strategy": result.name,
                "data_period": _result_data_period(result),
                "annual_no_cost": f"{result.annual_return_gross:.2%}",
                "annual_with_cost": f"{result.annual_return_net:.2%}",
                "max_drawdown": f"{result.max_drawdown:.2%}",
                "win_rate": f"{result.win_rate:.2%}",
                "trade_count": str(result.total_trade_count),
                "avg_hold": f"{result.avg_holding_days:.2f}",
                "risk_return": f"{result.risk_return_ratio:.2f}",
                "net_2010_2017": f"{_period_net_return(result.period_stats, '2010-2017'):.2%}",
                "net_2018_2025": f"{_period_net_return(result.period_stats, '2018-2025'):.2%}",
            }
        )

    comparison = pd.DataFrame(rows)
    print("\n=== Extended Ticker and Switching Comparison (C4, L5xS0, K2, No Averaging) ===")
    print(comparison.to_string(index=False))


def print_extended_detail(result: BacktestResult) -> None:
    print(f"\n=== {result.name} ===")
    print(f"Data Period: {_result_data_period(result)}")
    print(f"Annual Return (No Cost): {result.annual_return_gross:.2%}")
    print(f"Annual Return (With Cost): {result.annual_return_net:.2%}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")
    print(f"Win Rate: {result.win_rate:.2%}")
    print(f"Total Trade Count: {result.total_trade_count}")
    print(f"Average Holding Days: {result.avg_holding_days:.2f}")
    print(f"Risk Return Ratio: {result.risk_return_ratio:.2f}")
    print_yearly_table(result)
    print_period_table(result)


def print_correlation_analysis(correlation_df: pd.DataFrame) -> None:
    if correlation_df.empty:
        print("\n=== Correlation Analysis ===")
        print("No correlation rows were generated.")
        return

    rows: list[dict[str, str]] = []
    for metric in correlation_df.itertuples(index=False):
        rows.append(
            {
                "metric": str(metric.metric),
                "correlation": f"{float(metric.value):.4f}",
                "observations": str(int(metric.observations)),
            }
        )

    print("\n=== Correlation Analysis ===")
    print(pd.DataFrame(rows).to_string(index=False))


def print_longterm_comparison_table(results: list[BacktestResult]) -> None:
    rows: list[dict[str, str]] = []

    for result in results:
        rows.append(
            {
                "strategy": result.name,
                "data_period": _result_data_period(result),
                "annual_no_cost": f"{result.annual_return_gross:.2%}",
                "annual_with_cost": f"{result.annual_return_net:.2%}",
                "max_drawdown": f"{result.max_drawdown:.2%}",
                "win_rate": f"{result.win_rate:.2%}",
                "win_ci_95": f"[{result.win_rate_ci_lower:.2%}, {result.win_rate_ci_upper:.2%}]",
                "trade_count": str(result.total_trade_count),
                "avg_trades_yr": f"{result.avg_trades_per_year:.2f}",
                "avg_hold": f"{result.avg_holding_days:.2f}",
                "risk_return": f"{result.risk_return_ratio:.2f}",
            }
        )

    print("\n=== Long-term Comparison (C4, L5, Fixed 20%) ===")
    print(pd.DataFrame(rows).to_string(index=False))


def print_longterm_detail(result: BacktestResult) -> None:
    print(f"\n=== {result.name} ===")
    print(f"Data Period: {_result_data_period(result)}")
    print(f"Annual Return (No Cost): {result.annual_return_gross:.2%}")
    print(f"Annual Return (With Cost): {result.annual_return_net:.2%}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")
    print(f"Win Rate: {result.win_rate:.2%}")
    print(f"Win Rate 95% CI: [{result.win_rate_ci_lower:.2%}, {result.win_rate_ci_upper:.2%}]")
    print(f"Total Trade Count: {result.total_trade_count}")
    print(f"Average Trades / Year: {result.avg_trades_per_year:.2f}")
    print(f"Average Holding Days: {result.avg_holding_days:.2f}")
    print(f"Average Return / Trade: {result.avg_return_per_trade:.2%}")
    print(f"Average Cost / Trade: {result.avg_cost_per_trade:.2%}")
    print(f"Average Net / Trade: {result.avg_net_per_trade:.2%}")
    print(f"Risk Return Ratio: {result.risk_return_ratio:.2f}")
    print_yearly_table(result)
    print_period_table(result)


def print_monte_comparison_table(results: list[BacktestResult]) -> None:
    rows: list[dict[str, str]] = []

    for result in results:
        periods = result.period_stats.to_dict("records")
        first_period = periods[0] if len(periods) > 0 else {}
        second_period = periods[1] if len(periods) > 1 else {}
        rows.append(
            {
                "strategy": result.name,
                "annual_no_cost": f"{result.annual_return_gross:.2%}",
                "annual_with_cost": f"{result.annual_return_net:.2%}",
                "max_drawdown": f"{result.max_drawdown:.2%}",
                "win_rate": f"{result.win_rate:.2%}",
                "trade_count": str(result.total_trade_count),
                "risk_return": f"{result.risk_return_ratio:.2f}",
                "avg_alloc": f"{result.avg_allocation_ratio:.2%}",
                "first_half": f"{float(first_period.get('net_annual_return', 0.0)):.2%}",
                "second_half": f"{float(second_period.get('net_annual_return', 0.0)):.2%}",
            }
        )

    print("\n=== 2558 Allocation Comparison (C4, L5xS0) ===")
    print(pd.DataFrame(rows).to_string(index=False))


def print_monte_detail(result: BacktestResult) -> None:
    print(f"\n=== {result.name} ===")
    print(f"Annual Return (No Cost): {result.annual_return_gross:.2%}")
    print(f"Annual Return (With Cost): {result.annual_return_net:.2%}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")
    print(f"Win Rate: {result.win_rate:.2%}")
    print(f"Total Trade Count: {result.total_trade_count}")
    print(f"Risk Return Ratio: {result.risk_return_ratio:.2f}")
    print(f"Average Allocation Ratio: {result.avg_allocation_ratio:.2%}")
    print_yearly_table(result)
    print_period_table(result)


def print_comparison_table(results: list[BacktestResult]) -> None:
    rows: list[dict[str, str]] = []

    for result in results:
        rows.append(
            {
                "strategy": result.name,
                "annual_no_cost": f"{result.annual_return_gross:.2%}",
                "annual_with_cost": f"{result.annual_return_net:.2%}",
                "max_drawdown": f"{result.max_drawdown:.2%}",
                "win_rate": f"{result.win_rate:.2%}",
                "trade_count": str(result.total_trade_count),
                "avg_hold": f"{result.avg_holding_days:.2f}",
                "avg_return": f"{result.avg_return_per_trade:.2%}",
                "avg_cost": f"{result.avg_cost_per_trade:.2%}",
                "avg_net": f"{result.avg_net_per_trade:.2%}",
                "risk_return": f"{result.risk_return_ratio:.2f}",
                "both_positive": "Y" if result.both_periods_positive else "N",
                "dd_ok": "Y" if result.max_drawdown_within_limit else "N",
                "trades_ok": "Y" if result.trade_count_meets_minimum else "N",
            }
        )

    comparison = pd.DataFrame(rows)
    print("\n=== All Patterns Comparison ===")
    print(comparison.to_string(index=False))


def print_yearly_table(result: BacktestResult) -> None:
    yearly = result.yearly_returns.copy()
    yearly["gross_return"] = yearly["gross_return"].map(lambda value: f"{value:.2%}")
    yearly["net_return"] = yearly["net_return"].map(lambda value: f"{value:.2%}")
    print("\nYearly Performance")
    print(yearly.to_string(index=False))


def print_period_table(result: BacktestResult) -> None:
    periods = result.period_stats.copy()
    periods["gross_annual_return"] = periods["gross_annual_return"].map(lambda value: f"{value:.2%}")
    periods["net_annual_return"] = periods["net_annual_return"].map(lambda value: f"{value:.2%}")
    periods["win_rate"] = periods["win_rate"].map(lambda value: f"{value:.2%}")
    periods["avg_holding_days"] = periods["avg_holding_days"].map(lambda value: f"{value:.2f}")
    print("\nSubperiod Performance")
    print(periods.to_string(index=False))


def print_detail(result: BacktestResult) -> None:
    print(f"\n=== {result.name} ===")
    print(f"Annual Return (No Cost): {result.annual_return_gross:.2%}")
    print(f"Annual Return (With Cost): {result.annual_return_net:.2%}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")
    print(f"Win Rate: {result.win_rate:.2%}")
    print(f"Total Trade Count: {result.total_trade_count}")
    print(f"Average Holding Days: {result.avg_holding_days:.2f}")
    print(f"Average Return / Trade: {result.avg_return_per_trade:.2%}")
    print(f"Average Cost / Trade: {result.avg_cost_per_trade:.2%}")
    print(f"Average Net / Trade: {result.avg_net_per_trade:.2%}")
    print(f"Risk Return Ratio: {result.risk_return_ratio:.2f}")
    print_yearly_table(result)
    print_period_table(result)


def main() -> None:
    print("Downloading VIX and 1570.T data...")
    market_data = download_market_data()
    open_frame = market_data["open"]
    close_frame = market_data["close"]
    trade_dates = open_frame[ETF_TICKER].dropna().index.intersection(close_frame[ETF_TICKER].dropna().index)
    signals = build_entry_signals(market_data["vix_close"], trade_dates)

    if signals.empty:
        raise ValueError("No entry signals matched the VIX conditions.")

    print("Running VIX rebound backtests...")
    print(f"Signal count: {len(signals)}")
    print(f"Round-trip cost: {ROUND_TRIP_COST:.2%}")
    print(f"Pattern count: {len(PATTERNS)} x averaging On/Off = {len(PATTERNS) * 2}")

    results: list[BacktestResult] = []
    for pattern in PATTERNS:
        for averaging in (False, True):
            results.append(run_backtest(pattern, averaging, signals, market_data))

    results = sorted(
        results,
        key=lambda result: (
            result.annual_return_net,
            int(result.both_periods_positive),
            int(result.max_drawdown_within_limit),
            int(result.trade_count_meets_minimum),
            result.max_drawdown,
            result.total_trade_count,
        ),
        reverse=True,
    )

    best_result = select_best_result(results)
    results_df = build_results_csv(results, best_result)
    results_df.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")
    save_plot(results)

    print_best_result(best_result)
    print_comparison_table(results)
    for result in results:
        print_detail(result)

    l5s0_pattern = next(pattern for pattern in PATTERNS if pattern.label == "L5xS0")
    condition_results: list[BacktestResult] = []

    print("\nRunning condition-variation comparison...")
    for condition in CONDITION_PATTERNS:
        condition_signals = build_entry_signals(
            market_data["vix_close"],
            trade_dates,
            pct_change_threshold=condition.pct_change_threshold,
            absolute_vix_threshold=condition.absolute_vix_threshold,
        )
        if condition_signals.empty:
            continue

        result = run_backtest(l5s0_pattern, True, condition_signals, market_data)
        result.name = f"{condition.code} | {condition.description} | Averaging On"
        result.trades["strategy"] = result.name
        result.daily_results["strategy"] = result.name
        condition_results.append(result)

    condition_results = sorted(condition_results, key=lambda result: result.annual_return_net, reverse=True)
    if condition_results:
        condition_df = build_condition_results_csv(condition_results)
        condition_df.to_csv(CONDITIONS_CSV, index=False, encoding="utf-8-sig")
        save_condition_plot(condition_results)
        print_condition_comparison_table(condition_results)
        print_condition_tradeoff_table(condition_results)
        print(f"\nCondition CSV saved to: {CONDITIONS_CSV.resolve()}")
        print(f"Condition plot saved to: {CONDITIONS_PNG.resolve()}")

    c4_condition = next(condition for condition in CONDITION_PATTERNS if condition.code == "C4")
    c4_signals = build_entry_signals(
        market_data["vix_close"],
        trade_dates,
        pct_change_threshold=c4_condition.pct_change_threshold,
        absolute_vix_threshold=c4_condition.absolute_vix_threshold,
    )
    if c4_signals.empty:
        raise ValueError("No entry signals matched the fixed C4 condition.")
    nanpin_results: list[BacktestResult] = []

    print("\nRunning nanpin and Kelly comparison...")
    for nanpin_pattern in NANPIN_PATTERNS:
        for capital_mode in CAPITAL_MODES:
            nanpin_results.append(run_nanpin_backtest(nanpin_pattern, capital_mode, c4_signals, market_data))

    nanpin_results = sorted(
        nanpin_results,
        key=lambda result: (
            result.annual_return_net,
            result.max_drawdown,
            result.total_trade_count,
        ),
        reverse=True,
    )
    nanpin_df = build_nanpin_results_csv(nanpin_results)
    nanpin_df.to_csv(NANPIN_CSV, index=False, encoding="utf-8-sig")
    save_nanpin_plot(nanpin_results)
    print_nanpin_comparison_table(nanpin_results)
    print(f"\nNanpin CSV saved to: {NANPIN_CSV.resolve()}")
    print(f"Nanpin plot saved to: {NANPIN_PNG.resolve()}")

    kelly_results = [run_kelly_backtest(kelly_pattern, c4_signals, market_data) for kelly_pattern in KELLY_PATTERNS]
    kelly_results = sorted(
        kelly_results,
        key=lambda result: (
            result.annual_return_net,
            result.max_drawdown,
            result.total_trade_count,
        ),
        reverse=True,
    )
    kelly_df = build_kelly_results_csv(kelly_results)
    kelly_df.to_csv(KELLY_CSV, index=False, encoding="utf-8-sig")
    save_kelly_plot(kelly_results)
    print_kelly_comparison_table(kelly_results)
    for result in kelly_results:
        print_kelly_detail(result)
    print(f"\nKelly CSV saved to: {KELLY_CSV.resolve()}")
    print(f"Kelly plot saved to: {KELLY_PNG.resolve()}")

    monte_trade_dates = _common_trade_dates_for_tickers(("2558.T",), open_frame, close_frame)
    monte_signals = build_entry_signals(
        market_data["vix_close"],
        monte_trade_dates,
        pct_change_threshold=c4_condition.pct_change_threshold,
        absolute_vix_threshold=c4_condition.absolute_vix_threshold,
    )
    if monte_signals.empty:
        raise ValueError("No entry signals matched the 2558 Monte comparison condition.")
    monte_results = [run_monte_backtest(monte_pattern, monte_signals, market_data) for monte_pattern in MONTE_PATTERNS]
    monte_results = sorted(
        monte_results,
        key=lambda result: (
            result.annual_return_net,
            result.risk_return_ratio,
            result.avg_allocation_ratio,
        ),
        reverse=True,
    )
    monte_df = build_monte_results_csv(monte_results)
    monte_df.to_csv(MONTE_CSV, index=False, encoding="utf-8-sig")
    save_monte_plot(monte_results)
    print_monte_comparison_table(monte_results)
    for result in monte_results:
        print_monte_detail(result)
    print(f"\nMonte CSV saved to: {MONTE_CSV.resolve()}")
    print(f"Monte plot saved to: {MONTE_PNG.resolve()}")

    ticker_results: list[BacktestResult] = []
    print("\nRunning ticker comparison...")
    for ticker_pattern in TICKER_COMPARISON_PATTERNS:
        ticker_trade_dates = _common_trade_dates_for_tickers(ticker_pattern.tickers, open_frame, close_frame)
        if len(ticker_trade_dates) == 0:
            continue
        ticker_signals = build_entry_signals(
            market_data["vix_close"],
            ticker_trade_dates,
            pct_change_threshold=c4_condition.pct_change_threshold,
            absolute_vix_threshold=c4_condition.absolute_vix_threshold,
        )
        if ticker_signals.empty:
            continue
        ticker_results.append(run_ticker_comparison_backtest(ticker_pattern, ticker_signals, market_data))

    ticker_results = sorted(
        ticker_results,
        key=lambda result: (
            result.annual_return_net,
            result.risk_return_ratio,
            result.total_trade_count,
        ),
        reverse=True,
    )
    if not ticker_results:
        raise ValueError("No ticker comparison results were generated.")
    ticker_df = build_ticker_results_csv(ticker_results)
    ticker_df.to_csv(TICKERS_CSV, index=False, encoding="utf-8-sig")
    save_ticker_plot(ticker_results)
    print_ticker_comparison_table(ticker_results)
    for result in ticker_results:
        print_ticker_detail(result)
    print(f"\nTicker CSV saved to: {TICKERS_CSV.resolve()}")
    print(f"Ticker plot saved to: {TICKERS_PNG.resolve()}")

    extended_results: list[BacktestResult] = []
    print("\nRunning extended ticker and switching comparison...")
    for ticker_pattern in TICKER_COMPARISON_PATTERNS:
        ticker_trade_dates = _common_trade_dates_for_tickers(ticker_pattern.tickers, open_frame, close_frame)
        if len(ticker_trade_dates) == 0:
            continue
        ticker_signals = build_entry_signals(
            market_data["vix_close"],
            ticker_trade_dates,
            pct_change_threshold=c4_condition.pct_change_threshold,
            absolute_vix_threshold=c4_condition.absolute_vix_threshold,
        )
        if ticker_signals.empty:
            continue
        extended_results.append(run_ticker_comparison_backtest(ticker_pattern, ticker_signals, market_data))

    switch_trade_dates = _common_trade_dates_for_tickers(("1570.T", "2558.T"), open_frame, close_frame)
    switch_signals = build_entry_signals(
        market_data["vix_close"],
        switch_trade_dates,
        pct_change_threshold=c4_condition.pct_change_threshold,
        absolute_vix_threshold=c4_condition.absolute_vix_threshold,
    )
    for switch_pattern in SWITCH_PATTERNS:
        if switch_signals.empty:
            continue
        extended_results.append(run_switch_backtest(switch_pattern, switch_signals, market_data))

    extended_results = sorted(
        extended_results,
        key=lambda result: (
            result.annual_return_net,
            result.risk_return_ratio,
            result.total_trade_count,
        ),
        reverse=True,
    )
    if not extended_results:
        raise ValueError("No extended comparison results were generated.")

    correlation_df = build_signal_window_correlation_analysis(switch_signals, market_data)
    extended_df = build_extended_results_csv(extended_results, correlation_df)
    extended_df.to_csv(EXTENDED_CSV, index=False, encoding="utf-8-sig")
    save_extended_plot(extended_results)
    print_extended_comparison_table(extended_results)
    print_correlation_analysis(correlation_df)
    for result in extended_results:
        print_extended_detail(result)
    print(f"\nExtended CSV saved to: {EXTENDED_CSV.resolve()}")
    print(f"Extended plot saved to: {EXTENDED_PNG.resolve()}")

    print("\nRunning long-term comparison...")
    longterm_market_data = download_longterm_market_data()
    longterm_results = [
        run_longterm_backtest(longterm_pattern, longterm_market_data, c4_condition)
        for longterm_pattern in LONGTERM_PATTERNS
    ]
    longterm_results = sorted(
        longterm_results,
        key=lambda result: (
            result.annual_return_net,
            result.risk_return_ratio,
            result.total_trade_count,
        ),
        reverse=True,
    )
    longterm_df = build_longterm_results_csv(longterm_results)
    longterm_df.to_csv(LONGTERM_CSV, index=False, encoding="utf-8-sig")
    longterm_trades_df = build_longterm_trades_csv(longterm_results)
    longterm_trades_df.to_csv(TRADES_CSV, index=False, encoding="utf-8-sig")
    save_longterm_plot(longterm_results)
    print_longterm_comparison_table(longterm_results)
    for result in longterm_results:
        print_longterm_detail(result)
    print(f"\nLong-term CSV saved to: {LONGTERM_CSV.resolve()}")
    print(f"Long-term trades saved to: {TRADES_CSV.resolve()}")
    print(f"Long-term plot saved to: {LONGTERM_PNG.resolve()}")

    print(f"\nCSV saved to: {RESULTS_CSV.resolve()}")
    print(f"Plot saved to: {RESULTS_PNG.resolve()}")


if __name__ == "__main__":
    main()
