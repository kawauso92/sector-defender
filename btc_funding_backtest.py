from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

START = pd.Timestamp("2020-01-01 00:00:00", tz="UTC")
END = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
ROUND_TRIP_COST = 0.001
FIXED_ALLOCATION_RATIO = 0.20
ALLOCATION_FLOOR = 0.05
ALLOCATION_CAP = 0.60
KELLY_WARMUP_TRADE_COUNT = 10
FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
RESULTS_CSV = Path("btc_results.csv")
RESULTS_PNG = Path("btc_results.png")
USER_AGENT = {"User-Agent": "Mozilla/5.0"}
MONTE_INITIAL_SEQUENCE = [1.0, 2.0, 3.0]


@dataclass(frozen=True)
class ExitPattern:
    code: str
    description: str
    take_profit: float | None
    stop_loss: float | None
    hold_hours: int | None


@dataclass(frozen=True)
class MoneyPattern:
    code: str
    description: str


@dataclass
class BacktestResult:
    name: str
    exit_code: str
    money_code: str
    trades: pd.DataFrame
    timeline: pd.DataFrame
    annual_return_gross: float
    annual_return_net: float
    max_drawdown: float
    win_rate: float
    win_rate_ci_lower: float
    win_rate_ci_upper: float
    total_trade_count: int
    avg_trades_per_year: float
    avg_holding_hours: float
    risk_return_ratio: float
    avg_allocation_ratio: float
    yearly_returns: pd.DataFrame
    period_stats: pd.DataFrame
    leverage_annuals: dict[int, float]
    both_periods_positive: bool


EXIT_PATTERNS = [
    ExitPattern("F1", "Hold until next funding confirmation", None, None, 8),
    ExitPattern("F2", "Take profit +2%, stop loss -1%", 0.02, 0.01, None),
    ExitPattern("F3", "Take profit +5%, stop loss -2%", 0.05, 0.02, None),
    ExitPattern("F4", "Take profit +3%, no stop loss", 0.03, None, None),
]

MONEY_PATTERNS = [
    MoneyPattern("M0", "Fixed 20%"),
    MoneyPattern("M1", "Decomposition Monte Carlo"),
    MoneyPattern("M2", "Half Kelly"),
]


def _binance_get(url: str, params: dict[str, Any]) -> list[Any]:
    response = requests.get(url, params=params, timeout=30, headers=USER_AGENT)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected response from Binance: {payload}")
    return payload


def fetch_funding_rates() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    start_ms = int(START.timestamp() * 1000)
    end_ms = int(END.timestamp() * 1000)

    while start_ms <= end_ms:
        payload = _binance_get(
            FUNDING_URL,
            {
                "symbol": "BTCUSDT",
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 1000,
            },
        )
        if not payload:
            break

        for item in payload:
            funding_time = pd.to_datetime(int(item["fundingTime"]), unit="ms", utc=True)
            if funding_time < START or funding_time > END:
                continue
            rows.append(
                {
                    "funding_time": funding_time,
                    "funding_rate": float(item["fundingRate"]),
                }
            )

        last_time = int(payload[-1]["fundingTime"])
        if last_time >= end_ms:
            break
        start_ms = last_time + 1

    funding = pd.DataFrame(rows).drop_duplicates(subset=["funding_time"]).sort_values("funding_time")
    if funding.empty:
        raise ValueError("No funding-rate data was downloaded.")
    return funding.reset_index(drop=True)


def fetch_hourly_klines() -> pd.DataFrame:
    rows: list[list[Any]] = []
    start_ms = int(START.timestamp() * 1000)
    end_ms = int(END.timestamp() * 1000)

    while start_ms <= end_ms:
        payload = _binance_get(
            KLINES_URL,
            {
                "symbol": "BTCUSDT",
                "interval": "1h",
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 1500,
            },
        )
        if not payload:
            break

        rows.extend(payload)
        last_open_time = int(payload[-1][0])
        if last_open_time >= end_ms:
            break
        start_ms = last_open_time + 1

    if not rows:
        raise ValueError("No BTC hourly price data was downloaded.")

    hourly = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )
    hourly["open_time"] = pd.to_datetime(hourly["open_time"], unit="ms", utc=True)
    hourly["close_time"] = pd.to_datetime(hourly["close_time"], unit="ms", utc=True)
    for column in ["open", "high", "low", "close", "volume"]:
        hourly[column] = pd.to_numeric(hourly[column], errors="coerce")
    hourly = hourly[["open_time", "close_time", "open", "high", "low", "close", "volume"]].copy()
    hourly = hourly.dropna().drop_duplicates(subset=["open_time"]).sort_values("open_time")
    return hourly.reset_index(drop=True)


def build_8h_bars(hourly: pd.DataFrame) -> pd.DataFrame:
    frame = hourly.set_index("open_time")[["open", "high", "low", "close", "volume", "close_time"]].copy()
    bars = frame.resample("8h", label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "close_time": "last",
        }
    )
    bars = bars.dropna().reset_index().rename(columns={"open_time": "bar_time"})
    return bars


def build_signals(funding: pd.DataFrame, bars_8h: pd.DataFrame) -> pd.DataFrame:
    bar_times = pd.DatetimeIndex(bars_8h["bar_time"])
    rows: list[dict[str, Any]] = []

    for item in funding.itertuples(index=False):
        funding_time = pd.Timestamp(item.funding_time)
        if item.funding_rate >= 0.001:
            side = "long"
        elif item.funding_rate <= -0.0005:
            side = "short"
        else:
            continue

        entry_pos = bar_times.searchsorted(funding_time, side="right")
        if entry_pos >= len(bar_times):
            continue

        rows.append(
            {
                "funding_time": funding_time,
                "funding_rate": float(item.funding_rate),
                "side": side,
                "entry_time": pd.Timestamp(bar_times[entry_pos]),
            }
        )

    signals = pd.DataFrame(rows)
    if signals.empty:
        raise ValueError("No funding signals matched the configured thresholds.")
    signals = signals.sort_values(["entry_time", "funding_time"]).drop_duplicates(subset=["entry_time"], keep="last")
    return signals.reset_index(drop=True)


def print_signal_summary(funding: pd.DataFrame, signals: pd.DataFrame) -> None:
    funding_counts = funding.assign(year=funding["funding_time"].dt.year).groupby("year").size().reindex(range(2020, 2026), fill_value=0)
    signal_counts = (
        signals.assign(year=signals["entry_time"].dt.year)
        .groupby(["year", "side"])
        .size()
        .unstack(fill_value=0)
        .reindex(range(2020, 2026), fill_value=0)
    )
    if "long" not in signal_counts.columns:
        signal_counts["long"] = 0
    if "short" not in signal_counts.columns:
        signal_counts["short"] = 0
    signal_counts = signal_counts[["long", "short"]]

    summary = pd.DataFrame(
        {
            "funding_obs": funding_counts,
            "long_signals": signal_counts["long"],
            "short_signals": signal_counts["short"],
            "total_signals": signal_counts["long"] + signal_counts["short"],
        }
    ).reset_index(names="year")

    print("\n=== Funding And Signal Counts By Year ===")
    print(summary.to_string(index=False))

    recent_zero = summary[(summary["year"] >= 2023) & (summary["total_signals"] == 0)]
    if not recent_zero.empty:
        data_start = funding["funding_time"].min()
        data_end = funding["funding_time"].max()
        print(
            "\n2023-2025 signal count is zero while funding data exists for the full period: "
            f"{data_start} to {data_end}"
        )


def _annualized_return(return_series: pd.Series, timestamps: pd.Series) -> float:
    if return_series.empty:
        return 0.0
    equity_curve = (1.0 + return_series).cumprod()
    span_seconds = max((pd.Timestamp(timestamps.iloc[-1]) - pd.Timestamp(timestamps.iloc[0])).total_seconds(), 3600.0)
    years = span_seconds / (365.25 * 24 * 3600)
    return float(equity_curve.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0


def _build_yearly_returns(timeline: pd.DataFrame) -> pd.DataFrame:
    grouped = timeline.groupby(timeline["timestamp"].dt.year)
    gross = grouped["gross_return"].apply(lambda series: (1.0 + series).prod() - 1.0).reindex(range(2020, 2026), fill_value=0.0)
    net = grouped["net_return"].apply(lambda series: (1.0 + series).prod() - 1.0).reindex(range(2020, 2026), fill_value=0.0)
    return pd.DataFrame({"year": gross.index, "gross_return": gross.values, "net_return": net.values})


def _build_period_stats(timeline: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    periods = [
        ("2020-2022", pd.Timestamp("2020-01-01 00:00:00", tz="UTC"), pd.Timestamp("2022-12-31 23:59:59", tz="UTC")),
        ("2023-2025", pd.Timestamp("2023-01-01 00:00:00", tz="UTC"), pd.Timestamp("2025-12-31 23:59:59", tz="UTC")),
    ]
    rows: list[dict[str, Any]] = []

    for label, start_time, end_time in periods:
        period_timeline = timeline[(timeline["timestamp"] >= start_time) & (timeline["timestamp"] <= end_time)].copy()
        period_trades = trades[(trades["exit_time"] >= start_time) & (trades["exit_time"] <= end_time)].copy()
        rows.append(
            {
                "period": label,
                "gross_annual_return": _annualized_return(period_timeline["gross_return"], period_timeline["timestamp"]) if not period_timeline.empty else 0.0,
                "net_annual_return": _annualized_return(period_timeline["net_return"], period_timeline["timestamp"]) if not period_timeline.empty else 0.0,
                "win_rate": float((period_trades["net_return"] > 0).mean()) if not period_trades.empty else 0.0,
                "trade_count": int(len(period_trades)),
                "avg_holding_hours": float(period_trades["holding_hours"].mean()) if not period_trades.empty else 0.0,
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


def _compute_half_kelly_fraction(history_returns: list[float]) -> float:
    if len(history_returns) < KELLY_WARMUP_TRADE_COUNT:
        return FIXED_ALLOCATION_RATIO

    wins = [value for value in history_returns if value > 0]
    losses = [abs(value) for value in history_returns if value < 0]
    if not wins:
        return ALLOCATION_FLOOR
    if not losses:
        return ALLOCATION_CAP

    win_rate = len(wins) / len(history_returns)
    loss_rate = 1.0 - win_rate
    payoff_ratio = np.mean(wins) / np.mean(losses) if np.mean(losses) > 0 else np.inf
    if not np.isfinite(payoff_ratio) or payoff_ratio <= 0:
        return FIXED_ALLOCATION_RATIO

    full_kelly = win_rate - (loss_rate / payoff_ratio)
    half_kelly = full_kelly / 2.0
    return float(min(ALLOCATION_CAP, max(ALLOCATION_FLOOR, half_kelly)))


def _resolve_money_fraction(
    money_pattern: MoneyPattern,
    history_returns: list[float],
    monte_sequence: list[float],
) -> tuple[float, float | None]:
    if money_pattern.code == "M0":
        return FIXED_ALLOCATION_RATIO, None
    if money_pattern.code == "M2":
        return _compute_half_kelly_fraction(history_returns), None

    active_sequence = monte_sequence[:] if monte_sequence else MONTE_INITIAL_SEQUENCE[:]
    if len(active_sequence) == 1:
        wager_unit = float(active_sequence[0])
    else:
        wager_unit = float(active_sequence[0] + active_sequence[-1])
    raw_fraction = wager_unit / 10.0
    return float(min(ALLOCATION_CAP, max(ALLOCATION_FLOOR, raw_fraction))), wager_unit


def _update_monte_sequence(sequence: list[float], won_trade: bool, wager_unit: float | None) -> list[float]:
    active_sequence = sequence[:] if sequence else MONTE_INITIAL_SEQUENCE[:]
    if wager_unit is None:
        return active_sequence

    if won_trade:
        if len(active_sequence) == 1:
            active_sequence = []
        else:
            active_sequence = active_sequence[1:-1]
    else:
        active_sequence.append(float(wager_unit))

    return active_sequence if active_sequence else MONTE_INITIAL_SEQUENCE[:]


def _resolve_take_profit_exit(side: str, bar_open: float, bar_high: float, bar_low: float, take_profit_level: float) -> float:
    if side == "long":
        if bar_open >= take_profit_level:
            return bar_open
        return take_profit_level if bar_high >= take_profit_level else np.nan
    if bar_open <= take_profit_level:
        return bar_open
    return take_profit_level if bar_low <= take_profit_level else np.nan


def _resolve_stop_loss_exit(side: str, bar_open: float, bar_high: float, bar_low: float, stop_loss_level: float) -> float:
    if side == "long":
        if bar_open <= stop_loss_level:
            return bar_open
        return stop_loss_level if bar_low <= stop_loss_level else np.nan
    if bar_open >= stop_loss_level:
        return bar_open
    return stop_loss_level if bar_high >= stop_loss_level else np.nan


def simulate_trade(
    signal: pd.Series,
    exit_pattern: ExitPattern,
    hourly: pd.DataFrame,
    bars_8h: pd.DataFrame,
) -> dict[str, Any] | None:
    entry_time = pd.Timestamp(signal["entry_time"])
    side = str(signal["side"])
    direction = 1.0 if side == "long" else -1.0

    entry_bar = bars_8h.loc[bars_8h["bar_time"] == entry_time]
    if entry_bar.empty:
        return None
    entry_price = float(entry_bar.iloc[0]["open"])
    if entry_price <= 0:
        return None

    if exit_pattern.code == "F1":
        exit_time = pd.Timestamp(entry_bar.iloc[0]["close_time"])
        exit_price = float(entry_bar.iloc[0]["close"])
        raw_return = direction * ((exit_price / entry_price) - 1.0)
        return {
            "funding_time": pd.Timestamp(signal["funding_time"]),
            "entry_time": entry_time,
            "exit_time": exit_time,
            "side": side,
            "funding_rate": float(signal["funding_rate"]),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "gross_return": float(raw_return),
            "net_return": float(raw_return - ROUND_TRIP_COST),
            "holding_hours": float((exit_time - entry_time).total_seconds() / 3600.0),
            "exit_reason": "next_funding",
        }

    max_exit_time = entry_time + pd.Timedelta(days=30) if exit_pattern.code == "F4" else None
    hourly_slice = hourly[hourly["open_time"] >= entry_time].copy()
    if max_exit_time is not None:
        hourly_slice = hourly_slice[hourly_slice["close_time"] <= max_exit_time].copy()
    take_profit_level = None if exit_pattern.take_profit is None else entry_price * (1.0 + (exit_pattern.take_profit * direction))
    stop_loss_level = None if exit_pattern.stop_loss is None else entry_price * (1.0 - (exit_pattern.stop_loss * direction))

    for bar in hourly_slice.itertuples(index=False):
        take_profit_price = np.nan
        stop_loss_price = np.nan

        if take_profit_level is not None:
            take_profit_price = _resolve_take_profit_exit(side, float(bar.open), float(bar.high), float(bar.low), float(take_profit_level))
        if stop_loss_level is not None:
            stop_loss_price = _resolve_stop_loss_exit(side, float(bar.open), float(bar.high), float(bar.low), float(stop_loss_level))

        if not np.isnan(stop_loss_price) and not np.isnan(take_profit_price):
            exit_price = float(stop_loss_price)
            exit_reason = "stop_loss_priority"
        elif not np.isnan(stop_loss_price):
            exit_price = float(stop_loss_price)
            exit_reason = "stop_loss"
        elif not np.isnan(take_profit_price):
            exit_price = float(take_profit_price)
            exit_reason = "take_profit"
        else:
            continue

        exit_time = pd.Timestamp(bar.close_time)
        raw_return = direction * ((exit_price / entry_price) - 1.0)
        return {
            "funding_time": pd.Timestamp(signal["funding_time"]),
            "entry_time": entry_time,
            "exit_time": exit_time,
            "side": side,
            "funding_rate": float(signal["funding_rate"]),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "gross_return": float(raw_return),
            "net_return": float(raw_return - ROUND_TRIP_COST),
            "holding_hours": float((exit_time - entry_time).total_seconds() / 3600.0),
            "exit_reason": exit_reason,
        }

    last_bar = hourly_slice.iloc[-1]
    exit_time = pd.Timestamp(last_bar["close_time"])
    exit_price = float(last_bar["close"])
    raw_return = direction * ((exit_price / entry_price) - 1.0)
    return {
        "funding_time": pd.Timestamp(signal["funding_time"]),
        "entry_time": entry_time,
        "exit_time": exit_time,
        "side": side,
        "funding_rate": float(signal["funding_rate"]),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "gross_return": float(raw_return),
        "net_return": float(raw_return - ROUND_TRIP_COST),
        "holding_hours": float((exit_time - entry_time).total_seconds() / 3600.0),
        "exit_reason": "max_hold_30d" if exit_pattern.code == "F4" else "forced_end_of_sample",
    }


def _apply_mark_to_market_timeline(
    timeline: pd.DataFrame,
    hourly: pd.DataFrame,
    trades: pd.DataFrame,
) -> pd.DataFrame:
    updated = timeline.copy()
    gross_equity = 1.0
    net_equity = 1.0

    for trade in trades.sort_values("entry_time").itertuples(index=False):
        trade_hourly = hourly[
            (hourly["open_time"] >= pd.Timestamp(trade.entry_time))
            & (hourly["close_time"] <= pd.Timestamp(trade.exit_time))
        ].copy()
        if trade_hourly.empty:
            continue

        direction = 1.0 if str(trade.side) == "long" else -1.0
        entry_price = float(trade.entry_price)
        exposure_multiplier = float(trade.allocation_ratio) / FIXED_ALLOCATION_RATIO
        start_gross_equity = gross_equity
        start_net_equity = net_equity
        prev_gross_equity = gross_equity
        prev_net_equity = net_equity

        for row in trade_hourly.itertuples(index=False):
            is_final = pd.Timestamp(row.close_time) == pd.Timestamp(trade.exit_time)
            mark_price = float(trade.exit_price) if is_final else float(row.close)
            raw_trade_return = direction * ((mark_price / entry_price) - 1.0)
            gross_equity = start_gross_equity * (1.0 + raw_trade_return * exposure_multiplier)
            net_trade_return = raw_trade_return - (ROUND_TRIP_COST if is_final else 0.0)
            net_equity = start_net_equity * (1.0 + net_trade_return * exposure_multiplier)

            gross_increment = gross_equity / prev_gross_equity - 1.0
            net_increment = net_equity / prev_net_equity - 1.0

            mask = updated["timestamp"] == pd.Timestamp(row.close_time)
            updated.loc[mask, "gross_return"] += float(gross_increment)
            updated.loc[mask, "net_return"] += float(net_increment)
            if is_final:
                updated.loc[mask, "trade_count"] += 1

            prev_gross_equity = gross_equity
            prev_net_equity = net_equity

    updated["gross_equity_curve"] = (1.0 + updated["gross_return"]).cumprod()
    updated["net_equity_curve"] = (1.0 + updated["net_return"]).cumprod()
    updated["drawdown"] = updated["net_equity_curve"] / updated["net_equity_curve"].cummax() - 1.0
    return updated


def run_backtest(
    exit_pattern: ExitPattern,
    money_pattern: MoneyPattern,
    signals: pd.DataFrame,
    hourly: pd.DataFrame,
    bars_8h: pd.DataFrame,
) -> BacktestResult:
    history_returns: list[float] = []
    monte_sequence = MONTE_INITIAL_SEQUENCE[:]
    last_exit_time: pd.Timestamp | None = None
    trade_rows: list[dict[str, Any]] = []

    timeline = pd.DataFrame({"timestamp": pd.DatetimeIndex(hourly["close_time"])})
    timeline["gross_return"] = 0.0
    timeline["net_return"] = 0.0
    timeline["trade_count"] = 0

    for signal in signals.itertuples(index=False):
        signal_series = pd.Series(signal._asdict())
        entry_time = pd.Timestamp(signal_series["entry_time"])
        if last_exit_time is not None and entry_time <= last_exit_time:
            continue

        trade = simulate_trade(signal_series, exit_pattern, hourly, bars_8h)
        if trade is None:
            continue

        allocation_ratio, monte_wager_unit = _resolve_money_fraction(money_pattern, history_returns, monte_sequence)
        exposure_multiplier = allocation_ratio / FIXED_ALLOCATION_RATIO
        gross_portfolio_return = float(trade["gross_return"]) * exposure_multiplier
        net_portfolio_return = float(trade["net_return"]) * exposure_multiplier
        won_trade = float(trade["net_return"]) > 0

        if money_pattern.code == "M1":
            monte_before = ",".join(str(int(value) if float(value).is_integer() else value) for value in monte_sequence)
            monte_sequence = _update_monte_sequence(monte_sequence, won_trade, monte_wager_unit)
            monte_after = ",".join(str(int(value) if float(value).is_integer() else value) for value in monte_sequence)
        else:
            monte_before = ""
            monte_after = ""

        trade["allocation_ratio"] = float(allocation_ratio)
        trade["gross_portfolio_return"] = float(gross_portfolio_return)
        trade["net_portfolio_return"] = float(net_portfolio_return)
        trade["cost_rate"] = float(ROUND_TRIP_COST * exposure_multiplier)
        trade["monte_sequence_before"] = monte_before
        trade["monte_sequence_after"] = monte_after
        trade_rows.append(trade)

        history_returns.append(float(trade["net_return"]))
        last_exit_time = pd.Timestamp(trade["exit_time"])

    trades = pd.DataFrame(trade_rows)
    if trades.empty:
        raise ValueError(f"No trades were generated for {exit_pattern.code} x {money_pattern.code}.")

    timeline = _apply_mark_to_market_timeline(timeline, hourly, trades)

    annual_return_gross = _annualized_return(timeline["gross_return"], timeline["timestamp"])
    annual_return_net = _annualized_return(timeline["net_return"], timeline["timestamp"])
    max_drawdown = float(timeline["drawdown"].min()) if not timeline.empty else 0.0
    win_count = int((trades["net_return"] > 0).sum())
    total_trade_count = int(len(trades))
    win_rate = win_count / total_trade_count if total_trade_count > 0 else 0.0
    win_rate_ci_lower, win_rate_ci_upper = _wilson_interval(win_count, total_trade_count)
    avg_holding_hours = float(trades["holding_hours"].mean()) if not trades.empty else 0.0
    avg_trades_per_year = total_trade_count / 6.0
    avg_allocation_ratio = float(trades["allocation_ratio"].mean()) if not trades.empty else 0.0
    risk_return_ratio = annual_return_net / abs(max_drawdown) if max_drawdown < 0 else 0.0
    yearly_returns = _build_yearly_returns(timeline)
    period_stats = _build_period_stats(timeline, trades)
    both_periods_positive = bool((period_stats["net_annual_return"] > 0).all())

    leverage_annuals: dict[int, float] = {}
    for leverage in (5, 10, 20):
        leveraged_net = (timeline["net_return"] * leverage).clip(lower=-0.999999)
        leverage_annuals[leverage] = _annualized_return(leveraged_net, timeline["timestamp"])

    strategy_name = f"{exit_pattern.code} x {money_pattern.code}"
    trades.insert(0, "strategy", strategy_name)
    trades.insert(1, "exit_pattern", exit_pattern.code)
    trades.insert(2, "money_pattern", money_pattern.code)
    timeline.insert(0, "strategy", strategy_name)

    return BacktestResult(
        name=strategy_name,
        exit_code=exit_pattern.code,
        money_code=money_pattern.code,
        trades=trades,
        timeline=timeline,
        annual_return_gross=annual_return_gross,
        annual_return_net=annual_return_net,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        win_rate_ci_lower=win_rate_ci_lower,
        win_rate_ci_upper=win_rate_ci_upper,
        total_trade_count=total_trade_count,
        avg_trades_per_year=avg_trades_per_year,
        avg_holding_hours=avg_holding_hours,
        risk_return_ratio=risk_return_ratio,
        avg_allocation_ratio=avg_allocation_ratio,
        yearly_returns=yearly_returns,
        period_stats=period_stats,
        leverage_annuals=leverage_annuals,
        both_periods_positive=both_periods_positive,
    )


def select_best_result(results: list[BacktestResult]) -> BacktestResult:
    positive_both = [result for result in results if result.both_periods_positive]
    ranked = positive_both if positive_both else results
    return sorted(
        ranked,
        key=lambda item: (item.annual_return_net, item.risk_return_ratio, item.total_trade_count),
        reverse=True,
    )[0]


def build_results_csv(results: list[BacktestResult], best_result: BacktestResult) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for result in results:
        row: dict[str, Any] = {
            "strategy": result.name,
            "exit_pattern": result.exit_code,
            "money_pattern": result.money_code,
            "is_best": result.name == best_result.name,
            "annual_return_no_cost": result.annual_return_gross,
            "annual_return_with_cost": result.annual_return_net,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "win_rate_ci_lower": result.win_rate_ci_lower,
            "win_rate_ci_upper": result.win_rate_ci_upper,
            "total_trade_count": result.total_trade_count,
            "avg_trades_per_year": result.avg_trades_per_year,
            "avg_holding_hours": result.avg_holding_hours,
            "risk_return_ratio": result.risk_return_ratio,
            "avg_allocation_ratio": result.avg_allocation_ratio,
            "lev5_annual_return": result.leverage_annuals[5],
            "lev10_annual_return": result.leverage_annuals[10],
            "lev20_annual_return": result.leverage_annuals[20],
        }

        for yearly in result.yearly_returns.itertuples(index=False):
            row[f"gross_{yearly.year}"] = yearly.gross_return
            row[f"net_{yearly.year}"] = yearly.net_return

        for period in result.period_stats.itertuples(index=False):
            period_key = str(period.period).replace("-", "_")
            row[f"net_{period_key}"] = period.net_annual_return
            row[f"win_{period_key}"] = period.win_rate
            row[f"trades_{period_key}"] = period.trade_count
            row[f"avg_hold_{period_key}"] = period.avg_holding_hours

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["annual_return_with_cost", "risk_return_ratio", "avg_allocation_ratio"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def save_plot(results: list[BacktestResult]) -> None:
    top_results = sorted(
        results,
        key=lambda item: (item.annual_return_net, item.risk_return_ratio, item.total_trade_count),
        reverse=True,
    )[:3]
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    for result in top_results:
        axes[0].plot(result.timeline["timestamp"], result.timeline["gross_equity_curve"], label=result.name, linewidth=1.6)
        axes[1].plot(result.timeline["timestamp"], result.timeline["net_equity_curve"], label=result.name, linewidth=1.6)

    axes[0].set_title("BTC Funding Strategy No Cost (Top 3)")
    axes[0].set_ylabel("Gross Equity")
    axes[1].set_title("BTC Funding Strategy With Cost (Top 3)")
    axes[1].set_ylabel("Net Equity")
    axes[1].set_xlabel("Timestamp (UTC)")

    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(RESULTS_PNG, dpi=150)
    plt.close(fig)


def print_best_result(result: BacktestResult) -> None:
    print("\n=== Best Pattern ===")
    print(f"Strategy: {result.name}")
    print(f"Annual Return (No Cost): {result.annual_return_gross:.2%}")
    print(f"Annual Return (With Cost): {result.annual_return_net:.2%}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")
    print(f"Win Rate: {result.win_rate:.2%}")
    print(f"Win Rate 95% CI: [{result.win_rate_ci_lower:.2%}, {result.win_rate_ci_upper:.2%}]")
    print(f"Total Trade Count: {result.total_trade_count}")
    print(f"Average Trades / Year: {result.avg_trades_per_year:.2f}")
    print(f"Average Holding Hours: {result.avg_holding_hours:.2f}")
    print(f"Risk Return Ratio: {result.risk_return_ratio:.2f}")
    print(f"Average Allocation Ratio: {result.avg_allocation_ratio:.2%}")
    print(f"Leverage 5x Annual Return: {result.leverage_annuals[5]:.2%}")
    print(f"Leverage 10x Annual Return: {result.leverage_annuals[10]:.2%}")
    print(f"Leverage 20x Annual Return: {result.leverage_annuals[20]:.2%}")


def print_comparison_table(results: list[BacktestResult]) -> None:
    rows: list[dict[str, str]] = []

    for result in results:
        first_half = float(result.period_stats.loc[result.period_stats["period"] == "2020-2022", "net_annual_return"].iloc[0])
        second_half = float(result.period_stats.loc[result.period_stats["period"] == "2023-2025", "net_annual_return"].iloc[0])
        rows.append(
            {
                "strategy": result.name,
                "annual_no_cost": f"{result.annual_return_gross:.2%}",
                "annual_with_cost": f"{result.annual_return_net:.2%}",
                "max_drawdown": f"{result.max_drawdown:.2%}",
                "win_rate": f"{result.win_rate:.2%}",
                "win_ci_95": f"[{result.win_rate_ci_lower:.2%}, {result.win_rate_ci_upper:.2%}]",
                "trade_count": str(result.total_trade_count),
                "avg_trades_yr": f"{result.avg_trades_per_year:.2f}",
                "avg_hold_hr": f"{result.avg_holding_hours:.2f}",
                "risk_return": f"{result.risk_return_ratio:.2f}",
                "avg_alloc": f"{result.avg_allocation_ratio:.2%}",
                "net_2020_2022": f"{first_half:.2%}",
                "net_2023_2025": f"{second_half:.2%}",
                "lev5": f"{result.leverage_annuals[5]:.2%}",
                "lev10": f"{result.leverage_annuals[10]:.2%}",
                "lev20": f"{result.leverage_annuals[20]:.2%}",
            }
        )

    print("\n=== BTC Funding Strategy Comparison ===")
    print(pd.DataFrame(rows).to_string(index=False))


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
    periods["avg_holding_hours"] = periods["avg_holding_hours"].map(lambda value: f"{value:.2f}")
    print("\nSubperiod Performance")
    print(periods.to_string(index=False))


def print_detail(result: BacktestResult) -> None:
    print(f"\n=== {result.name} ===")
    print(f"Annual Return (No Cost): {result.annual_return_gross:.2%}")
    print(f"Annual Return (With Cost): {result.annual_return_net:.2%}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")
    print(f"Win Rate: {result.win_rate:.2%}")
    print(f"Win Rate 95% CI: [{result.win_rate_ci_lower:.2%}, {result.win_rate_ci_upper:.2%}]")
    print(f"Total Trade Count: {result.total_trade_count}")
    print(f"Average Trades / Year: {result.avg_trades_per_year:.2f}")
    print(f"Average Holding Hours: {result.avg_holding_hours:.2f}")
    print(f"Risk Return Ratio: {result.risk_return_ratio:.2f}")
    print(f"Average Allocation Ratio: {result.avg_allocation_ratio:.2%}")
    print(f"Leverage 5x Annual Return: {result.leverage_annuals[5]:.2%}")
    print(f"Leverage 10x Annual Return: {result.leverage_annuals[10]:.2%}")
    print(f"Leverage 20x Annual Return: {result.leverage_annuals[20]:.2%}")
    print_yearly_table(result)
    print_period_table(result)


def main() -> None:
    print("Downloading Binance funding-rate history...")
    funding = fetch_funding_rates()
    print("Downloading Binance hourly BTC price history...")
    hourly = fetch_hourly_klines()
    bars_8h = build_8h_bars(hourly)
    signals = build_signals(funding, bars_8h)
    print_signal_summary(funding, signals)

    print("Running BTC funding backtests...")
    print(f"Funding observations: {len(funding)}")
    print(f"Signal count: {len(signals)}")
    print(f"Round-trip cost: {ROUND_TRIP_COST:.2%}")
    print(f"Pattern count: {len(EXIT_PATTERNS)} x {len(MONEY_PATTERNS)} = {len(EXIT_PATTERNS) * len(MONEY_PATTERNS)}")

    results: list[BacktestResult] = []
    for exit_pattern in EXIT_PATTERNS:
        for money_pattern in MONEY_PATTERNS:
            results.append(run_backtest(exit_pattern, money_pattern, signals, hourly, bars_8h))

    results = sorted(
        results,
        key=lambda item: (item.both_periods_positive, item.annual_return_net, item.risk_return_ratio, item.total_trade_count),
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

    print(f"\nCSV saved to: {RESULTS_CSV.resolve()}")
    print(f"Plot saved to: {RESULTS_PNG.resolve()}")


if __name__ == "__main__":
    main()
