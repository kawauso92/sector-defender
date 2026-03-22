from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

START = pd.Timestamp("2020-01-01 00:00:00", tz="UTC")
END = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
RESULTS_CSV = Path("btc_liquidation_stats.csv")
RESULTS_PNG = Path("btc_liquidation_stats.png")
STREAK_RESULTS_CSV = Path("btc_streak_stats.csv")
EXECUTION_RESULTS_CSV = Path("btc_execution_stats.csv")
EXECUTION_RESULTS_PNG = Path("btc_execution_stats.png")
H1_CACHE_CSV = Path("btc_1h_cache.csv")
M15_CACHE_CSV = Path("btc_15m_cache.csv")
USER_AGENT = {"User-Agent": "Mozilla/5.0"}
REQUEST_SLEEP_SEC = 0.5
HORIZONS = [
    ("15m", pd.Timedelta(minutes=15)),
    ("30m", pd.Timedelta(minutes=30)),
    ("1h", pd.Timedelta(hours=1)),
    ("2h", pd.Timedelta(hours=2)),
    ("4h", pd.Timedelta(hours=4)),
]
THRESHOLDS = [
    ("drop_2pct", 0.02),
    ("drop_3pct", 0.03),
    ("drop_5pct", 0.05),
]
ACTIVE_THRESHOLDS = [
    ("drop_2pct", 0.02),
    ("drop_3pct", 0.03),
]
SIDE_LABELS = {
    "long_liquidation_proxy": "Long liquidation proxy",
    "short_liquidation_proxy": "Short liquidation proxy",
}
EXECUTION_PATTERNS = [
    {"code": "E0", "name": "Market", "mode": "market", "offset": 0.0, "fee_pct": 0.10},
    {"code": "E1", "name": "Limit -0.3%", "mode": "pullback", "offset": -0.003, "fee_pct": -0.02},
    {"code": "E2", "name": "Limit -0.5%", "mode": "pullback", "offset": -0.005, "fee_pct": -0.02},
    {"code": "E3", "name": "Limit +0.3%", "mode": "breakout", "offset": 0.003, "fee_pct": -0.02},
]
TP_SL_COMBINATIONS = [
    {"label": "TP0.7% SL0.7%", "tp": 0.007, "sl": 0.007},
    {"label": "TP1.0% SL0.7%", "tp": 0.010, "sl": 0.007},
    {"label": "TP1.4% SL0.7%", "tp": 0.014, "sl": 0.007},
    {"label": "TP1.4% SL1.0%", "tp": 0.014, "sl": 0.010},
    {"label": "TP1.4% SL1.4%", "tp": 0.014, "sl": 0.014},
    {"label": "TP2.0% SL1.0%", "tp": 0.020, "sl": 0.010},
    {"label": "TP2.0% SL1.4%", "tp": 0.020, "sl": 0.014},
    {"label": "No SL, 1h timeout", "tp": None, "sl": None},
]
STOP_LOSS_BY_THRESHOLD = {
    "drop_2pct": 0.65,
    "drop_3pct": 0.70,
}
EXECUTION_THRESHOLD = "drop_3pct"
E1_OFFSET = -0.003
E1_FEE_PCT = -0.02
EXECUTION_LOOKAHEAD = pd.Timedelta(hours=2)
EXECUTION_HOLD = pd.Timedelta(hours=1)


def _binance_get(url: str, params: dict[str, Any]) -> list[Any]:
    response = requests.get(url, params=params, timeout=30, headers=USER_AGENT)
    response.raise_for_status()
    payload = response.json()
    time.sleep(REQUEST_SLEEP_SEC)
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected Binance payload: {payload}")
    return payload


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    phat = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    centre = (phat + z2 / (2.0 * total)) / denominator
    margin = z * np.sqrt(((phat * (1.0 - phat)) + z2 / (4.0 * total)) / total) / denominator
    return max(0.0, float(centre - margin)), min(1.0, float(centre + margin))


def _run_lengths(outcomes: list[bool], target: bool) -> list[int]:
    lengths: list[int] = []
    current = 0
    for outcome in outcomes:
        if outcome == target:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def fetch_klines(interval: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows: list[list[Any]] = []
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    while start_ms <= end_ms:
        payload = _binance_get(
            KLINES_URL,
            {
                "symbol": "BTCUSDT",
                "interval": interval,
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
        raise ValueError(f"No BTCUSDT {interval} data was downloaded.")

    frame = pd.DataFrame(
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
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open_time", "close_time", "close"])
    frame = frame.drop_duplicates(subset=["open_time"]).sort_values("open_time")
    return frame.reset_index(drop=True)


def load_cached_klines(cache_path: Path) -> pd.DataFrame | None:
    if not cache_path.exists():
        return None

    frame = pd.read_csv(cache_path)
    if frame.empty:
        return None

    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time"], utc=True)
    for column in ["open", "high", "low", "close", "volume"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open_time", "close_time", "close"])
    frame = frame.drop_duplicates(subset=["open_time"]).sort_values("open_time")
    return frame.reset_index(drop=True)


def fetch_or_load_klines(interval: str, cache_path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    cached = load_cached_klines(cache_path)
    if cached is not None:
        return cached

    frame = fetch_klines(interval, start, end)
    frame.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return frame


def build_proxy_events(hourly_prices: pd.DataFrame) -> pd.DataFrame:
    hourly = hourly_prices.copy()
    hourly["prev_close"] = hourly["close"].shift(1)
    hourly["hourly_change"] = (hourly["close"] / hourly["prev_close"]) - 1.0
    hourly = hourly.dropna(subset=["prev_close", "hourly_change"]).copy()

    rows: list[dict[str, Any]] = []
    for item in hourly.itertuples(index=False):
        change = float(item.hourly_change)
        for threshold_label, threshold_value in ACTIVE_THRESHOLDS:
            if change <= -threshold_value:
                rows.append(
                    {
                        "event_time": pd.Timestamp(item.close_time),
                        "side": "long_liquidation_proxy",
                        "threshold": threshold_label,
                        "threshold_pct": threshold_value * 100.0,
                        "shock_return_pct": change * 100.0,
                    }
                )
            if change >= threshold_value:
                rows.append(
                    {
                        "event_time": pd.Timestamp(item.close_time),
                        "side": "short_liquidation_proxy",
                        "threshold": threshold_label,
                        "threshold_pct": threshold_value * 100.0,
                        "shock_return_pct": change * 100.0,
                    }
                )

    events = pd.DataFrame(rows)
    if events.empty:
        raise ValueError("No proxy liquidation events were detected from hourly BTC moves.")

    events = events.sort_values(["event_time", "threshold", "side"]).reset_index(drop=True)
    return events


def compute_event_returns(events: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    close_times = pd.DatetimeIndex(prices["close_time"])
    close_values = prices["close"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []

    for event in events.itertuples(index=False):
        base_pos = close_times.searchsorted(pd.Timestamp(event.event_time), side="left")
        if base_pos >= len(close_times):
            continue

        base_price = float(close_values[base_pos])
        base_time = pd.Timestamp(close_times[base_pos])

        for horizon_label, horizon_delta in HORIZONS:
            target_time = pd.Timestamp(event.event_time) + horizon_delta
            future_pos = close_times.searchsorted(target_time, side="left")
            if future_pos >= len(close_times):
                continue

            future_price = float(close_values[future_pos])
            raw_return = (future_price / base_price) - 1.0
            signal_return = raw_return if event.side == "long_liquidation_proxy" else -raw_return
            rows.append(
                {
                    "event_time": pd.Timestamp(event.event_time),
                    "side": event.side,
                    "threshold": event.threshold,
                    "threshold_pct": float(event.threshold_pct),
                    "shock_return_pct": float(event.shock_return_pct),
                    "base_time": base_time,
                    "horizon": horizon_label,
                    "future_time": pd.Timestamp(close_times[future_pos]),
                    "return_pct": signal_return * 100.0,
                }
            )

    event_returns = pd.DataFrame(rows)
    if event_returns.empty:
        raise ValueError("No future-return observations could be computed.")
    return event_returns


def build_summary_stats(event_returns: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for threshold_label, threshold_value in ACTIVE_THRESHOLDS:
        filtered = event_returns[event_returns["threshold"] == threshold_label].copy()
        for side in ["long_liquidation_proxy", "short_liquidation_proxy"]:
            side_frame = filtered[filtered["side"] == side].copy()
            for horizon_label, _ in HORIZONS:
                horizon_frame = side_frame[side_frame["horizon"] == horizon_label]["return_pct"].dropna()
                total = int(len(horizon_frame))
                wins = int((horizon_frame > 0).sum())
                ci_low, ci_high = _wilson_interval(wins, total)
                rows.append(
                    {
                        "threshold": threshold_label,
                        "threshold_pct": threshold_value * 100.0,
                        "side": side,
                        "horizon": horizon_label,
                        "sample_count": total,
                        "mean_return_pct": float(horizon_frame.mean()) if total else 0.0,
                        "median_return_pct": float(horizon_frame.median()) if total else 0.0,
                        "std_dev_pct": float(horizon_frame.std(ddof=0)) if total else 0.0,
                        "win_rate": float(wins / total) if total else 0.0,
                        "win_rate_ci_lower": ci_low,
                        "win_rate_ci_upper": ci_high,
                        "lower_25_pct": float(horizon_frame.quantile(0.25)) if total else 0.0,
                        "upper_25_pct": float(horizon_frame.quantile(0.75)) if total else 0.0,
                        "max_profit_pct": float(horizon_frame.max()) if total else 0.0,
                        "max_loss_pct": float(horizon_frame.min()) if total else 0.0,
                    }
                )

    return pd.DataFrame(rows)


def build_streak_stats(event_returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis = event_returns[
        (event_returns["side"] == "long_liquidation_proxy") & (event_returns["horizon"] == "1h")
    ].copy()
    if analysis.empty:
        raise ValueError("No 1h long-proxy observations available for streak analysis.")

    analysis = analysis.sort_values(["threshold", "event_time"]).reset_index(drop=True)
    analysis["is_win"] = analysis["return_pct"] >= 0.0
    analysis["month"] = analysis["event_time"].dt.strftime("%Y-%m")
    analysis["year"] = analysis["event_time"].dt.year.astype(int)

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    for threshold_label, _ in ACTIVE_THRESHOLDS:
        threshold_frame = analysis[analysis["threshold"] == threshold_label].copy()
        if threshold_frame.empty:
            continue

        outcomes = threshold_frame["is_win"].tolist()
        loss_streaks = _run_lengths(outcomes, False)
        win_streaks = _run_lengths(outcomes, True)
        total_loss_streaks = len(loss_streaks)
        max_loss_streak = max(loss_streaks, default=0)
        max_win_streak = max(win_streaks, default=0)
        stop_loss = STOP_LOSS_BY_THRESHOLD.get(threshold_label, 0.0)
        if max_loss_streak > 0 and stop_loss > 0:
            max_leverage = (0.5 ** (1.0 / max_loss_streak)) / stop_loss * 100.0
        else:
            max_leverage = 0.0

        monthly_rows: list[dict[str, Any]] = []
        for month, month_frame in threshold_frame.groupby("month", sort=True):
            month_loss_streaks = _run_lengths(month_frame["is_win"].tolist(), False)
            monthly_rows.append(
                {
                    "threshold": threshold_label,
                    "section": "monthly_max_loss_streak",
                    "period": month,
                    "value": max(month_loss_streaks, default=0),
                }
            )

        yearly_rows: list[dict[str, Any]] = []
        for year, year_frame in threshold_frame.groupby("year", sort=True):
            year_loss_streaks = _run_lengths(year_frame["is_win"].tolist(), False)
            yearly_rows.append(
                {
                    "threshold": threshold_label,
                    "section": "yearly_max_loss_streak",
                    "period": str(year),
                    "value": max(year_loss_streaks, default=0),
                }
            )

        worst_month = max(monthly_rows, key=lambda row: (row["value"], row["period"])) if monthly_rows else None
        average_loss_streak = float(np.mean(loss_streaks)) if loss_streaks else 0.0
        average_win_streak = float(np.mean(win_streaks)) if win_streaks else 0.0

        summary_rows.extend(
            [
                {"threshold": threshold_label, "metric": "sample_count", "value": float(len(threshold_frame))},
                {"threshold": threshold_label, "metric": "max_loss_streak", "value": float(max_loss_streak)},
                {"threshold": threshold_label, "metric": "avg_loss_streak", "value": average_loss_streak},
                {"threshold": threshold_label, "metric": "loss_streaks_total", "value": float(total_loss_streaks)},
                {"threshold": threshold_label, "metric": "loss_streaks_ge_2", "value": float(sum(length >= 2 for length in loss_streaks))},
                {"threshold": threshold_label, "metric": "loss_streaks_ge_3", "value": float(sum(length >= 3 for length in loss_streaks))},
                {"threshold": threshold_label, "metric": "loss_streaks_ge_5", "value": float(sum(length >= 5 for length in loss_streaks))},
                {"threshold": threshold_label, "metric": "max_win_streak", "value": float(max_win_streak)},
                {"threshold": threshold_label, "metric": "avg_win_streak", "value": average_win_streak},
                {"threshold": threshold_label, "metric": "recommended_max_leverage_x", "value": max_leverage},
            ]
        )

        for streak_length in sorted(set(loss_streaks)):
            probability = loss_streaks.count(streak_length) / total_loss_streaks if total_loss_streaks else 0.0
            detail_rows.append(
                {
                    "threshold": threshold_label,
                    "section": "loss_streak_probability",
                    "period": str(streak_length),
                    "value": probability,
                }
            )

        if worst_month is not None:
            detail_rows.append(
                {
                    "threshold": threshold_label,
                    "section": "worst_month",
                    "period": worst_month["period"],
                    "value": worst_month["value"],
                }
            )

        detail_rows.extend(monthly_rows)
        detail_rows.extend(yearly_rows)

    summary = pd.DataFrame(summary_rows)
    detail = pd.DataFrame(detail_rows)
    return summary, detail


def print_streak_summary(summary: pd.DataFrame, detail: pd.DataFrame) -> None:
    print("\n=== BTC Long 1h Streak Analysis ===")
    print("Scope: thresholds drop_2pct / drop_3pct, long only, 1h holding, win if return >= 0.")

    for threshold_label, _ in ACTIVE_THRESHOLDS:
        threshold_summary = summary[summary["threshold"] == threshold_label].copy()
        if threshold_summary.empty:
            continue
        metric_map = {row.metric: row.value for row in threshold_summary.itertuples(index=False)}
        threshold_detail = detail[detail["threshold"] == threshold_label].copy()
        worst_month = threshold_detail[threshold_detail["section"] == "worst_month"]
        worst_month_text = "-"
        if not worst_month.empty:
            worst = worst_month.iloc[0]
            worst_month_text = f"{worst['period']} ({int(worst['value'])})"

        yearly = threshold_detail[threshold_detail["section"] == "yearly_max_loss_streak"].copy()
        yearly_text = ", ".join(f"{row.period}:{int(row.value)}" for row in yearly.itertuples(index=False))

        loss_probs = threshold_detail[threshold_detail["section"] == "loss_streak_probability"].copy()
        loss_prob_text = ", ".join(
            f"{row.period}連敗:{row.value:.2%}" for row in loss_probs.itertuples(index=False)
        ) or "-"

        print(f"\n[{threshold_label}]")
        print(
            "Loss streaks: "
            f"max={int(metric_map.get('max_loss_streak', 0))}, "
            f"avg={metric_map.get('avg_loss_streak', 0.0):.2f}, "
            f">=2={int(metric_map.get('loss_streaks_ge_2', 0))}, "
            f">=3={int(metric_map.get('loss_streaks_ge_3', 0))}, "
            f">=5={int(metric_map.get('loss_streaks_ge_5', 0))}"
        )
        print(
            "Win streaks: "
            f"max={int(metric_map.get('max_win_streak', 0))}, "
            f"avg={metric_map.get('avg_win_streak', 0.0):.2f}"
        )
        print(f"Loss streak probabilities: {loss_prob_text}")
        print(f"Worst month: {worst_month_text}")
        print(f"Yearly max loss streak: {yearly_text or '-'}")
        print(f"Recommended max leverage: {metric_map.get('recommended_max_leverage_x', 0.0):.2f}x")


def save_streak_stats(summary: pd.DataFrame, detail: pd.DataFrame) -> None:
    summary_export = summary.copy()
    summary_export["section"] = "summary"
    summary_export["period"] = ""
    summary_export = summary_export.rename(columns={"metric": "item"})

    detail_export = detail.copy()
    detail_export["item"] = detail_export["section"]

    output = pd.concat(
        [
            summary_export[["threshold", "section", "item", "period", "value"]],
            detail_export[["threshold", "section", "item", "period", "value"]],
        ],
        ignore_index=True,
    )
    output.to_csv(STREAK_RESULTS_CSV, index=False, encoding="utf-8-sig")


def build_execution_price_frame(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices[["open_time", "close_time", "open", "high", "low", "close"]].dropna().copy()
    return frame.sort_values("open_time").reset_index(drop=True)


def find_e1_fills(events: pd.DataFrame, price_frame: pd.DataFrame) -> pd.DataFrame:
    signal_events = events[
        (events["threshold"] == EXECUTION_THRESHOLD) & (events["side"] == "long_liquidation_proxy")
    ].copy()
    if signal_events.empty:
        raise ValueError("No drop_3pct long-proxy events available for execution analysis.")

    close_times = pd.DatetimeIndex(price_frame["close_time"])
    fills: list[dict[str, Any]] = []
    for event in signal_events.itertuples(index=False):
        signal_time = pd.Timestamp(event.event_time)
        signal_pos = close_times.searchsorted(signal_time, side="left")
        if signal_pos >= len(close_times):
            continue

        signal_price = float(price_frame.iloc[signal_pos]["close"])
        limit_price = signal_price * (1.0 + E1_OFFSET)
        window = price_frame[
            (price_frame["open_time"] >= signal_time)
            & (price_frame["open_time"] < signal_time + EXECUTION_LOOKAHEAD)
        ].copy()

        for bar in window.itertuples(index=False):
            if float(bar.low) <= limit_price:
                fills.append(
                    {
                        "signal_time": signal_time,
                        "signal_price": signal_price,
                        "entry_time": pd.Timestamp(bar.open_time),
                        "entry_price": limit_price,
                    }
                )
                break

    return pd.DataFrame(fills)


def analyze_execution_patterns(events: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    signal_events = events[
        (events["threshold"] == EXECUTION_THRESHOLD) & (events["side"] == "long_liquidation_proxy")
    ].copy()
    if signal_events.empty:
        raise ValueError("No drop_3pct long-proxy events available for execution analysis.")

    price_frame = build_execution_price_frame(prices)
    open_times = pd.DatetimeIndex(price_frame["open_time"])
    close_times = pd.DatetimeIndex(price_frame["close_time"])

    rows: list[dict[str, Any]] = []
    for pattern in EXECUTION_PATTERNS:
        signal_count = int(len(signal_events))
        filled_returns: list[float] = []
        monthly_returns: dict[str, float] = {}
        fill_count = 0

        for event in signal_events.itertuples(index=False):
            signal_time = pd.Timestamp(event.event_time)
            signal_pos = close_times.searchsorted(signal_time, side="left")
            if signal_pos >= len(close_times):
                continue

            signal_price = float(price_frame.iloc[signal_pos]["close"])
            fill_price = np.nan
            fill_time = pd.NaT

            if pattern["mode"] == "market":
                fill_pos = open_times.searchsorted(signal_time, side="left")
                if fill_pos < len(open_times):
                    fill_time = pd.Timestamp(open_times[fill_pos])
                    fill_price = float(price_frame.iloc[fill_pos]["open"])
            else:
                target_price = signal_price * (1.0 + float(pattern["offset"]))
                window = price_frame[
                    (price_frame["open_time"] >= signal_time)
                    & (price_frame["open_time"] < signal_time + EXECUTION_LOOKAHEAD)
                ].copy()
                for bar in window.itertuples(index=False):
                    if pattern["mode"] == "pullback" and float(bar.low) <= target_price:
                        fill_time = pd.Timestamp(bar.open_time)
                        fill_price = target_price
                        break
                    if pattern["mode"] == "breakout" and float(bar.high) >= target_price:
                        fill_time = pd.Timestamp(bar.open_time)
                        fill_price = target_price
                        break

            if pd.isna(fill_time) or np.isnan(fill_price):
                continue

            exit_pos = close_times.searchsorted(fill_time + EXECUTION_HOLD, side="left")
            if exit_pos >= len(close_times):
                continue

            exit_time = pd.Timestamp(close_times[exit_pos])
            exit_price = float(price_frame.iloc[exit_pos]["close"])
            raw_return = (exit_price / fill_price) - 1.0
            net_return = raw_return - (float(pattern["fee_pct"]) / 100.0)
            filled_returns.append(net_return)
            fill_count += 1

            month_key = fill_time.strftime("%Y-%m")
            monthly_returns[month_key] = monthly_returns.get(month_key, 0.0) + net_return

        skipped = signal_count - fill_count
        fill_rate = fill_count / signal_count if signal_count else 0.0
        win_rate = float(np.mean([value >= 0.0 for value in filled_returns])) if filled_returns else 0.0
        avg_net_return = float(np.mean(filled_returns)) if filled_returns else 0.0
        expectancy_per_signal = avg_net_return * fill_rate
        monthly_values = list(monthly_returns.values())
        monthly_expectancy = float(np.mean(monthly_values)) if monthly_values else 0.0
        annualized = (1.0 + monthly_expectancy) ** 12 - 1.0 if monthly_expectancy > -1.0 else -1.0

        rows.append(
            {
                "pattern": pattern["code"],
                "pattern_name": pattern["name"],
                "signal_count": signal_count,
                "fill_count": fill_count,
                "fill_rate": fill_rate,
                "win_rate": win_rate,
                "avg_net_return_pct": avg_net_return * 100.0,
                "expectancy_per_signal_pct": expectancy_per_signal * 100.0,
                "monthly_expected_return_pct": monthly_expectancy * 100.0,
                "annualized_return_pct": annualized * 100.0,
                "skipped_count": skipped,
                "fee_pct": float(pattern["fee_pct"]),
            }
        )

    return pd.DataFrame(rows)


def analyze_tp_sl_combinations(events: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    price_frame = build_execution_price_frame(prices)
    fills = find_e1_fills(events, price_frame)
    if fills.empty:
        raise ValueError("No E1 fills were generated for TP/SL analysis.")

    close_times = pd.DatetimeIndex(price_frame["close_time"])
    rows: list[dict[str, Any]] = []

    for combo in TP_SL_COMBINATIONS:
        realized_returns: list[float] = []
        tp_hits = 0

        for fill in fills.itertuples(index=False):
            entry_time = pd.Timestamp(fill.entry_time)
            entry_price = float(fill.entry_price)
            holding_bars = price_frame[
                (price_frame["open_time"] >= entry_time)
                & (price_frame["open_time"] < entry_time + EXECUTION_HOLD)
            ].copy()
            if holding_bars.empty:
                continue

            exit_price = float(holding_bars.iloc[-1]["close"])
            tp_hit = False

            if combo["tp"] is None and combo["sl"] is None:
                raw_return = (exit_price / entry_price) - 1.0
                realized_returns.append(raw_return - (E1_FEE_PCT / 100.0))
                continue

            take_profit_price = entry_price * (1.0 + float(combo["tp"]))
            stop_loss_price = entry_price * (1.0 - float(combo["sl"]))

            for bar in holding_bars.itertuples(index=False):
                bar_low = float(bar.low)
                bar_high = float(bar.high)
                if bar_low <= stop_loss_price and bar_high >= take_profit_price:
                    exit_price = stop_loss_price
                    break
                if bar_low <= stop_loss_price:
                    exit_price = stop_loss_price
                    break
                if bar_high >= take_profit_price:
                    exit_price = take_profit_price
                    tp_hit = True
                    break

            raw_return = (exit_price / entry_price) - 1.0
            realized_returns.append(raw_return - (E1_FEE_PCT / 100.0))
            if tp_hit:
                tp_hits += 1

        sample_count = len(realized_returns)
        win_rate = tp_hits / sample_count if sample_count else 0.0
        expectancy = float(np.mean(realized_returns)) if realized_returns else 0.0
        max_loss = float(np.min(realized_returns)) if realized_returns else 0.0
        rows.append(
            {
                "combo": combo["label"],
                "trade_count": sample_count,
                "tp_hit_rate": win_rate,
                "expectancy_per_trade_pct": expectancy * 100.0,
                "max_loss_pct": max_loss * 100.0,
            }
        )

    return pd.DataFrame(rows)


def print_execution_summary(execution: pd.DataFrame) -> None:
    formatted = execution.copy()
    for column in [
        "fill_rate",
        "win_rate",
    ]:
        formatted[column] = formatted[column].map(lambda value: f"{value:.2%}")
    for column in [
        "avg_net_return_pct",
        "expectancy_per_signal_pct",
        "monthly_expected_return_pct",
        "annualized_return_pct",
    ]:
        formatted[column] = formatted[column].map(lambda value: f"{value:.3f}%")
    formatted["fee_pct"] = formatted["fee_pct"].map(lambda value: f"{value:.2f}%")
    formatted = formatted[
        [
            "pattern",
            "pattern_name",
            "signal_count",
            "fill_count",
            "fill_rate",
            "win_rate",
            "avg_net_return_pct",
            "expectancy_per_signal_pct",
            "monthly_expected_return_pct",
            "annualized_return_pct",
            "skipped_count",
            "fee_pct",
        ]
    ]
    print("\n=== BTC Execution Pattern Comparison ===")
    print("Scope: drop_3pct, long only, 1h hold after fill.")
    print(formatted.to_string(index=False))


def print_tp_sl_summary(tp_sl_summary: pd.DataFrame) -> None:
    formatted = tp_sl_summary.copy()
    formatted["tp_hit_rate"] = formatted["tp_hit_rate"].map(lambda value: f"{value:.2%}")
    for column in ["expectancy_per_trade_pct", "max_loss_pct"]:
        formatted[column] = formatted[column].map(lambda value: f"{value:.3f}%")
    formatted = formatted[
        [
            "combo",
            "trade_count",
            "tp_hit_rate",
            "expectancy_per_trade_pct",
            "max_loss_pct",
        ]
    ]
    print("\n=== BTC TP/SL Width Analysis ===")
    print("Scope: drop_3pct, E1 limit -0.3%, long only, 1h hold, stop-loss priority if TP/SL hit in same bar.")
    print(formatted.to_string(index=False))


def save_execution_plot(execution: pd.DataFrame) -> None:
    ordered = execution.sort_values("pattern").copy()
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(ordered["pattern"], ordered["expectancy_per_signal_pct"], color=["#2b6cb0", "#2f855a", "#d69e2e", "#c53030"])
    axis.set_title("Execution Pattern Expectancy per Signal")
    axis.set_ylabel("Expectancy (%)")
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(EXECUTION_RESULTS_PNG, dpi=150)
    plt.close(fig)


def print_summary(summary: pd.DataFrame) -> None:
    formatted = summary.copy()
    formatted["side"] = formatted["side"].map(SIDE_LABELS)
    for column in [
        "mean_return_pct",
        "median_return_pct",
        "std_dev_pct",
        "lower_25_pct",
        "upper_25_pct",
        "max_profit_pct",
        "max_loss_pct",
    ]:
        formatted[column] = formatted[column].map(lambda value: f"{value:.3f}%")
    formatted["win_rate"] = formatted["win_rate"].map(lambda value: f"{value:.2%}")
    formatted["win_rate_ci_95"] = formatted.apply(
        lambda row: f"[{row['win_rate_ci_lower']:.2%}, {row['win_rate_ci_upper']:.2%}]",
        axis=1,
    )
    formatted = formatted[
        [
            "threshold",
            "side",
            "horizon",
            "sample_count",
            "mean_return_pct",
            "median_return_pct",
            "std_dev_pct",
            "win_rate",
            "win_rate_ci_95",
            "lower_25_pct",
            "upper_25_pct",
            "max_profit_pct",
            "max_loss_pct",
        ]
    ]
    print("\n=== BTC Shock-Reversal Distribution Summary ===")
    print("Proxy events: 1h BTC move <= -threshold for long-liquidation proxy, >= +threshold for short-liquidation proxy.")
    print(formatted.to_string(index=False))


def save_plot(event_returns: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, len(ACTIVE_THRESHOLDS), figsize=(6 * len(ACTIVE_THRESHOLDS), 10), sharey=True, squeeze=False)

    for col_idx, (threshold_label, threshold_value) in enumerate(ACTIVE_THRESHOLDS):
        filtered = event_returns[event_returns["threshold"] == threshold_label].copy()
        for row_idx, side in enumerate(["long_liquidation_proxy", "short_liquidation_proxy"]):
            axis = axes[row_idx, col_idx]
            datasets: list[pd.Series] = []
            labels: list[str] = []
            for horizon_label, _ in HORIZONS:
                sample = filtered[(filtered["side"] == side) & (filtered["horizon"] == horizon_label)]["return_pct"].dropna()
                datasets.append(sample if not sample.empty else pd.Series([np.nan]))
                labels.append(horizon_label)

            axis.boxplot(datasets, labels=labels, showfliers=False)
            axis.set_title(f"{SIDE_LABELS[side]} | {threshold_value:.0%} threshold")
            axis.set_ylabel("Return (%)")
            axis.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_PNG, dpi=150)
    plt.close(fig)


def main() -> None:
    print("Loading BTCUSDT 1h candles for proxy event detection...")
    hourly_prices = fetch_or_load_klines("1h", H1_CACHE_CSV, START, END)
    events = build_proxy_events(hourly_prices)
    print(f"Detected {len(events):,} proxy events for validation threshold(s): {', '.join(label for label, _ in ACTIVE_THRESHOLDS)}")

    print("Loading BTCUSDT 15m candles from cache or Binance...")
    prices = fetch_or_load_klines("15m", M15_CACHE_CSV, START, END)

    event_returns = compute_event_returns(events, prices)
    summary = build_summary_stats(event_returns)
    streak_summary, streak_detail = build_streak_stats(event_returns)
    execution_summary = analyze_execution_patterns(events, prices)
    tp_sl_summary = analyze_tp_sl_combinations(events, prices)

    print_summary(summary)
    print_streak_summary(streak_summary, streak_detail)
    print_execution_summary(execution_summary)
    print_tp_sl_summary(tp_sl_summary)
    summary.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")
    save_plot(event_returns)
    save_streak_stats(streak_summary, streak_detail)
    execution_summary.to_csv(EXECUTION_RESULTS_CSV, index=False, encoding="utf-8-sig")
    save_execution_plot(execution_summary)

    print(f"\nCSV saved to: {RESULTS_CSV.resolve()}")
    print(f"Streak CSV saved to: {STREAK_RESULTS_CSV.resolve()}")
    print(f"Execution CSV saved to: {EXECUTION_RESULTS_CSV.resolve()}")
    print(f"Plot saved to: {RESULTS_PNG.resolve()}")
    print(f"Execution plot saved to: {EXECUTION_RESULTS_PNG.resolve()}")
    print(f"1h cache: {H1_CACHE_CSV.resolve()}")
    print(f"15m cache: {M15_CACHE_CSV.resolve()}")


if __name__ == "__main__":
    main()
