from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

H1_CACHE_CSV = Path("btc_1h_cache.csv")
M15_CACHE_CSV = Path("btc_15m_cache.csv")
RESULTS_CSV = Path("btc_drop_results.csv")
RESULTS_PNG = Path("btc_drop_results.png")

START = pd.Timestamp("2020-01-01 00:00:00", tz="UTC")
END = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")

SIGNAL_DROP = -0.03
ENTRY_OFFSET = -0.003
ENTRY_WINDOW = pd.Timedelta(hours=2)
TAKE_PROFIT = 0.014
STOP_LOSS = -0.014
MAX_HOLD = pd.Timedelta(hours=1)
ROUND_TRIP_FEE = -0.0002

LEVERAGES = [1, 5, 7, 10]
MONEY_MANAGEMENT = ["M0", "M1", "M2"]


def load_cache(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time"], utc=True)
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open_time", "close_time", "open", "high", "low", "close"])
    frame = frame.sort_values("open_time").drop_duplicates(subset=["open_time"])
    return frame.reset_index(drop=True)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    phat = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    centre = (phat + z2 / (2.0 * total)) / denominator
    margin = z * np.sqrt(((phat * (1.0 - phat)) + z2 / (4.0 * total)) / total) / denominator
    return max(0.0, float(centre - margin)), min(1.0, float(centre + margin))


def annualized_return(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    start = pd.Timestamp(equity.index[0])
    end = pd.Timestamp(equity.index[-1])
    days = max((end - start).total_seconds() / 86400.0, 1.0)
    final_value = float(equity.iloc[-1])
    if final_value <= 0:
        return -1.0
    return final_value ** (365.25 / days) - 1.0


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def period_return(equity: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float:
    window = equity[(equity.index >= start) & (equity.index <= end)]
    if window.empty:
        return 0.0
    start_value = float(window.iloc[0])
    end_value = float(window.iloc[-1])
    if start_value == 0:
        return 0.0
    return end_value / start_value - 1.0


def build_signals(hourly: pd.DataFrame) -> pd.DataFrame:
    frame = hourly.copy()
    frame["prev_close"] = frame["close"].shift(1)
    frame["hourly_return"] = frame["close"] / frame["prev_close"] - 1.0
    frame = frame.dropna(subset=["prev_close", "hourly_return"])
    frame = frame[frame["hourly_return"] <= SIGNAL_DROP].copy()
    frame["signal_time"] = frame["close_time"]
    frame["signal_price"] = frame["close"]
    return frame[["signal_time", "signal_price", "hourly_return"]].reset_index(drop=True)


def run_base_backtest(signals: pd.DataFrame, bars_15m: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    bars = bars_15m.copy().sort_values("open_time").reset_index(drop=True)
    next_available_time = START
    trades: list[dict[str, Any]] = []
    skipped_count = 0

    for signal in signals.itertuples(index=False):
        signal_time = pd.Timestamp(signal.signal_time)
        signal_price = float(signal.signal_price)

        if signal_time < next_available_time:
            skipped_count += 1
            continue

        limit_price = signal_price * (1.0 + ENTRY_OFFSET)
        order_window = bars[
            (bars["open_time"] >= signal_time) & (bars["open_time"] < signal_time + ENTRY_WINDOW)
        ].copy()

        fill_bar = None
        for bar in order_window.itertuples(index=False):
            if float(bar.low) <= limit_price:
                fill_bar = bar
                break

        if fill_bar is None:
            skipped_count += 1
            next_available_time = signal_time + ENTRY_WINDOW
            continue

        entry_time = pd.Timestamp(fill_bar.open_time)
        take_profit_price = limit_price * (1.0 + TAKE_PROFIT)
        stop_loss_price = limit_price * (1.0 + STOP_LOSS)

        holding_bars = bars[
            (bars["open_time"] >= entry_time) & (bars["open_time"] < entry_time + MAX_HOLD)
        ].copy()
        if holding_bars.empty:
            skipped_count += 1
            next_available_time = signal_time + ENTRY_WINDOW
            continue

        exit_price = float(holding_bars.iloc[-1]["close"])
        exit_time = pd.Timestamp(holding_bars.iloc[-1]["close_time"])
        exit_reason = "timeout"

        for bar in holding_bars.itertuples(index=False):
            bar_low = float(bar.low)
            bar_high = float(bar.high)
            if bar_low <= stop_loss_price and bar_high >= take_profit_price:
                exit_price = stop_loss_price
                exit_time = pd.Timestamp(bar.close_time)
                exit_reason = "stop_loss_same_bar"
                break
            if bar_low <= stop_loss_price:
                exit_price = stop_loss_price
                exit_time = pd.Timestamp(bar.close_time)
                exit_reason = "stop_loss"
                break
            if bar_high >= take_profit_price:
                exit_price = take_profit_price
                exit_time = pd.Timestamp(bar.close_time)
                exit_reason = "take_profit"
                break

        gross_return = exit_price / limit_price - 1.0
        net_return = gross_return - ROUND_TRIP_FEE
        next_available_time = exit_time

        trades.append(
            {
                "signal_time": signal_time,
                "signal_price": signal_price,
                "hourly_return": float(signal.hourly_return),
                "entry_time": entry_time,
                "entry_price": limit_price,
                "exit_time": exit_time,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "gross_return": gross_return,
                "net_return": net_return,
            }
        )

    return pd.DataFrame(trades), skipped_count


def compute_fraction_m0(_: list[float]) -> float:
    return 0.20


def compute_fraction_m1(state: dict[str, Any]) -> float:
    seq = state["sequence"]
    if not seq:
        seq[:] = [1, 2, 3]
    if len(seq) == 1:
        units = seq[0]
    else:
        units = seq[0] + seq[-1]
    fraction = units / 10.0
    return float(np.clip(fraction, 0.05, 0.60))


def update_m1_state(state: dict[str, Any], is_win: bool) -> None:
    seq = state["sequence"]
    if not seq:
        seq[:] = [1, 2, 3]
    if len(seq) == 1:
        units = seq[0]
    else:
        units = seq[0] + seq[-1]

    if is_win:
        if len(seq) == 1:
            seq.clear()
        else:
            seq.pop(0)
            if seq:
                seq.pop(-1)
        if not seq:
            seq[:] = [1, 2, 3]
    else:
        seq.append(units)


def compute_fraction_m2(history_returns: list[float]) -> float:
    if len(history_returns) < 10:
        return 0.20

    wins = [value for value in history_returns if value > 0]
    losses = [abs(value) for value in history_returns if value < 0]
    if not wins or not losses:
        return 0.20

    win_rate = len(wins) / len(history_returns)
    loss_rate = 1.0 - win_rate
    avg_win = float(np.mean(wins))
    avg_loss = float(np.mean(losses))
    if avg_loss <= 0:
        return 0.20

    payoff = avg_win / avg_loss
    if payoff <= 0:
        return 0.20

    full_kelly = win_rate - (loss_rate / payoff)
    half_kelly = full_kelly / 2.0
    return float(np.clip(half_kelly, 0.05, 0.60))


def longest_losing_streak(outcomes: list[bool]) -> int:
    longest = 0
    current = 0
    for outcome in outcomes:
        if outcome:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def build_equity_series(base_index: pd.DatetimeIndex, updates: list[tuple[pd.Timestamp, float]]) -> pd.Series:
    base_series = pd.Series(1.0, index=base_index)
    if not updates:
        return base_series

    points = pd.Series(
        [value for _, value in updates],
        index=pd.DatetimeIndex([timestamp for timestamp, _ in updates]),
        dtype=float,
    )
    points = points.groupby(level=0).last()
    combined_index = base_index.union(points.index).sort_values()
    series = pd.Series(index=combined_index, dtype=float)
    series.iloc[0] = 1.0
    series.loc[points.index] = points.values
    series = series.ffill().fillna(1.0)
    return series.reindex(base_index, method="ffill").fillna(1.0)


def compute_yearly_returns(equity: pd.Series) -> dict[str, float]:
    returns: dict[str, float] = {}
    for year in range(2020, 2026):
        start = pd.Timestamp(f"{year}-01-01 00:00:00", tz="UTC")
        end = pd.Timestamp(f"{year}-12-31 23:59:59", tz="UTC")
        returns[str(year)] = period_return(equity, start, end)
    return returns


def simulate_pattern(
    trades: pd.DataFrame,
    skipped_count: int,
    base_index: pd.DatetimeIndex,
    management: str,
    leverage: int,
) -> tuple[dict[str, Any], pd.Series, pd.Series]:
    gross_equity = 1.0
    net_equity = 1.0
    gross_updates = [(base_index[0], gross_equity)]
    net_updates = [(base_index[0], net_equity)]

    history_returns: list[float] = []
    outcomes: list[bool] = []
    m1_state = {"sequence": [1, 2, 3]}
    trade_rows: list[dict[str, Any]] = []

    for trade in trades.itertuples(index=False):
        if management == "M0":
            fraction = compute_fraction_m0(history_returns)
        elif management == "M1":
            fraction = compute_fraction_m1(m1_state)
        else:
            fraction = compute_fraction_m2(history_returns)

        gross_trade_pnl = fraction * leverage * float(trade.gross_return)
        net_trade_pnl = fraction * leverage * float(trade.net_return)

        gross_equity *= max(0.0, 1.0 + gross_trade_pnl)
        net_equity *= max(0.0, 1.0 + net_trade_pnl)

        exit_time = pd.Timestamp(trade.exit_time)
        gross_updates.append((exit_time, gross_equity))
        net_updates.append((exit_time, net_equity))

        is_win = net_trade_pnl > 0.0
        outcomes.append(is_win)
        history_returns.append(float(trade.net_return))
        if management == "M1":
            update_m1_state(m1_state, is_win)

        trade_rows.append(
            {
                "exit_time": exit_time,
                "fraction": fraction,
                "gross_trade_pnl": gross_trade_pnl,
                "net_trade_pnl": net_trade_pnl,
                "gross_return": float(trade.gross_return),
                "net_return": float(trade.net_return),
                "is_win": is_win,
            }
        )

    gross_series = build_equity_series(base_index, gross_updates)
    net_series = build_equity_series(base_index, net_updates)
    trade_frame = pd.DataFrame(trade_rows)

    gross_ann = annualized_return(gross_series)
    net_ann = annualized_return(net_series)
    max_dd = max_drawdown(net_series)
    risk_return = net_ann / abs(max_dd) if max_dd < 0 else 0.0

    filled_count = int(len(trades))
    total_signals = filled_count + skipped_count
    wins = int(sum(outcomes))
    win_rate = wins / filled_count if filled_count else 0.0
    ci_low, ci_high = wilson_interval(wins, filled_count)
    avg_fraction = float(trade_frame["fraction"].mean()) if not trade_frame.empty else 0.0

    gross_yearly = compute_yearly_returns(gross_series)
    net_yearly = compute_yearly_returns(net_series)
    first_half_start = pd.Timestamp("2020-01-01 00:00:00", tz="UTC")
    first_half_end = pd.Timestamp("2022-12-31 23:59:59", tz="UTC")
    second_half_start = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
    second_half_end = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")

    result = {
        "pattern": f"{management}xL{leverage}",
        "money_management": management,
        "leverage": leverage,
        "annual_return_gross": gross_ann,
        "annual_return_net": net_ann,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "trade_count": filled_count,
        "skipped_count": skipped_count,
        "fill_rate": filled_count / total_signals if total_signals else 0.0,
        "max_losing_streak": longest_losing_streak(outcomes),
        "risk_return_ratio": risk_return,
        "avg_fraction": avg_fraction,
        "first_half_return_gross": period_return(gross_series, first_half_start, first_half_end),
        "first_half_return_net": period_return(net_series, first_half_start, first_half_end),
        "second_half_return_gross": period_return(gross_series, second_half_start, second_half_end),
        "second_half_return_net": period_return(net_series, second_half_start, second_half_end),
        "win_rate_ci_lower": ci_low,
        "win_rate_ci_upper": ci_high,
    }
    for year in range(2020, 2026):
        result[f"gross_{year}"] = gross_yearly[str(year)]
        result[f"net_{year}"] = net_yearly[str(year)]

    return result, gross_series, net_series


def select_best(results: pd.DataFrame) -> tuple[pd.Series, bool]:
    qualified = results[
        (results["first_half_return_net"] > 0.0) & (results["second_half_return_net"] > 0.0)
    ].copy()
    used_strict_filter = not qualified.empty
    if qualified.empty:
        qualified = results.copy()
    qualified = qualified.sort_values("annual_return_net", ascending=False)
    return qualified.iloc[0], used_strict_filter


def print_results(results: pd.DataFrame, best: pd.Series, used_strict_filter: bool) -> None:
    print("\n=== BTC Drop Rebound Backtest ===")
    print("Scope: hourly drop <= -3%, E1 limit -0.3%, TP +1.4%, SL -1.4%, max hold 1h, single-position processing.")
    print(
        "Best pattern: "
        f"{best['pattern']} | annual net {best['annual_return_net']:.2%} | "
        f"first half {best['first_half_return_net']:.2%} | second half {best['second_half_return_net']:.2%}"
    )
    if not used_strict_filter:
        print("Selection note: no pattern had both 2020-2022 and 2023-2025 net returns above zero, so fallback = highest annual net return.")

    formatted = results.copy()
    for column in [
        "annual_return_gross",
        "annual_return_net",
        "max_drawdown",
        "win_rate",
        "fill_rate",
        "risk_return_ratio",
        "first_half_return_net",
        "second_half_return_net",
    ]:
        if column == "risk_return_ratio":
            formatted[column] = formatted[column].map(lambda value: f"{value:.2f}")
        else:
            formatted[column] = formatted[column].map(lambda value: f"{value:.2%}")
    formatted["win_rate_ci_95"] = formatted.apply(
        lambda row: f"[{row['win_rate_ci_lower']:.2%}, {row['win_rate_ci_upper']:.2%}]",
        axis=1,
    )
    formatted = formatted[
        [
            "pattern",
            "annual_return_gross",
            "annual_return_net",
            "max_drawdown",
            "win_rate",
            "win_rate_ci_95",
            "trade_count",
            "skipped_count",
            "fill_rate",
            "max_losing_streak",
            "risk_return_ratio",
            "first_half_return_net",
            "second_half_return_net",
        ]
    ]
    print(formatted.to_string(index=False))

    yearly_columns = ["pattern"] + [f"net_{year}" for year in range(2020, 2026)]
    yearly = results[yearly_columns].copy()
    for column in yearly_columns[1:]:
        yearly[column] = yearly[column].map(lambda value: f"{value:.2%}")
    print("\n=== Net Yearly Performance ===")
    print(yearly.to_string(index=False))


def save_plot(curves: dict[str, tuple[pd.Series, pd.Series]]) -> None:
    selected = {
        name: series
        for name, series in curves.items()
        if name.startswith("M1xL")
    }

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for name, (gross, net) in sorted(selected.items()):
        axes[0].plot(gross.index, gross.values - 1.0, label=name)
        axes[1].plot(net.index, net.values - 1.0, label=name)

    axes[0].set_title("M1 Gross Cumulative Return")
    axes[1].set_title("M1 Net Cumulative Return")
    for axis in axes:
        axis.set_ylabel("Cumulative Return")
        axis.grid(alpha=0.3)
        axis.legend()

    fig.tight_layout()
    fig.savefig(RESULTS_PNG, dpi=150)
    plt.close(fig)


def main() -> None:
    hourly = load_cache(H1_CACHE_CSV)
    bars_15m = load_cache(M15_CACHE_CSV)

    signals = build_signals(hourly)
    base_trades, skipped_count = run_base_backtest(signals, bars_15m)
    if base_trades.empty:
        raise ValueError("No filled trades were generated from the cached BTC data.")

    base_index = pd.DatetimeIndex(bars_15m["open_time"])
    results: list[dict[str, Any]] = []
    curves: dict[str, tuple[pd.Series, pd.Series]] = {}

    for management in MONEY_MANAGEMENT:
        for leverage in LEVERAGES:
            result, gross_curve, net_curve = simulate_pattern(
                base_trades,
                skipped_count,
                base_index,
                management,
                leverage,
            )
            results.append(result)
            curves[result["pattern"]] = (gross_curve, net_curve)

    results_frame = pd.DataFrame(results).sort_values("annual_return_net", ascending=False).reset_index(drop=True)
    best, used_strict_filter = select_best(results_frame)
    best_pattern = str(best["pattern"])
    results_frame["sort_key"] = (results_frame["pattern"] != best_pattern).astype(int)
    results_frame = results_frame.sort_values(["sort_key", "annual_return_net"], ascending=[True, False]).drop(columns=["sort_key"]).reset_index(drop=True)

    print_results(results_frame, best, used_strict_filter)
    results_frame.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")
    save_plot(curves)

    print(f"\nSignals: {len(signals)} | Filled trades: {len(base_trades)} | Skipped: {skipped_count}")
    print(f"CSV saved to: {RESULTS_CSV.resolve()}")
    print(f"Plot saved to: {RESULTS_PNG.resolve()}")


if __name__ == "__main__":
    main()
