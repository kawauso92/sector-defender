from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

BACKTEST_START = pd.Timestamp("2015-01-01")
BACKTEST_END = pd.Timestamp("2025-12-31")
FETCH_START = BACKTEST_START - pd.Timedelta(days=180)
FETCH_END = BACKTEST_END + pd.Timedelta(days=1)
ROLLING_WINDOW = 60
SELECTION_RATIO = 0.30
THRESHOLD_NONE = 0.0
THRESHOLD_FILTERED = 1.0
ROUND_TRIP_COST = 0.0005
RESULTS_CSV = Path("stocks_results.csv")
RESULTS_PNG = Path("stocks_results.png")

JP_WATCHLIST = {
    "XLK": ["8035.T", "6758.T", "6861.T"],
    "XLF": ["8306.T", "8316.T", "8411.T"],
    "XLE": ["5019.T", "5020.T", "1605.T"],
    "XLB": ["4188.T", "3402.T", "5401.T"],
    "XLI": ["6501.T", "7011.T", "6301.T"],
    "XLY": ["7203.T", "7267.T", "7269.T"],
    "XLP": ["2914.T", "2502.T", "2503.T"],
    "XLV": ["4502.T", "4519.T", "4523.T"],
    "XLU": ["9501.T", "9502.T", "9503.T"],
    "XLC": ["9984.T", "9433.T", "9432.T"],
    "XLRE": ["8801.T", "8802.T", "3289.T"],
}

US_ETFS = list(JP_WATCHLIST.keys())
JP_STOCKS = [ticker for tickers in JP_WATCHLIST.values() for ticker in tickers]


@dataclass
class BacktestResult:
    name: str
    threshold: float
    daily_results: pd.DataFrame
    annual_return_gross: float
    annual_return_net: float
    max_drawdown: float
    win_rate: float
    total_trade_days: int
    skip_days: int
    avg_return_per_trade: float
    avg_cost_per_trade: float
    avg_net_per_trade: float
    risk_return_ratio: float
    yearly_returns: pd.DataFrame
    period_stats: pd.DataFrame


def _normalize_download_frame(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw.empty:
        raise ValueError("yfinance からデータを取得できませんでした。")

    if isinstance(raw.columns, pd.MultiIndex):
        return raw

    if len(tickers) != 1:
        raise ValueError("複数ティッカー取得時の列形式が想定外です。")

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
        raise ValueError(f"{field} データを取得できませんでした。")

    return pd.DataFrame(values).sort_index()


def download_market_data() -> dict[str, pd.DataFrame]:
    tickers = US_ETFS + JP_STOCKS
    raw = yf.download(
        tickers=tickers,
        start=FETCH_START.strftime("%Y-%m-%d"),
        end=FETCH_END.strftime("%Y-%m-%d"),
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    history = _normalize_download_frame(raw, tickers)
    return {
        "open": _build_field_frame(history, tickers, "Open"),
        "close": _build_field_frame(history, tickers, "Close"),
    }


def compute_predicted_scores(us_closes: pd.DataFrame) -> pd.DataFrame:
    us_returns = us_closes.sort_index().pct_change()
    rolling_mean = us_returns.shift(1).rolling(ROLLING_WINDOW).mean()
    rolling_std = us_returns.shift(1).rolling(ROLLING_WINDOW).std(ddof=0)
    us_z_scores = (us_returns - rolling_mean) / rolling_std.replace(0, np.nan)

    mapped_scores: dict[str, pd.Series] = {}
    for sector, tickers in JP_WATCHLIST.items():
        if sector not in us_z_scores.columns:
            continue
        sector_scores = us_z_scores[sector]
        for ticker in tickers:
            mapped_scores[ticker] = sector_scores

    predicted_scores = pd.DataFrame(mapped_scores).sort_index()
    return predicted_scores.loc[predicted_scores.index >= BACKTEST_START]


def _next_trade_date(signal_date: pd.Timestamp, jp_trade_dates: pd.DatetimeIndex) -> pd.Timestamp | None:
    position = jp_trade_dates.searchsorted(signal_date + pd.Timedelta(days=1))
    if position >= len(jp_trade_dates):
        return None
    return jp_trade_dates[position]


def _select_candidates(scores: pd.Series, threshold: float) -> tuple[pd.Series, pd.Series]:
    valid_scores = scores.dropna().sort_values(ascending=False)
    if valid_scores.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    selection_count = max(1, int(np.ceil(len(valid_scores) * SELECTION_RATIO)))
    long_candidates = valid_scores.head(selection_count)
    short_candidates = valid_scores.tail(selection_count).sort_values()

    if threshold > 0:
        long_candidates = long_candidates[long_candidates.abs() >= threshold]
        short_candidates = short_candidates[short_candidates.abs() >= threshold]

    return long_candidates, short_candidates


def _resolve_open_to_close_return(
    ticker: str,
    trade_date: pd.Timestamp,
    side: str,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
) -> float | None:
    entry_open = opens.at[trade_date, ticker] if ticker in opens.columns else np.nan
    exit_close = closes.at[trade_date, ticker] if ticker in closes.columns else np.nan

    if pd.isna(entry_open) or pd.isna(exit_close) or entry_open == 0:
        return None

    gross_return = (exit_close / entry_open) - 1.0
    if side == "short":
        gross_return = -gross_return
    return float(gross_return)


def _annualized_return(return_series: pd.Series, trade_dates: pd.Series) -> float:
    if return_series.empty:
        return 0.0
    equity_curve = (1.0 + return_series).cumprod()
    span_days = max((trade_dates.iloc[-1] - trade_dates.iloc[0]).days, 1)
    years = span_days / 365.25
    return float(equity_curve.iloc[-1] ** (1 / years) - 1) if years > 0 else 0.0


def _build_yearly_returns(daily_results: pd.DataFrame) -> pd.DataFrame:
    grouped = daily_results.groupby(daily_results["trade_date"].dt.year)
    gross = grouped["gross_return"].apply(lambda s: (1 + s).prod() - 1).reindex(range(2015, 2026), fill_value=0.0)
    net = grouped["net_return"].apply(lambda s: (1 + s).prod() - 1).reindex(range(2015, 2026), fill_value=0.0)
    return pd.DataFrame({"year": gross.index, "gross_return": gross.values, "net_return": net.values})


def _build_period_stats(daily_results: pd.DataFrame) -> pd.DataFrame:
    periods = [
        ("2015-2020", pd.Timestamp("2015-01-01"), pd.Timestamp("2020-12-31")),
        ("2021-2025", pd.Timestamp("2021-01-01"), pd.Timestamp("2025-12-31")),
    ]
    rows: list[dict[str, Any]] = []

    for label, start, end in periods:
        period_df = daily_results[(daily_results["trade_date"] >= start) & (daily_results["trade_date"] <= end)].copy()
        active_df = period_df[period_df["filled_count"] > 0].copy()
        gross_annual = _annualized_return(period_df["gross_return"], period_df["trade_date"]) if not period_df.empty else 0.0
        net_annual = _annualized_return(period_df["net_return"], period_df["trade_date"]) if not period_df.empty else 0.0
        win_rate = float((active_df["net_return"] > 0).mean()) if not active_df.empty else 0.0

        rows.append(
            {
                "period": label,
                "gross_annual_return": gross_annual,
                "net_annual_return": net_annual,
                "win_rate": win_rate,
                "trade_days": int(len(active_df)),
                "skip_days": int(period_df["skip"].sum()) if not period_df.empty else 0,
            }
        )

    return pd.DataFrame(rows)


def run_backtest(
    name: str,
    threshold: float,
    predicted_scores: pd.DataFrame,
    market_data: dict[str, pd.DataFrame],
) -> BacktestResult:
    opens = market_data["open"].reindex(columns=JP_STOCKS)
    closes = market_data["close"].reindex(columns=JP_STOCKS)
    jp_trade_dates = opens.dropna(how="all").index.intersection(closes.dropna(how="all").index)
    records: list[dict[str, Any]] = []

    for signal_date, scores in predicted_scores.iterrows():
        trade_date = _next_trade_date(signal_date, jp_trade_dates)
        if trade_date is None or trade_date > BACKTEST_END:
            continue

        long_candidates, short_candidates = _select_candidates(scores, threshold)
        selected_count = len(long_candidates) + len(short_candidates)

        if selected_count == 0:
            records.append(
                {
                    "signal_date": signal_date,
                    "trade_date": trade_date,
                    "gross_return": 0.0,
                    "daily_cost": 0.0,
                    "net_return": 0.0,
                    "selected_long_count": 0,
                    "selected_short_count": 0,
                    "filled_long_count": 0,
                    "filled_short_count": 0,
                    "selected_count": 0,
                    "filled_count": 0,
                    "skip": True,
                }
            )
            continue

        position_returns: list[float] = []
        filled_long_count = 0
        filled_short_count = 0

        for ticker in long_candidates.index:
            gross_return = _resolve_open_to_close_return(ticker, trade_date, "long", opens, closes)
            if gross_return is None:
                continue
            position_returns.append(gross_return)
            filled_long_count += 1

        for ticker in short_candidates.index:
            gross_return = _resolve_open_to_close_return(ticker, trade_date, "short", opens, closes)
            if gross_return is None:
                continue
            position_returns.append(gross_return)
            filled_short_count += 1

        filled_count = filled_long_count + filled_short_count
        if filled_count == 0:
            records.append(
                {
                    "signal_date": signal_date,
                    "trade_date": trade_date,
                    "gross_return": 0.0,
                    "daily_cost": 0.0,
                    "net_return": 0.0,
                    "selected_long_count": len(long_candidates),
                    "selected_short_count": len(short_candidates),
                    "filled_long_count": 0,
                    "filled_short_count": 0,
                    "selected_count": selected_count,
                    "filled_count": 0,
                    "skip": True,
                }
            )
            continue

        gross_daily_return = float(np.mean(position_returns))
        net_daily_return = gross_daily_return - ROUND_TRIP_COST

        records.append(
            {
                "signal_date": signal_date,
                "trade_date": trade_date,
                "gross_return": gross_daily_return,
                "daily_cost": ROUND_TRIP_COST,
                "net_return": net_daily_return,
                "selected_long_count": len(long_candidates),
                "selected_short_count": len(short_candidates),
                "filled_long_count": filled_long_count,
                "filled_short_count": filled_short_count,
                "selected_count": selected_count,
                "filled_count": filled_count,
                "skip": False,
            }
        )

    daily_results = pd.DataFrame(records)
    if daily_results.empty:
        raise ValueError(f"{name} のバックテスト結果が空です。")

    daily_results["trade_date"] = pd.to_datetime(daily_results["trade_date"])
    daily_results = daily_results.sort_values(["trade_date", "signal_date"]).reset_index(drop=True)
    daily_results["gross_equity_curve"] = (1.0 + daily_results["gross_return"]).cumprod()
    daily_results["net_equity_curve"] = (1.0 + daily_results["net_return"]).cumprod()
    daily_results["drawdown"] = daily_results["net_equity_curve"] / daily_results["net_equity_curve"].cummax() - 1.0

    active_df = daily_results[daily_results["filled_count"] > 0].copy()
    annual_return_gross = _annualized_return(daily_results["gross_return"], daily_results["trade_date"])
    annual_return_net = _annualized_return(daily_results["net_return"], daily_results["trade_date"])
    max_drawdown = float(daily_results["drawdown"].min()) if not daily_results.empty else 0.0
    win_rate = float((active_df["net_return"] > 0).mean()) if not active_df.empty else 0.0
    total_trade_days = int(len(active_df))
    skip_days = int(daily_results["skip"].sum())
    avg_return_per_trade = float(active_df["gross_return"].mean()) if not active_df.empty else 0.0
    avg_cost_per_trade = float(active_df["daily_cost"].mean()) if not active_df.empty else 0.0
    avg_net_per_trade = float(active_df["net_return"].mean()) if not active_df.empty else 0.0
    risk_return_ratio = annual_return_net / abs(max_drawdown) if max_drawdown < 0 else 0.0
    yearly_returns = _build_yearly_returns(daily_results)
    period_stats = _build_period_stats(daily_results)

    return BacktestResult(
        name=name,
        threshold=threshold,
        daily_results=daily_results,
        annual_return_gross=annual_return_gross,
        annual_return_net=annual_return_net,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        total_trade_days=total_trade_days,
        skip_days=skip_days,
        avg_return_per_trade=avg_return_per_trade,
        avg_cost_per_trade=avg_cost_per_trade,
        avg_net_per_trade=avg_net_per_trade,
        risk_return_ratio=risk_return_ratio,
        yearly_returns=yearly_returns,
        period_stats=period_stats,
    )


def build_results_csv(results: list[BacktestResult]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for result in results:
        subset = result.daily_results.copy()
        subset.insert(0, "pattern", result.name)
        subset.insert(1, "threshold", result.threshold)
        frames.append(subset)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True).sort_values(["trade_date", "signal_date", "pattern"]).reset_index(drop=True)


def save_plot(results: list[BacktestResult]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    for result in results:
        axes[0].plot(result.daily_results["trade_date"], result.daily_results["gross_equity_curve"], label=result.name)
        axes[1].plot(result.daily_results["trade_date"], result.daily_results["net_equity_curve"], label=result.name)

    axes[0].set_title("Cumulative Return Without Cost")
    axes[1].set_title("Cumulative Return With Cost")
    axes[0].set_ylabel("Gross Equity")
    axes[1].set_ylabel("Net Equity")
    axes[1].set_xlabel("Trade Date")

    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()

    fig.tight_layout()
    fig.savefig(RESULTS_PNG, dpi=150)
    plt.close(fig)


def print_comparison_table(results: list[BacktestResult]) -> None:
    rows: list[dict[str, str]] = []

    for result in results:
        rows.append(
            {
                "pattern": result.name,
                "threshold": f"{result.threshold:.1f}",
                "annual_no_cost": f"{result.annual_return_gross:.2%}",
                "annual_with_cost": f"{result.annual_return_net:.2%}",
                "max_drawdown": f"{result.max_drawdown:.2%}",
                "win_rate": f"{result.win_rate:.2%}",
                "trade_days": str(result.total_trade_days),
                "skip_days": str(result.skip_days),
                "avg_return": f"{result.avg_return_per_trade:.2%}",
                "avg_cost": f"{result.avg_cost_per_trade:.2%}",
                "avg_net": f"{result.avg_net_per_trade:.2%}",
                "risk_return": f"{result.risk_return_ratio:.2f}",
            }
        )

    print("\n=== Pattern Comparison ===")
    print(pd.DataFrame(rows).to_string(index=False))


def print_yearly_table(result: BacktestResult) -> None:
    yearly = result.yearly_returns.copy()
    yearly["gross_return"] = yearly["gross_return"].map(lambda x: f"{x:.2%}")
    yearly["net_return"] = yearly["net_return"].map(lambda x: f"{x:.2%}")
    print("\nYearly Performance")
    print(yearly.to_string(index=False))


def print_period_table(result: BacktestResult) -> None:
    periods = result.period_stats.copy()
    periods["gross_annual_return"] = periods["gross_annual_return"].map(lambda x: f"{x:.2%}")
    periods["net_annual_return"] = periods["net_annual_return"].map(lambda x: f"{x:.2%}")
    periods["win_rate"] = periods["win_rate"].map(lambda x: f"{x:.2%}")
    print("\nSubperiod Performance")
    print(periods.to_string(index=False))


def print_detail(result: BacktestResult) -> None:
    print(f"\n=== {result.name} ===")
    print(f"Annual Return (No Cost): {result.annual_return_gross:.2%}")
    print(f"Annual Return (With Cost): {result.annual_return_net:.2%}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")
    print(f"Win Rate: {result.win_rate:.2%}")
    print(f"Total Trade Days: {result.total_trade_days}")
    print(f"Skip Days: {result.skip_days}")
    print(f"Average Return / Trade: {result.avg_return_per_trade:.2%}")
    print(f"Average Cost / Trade: {result.avg_cost_per_trade:.2%}")
    print(f"Average Net / Trade: {result.avg_net_per_trade:.2%}")
    print(f"Risk Return Ratio: {result.risk_return_ratio:.2f}")
    print_yearly_table(result)
    print_period_table(result)


def main() -> None:
    print("Downloading market data...")
    market_data = download_market_data()
    predicted_scores = compute_predicted_scores(market_data["close"].reindex(columns=US_ETFS))

    print("Running stock lead-lag backtest...")
    print(f"Universe size: {len(JP_STOCKS)} JP stocks")
    print(f"Round-trip cost: {ROUND_TRIP_COST:.2%}")

    results = [
        run_backtest(
            name="Pattern A",
            threshold=THRESHOLD_NONE,
            predicted_scores=predicted_scores,
            market_data=market_data,
        ),
        run_backtest(
            name="Pattern B",
            threshold=THRESHOLD_FILTERED,
            predicted_scores=predicted_scores,
            market_data=market_data,
        ),
    ]

    results_df = build_results_csv(results)
    results_df.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")
    save_plot(results)

    print_comparison_table(results)
    for result in results:
        print_detail(result)

    print(f"\nCSV saved to: {RESULTS_CSV.resolve()}")
    print(f"Plot saved to: {RESULTS_PNG.resolve()}")


if __name__ == "__main__":
    main()
