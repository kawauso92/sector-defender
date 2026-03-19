from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from config import JP_ETFS, ROLLING_WINDOW, US_ETFS, US_JP_MAP

BACKTEST_START = pd.Timestamp("2015-01-01")
BACKTEST_END = pd.Timestamp("2025-12-31")
FETCH_START = BACKTEST_START - pd.Timedelta(days=180)
FETCH_END = BACKTEST_END + pd.Timedelta(days=1)
SELECTION_RATIO = 0.30
THRESHOLD_NONE = 0.0
THRESHOLD_FILTERED = 1.0
MARKET_ROUND_TRIP_COST = 0.0010
LIMIT_ROUND_TRIP_COST = 0.0005
LIMIT_LONG_MULTIPLIER = 0.998
LIMIT_SHORT_MULTIPLIER = 1.002
RESULTS_CSV = Path("results.csv")
RESULTS_PNG = Path("results.png")


@dataclass
class BacktestResult:
    name: str
    order_type: str
    threshold: float
    round_trip_cost: float
    daily_results: pd.DataFrame
    annual_return_gross: float
    annual_return_net: float
    max_drawdown: float
    win_rate: float
    total_trade_days: int
    skip_days: int
    fill_rate: float | None
    avg_return_per_trade: float
    avg_cost_per_trade: float
    avg_net_per_trade: float
    risk_return_ratio: float
    yearly_returns: pd.DataFrame
    period_stats: pd.DataFrame


def _normalize_download_frame(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw.empty:
        raise ValueError("市場データを取得できませんでした。")

    if isinstance(raw.columns, pd.MultiIndex):
        return raw

    if len(tickers) != 1:
        raise ValueError("取得したデータ形式が想定と異なります。")

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
    tickers = US_ETFS + JP_ETFS
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
        "high": _build_field_frame(history, tickers, "High"),
        "low": _build_field_frame(history, tickers, "Low"),
        "close": _build_field_frame(history, tickers, "Close"),
        "volume": _build_field_frame(history, tickers, "Volume"),
    }


def compute_predicted_scores(us_closes: pd.DataFrame) -> pd.DataFrame:
    us_returns = us_closes.sort_index().pct_change()
    # 当日リターンを rolling 平均/標準偏差へ混ぜないため shift(1) を使い、look-ahead bias を避ける。
    rolling_mean = us_returns.shift(1).rolling(ROLLING_WINDOW).mean()
    rolling_std = us_returns.shift(1).rolling(ROLLING_WINDOW).std(ddof=0)
    us_z_scores = (us_returns - rolling_mean) / rolling_std.replace(0, np.nan)

    mapped_scores: dict[str, pd.Series] = {}
    for jp_ticker in JP_ETFS:
        source_tickers = [us_ticker for us_ticker, mapped in US_JP_MAP.items() if jp_ticker in mapped]
        if not source_tickers:
            continue
        mapped_scores[jp_ticker] = us_z_scores[source_tickers].mean(axis=1, skipna=True)

    predicted_scores = pd.DataFrame(mapped_scores).sort_index()
    return predicted_scores.loc[predicted_scores.index >= BACKTEST_START]


def _next_trade_date(signal_date: pd.Timestamp, jp_trade_dates: pd.DatetimeIndex) -> pd.Timestamp | None:
    # US シグナル日より後の最初の JP 営業日を使い、日米営業日ズレを保守的に扱う。
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


def _resolve_market_return(
    ticker: str,
    trade_date: pd.Timestamp,
    side: str,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
) -> tuple[float | None, bool]:
    entry_open = opens.at[trade_date, ticker] if ticker in opens.columns else np.nan
    exit_close = closes.at[trade_date, ticker] if ticker in closes.columns else np.nan

    if pd.isna(entry_open) or pd.isna(exit_close) or entry_open == 0:
        return None, True

    gross_return = (exit_close / entry_open) - 1.0
    if side == "short":
        gross_return = -gross_return
    return float(gross_return), False


def _resolve_limit_return(
    ticker: str,
    trade_date: pd.Timestamp,
    side: str,
    opens: pd.DataFrame,
    highs: pd.DataFrame,
    lows: pd.DataFrame,
    closes: pd.DataFrame,
) -> tuple[float | None, bool]:
    day_open = opens.at[trade_date, ticker] if ticker in opens.columns else np.nan
    day_close = closes.at[trade_date, ticker] if ticker in closes.columns else np.nan
    day_high = highs.at[trade_date, ticker] if ticker in highs.columns else np.nan
    day_low = lows.at[trade_date, ticker] if ticker in lows.columns else np.nan

    # 指値基準は当日 Open。Open 欠損日は非約定として扱う。
    if pd.isna(day_open) or day_open == 0 or pd.isna(day_close) or pd.isna(day_high) or pd.isna(day_low):
        return None, True

    if side == "long":
        limit_price = day_open * LIMIT_LONG_MULTIPLIER
        if day_low > limit_price:
            return None, False
        gross_return = (day_close / day_open) - 1.0
    else:
        limit_price = day_open * LIMIT_SHORT_MULTIPLIER
        if day_high < limit_price:
            return None, False
        gross_return = -((day_close / day_open) - 1.0)

    return float(gross_return), False


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
            }
        )

    return pd.DataFrame(rows)


def run_backtest(
    name: str,
    order_type: str,
    threshold: float,
    predicted_scores: pd.DataFrame,
    market_data: dict[str, pd.DataFrame],
    round_trip_cost: float,
) -> BacktestResult:
    opens = market_data["open"].reindex(columns=JP_ETFS)
    highs = market_data["high"].reindex(columns=JP_ETFS)
    lows = market_data["low"].reindex(columns=JP_ETFS)
    closes = market_data["close"].reindex(columns=JP_ETFS)
    jp_trade_dates = opens.dropna(how="all").index.intersection(closes.dropna(how="all").index)
    records: list[dict[str, Any]] = []
    total_selected_positions = 0
    total_filled_positions = 0

    for signal_date, scores in predicted_scores.iterrows():
        trade_date = _next_trade_date(signal_date, jp_trade_dates)
        if trade_date is None or trade_date > BACKTEST_END:
            continue

        long_candidates, short_candidates = _select_candidates(scores, threshold)
        selected_count = len(long_candidates) + len(short_candidates)
        total_selected_positions += selected_count

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
                    "data_issue_count": 0,
                }
            )
            continue

        position_returns: list[float] = []
        filled_long_count = 0
        filled_short_count = 0
        data_issue_count = 0

        for ticker, _score in long_candidates.items():
            if order_type == "market":
                gross_return, has_issue = _resolve_market_return(ticker, trade_date, "long", opens, closes)
            else:
                gross_return, has_issue = _resolve_limit_return(ticker, trade_date, "long", opens, highs, lows, closes)

            if gross_return is None:
                data_issue_count += int(has_issue)
                continue

            position_returns.append(gross_return)
            filled_long_count += 1

        for ticker, _score in short_candidates.items():
            if order_type == "market":
                gross_return, has_issue = _resolve_market_return(ticker, trade_date, "short", opens, closes)
            else:
                gross_return, has_issue = _resolve_limit_return(ticker, trade_date, "short", opens, highs, lows, closes)

            if gross_return is None:
                data_issue_count += int(has_issue)
                continue

            position_returns.append(gross_return)
            filled_short_count += 1

        filled_count = filled_long_count + filled_short_count
        total_filled_positions += filled_count

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
                    "data_issue_count": data_issue_count,
                }
            )
            continue

        gross_daily_return = float(np.mean(position_returns))
        net_daily_return = gross_daily_return - round_trip_cost

        records.append(
            {
                "signal_date": signal_date,
                "trade_date": trade_date,
                "gross_return": gross_daily_return,
                "daily_cost": round_trip_cost,
                "net_return": net_daily_return,
                "selected_long_count": len(long_candidates),
                "selected_short_count": len(short_candidates),
                "filled_long_count": filled_long_count,
                "filled_short_count": filled_short_count,
                "selected_count": selected_count,
                "filled_count": filled_count,
                "skip": False,
                "data_issue_count": data_issue_count,
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
    fill_rate = (total_filled_positions / total_selected_positions) if total_selected_positions > 0 else None
    avg_return_per_trade = float(active_df["gross_return"].mean()) if not active_df.empty else 0.0
    avg_cost_per_trade = float(active_df["daily_cost"].mean()) if not active_df.empty else 0.0
    avg_net_per_trade = float(active_df["net_return"].mean()) if not active_df.empty else 0.0
    risk_return_ratio = annual_return_net / abs(max_drawdown) if max_drawdown < 0 else 0.0
    yearly_returns = _build_yearly_returns(daily_results)
    period_stats = _build_period_stats(daily_results)

    return BacktestResult(
        name=name,
        order_type=order_type,
        threshold=threshold,
        round_trip_cost=round_trip_cost,
        daily_results=daily_results,
        annual_return_gross=annual_return_gross,
        annual_return_net=annual_return_net,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        total_trade_days=total_trade_days,
        skip_days=skip_days,
        fill_rate=fill_rate if order_type == "limit" else None,
        avg_return_per_trade=avg_return_per_trade,
        avg_cost_per_trade=avg_cost_per_trade,
        avg_net_per_trade=avg_net_per_trade,
        risk_return_ratio=risk_return_ratio,
        yearly_returns=yearly_returns,
        period_stats=period_stats,
    )


def build_results_csv(results: list[BacktestResult]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None

    for result in results:
        key = result.name.lower().replace(" ", "_")
        subset = result.daily_results[
            [
                "trade_date",
                "signal_date",
                "gross_return",
                "daily_cost",
                "net_return",
                "gross_equity_curve",
                "net_equity_curve",
                "selected_count",
                "filled_count",
                "selected_long_count",
                "selected_short_count",
                "filled_long_count",
                "filled_short_count",
                "skip",
                "data_issue_count",
            ]
        ].rename(
            columns={
                "signal_date": f"signal_date_{key}",
                "gross_return": f"gross_return_{key}",
                "daily_cost": f"daily_cost_{key}",
                "net_return": f"net_return_{key}",
                "gross_equity_curve": f"gross_equity_curve_{key}",
                "net_equity_curve": f"net_equity_curve_{key}",
                "selected_count": f"selected_count_{key}",
                "filled_count": f"filled_count_{key}",
                "selected_long_count": f"selected_long_count_{key}",
                "selected_short_count": f"selected_short_count_{key}",
                "filled_long_count": f"filled_long_count_{key}",
                "filled_short_count": f"filled_short_count_{key}",
                "skip": f"skip_{key}",
                "data_issue_count": f"data_issue_count_{key}",
            }
        )

        if merged is None:
            merged = subset
        else:
            merged = merged.merge(subset, on="trade_date", how="outer")

    if merged is None:
        return pd.DataFrame()

    return merged.sort_values("trade_date").reset_index(drop=True)


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
                "annual_no_cost": f"{result.annual_return_gross:.2%}",
                "annual_with_cost": f"{result.annual_return_net:.2%}",
                "max_drawdown": f"{result.max_drawdown:.2%}",
                "win_rate": f"{result.win_rate:.2%}",
                "trade_days": str(result.total_trade_days),
                "skip_days": str(result.skip_days),
                "fill_rate": f"{result.fill_rate:.2%}" if result.fill_rate is not None else "-",
                "avg_return": f"{result.avg_return_per_trade:.2%}",
                "avg_cost": f"{result.avg_cost_per_trade:.2%}",
                "avg_net": f"{result.avg_net_per_trade:.2%}",
                "risk_return": f"{result.risk_return_ratio:.2f}",
            }
        )

    print("\n=== 4パターン比較 ===")
    print(pd.DataFrame(rows).to_string(index=False))


def print_yearly_table(result: BacktestResult) -> None:
    yearly = result.yearly_returns.copy()
    yearly["gross_return"] = yearly["gross_return"].map(lambda x: f"{x:.2%}")
    yearly["net_return"] = yearly["net_return"].map(lambda x: f"{x:.2%}")
    print("\n年別パフォーマンス")
    print(yearly.to_string(index=False))


def print_period_table(result: BacktestResult) -> None:
    periods = result.period_stats.copy()
    periods["gross_annual_return"] = periods["gross_annual_return"].map(lambda x: f"{x:.2%}")
    periods["net_annual_return"] = periods["net_annual_return"].map(lambda x: f"{x:.2%}")
    periods["win_rate"] = periods["win_rate"].map(lambda x: f"{x:.2%}")
    print("\n前半・後半成績")
    print(periods.to_string(index=False))


def print_detail(result: BacktestResult) -> None:
    print(f"\n=== {result.name} ===")
    print(f"年率リターン（コストなし）: {result.annual_return_gross:.2%}")
    print(f"年率リターン（コストあり）: {result.annual_return_net:.2%}")
    print(f"最大ドローダウン        : {result.max_drawdown:.2%}")
    print(f"勝率                    : {result.win_rate:.2%}")
    print(f"総取引日数              : {result.total_trade_days}")
    print(f"見送り日数              : {result.skip_days}")
    if result.fill_rate is not None:
        print(f"約定率                  : {result.fill_rate:.2%}")
    print(f"1取引あたり平均リターン : {result.avg_return_per_trade:.2%}")
    print(f"1取引あたりコスト       : {result.avg_cost_per_trade:.2%}")
    print(f"1取引あたり純損益       : {result.avg_net_per_trade:.2%}")
    print(f"リスクリターン比        : {result.risk_return_ratio:.2f}")
    print_yearly_table(result)
    print_period_table(result)


def main() -> None:
    print("市場データを取得しています...")
    market_data = download_market_data()
    predicted_scores = compute_predicted_scores(market_data["close"].reindex(columns=US_ETFS))

    print("バックテストを実行しています...")
    print(f"成行コスト仮定: 往復 {MARKET_ROUND_TRIP_COST:.2%}")
    print(f"指値コスト仮定: 往復 {LIMIT_ROUND_TRIP_COST:.2%}（スリッページなし）")

    results = [
        run_backtest(
            name="Pattern A",
            order_type="market",
            threshold=THRESHOLD_NONE,
            predicted_scores=predicted_scores,
            market_data=market_data,
            round_trip_cost=MARKET_ROUND_TRIP_COST,
        ),
        run_backtest(
            name="Pattern B",
            order_type="market",
            threshold=THRESHOLD_FILTERED,
            predicted_scores=predicted_scores,
            market_data=market_data,
            round_trip_cost=MARKET_ROUND_TRIP_COST,
        ),
        run_backtest(
            name="Pattern C",
            order_type="limit",
            threshold=THRESHOLD_NONE,
            predicted_scores=predicted_scores,
            market_data=market_data,
            round_trip_cost=LIMIT_ROUND_TRIP_COST,
        ),
        run_backtest(
            name="Pattern D",
            order_type="limit",
            threshold=THRESHOLD_FILTERED,
            predicted_scores=predicted_scores,
            market_data=market_data,
            round_trip_cost=LIMIT_ROUND_TRIP_COST,
        ),
    ]

    results_df = build_results_csv(results)
    results_df.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")
    save_plot(results)

    print_comparison_table(results)
    for result in results:
        print_detail(result)

    print(f"\n保存ファイル: {RESULTS_CSV.resolve()}")
    print(f"保存ファイル: {RESULTS_PNG.resolve()}")


if __name__ == "__main__":
    main()
