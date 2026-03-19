from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    DATA_PERIOD,
    JP_ETF_LABELS,
    JP_ETFS,
    LOW_VOLUME_THRESHOLD_RATIO,
    MAX_POSITION_BUDGET_RATIO,
    ROLLING_WINDOW,
    TAKE_PROFIT_RATIO,
    US_ETF_LABELS,
    US_ETFS,
    US_JP_MAP,
    VIX_ALERT_LEVEL,
    VIX_SIGNAL_DAMPING,
    VIX_SKIP_LEVEL,
    VIX_TICKER,
    VOLUME_WINDOW,
    WEAK_SIGNAL_THRESHOLD,
)


AUTO_NORMAL_LABEL = "\U0001F7E2 \u81ea\u52d5\uff1a\u901a\u5e38\u30e2\u30fc\u30c9"
AUTO_SHORT_VIX_LABEL = "\U0001F534 \u81ea\u52d5\uff1a\u30b7\u30e7\u30fc\u30c8\u512a\u5148\u30e2\u30fc\u30c9\uff08VIX\u8b66\u6212\uff09"
AUTO_SHORT_BROAD_LABEL = "\U0001F534 \u81ea\u52d5\uff1a\u30b7\u30e7\u30fc\u30c8\u512a\u5148\u30e2\u30fc\u30c9\uff08\u5168\u9762\u5b89\u5730\u5408\u3044\uff09"
AUTO_SHORT_MIXED_LABEL = "\U0001F534 \u81ea\u52d5\uff1a\u30b7\u30e7\u30fc\u30c8\u512a\u5148\u30e2\u30fc\u30c9\uff08VIX\u8b66\u6212\u30fb\u5168\u9762\u5b89\u5730\u5408\u3044\uff09"
MANUAL_NORMAL_LABEL = "\u2699\ufe0f \u624b\u52d5\uff1a\u901a\u5e38\u30e2\u30fc\u30c9"
MANUAL_SHORT_LABEL = "\u2699\ufe0f \u624b\u52d5\uff1a\u30b7\u30e7\u30fc\u30c8\u512a\u5148\u30e2\u30fc\u30c9"
SKIP_TITLE = "\U0001F6AB \u672c\u65e5\u306f\u898b\u9001\u308a\u63a8\u5968"
SKIP_SIGNAL_TITLE = "\U0001F6AB \u672c\u65e5\u306f\u898b\u9001\u308a\u63a8\u5968\uff08\u30b7\u30b0\u30ca\u30eb\u5f37\u5ea6\u4e0d\u8db3\uff09"
LOW_VOLUME_WARNING = "\u26a0\ufe0f \u51fa\u6765\u9ad8\u6ce8\u610f"


@dataclass
class SignalPackage:
    long_candidates: pd.DataFrame
    short_candidates: pd.DataFrame
    skipped_candidates: pd.DataFrame
    ranking_table: pd.DataFrame
    us_summary: pd.DataFrame
    risk_messages: list[str]
    latest_market_timestamp: pd.Timestamp
    fetched_at: pd.Timestamp
    vix_value: float | None
    vix_damping_factor: float
    all_us_scores_negative: bool
    market_mode: str
    market_mode_label: str
    skip_trading: bool
    skip_title: str | None
    skip_reason: str | None
    us_data_unavailable: bool
    long_budget: float
    short_budget: float
    buffer_budget: float
    qualified_long_count: int
    qualified_short_count: int
    min_signal_threshold: float
    candidate_count: int


def _normalize_download_frame(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw.empty:
        raise ValueError("\u5e02\u5834\u30c7\u30fc\u30bf\u3092\u53d6\u5f97\u3067\u304d\u307e\u305b\u3093\u3067\u3057\u305f\u3002")

    if isinstance(raw.columns, pd.MultiIndex):
        return raw

    if len(tickers) != 1:
        raise ValueError("\u8907\u6570\u30c6\u30a3\u30c3\u30ab\u30fc\u306e\u30c7\u30fc\u30bf\u69cb\u9020\u304c\u60f3\u5b9a\u5916\u3067\u3059\u3002")

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
        raise ValueError(f"{field} \u30c7\u30fc\u30bf\u3092\u53d6\u5f97\u3067\u304d\u307e\u305b\u3093\u3067\u3057\u305f\u3002")

    return pd.DataFrame(values).sort_index()


def fetch_market_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    tickers = US_ETFS + JP_ETFS + [VIX_TICKER]
    raw = yf.download(
        tickers=tickers,
        period=DATA_PERIOD,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    history = _normalize_download_frame(raw, tickers)
    closes = _build_field_frame(history, tickers, "Close")
    volumes = _build_field_frame(history, tickers, "Volume")
    return closes, volumes


def _calculate_us_z_scores(us_closes: pd.DataFrame) -> pd.DataFrame:
    returns = us_closes.sort_index().pct_change().dropna(how="all")
    rows: list[dict[str, Any]] = []

    for ticker in US_ETFS:
        series = returns.get(ticker)
        if series is None:
            continue

        series = series.dropna()
        if len(series) < ROLLING_WINDOW + 1:
            continue

        latest_return = float(series.iloc[-1])
        baseline = series.iloc[-(ROLLING_WINDOW + 1) : -1]
        mean = float(baseline.mean())
        std = float(baseline.std(ddof=0))
        z_score = 0.0 if std == 0 or np.isnan(std) else (latest_return - mean) / std

        rows.append(
            {
                "ticker": ticker,
                "sector": US_ETF_LABELS.get(ticker, ticker),
                "latest_return": latest_return,
                "mean_return": mean,
                "std_return": std,
                "z_score": float(z_score),
            }
        )

    if not rows:
        raise ValueError("\u7c73\u56fdETF\u306e\u30b7\u30b0\u30ca\u30eb\u8a08\u7b97\u306b\u5fc5\u8981\u306a\u30c7\u30fc\u30bf\u304c\u4e0d\u8db3\u3057\u3066\u3044\u307e\u3059\u3002")

    return pd.DataFrame(rows).sort_values("z_score", ascending=False).reset_index(drop=True)


def _map_scores_to_japan(us_summary: pd.DataFrame) -> pd.DataFrame:
    mapped_scores: dict[str, list[float]] = {ticker: [] for ticker in JP_ETFS}

    for row in us_summary.itertuples(index=False):
        for jp_ticker in US_JP_MAP.get(row.ticker, []):
            mapped_scores.setdefault(jp_ticker, []).append(float(row.z_score))

    rows: list[dict[str, Any]] = []
    for ticker in JP_ETFS:
        source_scores = mapped_scores.get(ticker, [])
        predicted_score = float(np.mean(source_scores)) if source_scores else np.nan
        rows.append(
            {
                "ticker": ticker,
                "sector": JP_ETF_LABELS.get(ticker, ticker),
                "predicted_score": predicted_score,
                "source_count": len(source_scores),
            }
        )

    ranking = pd.DataFrame(rows).dropna(subset=["predicted_score"])
    if ranking.empty:
        raise ValueError("\u65e5\u672cETF\u3078\u30de\u30c3\u30d4\u30f3\u30b0\u53ef\u80fd\u306a\u30b7\u30b0\u30ca\u30eb\u304c\u3042\u308a\u307e\u305b\u3093\u3002")

    return ranking


def _append_japan_market_data(ranking: pd.DataFrame, jp_closes: pd.DataFrame, jp_volumes: pd.DataFrame) -> pd.DataFrame:
    if jp_closes.empty:
        raise ValueError("\u65e5\u672cETF\u306e\u7d42\u5024\u30c7\u30fc\u30bf\u3092\u53d6\u5f97\u3067\u304d\u307e\u305b\u3093\u3067\u3057\u305f\u3002")

    latest_prices = jp_closes.ffill().iloc[-1]

    if jp_volumes.empty:
        latest_volumes = pd.Series(index=jp_closes.columns, dtype=float)
        volume_average = pd.Series(index=jp_closes.columns, dtype=float)
    else:
        latest_volumes = jp_volumes.ffill().iloc[-1]
        volume_average = jp_volumes.ffill().tail(VOLUME_WINDOW).mean()

    enriched = ranking.copy()
    enriched["current_price"] = enriched["ticker"].map(latest_prices.to_dict())
    enriched["reference_close"] = enriched["current_price"]
    enriched["latest_volume"] = enriched["ticker"].map(latest_volumes.to_dict())
    enriched["avg_volume_5d"] = enriched["ticker"].map(volume_average.to_dict())
    enriched["low_volume_warning"] = (
        enriched["latest_volume"].fillna(0) < enriched["avg_volume_5d"].fillna(np.inf) * LOW_VOLUME_THRESHOLD_RATIO
    )
    enriched["volume_warning"] = np.where(enriched["low_volume_warning"], LOW_VOLUME_WARNING, "")
    enriched = enriched.dropna(subset=["current_price"])
    return enriched


def _apply_vix_filter(ranking: pd.DataFrame, vix_series: pd.Series) -> tuple[pd.DataFrame, float | None, float, list[str]]:
    vix_values = vix_series.dropna()
    vix_value = float(vix_values.iloc[-1]) if not vix_values.empty else None
    damping_factor = 1.0
    risk_messages: list[str] = []

    if vix_value is not None and vix_value > VIX_ALERT_LEVEL:
        damping_factor = VIX_SIGNAL_DAMPING
        risk_messages.append(
            f"VIX \u304c {vix_value:.2f} \u3068\u8b66\u6212\u6c34\u6e96 {VIX_ALERT_LEVEL:.0f} \u3092\u4e0a\u56de\u3063\u305f\u305f\u3081\u3001\u30b7\u30b0\u30ca\u30eb\u5f37\u5ea6\u3092 {damping_factor:.0%} \u306b\u6e1b\u8870\u3057\u3066\u3044\u307e\u3059\u3002"
        )
    elif vix_value is not None:
        risk_messages.append(
            f"VIX \u306f {vix_value:.2f} \u3067\u8b66\u6212\u6c34\u6e96 {VIX_ALERT_LEVEL:.0f} \u672a\u6e80\u3067\u3059\u3002"
        )

    adjusted = ranking.copy()
    adjusted["signal_score"] = adjusted["predicted_score"] * damping_factor
    return adjusted, vix_value, damping_factor, risk_messages


def _detect_us_data_unavailable(us_closes: pd.DataFrame, fetched_at: pd.Timestamp) -> bool:
    latest_us_timestamp = us_closes.dropna(how="all").index.max() if not us_closes.empty else pd.NaT
    if pd.isna(latest_us_timestamp):
        return True

    eastern_now = fetched_at.tz_convert("US/Eastern")
    if eastern_now.weekday() >= 5 or eastern_now.hour < 18:
        return False

    return pd.Timestamp(latest_us_timestamp).date() < eastern_now.date()


def _resolve_market_mode(
    total_capital: float,
    vix_value: float | None,
    all_us_scores_negative: bool,
    manual_mode: str | None,
) -> tuple[str, str, float, float, float]:
    auto_vix = vix_value is not None and vix_value > VIX_ALERT_LEVEL
    auto_broad_weakness = all_us_scores_negative
    auto_mode = "short_priority" if auto_vix or auto_broad_weakness else "normal"
    market_mode = manual_mode or auto_mode

    if market_mode == "short_priority":
        long_ratio, short_ratio = 0.30, 0.60
    else:
        long_ratio, short_ratio = 0.60, 0.30
    buffer_ratio = 0.10

    if manual_mode == "short_priority":
        market_mode_label = MANUAL_SHORT_LABEL
    elif manual_mode == "normal":
        market_mode_label = MANUAL_NORMAL_LABEL
    elif auto_vix and auto_broad_weakness:
        market_mode_label = AUTO_SHORT_MIXED_LABEL
    elif auto_vix:
        market_mode_label = AUTO_SHORT_VIX_LABEL
    elif auto_broad_weakness:
        market_mode_label = AUTO_SHORT_BROAD_LABEL
    else:
        market_mode_label = AUTO_NORMAL_LABEL

    return (
        market_mode,
        market_mode_label,
        total_capital * long_ratio,
        total_capital * short_ratio,
        total_capital * buffer_ratio,
    )


def _prepare_candidate_tables(
    ranking: pd.DataFrame,
    candidate_count: int,
    min_signal_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = ranking.sort_values("signal_score", ascending=False).reset_index(drop=True).copy()
    candidate_count = min(candidate_count, len(ordered))
    short_start_index = max(candidate_count, len(ordered) - candidate_count)

    ordered["direction_code"] = "skip"
    ordered["section"] = "skip"
    ordered["budget_code"] = "skip"

    long_mask = ordered.index < candidate_count
    short_mask = ordered.index >= short_start_index

    ordered.loc[long_mask, "section"] = "long_pool"
    ordered.loc[short_mask, "section"] = "short_pool"

    long_qualified = long_mask & (ordered["signal_score"].abs() >= min_signal_threshold)
    short_qualified = short_mask & (ordered["signal_score"].abs() >= min_signal_threshold)

    ordered.loc[long_mask & ~long_qualified, "budget_code"] = "below_threshold"
    ordered.loc[short_mask & ~short_qualified, "budget_code"] = "below_threshold"

    ordered.loc[long_qualified, "direction_code"] = "long"
    ordered.loc[long_qualified, "section"] = "long"
    ordered.loc[long_qualified, "budget_code"] = "candidate"

    ordered.loc[short_qualified, "direction_code"] = "short"
    ordered.loc[short_qualified, "section"] = "short"
    ordered.loc[short_qualified, "budget_code"] = "candidate"

    long_pool = (
        ordered.loc[ordered["section"].isin(["long_pool", "long"])]
        .sort_values("signal_score", ascending=False)
        .reset_index(drop=True)
    )
    short_pool = (
        ordered.loc[ordered["section"].isin(["short_pool", "short"])]
        .sort_values("signal_score")
        .reset_index(drop=True)
    )
    long_candidates = (
        ordered.loc[ordered["section"] == "long"]
        .sort_values("signal_score", ascending=False)
        .reset_index(drop=True)
    )
    short_candidates = (
        ordered.loc[ordered["section"] == "short"]
        .sort_values("signal_score")
        .reset_index(drop=True)
    )
    skipped_candidates = (
        ordered.loc[ordered["direction_code"] == "skip"]
        .sort_values("signal_score", ascending=False)
        .reset_index(drop=True)
    )

    return long_pool, short_pool, long_candidates, short_candidates, skipped_candidates


def _allocate_side_budget(
    candidates: pd.DataFrame,
    side_budget: float,
    total_capital: float,
    stop_loss_rate: float,
    side: str,
) -> pd.DataFrame:
    allocated = candidates.copy()
    if allocated.empty:
        allocated["allocation_ratio"] = pd.Series(dtype=float)
        allocated["recommended_units"] = pd.Series(dtype=int)
        allocated["recommended_amount"] = pd.Series(dtype=float)
        allocated["cumulative_amount"] = pd.Series(dtype=float)
        allocated["stop_loss_line"] = pd.Series(dtype=float)
        allocated["take_profit_line"] = pd.Series(dtype=float)
        allocated["budget_code"] = pd.Series(dtype=str)
        return allocated

    signal_abs_sum = allocated["signal_score"].abs().sum()
    fallback_ratio = 1 / len(allocated)
    max_position_budget = side_budget * MAX_POSITION_BUDGET_RATIO
    spent_budget = 0.0

    allocation_ratios: list[float] = []
    recommended_units: list[int] = []
    recommended_amounts: list[float] = []
    cumulative_amounts: list[float] = []
    stop_loss_lines: list[float] = []
    take_profit_lines: list[float] = []
    status_codes: list[str] = []

    for row in allocated.itertuples(index=False):
        current_price = float(row.current_price)
        allocation_ratio = (abs(float(row.signal_score)) / signal_abs_sum) if signal_abs_sum > 0 else fallback_ratio
        target_amount = min(side_budget * allocation_ratio, max_position_budget)

        if current_price > total_capital or current_price > side_budget:
            status_code = "insufficient_funds"
            units = 0
            amount = 0.0
        else:
            units = max(1, int(np.floor(target_amount / current_price)))
            amount = float(units * current_price)
            if amount <= max(side_budget - spent_budget, 0):
                status_code = "orderable"
                spent_budget += amount
            else:
                status_code = "over_budget"
                units = 0
                amount = 0.0

        allocation_ratios.append(allocation_ratio)
        recommended_units.append(units)
        recommended_amounts.append(amount)
        cumulative_amounts.append(spent_budget)
        status_codes.append(status_code)

        if status_code != "orderable":
            stop_loss_lines.append(np.nan)
            take_profit_lines.append(np.nan)
        elif side == "long":
            stop_loss_lines.append(current_price * (1 - stop_loss_rate))
            take_profit_lines.append(current_price * (1 + TAKE_PROFIT_RATIO))
        else:
            stop_loss_lines.append(current_price * (1 + stop_loss_rate))
            take_profit_lines.append(current_price * (1 - TAKE_PROFIT_RATIO))

    allocated["allocation_ratio"] = allocation_ratios
    allocated["recommended_units"] = recommended_units
    allocated["recommended_amount"] = recommended_amounts
    allocated["cumulative_amount"] = cumulative_amounts
    allocated["stop_loss_line"] = stop_loss_lines
    allocated["take_profit_line"] = take_profit_lines
    allocated["budget_code"] = status_codes
    return allocated


def _decorate_skipped(skipped_candidates: pd.DataFrame) -> pd.DataFrame:
    skipped = skipped_candidates.copy()
    skipped["allocation_ratio"] = 0.0
    skipped["recommended_units"] = 0
    skipped["recommended_amount"] = 0.0
    skipped["cumulative_amount"] = 0.0
    skipped["stop_loss_line"] = np.nan
    skipped["take_profit_line"] = np.nan
    if "budget_code" not in skipped.columns:
        skipped["budget_code"] = "skip"
    return skipped


def _evaluate_skip_trading(
    long_pool: pd.DataFrame,
    short_pool: pd.DataFrame,
    long_candidates: pd.DataFrame,
    short_candidates: pd.DataFrame,
    vix_value: float | None,
    us_data_unavailable: bool,
    min_signal_threshold: float,
) -> tuple[bool, str | None, str | None]:
    pooled_scores = pd.concat([long_pool["signal_score"], short_pool["signal_score"]], ignore_index=True)
    weak_signals = bool((pooled_scores.abs() < WEAK_SIGNAL_THRESHOLD).all()) if not pooled_scores.empty else True

    if vix_value is not None and vix_value > VIX_SKIP_LEVEL:
        return True, SKIP_TITLE, f"\u7406\u7531\uff1aVIX \u304c {vix_value:.1f} \u3068\u6975\u5ea6\u306e\u8b66\u6212\u6c34\u6e96\u306e\u305f\u3081"
    if long_candidates.empty and short_candidates.empty:
        return (
            True,
            SKIP_SIGNAL_TITLE,
            f"\u7406\u7531\uff1a\u95be\u5024 {min_signal_threshold:.1f} \u4ee5\u4e0a\u306e\u5bfe\u8c61\u9298\u67c4\u304c\u3042\u308a\u307e\u305b\u3093",
        )
    if weak_signals:
        return (
            True,
            SKIP_TITLE,
            f"\u7406\u7531\uff1a\u5019\u88dc\u9298\u67c4\u306e\u30b9\u30b3\u30a2\u7d76\u5bfe\u5024\u304c\u3059\u3079\u3066 {WEAK_SIGNAL_THRESHOLD:.1f} \u672a\u6e80\u306e\u305f\u3081",
        )
    if us_data_unavailable:
        return True, SKIP_TITLE, "\u7406\u7531\uff1a\u7c73\u56fd\u5e02\u5834\u304c\u4f11\u5834\u3001\u307e\u305f\u306f\u6700\u65b0\u30c7\u30fc\u30bf\u3092\u53d6\u5f97\u3067\u304d\u306a\u3044\u305f\u3081"
    return False, None, None


def build_signal_package(
    total_capital: float,
    stop_loss_rate: float,
    manual_mode: str | None = None,
    candidate_count: int = 5,
    min_signal_threshold: float = 1.0,
) -> SignalPackage:
    fetched_at = pd.Timestamp.now(tz="Asia/Tokyo")
    closes, volumes = fetch_market_data()

    us_closes = closes.reindex(columns=US_ETFS).dropna(how="all")
    jp_closes = closes.reindex(columns=JP_ETFS).dropna(how="all")
    jp_volumes = volumes.reindex(columns=JP_ETFS).dropna(how="all")
    vix_series = closes.get(VIX_TICKER, pd.Series(dtype=float))

    us_summary = _calculate_us_z_scores(us_closes)
    all_us_scores_negative = bool((us_summary["z_score"] < 0).all())

    ranking = _map_scores_to_japan(us_summary)
    ranking = _append_japan_market_data(ranking, jp_closes, jp_volumes)
    ranking, vix_value, damping_factor, risk_messages = _apply_vix_filter(ranking, vix_series)

    us_data_unavailable = _detect_us_data_unavailable(us_closes, fetched_at)
    market_mode, market_mode_label, long_budget, short_budget, buffer_budget = _resolve_market_mode(
        total_capital=total_capital,
        vix_value=vix_value,
        all_us_scores_negative=all_us_scores_negative,
        manual_mode=manual_mode,
    )

    long_pool, short_pool, long_candidates, short_candidates, skipped_candidates = _prepare_candidate_tables(
        ranking=ranking,
        candidate_count=candidate_count,
        min_signal_threshold=min_signal_threshold,
    )
    long_candidates = _allocate_side_budget(long_candidates, long_budget, total_capital, stop_loss_rate, side="long")
    short_candidates = _allocate_side_budget(short_candidates, short_budget, total_capital, stop_loss_rate, side="short")
    skipped_candidates = _decorate_skipped(skipped_candidates)

    low_volume_tickers = ranking.loc[ranking["low_volume_warning"], "ticker"].tolist()
    if low_volume_tickers:
        risk_messages.append("\u51fa\u6765\u9ad8\u6ce8\u610f: " + ", ".join(low_volume_tickers))

    skip_trading, skip_title, skip_reason = _evaluate_skip_trading(
        long_pool=long_pool,
        short_pool=short_pool,
        long_candidates=long_candidates,
        short_candidates=short_candidates,
        vix_value=vix_value,
        us_data_unavailable=us_data_unavailable,
        min_signal_threshold=min_signal_threshold,
    )

    latest_market_timestamp = closes.dropna(how="all").index.max()
    ranking_table = pd.concat([long_candidates, short_candidates, skipped_candidates], ignore_index=True)

    return SignalPackage(
        long_candidates=long_candidates,
        short_candidates=short_candidates,
        skipped_candidates=skipped_candidates,
        ranking_table=ranking_table,
        us_summary=us_summary,
        risk_messages=risk_messages,
        latest_market_timestamp=latest_market_timestamp,
        fetched_at=fetched_at,
        vix_value=vix_value,
        vix_damping_factor=damping_factor,
        all_us_scores_negative=all_us_scores_negative,
        market_mode=market_mode,
        market_mode_label=market_mode_label,
        skip_trading=skip_trading,
        skip_title=skip_title,
        skip_reason=skip_reason,
        us_data_unavailable=us_data_unavailable,
        long_budget=long_budget,
        short_budget=short_budget,
        buffer_budget=buffer_budget,
        qualified_long_count=len(long_candidates),
        qualified_short_count=len(short_candidates),
        min_signal_threshold=min_signal_threshold,
        candidate_count=candidate_count,
    )
