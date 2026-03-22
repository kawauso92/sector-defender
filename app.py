from __future__ import annotations

import os
import pandas as pd
import streamlit as st

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from btc_logic import (
    BtcSignalPackage,
    apply_monte_carlo_result,
    append_trade_history_row,
    build_btc_signal_package,
    calculate_recommended_position,
    close_open_trade,
    format_monte_carlo_sequence,
    load_trade_history,
    monte_carlo_fraction,
    summarize_trade_history,
)
from config import APP_TITLE, DEFAULT_CAPITAL, DEFAULT_STOP_LOSS
from logic import SignalPackage, build_signal_package
from notify import (
    build_btc_notification_message,
    build_vix_notification_message,
    send_line_notify,
)
from vix_logic import (
    VixSignalPackage,
    calculate_vix_recommended_position,
    calculate_vix_units,
    load_vix_signal_package as build_vix_signal_package,
)

LONG_LIMIT_MULTIPLIER = 0.999
SHORT_LIMIT_MULTIPLIER = 1.001

TEXT_SUBTITLE = "\u7c73\u56fd\u696d\u7a2eETF\u306e z-score \u3092\u65e5\u672c\u696d\u7a2eETF\u3078\u30de\u30c3\u30d4\u30f3\u30b0\u3057\u3001\u30ed\u30f3\u30b0\u30fb\u30b7\u30e7\u30fc\u30c8\u5019\u88dc\u3092\u4e00\u89a7\u5316\u3057\u307e\u3059\u3002"
TEXT_REFRESH = "\u30c7\u30fc\u30bf\u66f4\u65b0"
TEXT_DISPLAY_TIME = "\u8868\u793a\u65e5\u6642"
TEXT_SETTINGS = "\u8a2d\u5b9a"
TEXT_MANUAL_OVERRIDE = "\u624b\u52d5\u3067\u5207\u308a\u66ff\u3048\u308b"
TEXT_MANUAL_MODE = "\u624b\u52d5\u30e2\u30fc\u30c9"
TEXT_NORMAL_MODE = "\u901a\u5e38\u30e2\u30fc\u30c9"
TEXT_SHORT_PRIORITY_MODE = "\u30b7\u30e7\u30fc\u30c8\u512a\u5148\u30e2\u30fc\u30c9"
TEXT_CAPITAL = "\u7dcf\u8cc7\u91d1 (\u5186)"
TEXT_STOP_LOSS = "\u640d\u5207\u308a\u7387 (%)"
TEXT_CANDIDATE_COUNT = "\u5019\u88dc\u9298\u67c4\u6570\uff08\u30ed\u30f3\u30b0/\u30b7\u30e7\u30fc\u30c8\u305d\u308c\u305e\u308c\uff09"
TEXT_MIN_THRESHOLD = "\u6700\u4f4e\u30b7\u30b0\u30ca\u30eb\u95be\u5024"
TEXT_THRESHOLD_COUNT = "\u95be\u5024{threshold:.1f}\u4ee5\u4e0a: \u30ed\u30f3\u30b0{long_count}\u4ef6\u30fb\u30b7\u30e7\u30fc\u30c8{short_count}\u4ef6"
TEXT_LAST_UPDATE = "\u6700\u7d42\u30c7\u30fc\u30bf\u66f4\u65b0\u6642\u523b"
TEXT_MARKET_DATE = "\u5e02\u5834\u30c7\u30fc\u30bf\u65e5\u4ed8"
TEXT_LOADING = "\u5e02\u5834\u30c7\u30fc\u30bf\u3068\u30b7\u30b0\u30ca\u30eb\u3092\u8a08\u7b97\u3057\u3066\u3044\u307e\u3059..."
TEXT_LOAD_ERROR = "\u30c7\u30fc\u30bf\u53d6\u5f97\u307e\u305f\u306f\u30b7\u30b0\u30ca\u30eb\u8a08\u7b97\u306b\u5931\u6557\u3057\u307e\u3057\u305f"
TEXT_SKIP_INFO = "\u898b\u9001\u308a\u63a8\u5968\u65e5\u3067\u3082\u53c2\u8003\u60c5\u5831\u3068\u3057\u3066\u30c6\u30fc\u30d6\u30eb\u3092\u8868\u793a\u3057\u307e\u3059\u304c\u3001\u6ce8\u6587\u306f\u975e\u63a8\u5968\u3067\u3059\u3002"
TEXT_TOTAL_CAPITAL = "\u7dcf\u8cc7\u91d1"
TEXT_LONG_BUDGET = "\u30ed\u30f3\u30b0\u4e88\u7b97"
TEXT_SHORT_BUDGET = "\u30b7\u30e7\u30fc\u30c8\u4e88\u7b97"
TEXT_BUFFER = "\u30d0\u30c3\u30d5\u30a1"
TEXT_US_SUMMARY = "\u7c73\u56fd\u5e02\u5834\u30b5\u30de\u30ea\u30fc"
TEXT_STRONG_TOP3 = "\u5f37\u3044\u696d\u7a2e TOP3"
TEXT_WEAK_TOP3 = "\u5f31\u3044\u696d\u7a2e TOP3"
TEXT_BROAD_WEAKNESS = "\u26a0\ufe0f \u5168\u9762\u5b89\u5730\u5408\u3044"
TEXT_RISK = "\u30ea\u30b9\u30af\u8b66\u544a"
TEXT_NO_RISK = "\u73fe\u5728\u306e\u30ea\u30b9\u30af\u8b66\u544a\u306f\u3042\u308a\u307e\u305b\u3093\u3002"
TEXT_ORDER_PRIORITY = "\u6ce8\u6587\u512a\u5148\u9806\u4f4d"
TEXT_SCORE_DESC = (
    "\u4e88\u6e2c\u30b9\u30b3\u30a2\uff1a\u7c73\u56fd\u696d\u7a2eETF\u306e60\u65e5\u5e73\u5747\u306b\u5bfe\u3059\u308b\u6a19\u6e96\u504f\u5dee\u3002"
    " \u30d7\u30e9\u30b9\u304c\u5f37\u3044\uff08\u30ed\u30f3\u30b0\u5019\u88dc\uff09\u3001\u30de\u30a4\u30ca\u30b9\u304c\u5f31\u3044\uff08\u30b7\u30e7\u30fc\u30c8\u5019\u88dc\uff09\u3002"
    " \u4eca\u65e5\u306f\u5168\u9762\u5b89\u306e\u305f\u3081\u5168\u30b9\u30b3\u30a2\u304c\u30de\u30a4\u30ca\u30b9\u3067\u3059\u304c\u3001\u76f8\u5bfe\u7684\u306b\u5f37\u3044\u4e0a\u4f4d\u304c\u30ed\u30f3\u30b0\u5019\u88dc\u3067\u3059\u3002"
)
TEXT_DAY_ORDER = "\u6ce8\u6587\u306f\u5f53\u65e5\u9650\u308a\u6709\u52b9\uff08\u30c7\u30a4\u30aa\u30fc\u30c0\u30fc\uff09\u3067\u8a2d\u5b9a\u3057\u3066\u304f\u3060\u3055\u3044"
TEXT_CLOSE_EXIT = "\u5f15\u3051\u306f15\u664215\u5206\u301c25\u5206\u306b\u6210\u884c\u6c7a\u6e08\u63a8\u5968"
TEXT_LONG = "\u30ed\u30f3\u30b0\u5019\u88dc"
TEXT_SHORT = "\u30b7\u30e7\u30fc\u30c8\u5019\u88dc"
TEXT_SKIP_LIST = "\u898b\u9001\u308a\u4e00\u89a7"
TEXT_NO_LONG = "\u30ed\u30f3\u30b0\u5019\u88dc\u306f\u3042\u308a\u307e\u305b\u3093\u3002"
TEXT_NO_SHORT = "\u30b7\u30e7\u30fc\u30c8\u5019\u88dc\u306f\u3042\u308a\u307e\u305b\u3093\u3002"
TEXT_NO_SKIP = "\u898b\u9001\u308a\u9298\u67c4\u306f\u3042\u308a\u307e\u305b\u3093\u3002"
TEXT_RELATIVE_STRENGTH = "\u76f8\u5bfe\u7684\u5f37\uff08\u5168\u4f53\u5f31\u5730\u5408\u3044\uff09"
TEXT_STATUS_ORDERABLE = "\u6ce8\u6587\u53ef"
TEXT_STATUS_OVER_BUDGET = "\u4e88\u7b97\u8d85\u904e"
TEXT_STATUS_INSUFFICIENT = "\u8cc7\u91d1\u4e0d\u8db3"
TEXT_STATUS_BELOW_THRESHOLD = "\u95be\u5024\u672a\u6e80"
TEXT_STATUS_SKIP = "\u898b\u9001\u308a"
TEXT_STATUS_DIVIDER = "\u533a\u5207\u308a"
TEXT_DIRECTION_LONG = "\U0001F7E2\u30ed\u30f3\u30b0"
TEXT_DIRECTION_SHORT = "\U0001F534\u30b7\u30e7\u30fc\u30c8"
TEXT_DIRECTION_SKIP = "\u26aa\u898b\u9001\u308a"
TEXT_PAGE_SELECT = "\u30da\u30fc\u30b8\u9078\u629e"
TEXT_PAGE_ETF = "\u65e5\u7c73ETF\u30ea\u30fc\u30c9\u30e9\u30b0\u30b7\u30b0\u30ca\u30eb"
TEXT_PAGE_BTC = "BTC\u6025\u843d\u30ea\u30d0\u30a6\u30f3\u30c9\u30b7\u30b0\u30ca\u30eb"
TEXT_PAGE_VIX = "VIX\u6025\u843d\u30ea\u30d0\u30a6\u30f3\u30c9\u30b7\u30b0\u30ca\u30eb\uff082558\u5bfe\u8c61\uff09"
TEXT_BTC_SUBTITLE = "BTC/USDT\u306e1\u6642\u9593\u8db3\u3067-3%\u4ee5\u4e0a\u306e\u6025\u843d\u3092\u691c\u77e5\u3057\u3001E1\u6307\u5024\u306e\u9006\u5f35\u308a\u6761\u4ef6\u3092\u8868\u793a\u3057\u307e\u3059\u3002"
TEXT_BTC_SIGNAL = "\U0001F534 BTC\u6025\u843d\u30b7\u30b0\u30ca\u30eb\u767a\u751f"
TEXT_BTC_NO_SIGNAL = "\u2705 \u73fe\u5728\u30b7\u30b0\u30ca\u30eb\u306a\u3057"
TEXT_VIX_SUBTITLE = "C4\u6761\u4ef6\uff08VIX\u524d\u65e5\u6bd4+10%\u4ee5\u4e0a\u304b\u3064VIX 25\u4ee5\u4e0a\uff09\u3067 2558.T \u306e\u9006\u5f35\u308a\u30b7\u30b0\u30ca\u30eb\u3092\u8868\u793a\u3057\u307e\u3059\u3002"
TEXT_VIX_SIGNAL = "\U0001F534 VIX\u30b7\u30b0\u30ca\u30eb\u767a\u751f"
TEXT_VIX_NO_SIGNAL = "\u2705 \u73fe\u5728\u30b7\u30b0\u30ca\u30eb\u306a\u3057"
TEXT_NOTIFY = "\u901a\u77e5\u8a2d\u5b9a"
TEXT_NOTIFY_MISSING = ".env\u306bLINE_NOTIFY_TOKEN\u3092\u8a2d\u5b9a\u3057\u3066\u304f\u3060\u3055\u3044"


st.set_page_config(page_title=APP_TITLE, layout="wide")


@st.cache_data(ttl=300, show_spinner=False)
def load_signal_package(
    total_capital: float,
    stop_loss_rate: float,
    manual_mode: str | None,
    candidate_count: int,
    min_signal_threshold: float,
) -> SignalPackage:
    return build_signal_package(
        total_capital=total_capital,
        stop_loss_rate=stop_loss_rate,
        manual_mode=manual_mode,
        candidate_count=candidate_count,
        min_signal_threshold=min_signal_threshold,
    )


@st.cache_data(ttl=300, show_spinner=False)
def load_btc_signal_package() -> BtcSignalPackage:
    return build_btc_signal_package()


@st.cache_data(ttl=300, show_spinner=False)
def load_vix_signal_page_package() -> VixSignalPackage:
    return build_vix_signal_package()


def format_currency(value: float) -> str:
    return f"\u00a5{value:,.0f}"


def format_price(value: float) -> str:
    return f"\u00a5{value:,.2f}"


def status_text(code: str) -> str:
    mapping = {
        "orderable": TEXT_STATUS_ORDERABLE,
        "over_budget": TEXT_STATUS_OVER_BUDGET,
        "insufficient_funds": TEXT_STATUS_INSUFFICIENT,
        "below_threshold": TEXT_STATUS_BELOW_THRESHOLD,
        "skip": TEXT_STATUS_SKIP,
        "candidate": "\u5019\u88dc",
    }
    return mapping.get(code, code)


def direction_text(code: str) -> str:
    mapping = {
        "long": TEXT_DIRECTION_LONG,
        "short": TEXT_DIRECTION_SHORT,
        "skip": TEXT_DIRECTION_SKIP,
    }
    return mapping.get(code, TEXT_DIRECTION_SKIP)


def format_units(units: int, budget_code: str) -> str:
    if budget_code != "orderable":
        return "-"
    return f"{units:,d}"


def format_amount(amount: float, budget_code: str) -> str:
    if budget_code != "orderable":
        return "-"
    return format_currency(amount)


def format_optional_price(value: float, budget_code: str) -> str:
    if budget_code != "orderable" or pd.isna(value):
        return "-"
    return format_price(value)


def format_limit_price(reference_close: float, direction_code: str, budget_code: str) -> str:
    if budget_code != "orderable" or pd.isna(reference_close):
        return "-"
    multiplier = LONG_LIMIT_MULTIPLIER if direction_code == "long" else SHORT_LIMIT_MULTIPLIER
    return format_price(reference_close * multiplier)


def insert_cutoff_marker(table: pd.DataFrame) -> pd.DataFrame:
    orderable = table["\u72b6\u614b"] == TEXT_STATUS_ORDERABLE
    if not orderable.any():
        return table

    cutoff_index = orderable[orderable].index[-1]
    marker = {column: "" for column in table.columns}
    marker["\u9298\u67c4"] = "\u2191\u3053\u3053\u307e\u3067\u6ce8\u6587\u63a8\u5968"
    marker["\u72b6\u614b"] = TEXT_STATUS_DIVIDER

    upper = table.iloc[: cutoff_index + 1]
    lower = table.iloc[cutoff_index + 1 :]
    return pd.concat([upper, pd.DataFrame([marker]), lower], ignore_index=True)


def prepare_candidate_table(df: pd.DataFrame, all_us_scores_negative: bool) -> pd.DataFrame:
    table = df.copy()
    table["direction_label"] = table["direction_code"].map(direction_text)
    table["score_label"] = table["signal_score"].map(lambda x: f"{x:.2f}")
    table["note"] = ""
    table.loc[(table["direction_code"] == "long") & all_us_scores_negative, "note"] = TEXT_RELATIVE_STRENGTH
    table["price_label"] = table["current_price"].map(format_price)
    table["allocation_label"] = table["allocation_ratio"].map(lambda x: f"{x * 100:.1f}%")
    table["amount_label"] = table.apply(lambda row: format_amount(row["recommended_amount"], row["budget_code"]), axis=1)
    table["units_label"] = table.apply(lambda row: format_units(int(row["recommended_units"]), row["budget_code"]), axis=1)
    table["limit_label"] = table.apply(
        lambda row: format_limit_price(row["reference_close"], row["direction_code"], row["budget_code"]),
        axis=1,
    )
    table["stop_label"] = table.apply(
        lambda row: format_optional_price(row["stop_loss_line"], row["budget_code"]),
        axis=1,
    )
    table["take_label"] = table.apply(
        lambda row: format_optional_price(row["take_profit_line"], row["budget_code"]),
        axis=1,
    )
    table["status_label"] = table["budget_code"].map(status_text)

    display = table[
        [
            "ticker",
            "sector",
            "direction_label",
            "score_label",
            "note",
            "price_label",
            "allocation_label",
            "amount_label",
            "units_label",
            "limit_label",
            "status_label",
            "stop_label",
            "take_label",
            "volume_warning",
        ]
    ].rename(
        columns={
            "ticker": "\u9298\u67c4",
            "sector": "\u696d\u7a2e",
            "direction_label": "\u65b9\u5411",
            "score_label": "\u4e88\u6e2c\u30b9\u30b3\u30a2",
            "note": "\u6ce8\u91c8",
            "price_label": "\u73fe\u5728\u4fa1\u683c",
            "allocation_label": "\u914d\u5206\u6bd4\u7387",
            "amount_label": "\u63a8\u5968\u91d1\u984d",
            "units_label": "\u63a8\u5968\u53e3\u6570",
            "limit_label": "\u6307\u5024\u4fa1\u683c",
            "status_label": "\u72b6\u614b",
            "stop_label": "\u640d\u5207\u308a\u30e9\u30a4\u30f3",
            "take_label": "\u5229\u78ba\u76ee\u5b89(+3%)",
            "volume_warning": "\u51fa\u6765\u9ad8\u8b66\u544a",
        }
    )
    return insert_cutoff_marker(display)


def prepare_skipped_table(df: pd.DataFrame) -> pd.DataFrame:
    table = df.copy()
    table["direction_label"] = table["direction_code"].map(direction_text)
    table["score_label"] = table["signal_score"].map(lambda x: f"{x:.2f}")
    table["price_label"] = table["current_price"].map(format_price)
    table["reason_label"] = table["budget_code"].map(status_text)
    return table[
        ["ticker", "sector", "direction_label", "score_label", "price_label", "reason_label", "volume_warning"]
    ].rename(
        columns={
            "ticker": "\u9298\u67c4",
            "sector": "\u696d\u7a2e",
            "direction_label": "\u65b9\u5411",
            "score_label": "\u4e88\u6e2c\u30b9\u30b3\u30a2",
            "price_label": "\u73fe\u5728\u4fa1\u683c",
            "reason_label": "\u7406\u7531",
            "volume_warning": "\u51fa\u6765\u9ad8\u8b66\u544a",
        }
    )


def style_candidate_table(df: pd.DataFrame, direction: str, skip_trading: bool):
    def row_style(row: pd.Series) -> list[str]:
        if skip_trading:
            return ["color: #7a7a7a; background-color: #f3f4f6"] * len(row)
        if row["\u72b6\u614b"] in {
            TEXT_STATUS_OVER_BUDGET,
            TEXT_STATUS_INSUFFICIENT,
            TEXT_STATUS_BELOW_THRESHOLD,
            TEXT_STATUS_SKIP,
        }:
            return ["color: #7a7a7a; background-color: #f3f4f6"] * len(row)
        if row["\u72b6\u614b"] == TEXT_STATUS_DIVIDER:
            return ["font-weight: 700; border-top: 2px solid #6b7280; background-color: #ffffff"] * len(row)
        if row["\u72b6\u614b"] == TEXT_STATUS_ORDERABLE and direction == "long":
            return ["background-color: #dcfce7; color: #166534"] * len(row)
        if row["\u72b6\u614b"] == TEXT_STATUS_ORDERABLE and direction == "short":
            return ["background-color: #fee2e2; color: #991b1b"] * len(row)
        return [""] * len(row)

    return df.style.apply(row_style, axis=1)


def style_skipped_table(df: pd.DataFrame, skip_trading: bool):
    background = "#f3f4f6" if skip_trading else "#f9fafb"
    return df.style.apply(lambda row: [f"color: #7a7a7a; background-color: {background}"] * len(row), axis=1)


def render_market_summary(package: SignalPackage) -> None:
    strongest = package.us_summary.head(3)
    weakest = package.us_summary.tail(3).sort_values("z_score")

    if package.all_us_scores_negative:
        st.warning(TEXT_BROAD_WEAKNESS)

    left, right = st.columns(2)
    with left:
        st.subheader(TEXT_US_SUMMARY)
        st.markdown(f"**{TEXT_STRONG_TOP3}**")
        for row in strongest.itertuples(index=False):
            st.write(f"{row.sector} ({row.ticker}) : {row.z_score:.2f}")

    with right:
        st.subheader(TEXT_WEAK_TOP3)
        for row in weakest.itertuples(index=False):
            st.write(f"{row.sector} ({row.ticker}) : {row.z_score:.2f}")


def format_usdt(value: float) -> str:
    return f"{value:,.2f} USDT"


def format_fx_rate(value: float) -> str:
    return f"¥{value:,.2f}/USD"


def resolve_line_notify_token() -> str:
    token = os.environ.get("LINE_NOTIFY_TOKEN", "").strip()
    if token:
        return token
    try:
        return str(st.secrets.get("LINE_NOTIFY_TOKEN", "")).strip()
    except Exception:
        return ""


def render_notification_settings() -> tuple[str, dict[str, bool]]:
    st.sidebar.markdown("---")
    st.sidebar.subheader(TEXT_NOTIFY)
    btc_notify = st.sidebar.checkbox("BTCシグナル通知", value=False)
    vix_notify = st.sidebar.checkbox("VIXシグナル通知", value=False)
    etf_notify = st.sidebar.checkbox("ETFリードラグ通知", value=False)
    token = resolve_line_notify_token()
    if not token:
        st.sidebar.info(TEXT_NOTIFY_MISSING)
    return token, {"btc": btc_notify, "vix": vix_notify, "etf": etf_notify}


def maybe_send_notification(session_key: str, enabled: bool, token: str, dedupe_key: str, message: str) -> None:
    if not enabled or not token:
        return
    if st.session_state.get(session_key) == dedupe_key:
        return
    sent, response_message = send_line_notify(token, message)
    if sent:
        st.session_state[session_key] = dedupe_key
        st.sidebar.success(response_message)
    else:
        st.sidebar.warning(response_message)


def render_etf_page() -> None:
    st.title(APP_TITLE)
    st.caption(TEXT_SUBTITLE)

    refresh_col, meta_col = st.columns([1, 3])
    with refresh_col:
        if st.button(TEXT_REFRESH, key="refresh_etf"):
            st.cache_data.clear()
            st.rerun()
    with meta_col:
        st.write(f"{TEXT_DISPLAY_TIME}: {pd.Timestamp.now(tz='Asia/Tokyo').strftime('%Y-%m-%d %H:%M JST')}")

    st.sidebar.header(TEXT_SETTINGS)
    mode_placeholder = st.sidebar.empty()
    manual_override = st.sidebar.checkbox(TEXT_MANUAL_OVERRIDE)
    manual_mode_label = st.sidebar.selectbox(
        TEXT_MANUAL_MODE,
        options=[TEXT_NORMAL_MODE, TEXT_SHORT_PRIORITY_MODE],
        index=0,
        disabled=not manual_override,
    )
    total_capital = st.sidebar.number_input(TEXT_CAPITAL, min_value=10000, step=10000, value=DEFAULT_CAPITAL)
    stop_loss_rate_pct = st.sidebar.slider(TEXT_STOP_LOSS, min_value=0.5, max_value=10.0, step=0.5, value=DEFAULT_STOP_LOSS * 100)
    candidate_count = st.sidebar.slider(TEXT_CANDIDATE_COUNT, min_value=1, max_value=8, value=5)
    min_signal_threshold = st.sidebar.slider(TEXT_MIN_THRESHOLD, min_value=0.0, max_value=2.0, step=0.1, value=1.0)
    count_placeholder = st.sidebar.empty()
    updated_placeholder = st.sidebar.empty()
    render_notification_settings()

    manual_mode = None
    if manual_override:
        manual_mode = "short_priority" if manual_mode_label == TEXT_SHORT_PRIORITY_MODE else "normal"

    try:
        with st.spinner(TEXT_LOADING):
            package = load_signal_package(
                total_capital=float(total_capital),
                stop_loss_rate=stop_loss_rate_pct / 100,
                manual_mode=manual_mode,
                candidate_count=int(candidate_count),
                min_signal_threshold=float(min_signal_threshold),
            )
    except Exception as exc:
        st.error(f"{TEXT_LOAD_ERROR}: {exc}")
        st.stop()

    mode_placeholder.caption(package.market_mode_label)
    count_placeholder.caption(
        TEXT_THRESHOLD_COUNT.format(
            threshold=package.min_signal_threshold,
            long_count=package.qualified_long_count,
            short_count=package.qualified_short_count,
        )
    )
    updated_placeholder.caption(
        f"{TEXT_LAST_UPDATE}: {package.fetched_at.strftime('%Y-%m-%d %H:%M:%S JST')} / "
        f"{TEXT_MARKET_DATE} {package.latest_market_timestamp.strftime('%Y-%m-%d')}"
    )

    if package.skip_trading:
        st.markdown(
            f"""
            <div style="padding: 16px; border-radius: 10px; background: #fff1f2; border: 1px solid #fca5a5; color: #991b1b; font-size: 28px; font-weight: 700; text-align: center; margin: 12px 0;">
                {package.skip_title}
            </div>
            """,
            unsafe_allow_html=True,
        )
        if package.skip_reason:
            st.warning(package.skip_reason)
        st.caption(TEXT_SKIP_INFO)

    budget_col1, budget_col2, budget_col3, budget_col4 = st.columns(4)
    budget_col1.metric(TEXT_TOTAL_CAPITAL, format_currency(float(total_capital)))
    budget_col2.metric(TEXT_LONG_BUDGET, format_currency(package.long_budget))
    budget_col3.metric(TEXT_SHORT_BUDGET, format_currency(package.short_budget))
    budget_col4.metric(TEXT_BUFFER, format_currency(package.buffer_budget))

    render_market_summary(package)

    st.subheader(TEXT_RISK)
    if package.risk_messages:
        for message in package.risk_messages:
            if "VIX" in message or "\u51fa\u6765\u9ad8" in message:
                st.warning(message)
            else:
                st.info(message)
    else:
        st.success(TEXT_NO_RISK)

    st.subheader(TEXT_ORDER_PRIORITY)
    st.caption(TEXT_SCORE_DESC)
    st.caption(TEXT_DAY_ORDER)
    st.caption(TEXT_CLOSE_EXIT)

    st.markdown(f"**{TEXT_LONG}**")
    if package.long_candidates.empty:
        st.info(TEXT_NO_LONG)
    else:
        long_table = prepare_candidate_table(package.long_candidates, package.all_us_scores_negative)
        st.dataframe(style_candidate_table(long_table, "long", package.skip_trading), use_container_width=True, hide_index=True)

    st.markdown(f"**{TEXT_SHORT}**")
    if package.short_candidates.empty:
        st.info(TEXT_NO_SHORT)
    else:
        short_table = prepare_candidate_table(package.short_candidates, package.all_us_scores_negative)
        st.dataframe(style_candidate_table(short_table, "short", package.skip_trading), use_container_width=True, hide_index=True)

    with st.expander(TEXT_SKIP_LIST, expanded=False):
        if package.skipped_candidates.empty:
            st.caption(TEXT_NO_SKIP)
        else:
            skipped_table = prepare_skipped_table(package.skipped_candidates)
            st.dataframe(style_skipped_table(skipped_table, package.skip_trading), use_container_width=True, hide_index=True)


def render_btc_page() -> None:
    st.title(TEXT_PAGE_BTC)
    st.caption(TEXT_BTC_SUBTITLE)

    refresh_col, meta_col = st.columns([1, 3])
    with refresh_col:
        if st.button(TEXT_REFRESH, key="refresh_btc"):
            st.cache_data.clear()
            st.rerun()
    with meta_col:
        st.write(f"{TEXT_DISPLAY_TIME}: {pd.Timestamp.now(tz='Asia/Tokyo').strftime('%Y-%m-%d %H:%M JST')}")

    st.sidebar.header("設定")
    collateral_jpy = st.sidebar.number_input("証拠金入力 (円)", min_value=10000, step=10000, value=100000)
    leverage = st.sidebar.selectbox("レバレッジ", options=[5, 7, 10], index=2)
    management_label = st.sidebar.selectbox("資金管理方式", options=["固定20%", "分解モンテカルロ法"], index=1)

    if "btc_monte_carlo_sequence" not in st.session_state:
        st.session_state["btc_monte_carlo_sequence"] = [1, 2, 3]
    if "btc_trade_action" not in st.session_state:
        st.session_state["btc_trade_action"] = ""

    button_col1, button_col2 = st.sidebar.columns(2)
    if button_col1.button("勝ち", key="btc_monte_win"):
        st.session_state["btc_monte_carlo_sequence"] = apply_monte_carlo_result(st.session_state["btc_monte_carlo_sequence"], True)
    if button_col2.button("負け", key="btc_monte_loss"):
        st.session_state["btc_monte_carlo_sequence"] = apply_monte_carlo_result(st.session_state["btc_monte_carlo_sequence"], False)
    st.sidebar.caption("モンテカルロリスト")
    st.sidebar.code(format_monte_carlo_sequence(st.session_state["btc_monte_carlo_sequence"]))
    token, notify_flags = render_notification_settings()

    try:
        with st.spinner("BTC 1時間足とシグナルを確認しています..."):
            package = load_btc_signal_package()
    except Exception as exc:
        st.error(f"BTCデータ取得に失敗しました: {exc}")
        st.stop()

    history = load_trade_history()
    open_positions = history[history["status"] == "open"].copy()
    current_signal_recorded = False
    if not history.empty:
        current_signal_recorded = history["signal_date"].eq(package.latest_timestamp).fillna(False).any()

    management_code = "fixed" if management_label == "固定20%" else "monte_carlo"
    position_plan = calculate_recommended_position(
        collateral_jpy=float(collateral_jpy),
        leverage=int(leverage),
        management=management_code,
        monte_carlo_sequence=st.session_state["btc_monte_carlo_sequence"],
        usd_jpy_rate=package.usd_jpy_rate,
    )

    fx_label = format_fx_rate(package.usd_jpy_rate)
    fx_time = package.usd_jpy_timestamp.tz_convert("Asia/Tokyo").strftime("%Y-%m-%d %H:%M JST")
    fallback_note = "（フォールバック値）" if package.usd_jpy_fallback_used else ""
    st.caption(
        f"最終BTCデータ: {package.latest_timestamp.tz_convert('Asia/Tokyo').strftime('%Y-%m-%d %H:%M JST')}"
        f" / キャッシュ更新: {'Yes' if package.cache_updated else 'No'}"
    )
    st.caption(f"為替レート：{fx_label}（{fx_time}）{fallback_note}")

    header_col1, header_col2 = st.columns(2)
    header_col1.metric("現在のBTC価格", f"${package.current_price:,.2f}")
    header_col2.metric("直近1時間の変化率", f"{package.latest_change_pct:+.2f}%")

    st.sidebar.caption(f"現在の投入比率: {position_plan['fraction'] * 100:.1f}%")
    if management_code == "monte_carlo":
        st.sidebar.caption(f"モンテカルロ比率: {monte_carlo_fraction(st.session_state['btc_monte_carlo_sequence']) * 100:.1f}%")

    if package.signal_active:
        maybe_send_notification(
            session_key="btc_last_notified_signal",
            enabled=notify_flags["btc"],
            token=token,
            dedupe_key=f"{package.latest_timestamp.isoformat()}_{package.signal_drop_pct:.2f}",
            message=build_btc_notification_message(
                drop_pct=package.signal_drop_pct,
                limit_price=package.limit_price,
                margin_jpy=position_plan["margin_jpy"],
                leverage=int(leverage),
            ),
        )
        st.markdown(
            "\n".join(
                [
                    "━━━━━━━━━━━━━━━━",
                    "🔴 BTCシグナル発生",
                    "━━━━━━━━━━━━━━━━",
                    f"現在価格：${package.current_price:,.2f}（¥{package.current_price * package.usd_jpy_rate:,.0f}）",
                    "",
                    "【注文指示】",
                    "指値注文（Limit Buy）",
                    f"注文価格：${package.limit_price:,.2f}（現在価格×0.997）",
                    "有効期限：2時間以内にキャンセル",
                    "",
                    "【決済ライン】",
                    f"利確（Take Profit）：${package.take_profit_price:,.2f}（+1.4%）",
                    f"損切（Stop Loss）：${package.stop_loss_price:,.2f}（-1.4%）",
                    "保有上限：1時間",
                    "",
                    "【推奨ロット】",
                    f"証拠金使用額：¥{position_plan['margin_jpy']:,.0f}",
                    f"ポジションサイズ：{position_plan['position_usdt']:,.2f} USDT",
                    f"（為替レート：{fx_label}）",
                    "━━━━━━━━━━━━━━━━",
                ]
            )
        )
    else:
        st.success(TEXT_BTC_NO_SIGNAL)
        st.markdown("**直近の急落履歴（過去5件）**")
        recent_drops = package.recent_drops.copy()
        recent_drops["急落時刻"] = pd.to_datetime(recent_drops["急落時刻"], utc=True).dt.tz_convert("Asia/Tokyo").dt.strftime("%Y-%m-%d %H:%M")
        recent_drops["終値(USDT)"] = recent_drops["終値(USDT)"].map(lambda value: f"{value:,.2f}")
        recent_drops["1時間変化率"] = recent_drops["1時間変化率"].map(lambda value: f"{value:.2f}%")
        st.dataframe(recent_drops, use_container_width=True, hide_index=True)

    st.subheader("推奨ロット")
    lot_col1, lot_col2 = st.columns(2)
    lot_col1.metric("推奨ポジションサイズ（USDT換算）", format_usdt(position_plan["position_usdt"]))
    lot_col2.metric("推奨証拠金使用額（円）", format_currency(position_plan["margin_jpy"]))

    st.subheader("売買記録")
    action_col1, action_col2, action_col3 = st.columns(3)
    if action_col1.button("✅ 約定した", disabled=(not package.signal_active) or current_signal_recorded, key="btc_fill_button"):
        st.session_state["btc_trade_action"] = "fill"
    if action_col2.button("❌ 見送り", disabled=(not package.signal_active) or current_signal_recorded, key="btc_skip_button"):
        append_trade_history_row(
            {
                "signal_date": package.latest_timestamp,
                "signal_price": package.current_price,
                "entry_type": "E1_limit_pullback",
                "take_profit": package.take_profit_price,
                "stop_loss": package.stop_loss_price,
                "status": "skipped",
            }
        )
        st.session_state["btc_trade_action"] = ""
        st.success("見送りを記録しました。")
        st.rerun()
    if action_col3.button("⏰ 期限切れ", disabled=(not package.signal_active) or current_signal_recorded, key="btc_expired_button"):
        append_trade_history_row(
            {
                "signal_date": package.latest_timestamp,
                "signal_price": package.current_price,
                "entry_type": "E1_limit_pullback",
                "take_profit": package.take_profit_price,
                "stop_loss": package.stop_loss_price,
                "status": "expired",
            }
        )
        st.session_state["btc_trade_action"] = ""
        st.success("期限切れを記録しました。")
        st.rerun()

    if st.session_state.get("btc_trade_action") == "fill":
        with st.form("btc_entry_form", clear_on_submit=False):
            entry_price = st.number_input("約定価格（USDT）", min_value=0.0, value=float(package.limit_price), step=10.0)
            entry_size_usdt = st.number_input("約定数量（USDT）", min_value=0.0, value=float(position_plan["position_usdt"]), step=100.0)
            entry_leverage = st.number_input("レバレッジ", min_value=1, max_value=50, value=int(leverage), step=1)
            submitted = st.form_submit_button("記録する")
            if submitted:
                entry_jpy = (float(entry_size_usdt) / max(float(entry_leverage), 1.0)) * package.usd_jpy_rate
                append_trade_history_row(
                    {
                        "signal_date": package.latest_timestamp,
                        "signal_price": package.current_price,
                        "entry_type": "E1_limit_pullback",
                        "entry_price": float(entry_price),
                        "entry_size_usdt": float(entry_size_usdt),
                        "leverage": float(entry_leverage),
                        "entry_jpy": entry_jpy,
                        "take_profit": package.take_profit_price,
                        "stop_loss": package.stop_loss_price,
                        "status": "open",
                    }
                )
                st.session_state["btc_trade_action"] = ""
                st.success("約定を記録しました。")
                st.rerun()

    if not open_positions.empty:
        latest_open = open_positions.iloc[-1]
        st.markdown("**保有中ポジションの決済記録**")
        st.caption(
            f"建値 ${float(latest_open['entry_price']):,.2f} / サイズ {float(latest_open['entry_size_usdt']):,.2f} USDT / "
            f"レバ {int(float(latest_open['leverage']))}倍"
        )
        with st.form("btc_exit_form", clear_on_submit=False):
            exit_price = st.number_input("決済価格入力", min_value=0.0, value=float(package.current_price), step=10.0)
            exit_reason = st.selectbox("決済理由", options=["利確", "損切り", "強制決済", "手動"], index=0)
            exit_submitted = st.form_submit_button("決済記録")
            if exit_submitted:
                close_open_trade(float(exit_price), str(exit_reason), package.usd_jpy_rate)
                st.success("決済を記録しました。")
                st.rerun()

    history = load_trade_history()
    summary_stats = summarize_trade_history(history)
    st.subheader("売買履歴")
    history_col1, history_col2, history_col3, history_col4 = st.columns(4)
    history_col1.metric("合計損益（円）", format_currency(summary_stats["total_pnl_jpy"]))
    history_col2.metric("合計損益（USDT）", format_usdt(summary_stats["total_pnl_usdt"]))
    history_col3.metric("勝率", f"{summary_stats['win_rate']:.2%}")
    history_col4.metric("平均損益（円）", format_currency(summary_stats["avg_pnl_jpy"]))

    history_display = history.sort_values("signal_date", ascending=False).head(10).copy()
    if not history_display.empty:
        history_display["signal_date"] = pd.to_datetime(history_display["signal_date"], utc=True, errors="coerce").dt.tz_convert("Asia/Tokyo").dt.strftime("%Y-%m-%d %H:%M")
        for column in ["signal_price", "entry_price", "entry_size_usdt", "entry_jpy", "take_profit", "stop_loss", "exit_price", "pnl_usdt", "pnl_jpy"]:
            history_display[column] = history_display[column].map(lambda value: f"{value:,.2f}" if pd.notna(value) else "-")
        history_display["pnl_pct"] = history_display["pnl_pct"].map(lambda value: f"{value:.2%}" if pd.notna(value) else "-")
        st.dataframe(history_display, use_container_width=True, hide_index=True)
        st.download_button(
            "CSVエクスポート",
            data=history.to_csv(index=False, encoding="utf-8-sig"),
            file_name="trade_history.csv",
            mime="text/csv",
        )
    else:
        st.info("売買履歴はまだありません。")

    st.subheader("バックテスト結果サマリー")
    summary = package.backtest_summary
    st.markdown("戦略概要：-3%急落後リバウンド狙い")
    st.markdown(
        f"実績（2020-2025）：年率{summary.annual_return_net * 100:.1f}%"
        f"・勝率{summary.win_rate * 100:.1f}%・最大連敗{summary.max_losing_streak}回"
    )


def render_vix_page() -> None:
    st.title(TEXT_PAGE_VIX)
    st.caption(TEXT_VIX_SUBTITLE)

    refresh_col, meta_col = st.columns([1, 3])
    with refresh_col:
        if st.button(TEXT_REFRESH, key="refresh_vix"):
            st.cache_data.clear()
            st.rerun()
    with meta_col:
        st.write(f"{TEXT_DISPLAY_TIME}: {pd.Timestamp.now(tz='Asia/Tokyo').strftime('%Y-%m-%d %H:%M JST')}")

    st.sidebar.header(TEXT_SETTINGS)
    collateral_jpy = st.sidebar.number_input("証拠金入力 (円)", min_value=10000, step=10000, value=100000, key="vix_collateral")
    st.sidebar.selectbox("資金管理方式", options=["ケリー基準"], index=0, key="vix_mgmt")
    is_holding = st.sidebar.checkbox("現在保有中", value=False, key="vix_holding")
    token, notify_flags = render_notification_settings()

    try:
        with st.spinner("VIX と 2558.T の最新データを取得しています..."):
            package = load_vix_signal_page_package()
    except Exception as exc:
        st.error(f"VIXデータ取得に失敗しました: {exc}")
        st.stop()

    recommendation = calculate_vix_recommended_position(float(collateral_jpy), bool(is_holding))
    recommended_units = calculate_vix_units(recommendation["amount_jpy"], package.limit_price)

    st.caption(
        f"最終VIXデータ: {package.vix_date.tz_convert('Asia/Tokyo').strftime('%Y-%m-%d %H:%M JST')}"
    )
    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("現在のVIX値", f"{package.current_vix:.2f}")
    metric_col2.metric("前日比変化率", f"{package.vix_change_pct:+.2f}%")

    if package.signal_active:
        maybe_send_notification(
            session_key="vix_last_notified_signal",
            enabled=notify_flags["vix"],
            token=token,
            dedupe_key=f"{package.vix_date.isoformat()}_{package.current_vix:.2f}_{package.vix_change_pct:.2f}",
            message=build_vix_notification_message(
                vix_value=package.current_vix,
                vix_change_pct=package.vix_change_pct,
                limit_price_jpy=package.limit_price,
                amount_jpy=recommendation["amount_jpy"],
            ),
        )
        st.error(TEXT_VIX_SIGNAL)
        st.markdown(f"**VIX現在値**: {package.current_vix:.2f}")
        st.markdown(f"**前日比**: {package.vix_change_pct:+.2f}%")
        st.markdown(f"**2558.T指値価格**: {format_price(package.limit_price)}")
        st.markdown("**利確**: VIXが20以下に戻った翌日決済")
        st.markdown("**損切り**: なし")
    else:
        st.success(TEXT_VIX_NO_SIGNAL)
        st.markdown("**直近5営業日のVIX値**")
        st.dataframe(package.recent_vix, use_container_width=True, hide_index=True)

    st.subheader("推奨ロット")
    lot_col1, lot_col2 = st.columns(2)
    lot_col1.metric("推奨金額", format_currency(recommendation["amount_jpy"]))
    lot_col2.metric("推奨口数", f"{recommended_units:,d} 口")
    st.caption(f"ケリー基準投入比率: {package.backtest_summary.kelly_fraction * 100:.1f}%")

    st.subheader("バックテスト結果サマリー")
    st.markdown("戦略：C4+L5×S0+ケリー基準")
    st.markdown(
        f"実績：年率{package.backtest_summary.annual_return * 100:.2f}%"
        f"・勝率{package.backtest_summary.win_rate * 100:.1f}%"
        f"・最大DD{package.backtest_summary.max_drawdown * 100:.2f}%"
    )


page = st.sidebar.radio(TEXT_PAGE_SELECT, options=[TEXT_PAGE_ETF, TEXT_PAGE_BTC, TEXT_PAGE_VIX], index=0)

if page == TEXT_PAGE_ETF:
    render_etf_page()
elif page == TEXT_PAGE_BTC:
    render_btc_page()
else:
    render_vix_page()
