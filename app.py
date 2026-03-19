from __future__ import annotations

from dotenv import load_dotenv
import pandas as pd
import streamlit as st

load_dotenv()

from config import APP_TITLE, DEFAULT_CAPITAL, DEFAULT_STOP_LOSS
from logic import SignalPackage, build_signal_package

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


st.title(APP_TITLE)
st.caption(TEXT_SUBTITLE)

refresh_col, meta_col = st.columns([1, 3])
with refresh_col:
    if st.button(TEXT_REFRESH):
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
