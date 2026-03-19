from __future__ import annotations

import os

APP_TITLE = "Sector Defender"

US_ETFS = [
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

JP_ETFS = [
    "1617.T",
    "1618.T",
    "1619.T",
    "1620.T",
    "1621.T",
    "1622.T",
    "1623.T",
    "1624.T",
    "1625.T",
    "1626.T",
    "1627.T",
    "1628.T",
    "1629.T",
    "1630.T",
    "1631.T",
    "1632.T",
    "1633.T",
]

US_JP_MAP = {
    "XLK": ["1625.T", "1626.T"],
    "XLF": ["1631.T", "1632.T"],
    "XLE": ["1618.T"],
    "XLB": ["1620.T", "1623.T"],
    "XLI": ["1619.T", "1624.T"],
    "XLY": ["1630.T"],
    "XLP": ["1617.T"],
    "XLV": ["1621.T"],
    "XLU": ["1627.T"],
    "XLC": ["1626.T"],
    "XLRE": ["1633.T"],
}

US_ETF_LABELS = {
    "XLB": "\u7d20\u6750",
    "XLC": "\u901a\u4fe1",
    "XLE": "\u30a8\u30cd\u30eb\u30ae\u30fc",
    "XLF": "\u91d1\u878d",
    "XLI": "\u8cc7\u672c\u8ca1",
    "XLK": "\u30c6\u30af\u30ce\u30ed\u30b8\u30fc",
    "XLP": "\u751f\u6d3b\u5fc5\u9700\u54c1",
    "XLRE": "\u4e0d\u52d5\u7523",
    "XLU": "\u516c\u76ca",
    "XLV": "\u30d8\u30eb\u30b9\u30b1\u30a2",
    "XLY": "\u4e00\u822c\u6d88\u8cbb\u8ca1",
}

JP_ETF_LABELS = {
    "1617.T": "\u98df\u54c1",
    "1618.T": "\u30a8\u30cd\u30eb\u30ae\u30fc\u8cc7\u6e90",
    "1619.T": "\u5efa\u8a2d\u30fb\u8cc7\u6750",
    "1620.T": "\u7d20\u6750\u30fb\u5316\u5b66",
    "1621.T": "\u533b\u85ac\u54c1",
    "1622.T": "\u81ea\u52d5\u8eca\u30fb\u8f38\u9001\u6a5f",
    "1623.T": "\u9244\u92fc\u30fb\u975e\u9244",
    "1624.T": "\u6a5f\u68b0",
    "1625.T": "\u96fb\u6a5f\u30fb\u7cbe\u5bc6",
    "1626.T": "\u60c5\u5831\u901a\u4fe1",
    "1627.T": "\u96fb\u529b\u30fb\u30ac\u30b9",
    "1628.T": "\u904b\u8f38\u30fb\u7269\u6d41",
    "1629.T": "\u5546\u793e\u30fb\u5378\u58f2",
    "1630.T": "\u5c0f\u58f2",
    "1631.T": "\u9280\u884c",
    "1632.T": "\u91d1\u878d",
    "1633.T": "\u4e0d\u52d5\u7523",
}

DEFAULT_CAPITAL = int(os.getenv("DEFAULT_CAPITAL", "200000"))
DEFAULT_STOP_LOSS = float(os.getenv("DEFAULT_STOP_LOSS", "0.02"))

DATA_PERIOD = os.getenv("DATA_PERIOD", "9mo")
ROLLING_WINDOW = int(os.getenv("ROLLING_WINDOW", "60"))
TAKE_PROFIT_RATIO = float(os.getenv("TAKE_PROFIT_RATIO", "0.03"))
MAX_POSITION_BUDGET_RATIO = float(os.getenv("MAX_POSITION_BUDGET_RATIO", "0.40"))
WEAK_SIGNAL_THRESHOLD = float(os.getenv("WEAK_SIGNAL_THRESHOLD", "0.50"))

VIX_TICKER = "^VIX"
VIX_ALERT_LEVEL = float(os.getenv("VIX_ALERT_LEVEL", "25"))
VIX_SKIP_LEVEL = float(os.getenv("VIX_SKIP_LEVEL", "30"))
VIX_SIGNAL_DAMPING = float(os.getenv("VIX_SIGNAL_DAMPING", "0.50"))
VOLUME_WINDOW = int(os.getenv("VOLUME_WINDOW", "5"))
LOW_VOLUME_THRESHOLD_RATIO = float(os.getenv("LOW_VOLUME_THRESHOLD_RATIO", "0.50"))
