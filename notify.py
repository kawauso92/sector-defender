from __future__ import annotations

import requests

LINE_NOTIFY_URL = "https://notify-api.line.me/api/notify"


def send_line_notify(token: str, message: str) -> tuple[bool, str]:
    if not token:
        return False, "LINE_NOTIFY_TOKEN is not set."

    try:
        response = requests.post(
            LINE_NOTIFY_URL,
            headers={"Authorization": f"Bearer {token}"},
            data={"message": message},
            timeout=20,
        )
        if response.ok:
            return True, "Notification sent."
        return False, f"LINE Notify failed: {response.status_code} {response.text}"
    except Exception as exc:
        return False, f"LINE Notify request failed: {exc}"


def build_btc_notification_message(
    drop_pct: float,
    limit_price: float,
    margin_jpy: float,
    leverage: int,
) -> str:
    return (
        "【BTCシグナル🔴】\n"
        f"BTC急落検知：{drop_pct:.2f}%\n"
        f"指値：{limit_price:,.2f} USD\n"
        "利確：+1.4% / 損切り：-1.4%\n"
        f"推奨証拠金：¥{margin_jpy:,.0f}（レバ{leverage}倍）\n"
        "有効期限：2時間以内"
    )


def build_vix_notification_message(
    vix_value: float,
    vix_change_pct: float,
    limit_price_jpy: float,
    amount_jpy: float,
) -> str:
    return (
        "【VIXシグナル🔴】\n"
        f"VIX急騰検知：{vix_change_pct:+.1f}%（現在値：{vix_value:.1f}）\n"
        f"2558.T指値：¥{limit_price_jpy:,.0f}\n"
        "利確：VIX20以下まで保有\n"
        f"推奨金額：¥{amount_jpy:,.0f}"
    )
