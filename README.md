# Sector Defender

日米ETFリードラグ、BTC急落リバウンド、VIX急騰リバウンドのシグナルを確認する Streamlit アプリです。

## ページ構成

- 日米ETFリードラグシグナル
- BTC急落リバウンドシグナル
- VIX急落リバウンドシグナル（2558対象）

## ローカル起動

```bash
git clone https://github.com/kawauso92/sector-defender
cd sector-defender
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

## 環境変数

`.env.example` を `.env` にコピーして使います。

```env
DEFAULT_CAPITAL=200000
DEFAULT_STOP_LOSS=0.02
ROLLING_WINDOW=60
DATA_PERIOD=9mo
VIX_ALERT_LEVEL=25
VIX_SKIP_LEVEL=30
VIX_SIGNAL_DAMPING=0.50
LONG_SHORT_CAPITAL_RATIO=0.30
TAKE_PROFIT_RATIO=0.03
MAX_POSITION_BUDGET_RATIO=0.40
WEAK_SIGNAL_THRESHOLD=0.50
LINE_NOTIFY_TOKEN=your_token_here
```

## Streamlit Cloudデプロイ手順

1. [https://share.streamlit.io](https://share.streamlit.io) にアクセス
2. GitHubアカウントでログイン
3. `New app` をクリック
4. `Repository`: `kawauso92/sector-defender`
5. `Branch`: `main`
6. `Main file path`: `app.py`
7. `Advanced settings` → `Secrets` に以下を追加

```toml
LINE_NOTIFY_TOKEN = "your_token"
```

8. `Deploy` をクリック

補足:

- `btc_1h_cache.csv` と `trade_history.csv` がなくても、アプリ側で初期化または再取得するようにしてあります。
- `LINE_NOTIFY_TOKEN` はローカルでは `.env`、Streamlit Cloud では Secrets から読み込みます。

## LINE Notify 設定

1. `.env` または Streamlit Cloud Secrets に `LINE_NOTIFY_TOKEN` を設定
2. サイドバーの通知設定で BTC/VIX 通知を ON
3. シグナル発生時に通知送信

## 補助ファイル

- `app.py`: Streamlit UI
- `btc_logic.py`: BTC シグナル判定、為替取得、履歴管理
- `vix_logic.py`: VIX シグナル判定
- `notify.py`: 通知送信
