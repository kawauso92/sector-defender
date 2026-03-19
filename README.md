# Sector Defender
日米業種ETFリードラグ投資シグナルWebアプリ

## 運用方針
- ロジック：米国業種ETF z-score → 日本業種ETFリードラグ
- 閾値：1.0以上のシグナルのみエントリー（Pattern B）
- 注文方法：指値（前日終値×0.999/1.001）・当日限り
- 決済：引け成行（15時15分〜25分）
- 資金配分：ロング60%・ショート30%・バッファ10%
  （ショート優先モード時はロング30%・ショート60%）

## バックテスト結果（2015-2025）
- 年率リターン（コストなし）：39.28%
- 年率リターン（コスト0.1%込み）：13.36%
- 最大ドローダウン：-20.41%
- 勝率：49.32%

## セットアップ
```bash
git clone https://github.com/kawauso92/sector-defender
cd sector-defender
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

## 別環境での再開
```bash
git pull
streamlit run app.py
```

## 注意事項
- データはyfinance（15〜20分遅延）
- 本番運用は自己責任で
- バックテスト結果は将来の利益を保証しない
