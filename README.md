# bybit-strategy-lab (MVP)

Мини-проект для Railway: берёт kline (свечи) по WebSocket с Bybit, сохраняет в БД и показывает:
- /health
- /bars?symbol=APTUSDT&tf=120&limit=500
- /chart?symbol=APTUSDT&tf=120&limit=500 (HTML свечной график Plotly)
- /signal (если включить стратегию)

## Backfill истории (последние N дней)

На старте сервера можно автоматически догрузить историю через REST `/v5/market/kline` (Bybit V5):
- данные возвращаются в обратной сортировке по startTime и `limit` до 1000 свечей на страницу
- backfill пагинирует назад во времени и UPSERT-ит в БД

## Railway Variables (минимум)
- SYMBOLS=APTUSDT (или APTUSDT,BTCUSDT)
- TF=120 (интервал в минутах для Bybit kline: 1,3,5,15,30,60,120,240...)
- BYBIT_WS_URL=wss://stream.bybit.com/v5/public/linear
- DB_PATH=data.db (если SQLite)

## Railway Postgres (рекомендуется)
Если в Railway подключён Postgres и есть `DATABASE_URL`, проект автоматически будет использовать Postgres.

## Backfill Variables (опционально)
- BACKFILL_ON_START=1
- BACKFILL_DAYS=365
- BYBIT_REST_URL=https://api.bybit.com (testnet: https://api-testnet.bybit.com)

## Strategy Variables (опционально)
- ENABLE_STRATEGY=1
- LOOKBACK=500 (сколько баров брать для расчёта)
- ATR_LEN=10
- ST_FACTOR=3
- ADX_LEN=14
- ADX_MIN=20

## Локальный запуск
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Открой:
- http://127.0.0.1:8000/health
- http://127.0.0.1:8000/chart?symbol=APTUSDT&tf=120&limit=300
