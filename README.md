# bybit-strategy-lab (MVP)

Мини-проект для Railway: берёт kline (свечи) по WebSocket с Bybit, сохраняет в SQLite и показывает:
- /health
- /bars?symbol=APTUSDT&tf=120&limit=500
- /chart?symbol=APTUSDT&tf=120&limit=500 (HTML свечной график Plotly)
- /signal (пока заглушка, таблица есть)

## Railway Variables (минимум)
- SYMBOLS=APTUSDT (или APTUSDT,BTCUSDT)
- TF=120 (интервал в минутах для Bybit kline: 1,3,5,15,30,60,120,240...)
- BYBIT_WS_URL=wss://stream.bybit.com/v5/public/linear
- DB_PATH=data.db

## Локальный запуск
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Открой:
- http://127.0.0.1:8000/health
- http://127.0.0.1:8000/chart?symbol=APTUSDT&tf=120&limit=300
