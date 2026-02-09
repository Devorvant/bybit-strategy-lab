import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    # Bybit
    BYBIT_WS_URL: str = os.getenv("BYBIT_WS_URL", "wss://stream.bybit.com/v5/public/linear")
    CATEGORY: str = os.getenv("BYBIT_CATEGORY", "linear")  # linear/spot/inverse/option
    SYMBOLS: list[str] = os.getenv("SYMBOLS", "APTUSDT").split(",")
    TF: str = os.getenv("TF", "120")  # kline interval (minutes)
    # Storage
    DB_PATH: str = os.getenv("DB_PATH", "data.db")
    # Strategy
    LOOKBACK: int = int(os.getenv("LOOKBACK", "500"))

settings = Settings()
