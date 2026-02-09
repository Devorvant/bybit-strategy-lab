import os
from dataclasses import dataclass, field

def _symbols_from_env() -> list[str]:
    raw = os.getenv("SYMBOLS", "APTUSDT")
    return [s.strip().upper() for s in raw.split(",") if s.strip()]

@dataclass(frozen=True)
class Settings:
    BYBIT_WS_URL: str = os.getenv("BYBIT_WS_URL", "wss://stream.bybit.com/v5/public/linear")
    CATEGORY: str = os.getenv("BYBIT_CATEGORY", "linear")
    SYMBOLS: list[str] = field(default_factory=_symbols_from_env)  # <-- фикс
    TF: str = os.getenv("TF", "120")  # minutes (Bybit kline interval)
    DB_PATH: str = os.getenv("DB_PATH", "data.db")
    LOOKBACK: int = int(os.getenv("LOOKBACK", "500"))

settings = Settings()
