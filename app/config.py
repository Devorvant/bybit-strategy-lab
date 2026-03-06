import os
from dataclasses import dataclass, field


def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}

def _symbols_from_env() -> list[str]:
    raw = os.getenv("SYMBOLS", "APTUSDT")
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def _tfs_from_env() -> list[str]:
    """Timeframes (Bybit kline interval) list.

    - If TFS is set: use it (comma-separated, e.g. "60,120,240")
    - Else: fallback to single TF
    """
    raw = os.getenv("TFS")
    if raw and raw.strip():
        return [s.strip() for s in raw.split(",") if s.strip()]
    return [os.getenv("TF", "120").strip()]

@dataclass(frozen=True)
class Settings:
    BYBIT_WS_URL: str = os.getenv("BYBIT_WS_URL", "wss://stream.bybit.com/v5/public/linear")
    BYBIT_REST_URL: str = os.getenv("BYBIT_REST_URL", "https://api.bybit.com")
    CATEGORY: str = os.getenv("BYBIT_CATEGORY", "linear")
    SYMBOLS: list[str] = field(default_factory=_symbols_from_env)  # <-- фикс
    TF: str = os.getenv("TF", "120")  # minutes (Bybit kline interval)
    # Optional: subscribe/backfill multiple timeframes (comma-separated)
    # Example: TFS=60,120,240
    TFS: list[str] = field(default_factory=_tfs_from_env)
    # If set (e.g. Railway Postgres), takes precedence over DB_PATH
    DATABASE_URL: str | None = os.getenv("DATABASE_URL")
    DB_PATH: str = os.getenv("DB_PATH", "data.db")
    LOOKBACK: int = int(os.getenv("LOOKBACK", "500"))

    # Data backfill
    BACKFILL_ON_START: bool = _bool_env("BACKFILL_ON_START", False)
    BACKFILL_DAYS: int = int(os.getenv("BACKFILL_DAYS", "365"))
    BACKFILL_SLEEP_MS: int = int(os.getenv("BACKFILL_SLEEP_MS", "120"))

    # Optional: disable strategy loop to keep "collector only" mode
    ENABLE_STRATEGY: bool = _bool_env("ENABLE_STRATEGY", False)

    # --- Strategy params (Supertrend + ADX) ---
    ATR_LEN: int = int(os.getenv("ATR_LEN", "10"))
    ST_FACTOR: float = float(os.getenv("ST_FACTOR", "3"))
    ADX_LEN: int = int(os.getenv("ADX_LEN", "14"))
    ADX_MIN: float = float(os.getenv("ADX_MIN", "20"))

settings = Settings()
