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


def _norm_env(v: str | None, default: str) -> str:
    s = (v or "").strip().lower()
    return s if s else default


def _data_env_from_env() -> str:
    # Safe default: testnet (prevents accidental mainnet usage)
    # Explicitly set DATA_ENV=mainnet to fetch real market data.
    return _norm_env(os.getenv("DATA_ENV"), "testnet")


def _trade_env_from_env() -> str:
    # Safe default follows legacy BYBIT_TESTNET flag (default was testnet in this project)
    raw = os.getenv("TRADE_ENV")
    if raw and raw.strip():
        return _norm_env(raw, "testnet")
    legacy_testnet = os.getenv("BYBIT_TESTNET", "1") != "0"
    return "testnet" if legacy_testnet else "mainnet"


def _default_ws_url(category: str, env: str) -> str:
    # Bybit v5 public WS endpoints differ by category.
    host = "stream-testnet.bybit.com" if env == "testnet" else "stream.bybit.com"
    cat = (category or "linear").strip().lower()
    if cat in {"linear", "spot", "inverse", "option"}:
        return f"wss://{host}/v5/public/{cat}"
    return f"wss://{host}/v5/public/linear"


def _default_rest_url(env: str) -> str:
    return "https://api-testnet.bybit.com" if env == "testnet" else "https://api.bybit.com"


def _ws_url_from_env() -> str:
    override = os.getenv("BYBIT_WS_URL")
    if override and override.strip():
        return override.strip()
    data_env = _data_env_from_env()
    category = os.getenv("BYBIT_CATEGORY", "linear")
    return _default_ws_url(category, data_env)


def _rest_url_from_env() -> str:
    override = os.getenv("BYBIT_REST_URL")
    if override and override.strip():
        return override.strip()
    data_env = _data_env_from_env()
    return _default_rest_url(data_env)

@dataclass(frozen=True)
class Settings:
    # Data collection environment (market data only)
    DATA_ENV: str = field(default_factory=_data_env_from_env)
    # Trade execution environment (orders/positions). Keys differ for testnet/mainnet.
    TRADE_ENV: str = field(default_factory=_trade_env_from_env)

    # Market data endpoints (derived from DATA_ENV unless overridden)
    BYBIT_WS_URL: str = field(default_factory=_ws_url_from_env)
    BYBIT_REST_URL: str = field(default_factory=_rest_url_from_env)
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
