
from fastapi import Header, HTTPException
import os


def get_trade_token() -> str:
    return os.getenv('TRADE_API_TOKEN', '').strip()


def require_trade_token(x_trade_token: str | None = Header(default=None)):
    """Optional protection for manual trade routes. If TRADE_API_TOKEN is set, header must match."""
    token = get_trade_token()
    if token and (x_trade_token or '').strip() != token:
        raise HTTPException(status_code=401, detail='bad trade token')
    return True
