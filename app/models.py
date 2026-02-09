from pydantic import BaseModel
from typing import Optional, List

class Bar(BaseModel):
    ts: int
    o: float
    h: float
    l: float
    c: float
    v: float

class BarsResponse(BaseModel):
    symbol: str
    tf: str
    bars: List[Bar]

class Signal(BaseModel):
    ts: int
    signal: str  # LONG/SHORT/FLAT
    note: Optional[str] = ""
