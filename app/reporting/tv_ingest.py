from fastapi import APIRouter, Request

router = APIRouter()

@router.post("/tv_ingest")
async def tv_ingest(request: Request):
    """Receives JSON lines from TradingView alerts (webhook)."""
    raw = await request.body()
    # Railway: easiest is to view logs; each alert is one line of JSON
    print(raw.decode("utf-8", errors="replace"))
    return {"ok": True}
