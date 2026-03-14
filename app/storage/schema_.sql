CREATE TABLE IF NOT EXISTS bars (
  symbol TEXT NOT NULL,
  tf TEXT NOT NULL,
  ts BIGINT NOT NULL,          -- ms epoch open time
  o REAL NOT NULL,
  h REAL NOT NULL,
  l REAL NOT NULL,
  c REAL NOT NULL,
  v REAL NOT NULL,
  PRIMARY KEY(symbol, tf, ts)
);

CREATE TABLE IF NOT EXISTS signals (
  symbol TEXT NOT NULL,
  tf TEXT NOT NULL,
  ts BIGINT NOT NULL,          -- ms (bar close time)
  signal TEXT NOT NULL,         -- LONG/SHORT/FLAT
  note TEXT,
  PRIMARY KEY(symbol, tf, ts)
);
