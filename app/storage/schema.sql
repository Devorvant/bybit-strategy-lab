CREATE TABLE IF NOT EXISTS bars (
  symbol TEXT NOT NULL,
  tf TEXT NOT NULL,
  ts BIGINT NOT NULL,
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
  ts BIGINT NOT NULL,
  signal TEXT NOT NULL,
  note TEXT,
  PRIMARY KEY(symbol, tf, ts)
);

CREATE TABLE IF NOT EXISTS strategy_decisions (
  id INTEGER PRIMARY KEY,
  ts BIGINT NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER) * 1000),
  run_id TEXT,
  symbol TEXT NOT NULL,
  tf TEXT,
  bar_ts BIGINT,
  opt_id INTEGER,
  strategy_name TEXT,
  signal TEXT,
  action TEXT,
  reason TEXT,
  signal_note TEXT,
  exchange_side TEXT,
  exchange_size REAL,
  cooldown_active BOOLEAN,
  cooldown_until REAL,
  blocked BOOLEAN,
  blocked_reason TEXT,
  auto_enabled BOOLEAN,
  execute_requested BOOLEAN,
  executed BOOLEAN,
  payload TEXT
);
CREATE INDEX IF NOT EXISTS idx_strategy_decisions_symbol_tf_ts ON strategy_decisions(symbol, tf, ts DESC);

CREATE TABLE IF NOT EXISTS execution_events (
  id INTEGER PRIMARY KEY,
  ts BIGINT NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER) * 1000),
  run_id TEXT,
  symbol TEXT NOT NULL,
  tf TEXT,
  bar_ts BIGINT,
  event_type TEXT,
  requested_action TEXT,
  side TEXT,
  qty REAL,
  price REAL,
  reduce_only BOOLEAN,
  ok BOOLEAN,
  error TEXT,
  order_id TEXT,
  order_link_id TEXT,
  response TEXT
);
CREATE INDEX IF NOT EXISTS idx_execution_events_symbol_tf_ts ON execution_events(symbol, tf, ts DESC);

CREATE TABLE IF NOT EXISTS exchange_snapshots (
  id INTEGER PRIMARY KEY,
  ts BIGINT NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER) * 1000),
  run_id TEXT,
  symbol TEXT NOT NULL,
  tf TEXT,
  bar_ts BIGINT,
  source TEXT,
  side TEXT,
  size REAL,
  avg_price REAL,
  mark_price REAL,
  unrealised_pnl REAL,
  leverage REAL,
  liq_price REAL,
  take_profit REAL,
  stop_loss REAL,
  has_position BOOLEAN,
  wallet_equity REAL,
  wallet_balance REAL,
  available_balance REAL,
  payload TEXT
);
CREATE INDEX IF NOT EXISTS idx_exchange_snapshots_symbol_tf_ts ON exchange_snapshots(symbol, tf, ts DESC);
