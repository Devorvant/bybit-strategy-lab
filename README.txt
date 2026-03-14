Файлы под замену:

1) app/trading/journal.py
   - журналирование теперь работает и с SQLite, и с Postgres
   - после вставок есть commit

2) app/storage/schema.sql
   - добавлены таблицы strategy_decisions, execution_events, exchange_snapshots

3) app/trading/chart_service.py
   - live.events получают source=manual/auto/exchange
   - есть fallback price/qty из response payload
   - external close / tp-sl close лучше различаются

4) templates/trade_chart.html
   - на верхний график добавлены отдельные traces:
     * Live Entry
     * Live Close
     * External / TP-SL Close
   - стратегия и факт визуально разделены

Что сделать после замены:
- перезапустить приложение
- если база SQLite уже старая и таблиц журналов нет, один раз инициализировать schema.sql на новой БД
  или выполнить DDL для новых таблиц вручную
