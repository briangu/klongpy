# Database Examples

KlongPy + DuckDB table interop and a lightweight table store.

## Requirements

```bash
pip install "klongpy[db,web]"
```

## Run

From this directory:

```bash
# Table store (DFS) server
kgpy dfs.kg -- /tmp/tables
```

```bash
# Web + DB server
kgpy server.kg
curl -X POST -d "t=1691111164807793000&p=100&v=10000" "http://localhost:8080/p"
curl "http://localhost:8080"
```
