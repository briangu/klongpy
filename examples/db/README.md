# Database Examples

The NYC taxi example uses the desktop runtime's DuckDB engine. This directory also contains a Python-hosted table store.

## NYC taxi trips

Download yellow and green trip records for the months you want to query. From the repository root, pass the two local paths or quoted globs to either command. DuckDB may download its matching Parquet extension on first use. [NYC TLC monthly Parquet files](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

```sh
kg examples/db/nyc-taxi.kg 'data/yellow_tripdata_*.parquet' 'data/green_tripdata_*.parquet'
kgpy examples/db/nyc-taxi.kg -- 'data/yellow_tripdata_*.parquet' 'data/green_tripdata_*.parquet'
```

The program prints four JSON arrays, one per line. They contain trip counts by cab type, average total fare by passenger count, trip counts by passenger count and pickup year, and trip counts by passenger count, year and rounded distance. Dates come from the pickup field inside each record. The distance report orders the busiest groups first within each year.

The program scans the fare, passenger, distance and pickup columns together. It stores a count of trips, a count of non-null fares and a fare total for each detailed group. Summing those counts and totals produces the coarser reports. A group with no recorded fare has a JSON `null` average. [Original four-query benchmark](https://tech.marksblogg.com/duckdb-1b-taxi-rides.html)

## Table store requirements

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
