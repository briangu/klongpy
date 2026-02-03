# Stats Logging Example

High-throughput stat ingestion over IPC, plus a head query client.

## Requirements

```bash
pip install pytz
```

## Run

Terminal 1 (server):

```bash
kgpy stats_server.kg
```

Terminal 2 (client generator):

```bash
kgpy stats_client.kg
```

Terminal 3 (tail the latest rows):

```bash
kgpy head.kg
```
