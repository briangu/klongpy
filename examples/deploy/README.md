# Docker Deployment Example

This directory contains a Dockerfile that runs `kgpy` with a script path
provided via `KG_FILE_PATH`.

## Build

Run from the repo root:

```bash
docker build -t klongpy-app -f examples/deploy/docker/Dockerfile .
```

## Run

Mount the repo and point `KG_FILE_PATH` at a script in the container:

```bash
docker run --rm -e KG_FILE_PATH=/usr/src/app/examples/web/server.kg \
  -v "$PWD":/usr/src/app -p 8888:8888 klongpy-app
```
