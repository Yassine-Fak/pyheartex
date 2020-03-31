#! /usr/bin/env bash

docker build -t gcharbon/pyheartex-example-server -f Dockerfile .
docker build -t gcharbon/pyheartex-example-worker -f Dockerfile.worker .
docker stack deploy -c docker-compose.yml pyheartex-example
