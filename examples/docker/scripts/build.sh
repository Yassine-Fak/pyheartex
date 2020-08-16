#! /usr/bin/env bash

docker build -t gcharbon/pyheartex-example-server -f Dockerfile .
docker build -t gcharbon/pyheartex-example-worker -f Dockerfile.worker .
# docker build -t label-studio -f Dockerfile.label-studio .
