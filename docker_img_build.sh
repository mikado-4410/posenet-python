#!/usr/bin/env bash

docker rmi -f posenet-python

docker build -t posenet-python -f Dockerfile .
