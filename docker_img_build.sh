#!/usr/bin/env bash

if [ -z "$1" ]; then
  echo "pass CPU or GPU as argument"
  echo "Docker image build failed..."
  exit 1
fi

if [ "$1" = "GPU" ]; then
  image="posenet-python-gpu"
  dockerfile="Dockerfile-gpu"
else
  image="posenet-python-cpu"
  dockerfile="Dockerfile"
fi

docker rmi -f "$image"

docker build -t "$image" -f "$dockerfile" .
