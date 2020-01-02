#!/usr/bin/env bash

if [ -z "$1" ]; then
  echo "pass CPU or GPU as argument"
  echo "Docker image build failed..."
  exit 1
fi

if [ "$1" = "GPU" ]; then
  image="posenet-python-gpu"
  version="--build-arg IMAGE_VERSION=2.0.0-gpu-py3-jupyter"
else
  image="posenet-python-cpu"
  version=""
fi

docker rmi -f "$image"

docker build -t "$image" $version .
