#!/usr/bin/env bash

WORK=$(pwd)

docker run --gpus all -it -v $WORK:/work posenet-python python "$@"
