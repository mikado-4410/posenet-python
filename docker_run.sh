#!/usr/bin/env bash

WORK=$(pwd)

if [ -z "$POSENET_PYTHON_DEVICE" ]; then
  echo "set the environment variable POSENET_PYTHON_DEVICE to CPU or GPU, or enter your choice below:"
  read -p "Enter your device (CPU or GPU): "  device
  if [ "$device" = "GPU" ]; then
    source <(echo "export POSENET_PYTHON_DEVICE=GPU");
  elif [ "$device" = "CPU" ]; then
    source <(echo "export POSENET_PYTHON_DEVICE=CPU");
  else
    echo "Device configuration failed..."
    exit 1
  fi
fi


echo "device is: $POSENET_PYTHON_DEVICE"

if [ "$POSENET_PYTHON_DEVICE" = "GPU" ]; then
  image="posenet-python-gpu"
else
  image="posenet-python-cpu"
fi

docker run --gpus all -it --rm -v $WORK:/work "$image" python "$@"
