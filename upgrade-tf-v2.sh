#!/usr/bin/env bash

WORK=$(dirname $(pwd))

docker run --gpus all -it -v $WORK:/work posenet-python tf_upgrade_v2 \
  --intree posenet-python/ \
  --outtree posenet-python_v2/ \
  --reportfile posenet-python/report.txt
