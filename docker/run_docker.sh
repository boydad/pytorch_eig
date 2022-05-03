#!/bin/bash

docker run --rm --gpus=all -it pyeig:22.03 \
       python -u /opt/pytorch_eig/test.py
