#!/usr/bin/env bash

# -it = interactive
# --rm = delete container when finished
# --network host = container will open ports on host itself
# --volume = bind mount
# --device = pass USB camera into container

# the specific nvcr.io path and container tag must come from:
#
# https://ngc.nvidia.com/catalog/containers/nvidia:dli:dli-nano-ai

sudo docker run \
    --runtime nvidia \
    -it \
    --rm \
    --network host \
    --volume "$HOME/nvdli-data:/nvdli-nano/data" \
    --device /dev/video0 \
    "nvcr.io/nvidia/dli/dli-nano-ai:v2.0.1-r32.4.4"
