#!/usr/bin/env bash

# -it = interactive
# --rm = delete container when finished
# --network host = container will open ports on host itself
# --volume = bind mount
# --device = pass USB camera into container

sudo docker run \
    --runtime nvidia \
    -it \
    --rm \
    --network host \
    --volume "$HOME/nvdli-data:/nvdli-nano/data" \
    --device /dev/video0 \
    "l4t-ml:r32.4.4-py3"
