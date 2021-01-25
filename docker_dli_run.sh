#!/usr/bin/env bash
#
#  Run the container for the NVIDIA Deep Learning Institute course on AI for
#  Jetson Nano.
#
#  This has been superseded by the docker-compose configuration in
#  docker-compose.yml.  See that file for directions on how to run.
#
# --runtime nvidia = necessary default isn't set in /etc/docker/daemon.json
# -it = interactive (use stdin, tty)
# --rm = delete container when finished
# --network host = container will open ports on host itself
# --volume = bind mount
# --device = pass USB camera into container
#
# The specific nvcr.io path and container tag must come from:
#
# https://ngc.nvidia.com/catalog/containers/nvidia:dli:dli-nano-ai
#
docker run \
    --runtime nvidia \
    -it \
    --rm \
    --network host \
    --volume "$HOME/nvdli-data:/nvdli-nano/data" \
    --device /dev/video0 \
    "nvcr.io/nvidia/dli/dli-nano-ai:v2.0.1-r32.4.4"
