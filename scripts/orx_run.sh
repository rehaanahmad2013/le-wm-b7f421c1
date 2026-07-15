#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

apt-get update -qq
apt-get install -y -qq --no-install-recommends \
  curl ca-certificates zstd libgl1 libglfw3 libglew2.2 libosmesa6 \
  libegl1 libx11-6 libxext6 libxrender1 ffmpeg >/dev/null

python -m pip install --disable-pip-version-check --quiet \
  'stable-worldmodel[train,format]==0.1.1' \
  'stable-pretraining==0.1.8' \
  'transformers>=4.50,<5' \
  'huggingface-hub>=0.30,<2' \
  'pygame>=2.6' 'pymunk>=6.11' 'shapely>=2.0' \
  'ogbench>=1.1' 'dm-control>=1.0.31' 'gymnasium-robotics>=1.3' \
  'opencv-python-headless>=4.10' 'scikit-learn>=1.6' 'scipy>=1.14' \
  'pandas>=2.2' 'matplotlib>=3.9' 'marimo>=0.17,<1'

python -u reproduce.py
