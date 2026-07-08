#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   GPU=0 \
#   UE_PATH=/path/to/UE \
#   ./envs/run.sh [TASK]
#
# TASK defaults to "full_loop", passed through to run/render.sh

GPU=${GPU:-0}
UE_PATH=${UE_PATH:?'UE_PATH is required (mounted to /home/ue_user/Unreal)'}
TASK=${1:-full_loop}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE="ue_render"

sudo docker run --gpus "device=${GPU}" --rm --runtime=nvidia \
  -e GPU="${GPU}" \
  -e POD_NAME="${POD_NAME:-}" \
  -v "${UE_PATH}:/home/ue_user/Unreal" \
  -v "${REPO_ROOT}:/home/ue_user/code/" \
  -v "${REPO_ROOT}/../recam:/home/ue_user/code/recam" \
  --workdir /home/ue_user/code/ \
  "${IMAGE}" \
  bash -c "
    # render.sh sources ~/miniconda3; miniforge3 is installed instead — create a compat symlink
    if [ ! -e ~/miniconda3 ] && [ -d ~/miniforge3 ]; then
      ln -s ~/miniforge3 ~/miniconda3
    fi
    bash run/render.sh '${TASK}'
  "
