#!/bin/sh
set -eu

cd "$(dirname "$0")"

# Detect nvcc
NVCC="${NVCC:-nvcc}"
if ! command -v "$NVCC" > /dev/null 2>&1; then
  # Try common CUDA toolkit paths
  for p in /usr/local/cuda/bin/nvcc /opt/cuda/bin/nvcc; do
    if [ -x "$p" ]; then
      NVCC="$p"
      break
    fi
  done
fi

if ! command -v "$NVCC" > /dev/null 2>&1 && [ ! -x "$NVCC" ]; then
  echo "nvcc not found. Install CUDA Toolkit first."
  echo "  Ubuntu: sudo apt install nvidia-cuda-toolkit"
  exit 1
fi

# Detect GPU arch — default to sm_89 for RTX 4090
ARCH="${CUDA_ARCH:-sm_89}"

echo "compiling with $NVCC for $ARCH ..."

$NVCC -std=c++17 -O3 \
  -arch="$ARCH" \
  hash256-cuda-miner.cu \
  -o hash256-cuda-miner

echo "built ./hash256-cuda-miner"
