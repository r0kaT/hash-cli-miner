#!/bin/sh
set -eu

cd "$(dirname "$0")"

# Detect nvcc
NVCC="${NVCC:-nvcc}"
if ! command -v "$NVCC" > /dev/null 2>&1; then
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

# Detect GPU arch — default to sm_90 for H100 (Hopper)
# sm_89 = RTX 4090, sm_80 = A100, sm_90 = H100/H200
ARCH="${CUDA_ARCH:-sm_90}"

echo "compiling with $NVCC for $ARCH ..."

$NVCC -std=c++17 -O3 \
  -arch="$ARCH" \
  -Xcompiler -pthread \
  hash256-cuda-miner.cu \
  -o hash256-cuda-miner \
  -lpthread

echo "built ./hash256-cuda-miner"
