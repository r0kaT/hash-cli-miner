# HASH256 GPU Miner

HASH256 proof-of-work CLI miner using NVIDIA CUDA GPU.

**Multi-GPU support: optimized for 8x H100**

## Nuggets

```
keccak256(challenge || nonce) < difficulty
```

Contract Address (Ethereum mainnet): `0xAC7b5d06fa1e77D08aea40d46cB7C5923A87A0cc`

## Installation

Requirements: Node.js, CUDA Toolkit (nvcc), NVIDIA Drivers

```bash
git clone https://github.com/fanqieBTC/hash256-cli.git
cd hash256-cli
npm install
npm run assets
npm run build:cuda
```

GPU architecture (default `sm_90` is for H100/H200):

```bash
CUDA_ARCH=sm_90 sh build-cuda.sh   # H100 / H200 (default)
CUDA_ARCH=sm_89 sh build-cuda.sh   # RTX 4090
CUDA_ARCH=sm_80 sh build-cuda.sh   # A100
CUDA_ARCH=sm_86 sh build-cuda.sh   # RTX 3090
```

## Configuration

```bash
cp .env.example .env
nano .env
```

```text
HASH256_RPC_URL=https://rpc.mevblocker.io/fast
PRIVATE_KEY=0xYourPrivateKey
```

## Usage

Check on-chain status:

```bash
node hash256-cli.js status
```

Benchmark (all GPUs):

```bash
node hash256-cli.js bench --engine cuda --seconds 10
```

Benchmark with specific GPU count:

```bash
node hash256-cli.js bench --engine cuda --seconds 10 --gpus 8
```

Mining (submitting transactions):

```bash
node hash256-cli.js mine --engine cuda --submit --loop
```

Mining with all 8 GPUs:

```bash
node hash256-cli.js mine --engine cuda --submit --loop --gpus 8 --batch 8000000000
```

Mining without submitting:

```bash
node hash256-cli.js mine --address 0xYourAddress --engine cuda --loop
```

### Multi-GPU Options

| Flag | Description | Default |
|------|-------------|---------|
| `--gpus N` | Number of GPUs to use (0 = all) | all available |
| `--batch N` | Total hash batch across all GPUs | 4,096,000,000 |
| `--progress-ms N` | Progress report interval (ms) | 1000 |
| `--cutoff-ms N` | Epoch cutoff timestamp (ms) | 0 (none) |

## 8x H100 Optimization Notes

- **Architecture**: `sm_90` (Hopper), 132 SMs per GPU
- **Kernel**: 512 threads/block, 32 iterations/thread, `__launch_bounds__` for register optimization
- **Memory**: Pinned (page-locked) host memory for fast async DMA transfers
- **Concurrency**: Double-buffered CUDA streams per GPU (2 streams × 8 GPUs = 16 concurrent pipelines)
- **Multi-GPU**: Per-GPU host threads with partitioned nonce ranges, atomic early-exit across GPUs
- **Expected throughput**: ~30-50 GH/s per H100 → 240-400 GH/s on 8x H100

## Ubuntu Server (systemd)

```bash
cp .env.example .env && nano .env
bash scripts/install-linux-service.sh
journalctl -u hash256-miner -f
```

## Stop

```bash
sudo systemctl stop hash256-miner
# or
pkill -f hash256-cuda-miner
```
