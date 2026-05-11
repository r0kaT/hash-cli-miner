# HASH256 GPU Miner

HASH256 proof-of-work CLI miner using NVIDIA CUDA GPU.

**Multi-GPU support: optimized for 8x H100 / 4x A100 (SXM4 & PCIe)**

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

## 4x A100 SXM4 Optimization Notes

A100 SXM4 (40GB or 80GB HBM2e, Ampere `sm_80`, 108 SMs, 400W, NVLink 3.0) runs the same kernel as H100. The `__launch_bounds__(512, 2)` configuration is compatible — A100 has a 64 KB register file per SM and easily fits 2 resident blocks/SM.

Build for A100:

```bash
CUDA_ARCH=sm_80 sh build-cuda.sh
```

Benchmark on all 4 GPUs:

```bash
node hash256-cli.js bench --engine cuda --seconds 10 --gpus 4
```

Mining on 4x A100 SXM4 (batch sized for ~108 SMs × 2 blocks × 512 threads × 64 iter × 8 waves ≈ 7.2B hashes/GPU per launch — total batch ~28-32B for the cluster):

```bash
node hash256-cli.js mine --engine cuda --submit --loop --gpus 4 --batch 28000000000
```

Lower-latency variant (smaller batch → faster early-exit when a solution is found, slightly lower steady-state H/s):

```bash
node hash256-cli.js mine --engine cuda --submit --loop --gpus 4 --batch 16000000000
```

- **Architecture**: `sm_80` (Ampere), 108 SMs per GPU (vs. 132 on H100 → ~82% of H100's compute width)
- **Concurrency**: 2 streams × 4 GPUs = 8 concurrent pipelines
- **NVLink**: Mining is embarrassingly parallel per-GPU — no inter-GPU traffic, so NVLink/NVSwitch is unused (PCIe-attached A100s perform the same)
- **Expected throughput**: ~12-20 GH/s per A100 SXM4 → **50-80 GH/s on 4x A100 SXM4**
- **Tuning tip**: if you see `[gpu*]` rates plateau below expected, try `--batch 32000000000` (deeper queue, better SM occupancy) or `--batch 12000000000` (snappier early-exit)

## 4x A100 PCIe Optimization Notes

A100 PCIe shares the GA100 silicon with SXM4 (108 SMs, `sm_80`, same 40/80 GB HBM2e) but runs at **250W TDP** vs SXM4's 400W. Under sustained Keccak load the chip is power-limited well before it is thermal-limited, so steady-state clocks land ~10–15% lower than SXM4. Mining is compute-bound on the SMs (no inter-GPU traffic), so PCIe 4.0 x16 vs NVLink makes **no difference** for this workload.

Build is identical to SXM4:

```bash
CUDA_ARCH=sm_80 sh build-cuda.sh
```

Benchmark on all 4 GPUs:

```bash
node hash256-cli.js bench --engine cuda --seconds 10 --gpus 4
```

Mining on 4x A100 PCIe (start a bit smaller than SXM4 since per-GPU H/s is lower — keeps early-exit responsive):

```bash
node hash256-cli.js mine --engine cuda --submit --loop --gpus 4 --batch 24000000000
```

Throughput-leaning variant (if your chassis cools well and you want maximum H/s per launch):

```bash
node hash256-cli.js mine --engine cuda --submit --loop --gpus 4 --batch 32000000000
```

- **Architecture**: `sm_80` (Ampere), 108 SMs per GPU — same as SXM4
- **Power**: 250W per card (vs 400W SXM4) → ~10–15% lower sustained throughput
- **Interconnect**: PCIe 4.0 x16 (or NVLink bridge between pairs); irrelevant for mining
- **Cooling**: passive-cooled PCIe variants need server airflow; consumer-style chassis can thermal-throttle quickly under 100% kernel load — watch `nvidia-smi -l 1` for `P0` clock drops
- **Expected throughput**: ~10-17 GH/s per A100 PCIe → **40-70 GH/s on 4x A100 PCIe**
- **Tuning tip**: if `nvidia-smi` shows the card pegged at the 250W power limit, you are at the chip's ceiling — increasing `--batch` further won't help. The only remaining lever is `nvidia-smi -pl <watts>` (if your firmware permits a higher cap, e.g. 300W on some SKUs)

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
