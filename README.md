# HASH256 GPU Miner

HASH256 proof-of-work CLI miner using NVIDIA CUDA GPU.

**RTX 4090 Test Result: ~4.6 GH/s**

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

Other GPU builds (default `sm_89` is for RTX 4090):

```bash
CUDA_ARCH=sm_86 sh build-cuda.sh   # RTX 3090
CUDA_ARCH=sm_80 sh build-cuda.sh   # A100
```

## Configuration procedures

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

Benchmark:

```bash
node hash256-cli.js bench --engine cuda --seconds 10
```

Mining (submitting transactions):

```bash
node hash256-cli.js mine --engine cuda --submit --loop
```

Mining without submitting:

```bash
node hash256-cli.js mine --address 0xYourAddress --engine cuda --loop
```

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
