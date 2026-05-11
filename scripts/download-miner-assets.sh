#!/bin/sh
set -eu

cd "$(dirname "$0")/.."
mkdir -p vendor

curl -fsSL https://hash256.fun/miner/hash_miner.js -o vendor/hash_miner.js
curl -fsSL https://hash256.fun/miner/hash_miner_bg.wasm -o vendor/hash_miner_bg.wasm

echo "downloaded official HASH256 WASM miner assets into ./vendor"
