#!/usr/bin/env node
import { spawn } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { config as loadDotEnv } from "dotenv";
import {
  bytesToHex,
  concat,
  createPublicClient,
  createWalletClient,
  formatEther,
  formatUnits,
  hexToBytes,
  http,
  isAddress,
  keccak256,
  parseGwei,
} from "viem";
import { privateKeyToAccount } from "viem/accounts";
import { mainnet } from "viem/chains";
import { HASH256_ABI, HASH256_ADDRESS } from "./hash256-abi.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
loadDotEnv({ path: path.join(__dirname, ".env"), quiet: true });

const DEFAULT_RPC = process.env.HASH256_RPC_URL || "https://rpc.mevblocker.io/fast";
const RPC_TIMEOUT_MS = Number(process.env.HASH256_RPC_TIMEOUT_MS || 20_000);
const OFFICIAL_WASM_JS = path.join(__dirname, "vendor", "hash_miner.js");
const OFFICIAL_WASM_BIN = path.join(__dirname, "vendor", "hash_miner_bg.wasm");
const CUDA_BIN = path.join(__dirname, "hash256-cuda-miner");
const DEFAULT_WASM_BATCH = 1_000_000;
const DEFAULT_CUDA_BATCH = 4_096_000_000;
const DEFAULT_MINE_ROUND_SECONDS = 60;

function parseArgs(argv) {
  const args = { _: [] };
  for (let i = 0; i < argv.length; i += 1) {
    const item = argv[i];
    if (!item.startsWith("--")) {
      args._.push(item);
      continue;
    }
    const eq = item.indexOf("=");
    if (eq !== -1) {
      args[item.slice(2, eq)] = item.slice(eq + 1);
      continue;
    }
    const key = item.slice(2);
    const next = argv[i + 1];
    if (!next || next.startsWith("--")) {
      args[key] = true;
    } else {
      args[key] = next;
      i += 1;
    }
  }
  return args;
}

function usage() {
  console.log(`HASH256 GPU Miner

Usage:
  node hash256-cli.js status [--address 0x...]
  node hash256-cli.js bench [--seconds 5] [--gpus 8]
  node hash256-cli.js tune [--seconds 4] [--batches 64000000,256000000,512000000]
  node hash256-cli.js solve --challenge 0x... --difficulty 0x...
  node hash256-cli.js mine --address 0x... [--loop] [--gpus 8]
  node hash256-cli.js mine --submit [--private-key 0x...] [--loop] [--gpus 8]

Environment:
  HASH256_RPC_URL     Ethereum RPC, default ${DEFAULT_RPC}
  PRIVATE_KEY         Optional private key for --submit; can be set in .env

Multi-GPU:
  --gpus N            Use N GPUs (default: all available). Optimized for 8xH100.
  --batch N           Total batch across all GPUs (default: ${DEFAULT_CUDA_BATCH.toLocaleString()}).

Notes:
  - --address is read-only; PRIVATE_KEY/--private-key is required to submit.
  - --loop refreshes chain challenge between mining rounds.
  - Requires built hash256-cuda-miner binary (run: sh build-cuda.sh).
  - For H100: use CUDA_ARCH=sm_90 (default). For A100: CUDA_ARCH=sm_80.
  - Submitting mine() costs Ethereum gas. Use --submit only with your own key.`);
}

function toBytes32Hex(value) {
  if (typeof value === "bigint") {
    if (value < 0n) throw new Error("uint256 cannot be negative");
    return `0x${value.toString(16).padStart(64, "0")}`;
  }
  if (typeof value !== "string") throw new Error("expected hex string");
  const hex = value.startsWith("0x") ? value.slice(2) : value;
  if (!/^[0-9a-fA-F]{64}$/.test(hex)) {
    throw new Error(`expected 32-byte hex, got ${hex.length / 2} bytes`);
  }
  return `0x${hex.toLowerCase()}`;
}

function randomPrefix24Hex() {
  return `0x${crypto.randomBytes(24).toString("hex")}`;
}

function formatHashRate(rate) {
  if (rate >= 1e9) return `${(rate / 1e9).toFixed(2)} GH/s`;
  if (rate >= 1e6) return `${(rate / 1e6).toFixed(2)} MH/s`;
  if (rate >= 1e3) return `${(rate / 1e3).toFixed(2)} KH/s`;
  return `${rate.toFixed(0)} H/s`;
}

function formatBig(value, decimals = 18) {
  const formatted = formatUnits(value, decimals).replace(/\.?0+$/, "");
  return formatted || "0";
}

function makePublicClient(rpcUrl) {
  return createPublicClient({
    chain: mainnet,
    transport: http(rpcUrl, { timeout: RPC_TIMEOUT_MS, retryCount: 2 }),
  });
}

async function fileExists(file) {
  try {
    await fs.access(file);
    return true;
  } catch {
    return false;
  }
}

async function loadOfficialWasmMiner() {
  if (!(await fileExists(OFFICIAL_WASM_JS)) || !(await fileExists(OFFICIAL_WASM_BIN))) {
    throw new Error("official WASM assets missing; run: npm run assets");
  }
  const mod = await import(pathToFileURL(OFFICIAL_WASM_JS).href);
  const wasmBytes = await fs.readFile(OFFICIAL_WASM_BIN);
  await mod.default({ module_or_path: wasmBytes });
  return mod;
}

async function mineWithWasm({ challenge, difficulty, batchSize, seconds, progressEveryMs }) {
  const mod = await loadOfficialWasmMiner();
  const challengeBytes = hexToBytes(toBytes32Hex(challenge));
  const difficultyBytes = hexToBytes(toBytes32Hex(difficulty));
  const prefixBytes = crypto.randomBytes(24);
  const miner = new mod.Miner(challengeBytes, difficultyBytes, prefixBytes);
  const startedAt = Date.now();
  let lastProgress = startedAt;
  let counter = 0n;
  let hashes = 0n;
  const batch = BigInt(batchSize);

  try {
    while (true) {
      const hit = miner.search(counter, batch);
      const now = Date.now();
      if (hit) {
        const hitHashes = hit.hashes ?? batch;
        hashes += hitHashes;
        return {
          nonce: bytesToHex(hit.nonce),
          hash: bytesToHex(hit.result),
          hashes,
          elapsedMs: now - startedAt,
          backend: "wasm",
        };
      }
      counter += batch;
      hashes += batch;
      if (now - lastProgress >= progressEveryMs) {
        const rate = Number(hashes) / Math.max(0.001, (now - startedAt) / 1000);
        console.log(`[wasm] ${hashes.toLocaleString()} hashes · ${formatHashRate(rate)}`);
        lastProgress = now;
      }
      if (seconds && now - startedAt >= seconds * 1000) {
        return {
          nonce: null,
          hash: null,
          hashes,
          elapsedMs: now - startedAt,
          backend: "wasm",
          expired: true,
        };
      }
      await new Promise((resolve) => setImmediate(resolve));
    }
  } finally {
    miner.free();
  }
}

async function mineWithMetal({ challenge, difficulty, batchSize, seconds, progressEveryMs, quiet }) {
  if (!(await fileExists(METAL_BIN))) {
    throw new Error("Metal miner missing; run: npm run build:metal");
  }
  const args = [
    "--challenge",
    toBytes32Hex(challenge),
    "--difficulty",
    toBytes32Hex(difficulty),
    "--prefix",
    randomPrefix24Hex(),
    "--batch",
    String(batchSize),
    "--progress-ms",
    String(progressEveryMs),
  ];
  if (seconds) args.push("--cutoff-ms", String(Date.now() + seconds * 1000));

  const child = spawn(METAL_BIN, args, {
    cwd: __dirname,
    stdio: ["ignore", "pipe", "pipe"],
  });

  let stderr = "";
  child.stderr.on("data", (chunk) => {
    stderr += chunk.toString();
  });

  return await new Promise((resolve, reject) => {
    let buffer = "";
    child.stdout.on("data", (chunk) => {
      buffer += chunk.toString();
      const lines = buffer.split(/\r?\n/);
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (!line.trim()) continue;
        let message;
        try {
          message = JSON.parse(line);
        } catch {
          console.log(line);
          continue;
        }
        if (message.type === "progress") {
          if (!quiet) {
            console.log(
              `[metal] ${BigInt(message.hashes).toLocaleString()} hashes · ${formatHashRate(
                Number(message.hashrate),
              )}`,
            );
          }
        }
        if (message.type === "found") {
          resolve({
            nonce: message.nonce,
            hash: message.hash,
            hashes: BigInt(message.hashes),
            elapsedMs: Number(message.elapsedMs),
            backend: "metal",
          });
        }
        if (message.type === "expired") {
          resolve({
            nonce: null,
            hash: null,
            hashes: BigInt(message.hashes),
            elapsedMs: Number(message.elapsedMs),
            backend: "metal",
            expired: true,
          });
        }
      }
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) return;
      reject(new Error(stderr.trim() || `metal miner exited with code ${code}`));
    });
  });
}

async function mineWithCuda({ challenge, difficulty, batchSize, seconds, progressEveryMs, quiet, gpus }) {
  if (!(await fileExists(CUDA_BIN))) {
    throw new Error("CUDA miner missing; run: sh build-cuda.sh");
  }
  const args = [
    "--challenge",
    toBytes32Hex(challenge),
    "--difficulty",
    toBytes32Hex(difficulty),
    "--prefix",
    randomPrefix24Hex(),
    "--batch",
    String(batchSize),
    "--progress-ms",
    String(progressEveryMs),
  ];
  if (gpus) args.push("--gpus", String(gpus));
  if (seconds) args.push("--cutoff-ms", String(Date.now() + seconds * 1000));

  const child = spawn(CUDA_BIN, args, {
    cwd: __dirname,
    stdio: ["ignore", "pipe", "pipe"],
  });

  let stderr = "";
  child.stderr.on("data", (chunk) => {
    stderr += chunk.toString();
  });

  return await new Promise((resolve, reject) => {
    let buffer = "";
    child.stdout.on("data", (chunk) => {
      buffer += chunk.toString();
      const lines = buffer.split(/\r?\n/);
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (!line.trim()) continue;
        let message;
        try {
          message = JSON.parse(line);
        } catch {
          console.log(line);
          continue;
        }
        if (message.type === "progress") {
          if (!quiet) {
            console.log(
              `[cuda] ${BigInt(message.hashes).toLocaleString()} hashes · ${formatHashRate(
                Number(message.hashrate),
              )}`,
            );
          }
        }
        if (message.type === "found") {
          resolve({
            nonce: message.nonce,
            hash: message.hash,
            hashes: BigInt(message.hashes),
            elapsedMs: Number(message.elapsedMs),
            backend: "cuda",
          });
        }
        if (message.type === "expired") {
          resolve({
            nonce: null,
            hash: null,
            hashes: BigInt(message.hashes),
            elapsedMs: Number(message.elapsedMs),
            backend: "cuda",
            expired: true,
          });
        }
      }
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) return;
      reject(new Error(stderr.trim() || `cuda miner exited with code ${code}`));
    });
  });
}

async function chooseEngine(engine) {
  if (engine && engine !== "auto") return engine;
  if (await fileExists(CUDA_BIN)) return "cuda";
  return "wasm";
}

async function mineSolution(options) {
  const engine = await chooseEngine(options.engine);
  const defaultBatch = engine === "cuda" ? DEFAULT_CUDA_BATCH : DEFAULT_WASM_BATCH;
  const batchSize = options.batchSize || defaultBatch;
  const nextOptions = { ...options, batchSize };
  if (engine === "cuda") return mineWithCuda(nextOptions);
  if (engine === "wasm") return mineWithWasm(nextOptions);
  throw new Error(`unknown engine: ${engine}`);
}


function verifySolution(challenge, nonce, difficulty) {
  const digest = keccak256(concat([hexToBytes(toBytes32Hex(challenge)), hexToBytes(toBytes32Hex(nonce))]));
  const ok = BigInt(digest) < BigInt(toBytes32Hex(difficulty));
  return { ok, digest };
}

async function readChainState(client, address) {
  const [genesis, mining] = await Promise.all([
    client.readContract({
      address: HASH256_ADDRESS,
      abi: HASH256_ABI,
      functionName: "genesisState",
    }),
    client.readContract({
      address: HASH256_ADDRESS,
      abi: HASH256_ABI,
      functionName: "miningState",
    }),
  ]);
  let balance = null;
  let challenge = null;
  if (address) {
    [balance, challenge] = await Promise.all([
      client.readContract({
        address: HASH256_ADDRESS,
        abi: HASH256_ABI,
        functionName: "balanceOf",
        args: [address],
      }),
      client.readContract({
        address: HASH256_ADDRESS,
        abi: HASH256_ABI,
        functionName: "getChallenge",
        args: [address],
      }),
    ]);
  }
  return { genesis, mining, balance, challenge };
}

async function printStatus(args) {
  const rpc = args.rpc || DEFAULT_RPC;
  const client = makePublicClient(rpc);
  const privateKey = args["private-key"] || process.env.PRIVATE_KEY;
  const address = args.address || (privateKey ? privateKeyToAccount(privateKey).address : null);
  const state = await readChainState(client, address);
  const [minted, remaining, ethRaised, complete] = state.genesis;
  const [era, reward, difficulty, miningMinted, miningRemaining, epoch, epochBlocksLeft] =
    state.mining;
  console.log(`HASH256 contract: ${HASH256_ADDRESS}`);
  console.log(`RPC: ${rpc}`);
  console.log(`genesis: ${complete ? "complete" : "not complete"} · sold ${formatBig(minted)} HASH · remaining ${formatBig(remaining)} HASH · raised ${formatEther(ethRaised)} ETH`);
  console.log(`mining: era ${era} · reward ${formatBig(reward)} HASH · epoch ${epoch} · rotates in ${epochBlocksLeft} blocks`);
  console.log(`difficulty target: ${toBytes32Hex(difficulty)}`);
  console.log(`mined supply: ${formatBig(miningMinted)} HASH · remaining ${formatBig(miningRemaining)} HASH`);
  if (address) {
    console.log(`miner: ${address}`);
    console.log(`balance: ${formatBig(state.balance)} HASH`);
    console.log(`challenge: ${state.challenge}`);
  }
}

async function runBench(args) {
  const seconds = Number(args.seconds || 5);
  const result = await mineSolution({
    challenge: args.challenge || `0x${"00".repeat(32)}`,
    difficulty: args.difficulty || `0x0000${"ff".repeat(30)}`,
    engine: args.engine || "auto",
    batchSize: args.batch ? Number(args.batch) : undefined,
    seconds,
    progressEveryMs: Number(args["progress-ms"] || 1000),
    gpus: args.gpus ? Number(args.gpus) : undefined,
  });
  const rate = Number(result.hashes) / Math.max(0.001, result.elapsedMs / 1000);
  console.log(`bench: ${result.backend} · ${result.hashes.toLocaleString()} hashes · ${formatHashRate(rate)}`);
  if (result.nonce) {
    console.log(`found nonce ${result.nonce}`);
    console.log(`hash ${result.hash}`);
  }
}

function parseBatchCandidates(value) {
  if (!value) return [8_000_000, 64_000_000, 128_000_000, 256_000_000, 512_000_000];
  return String(value)
    .split(",")
    .map((item) => Number(item.trim()))
    .filter((item) => Number.isFinite(item) && item > 0);
}

async function runTune(args) {
  const seconds = Number(args.seconds || 4);
  const candidates = parseBatchCandidates(args.batches);
  if (!candidates.length) throw new Error("tune needs at least one positive batch size");

  console.log(`tuning Metal batch size · ${seconds}s per candidate`);
  let best = null;
  for (const batch of candidates) {
    const result = await mineSolution({
      challenge: `0x${"00".repeat(32)}`,
      difficulty: `0x${"00".repeat(32)}`,
      engine: "metal",
      batchSize: batch,
      seconds,
      progressEveryMs: Number(args["progress-ms"] || 60_000),
      quiet: true,
    });
    const rate = Number(result.hashes) / Math.max(0.001, result.elapsedMs / 1000);
    console.log(`batch ${batch.toLocaleString()}: ${formatHashRate(rate)}`);
    if (!best || rate > best.rate) best = { batch, rate };
  }
  console.log(`best: --batch ${best.batch} · ${formatHashRate(best.rate)}`);
}

async function runSolve(args) {
  if (!args.challenge || !args.difficulty) {
    throw new Error("solve requires --challenge and --difficulty");
  }
  const result = await mineSolution({
    challenge: args.challenge,
    difficulty: args.difficulty,
    engine: args.engine || "auto",
    batchSize: args.batch ? Number(args.batch) : undefined,
    seconds: args.seconds ? Number(args.seconds) : 0,
    progressEveryMs: Number(args["progress-ms"] || 1000),
    gpus: args.gpus ? Number(args.gpus) : undefined,
  });
  if (!result.nonce) {
    console.log(`no solution before cutoff · ${result.hashes.toLocaleString()} hashes`);
    return;
  }
  const check = verifySolution(args.challenge, result.nonce, args.difficulty);
  console.log(`backend: ${result.backend}`);
  console.log(`nonce: ${result.nonce}`);
  console.log(`hash: ${check.digest}`);
  console.log(`valid: ${check.ok}`);
  console.log(`hashes: ${result.hashes.toLocaleString()} · elapsed ${(result.elapsedMs / 1000).toFixed(2)}s`);
}

async function submitMine({ client, privateKey, nonce, tipGwei, rpc }) {
  if (!privateKey) throw new Error("submit requires PRIVATE_KEY or --private-key");
  const account = privateKeyToAccount(privateKey);
  const wallet = createWalletClient({
    account,
    chain: mainnet,
    transport: http(rpc || DEFAULT_RPC, { timeout: RPC_TIMEOUT_MS, retryCount: 2 }),
  });

  let gas = 300000n;
  try {
    const estimate = await client.estimateContractGas({
      address: HASH256_ADDRESS,
      abi: HASH256_ABI,
      functionName: "mine",
      args: [BigInt(nonce)],
      account,
    });
    gas = (estimate * 3n) / 2n;
    if (gas < 200000n) gas = 200000n;
    if (gas > 400000n) gas = 400000n;
  } catch (error) {
    console.warn(`[warn] gas estimate failed, using fallback: ${error.shortMessage || error.message}`);
  }

  const priority = parseGwei(String(tipGwei || 2));
  let maxFeePerGas = priority + parseGwei("5");
  try {
    const block = await client.getBlock();
    if (block.baseFeePerGas) {
      maxFeePerGas = block.baseFeePerGas * 3n + priority;
    }
  } catch {}

  return await wallet.writeContract({
    address: HASH256_ADDRESS,
    abi: HASH256_ABI,
    functionName: "mine",
    args: [BigInt(nonce)],
    gas,
    maxFeePerGas,
    maxPriorityFeePerGas: priority,
  });
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function roundSecondsForState(args, state) {
  if (args.seconds) return Number(args.seconds);
  if (args["round-seconds"]) return Number(args["round-seconds"]);

  const blocksLeft = Number(state.mining[6]);
  if (Number.isFinite(blocksLeft) && blocksLeft > 0) {
    const secondsLeft = blocksLeft * 12;
    return Math.max(5, Math.min(DEFAULT_MINE_ROUND_SECONDS, secondsLeft - 8));
  }
  return DEFAULT_MINE_ROUND_SECONDS;
}

async function runMineRound({ args, client, rpc, privateKey, address }) {
  const state = await readChainState(client, address);
  if (!state.genesis[3] && !args.force) {
    console.log("Mining is not open yet: genesis is not complete. Use status to watch it.");
    return { stop: true };
  }

  const challenge = state.challenge;
  const difficulty = toBytes32Hex(state.mining[2]);
  const roundSeconds = roundSecondsForState(args, state);
  console.log(`miner: ${address}`);
  console.log(`epoch: ${state.mining[5]} · rotates in ${state.mining[6]} blocks · round ${roundSeconds}s`);
  console.log(`challenge: ${challenge}`);
  console.log(`difficulty: ${difficulty}`);

  const result = await mineSolution({
    challenge,
    difficulty,
    engine: args.engine || "auto",
    batchSize: args.batch ? Number(args.batch) : undefined,
    seconds: roundSeconds,
    progressEveryMs: Number(args["progress-ms"] || 1000),
    gpus: args.gpus ? Number(args.gpus) : undefined,
  });

  if (!result.nonce) {
    console.log(`no solution this round · ${result.hashes.toLocaleString()} hashes`);
    return { found: false };
  }

  const check = verifySolution(challenge, result.nonce, difficulty);
  console.log(`found nonce: ${result.nonce}`);
  console.log(`hash: ${check.digest}`);
  console.log(`valid: ${check.ok}`);
  if (!check.ok) throw new Error("local verification failed; refusing to submit");

  const fresh = await readChainState(client, address);
  const freshDifficulty = toBytes32Hex(fresh.mining[2]);
  if (fresh.challenge.toLowerCase() !== challenge.toLowerCase() || freshDifficulty !== difficulty) {
    console.log("challenge changed before submit; skipping stale nonce and refreshing.");
    return { found: true, stale: true };
  }

  const shouldSubmit = Boolean(args.submit);
  if (!shouldSubmit) {
    console.log("not submitted. Re-run with --submit and PRIVATE_KEY to call mine(nonce).");
    return { found: true, submitted: false, stop: true };
  }
  const tx = await submitMine({
    client,
    privateKey,
    nonce: result.nonce,
    tipGwei: Number(args.tip || 2),
    rpc,
  });
  console.log(`submitted: ${tx}`);
  console.log(`etherscan: https://etherscan.io/tx/${tx}`);
  return { found: true, submitted: true, tx };
}

async function runMine(args) {
  const rpc = args.rpc || DEFAULT_RPC;
  const client = makePublicClient(rpc);
  const privateKey = args["private-key"] || process.env.PRIVATE_KEY;
  const accountAddress = privateKey ? privateKeyToAccount(privateKey).address : null;
  const address = args.address || accountAddress;
  if (!address || !isAddress(address)) throw new Error("mine requires --address 0x... or PRIVATE_KEY");

  const loop = Boolean(args.loop || args.forever || args.watch);
  while (true) {
    const outcome = await runMineRound({ args, client, rpc, privateKey, address });
    if (!loop || outcome.stop) return;
    await sleep(Number(args["sleep-ms"] || 2000));
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const command = args._[0] || "help";
  if (command === "help" || args.help) {
    usage();
    return;
  }
  if (command === "status") return printStatus(args);
  if (command === "bench") return runBench(args);
  if (command === "tune") return runTune(args);
  if (command === "solve") return runSolve(args);
  if (command === "mine") return runMine(args);
  throw new Error(`unknown command: ${command}`);
}

main().catch((error) => {
  console.error(`[error] ${error.shortMessage || error.message}`);
  process.exitCode = 1;
});
