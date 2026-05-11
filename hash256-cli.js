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
const DEFAULT_CUDA_BATCH = 512_000_000;
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
  node hash256-cli.js bench [--seconds 5]
  node hash256-cli.js tune [--seconds 4] [--batches 64000000,256000000,512000000]
  node hash256-cli.js solve --challenge 0x... --difficulty 0x...
  node hash256-cli.js mine --address 0x... [--loop]
  node hash256-cli.js mine --submit [--private-key 0x...] [--loop]

Environment:
  HASH256_RPC_URL     Ethereum RPC, default ${DEFAULT_RPC}
  PRIVATE_KEY         Optional private key for --submit; can be set in .env
  HASH256_WORKERS     Number of worker processes (default: 1)

Notes:
  - --address is read-only; PRIVATE_KEY/--private-key is required to submit.
  - --loop refreshes chain challenge between mining rounds.
  - Requires built hash256-cuda-miner binary (run: sh build-cuda.sh).
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

async function mineWithWasm({ challenge, difficulty, batchSize, seconds, progressEveryMs, workers = 1 }) {
  const mod = await loadOfficialWasmMiner();
  const challengeBytes = hexToBytes(toBytes32Hex(challenge));
  const difficultyBytes = hexToBytes(toBytes32Hex(difficulty));
  const batch = BigInt(batchSize);
  const workerCount = Number(workers || process.env.HASH256_WORKERS || 1);

  // Create multiple miner instances (each with its own 24-byte prefix) and partition the
  // search space by advancing counters by batch*workerCount to avoid overlap.
  const miners = [];
  for (let i = 0; i < workerCount; i += 1) {
    const prefixBytes = crypto.randomBytes(24);
    const miner = new mod.Miner(challengeBytes, difficultyBytes, prefixBytes);
    miners.push({ miner, prefixBytes, counter: BigInt(i) * batch });
  }

  const startedAt = Date.now();
  let lastProgress = startedAt;
  let totalHashes = 0n;

  try {
    const workerPromises = miners.map((entry, idx) =>
      (async () => {
        const m = entry.miner;
        let counter = entry.counter;
        while (true) {
          const hit = m.search(counter, batch);
          const now = Date.now();
          if (hit) {
            const hitHashes = hit.hashes ?? batch;
            totalHashes += BigInt(hitHashes);
            return {
              nonce: bytesToHex(hit.nonce),
              hash: bytesToHex(hit.result),
              hashes: totalHashes,
              elapsedMs: now - startedAt,
              backend: "wasm",
            };
          }
          counter += batch * BigInt(workerCount);
          totalHashes += batch;

          if (now - lastProgress >= progressEveryMs) {
            const rate = Number(totalHashes) / Math.max(0.001, (now - startedAt) / 1000);
            console.log(`[wasm] ${totalHashes.toString()} hashes · ${formatHashRate(rate)}`);
            lastProgress = now;
          }
          if (seconds && now - startedAt >= seconds * 1000) {
            return {
              nonce: null,
              hash: null,
              hashes: totalHashes,
              elapsedMs: now - startedAt,
              backend: "wasm",
              expired: true,
            };
          }
          await new Promise((resolve) => setImmediate(resolve));
        }
      })(),
    );

    const result = await Promise.race(workerPromises);
    return result;
  } finally {
    for (const m of miners) {
      try {
        m.miner.free();
      } catch (e) {}
    }
  }
}

async function mineWithMetal({ challenge, difficulty, batchSize, seconds, progressEveryMs, quiet, workers = 1 }) {
  if (!(await fileExists(METAL_BIN))) {
    throw new Error("Metal miner missing; run: npm run build:metal");
  }
  const workerCount = Number(workers || process.env.HASH256_WORKERS || 1);

  // Helper to spawn one child with its own random prefix
  function spawnWorker() {
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
    return child;
  }

  if (workerCount <= 1) {
    // single worker behaviour (keep existing code path)
    const child = spawnWorker();
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
                `[metal] ${BigInt(message.hashes).toLocaleString()} hashes · ${formatHashRate(Number(message.hashrate))}`,
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

  // multiple workers: spawn many processes and race them
  const children = [];
  let settled = false;

  const promises = Array.from({ length: workerCount }, () => {
    return new Promise((resolve, reject) => {
      const child = spawnWorker();
      children.push(child);
      let stderr = "";
      child.stderr.on("data", (chunk) => (stderr += chunk.toString()));
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
                `[metal] ${BigInt(message.hashes).toLocaleString()} hashes · ${formatHashRate(Number(message.hashrate))}`,
              );
            }
          }
          if (message.type === "found") {
            if (settled) return;
            settled = true;
            resolve({
              nonce: message.nonce,
              hash: message.hash,
              hashes: BigInt(message.hashes),
              elapsedMs: Number(message.elapsedMs),
              backend: "metal",
            });
          }
          if (message.type === "expired") {
            if (settled) return;
            settled = true;
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
        if (!settled) reject(new Error(stderr.trim() || `metal miner exited with code ${code}`));
      });
    });
  });

  try {
    const result = await Promise.race(promises);
    // kill others
    for (const c of children) {
      try {
        c.kill();
      } catch (e) {}
    }
    return result;
  } finally {
    for (const c of children) try { c.kill(); } catch (e) {}
  }
}

async function mineWithCuda({ challenge, difficulty, batchSize, seconds, progressEveryMs, quiet, workers = 1 }) {
  if (!(await fileExists(CUDA_BIN))) {
    throw new Error("CUDA miner missing; run: sh build-cuda.sh");
  }
  const workerCount = Number(workers || process.env.HASH256_WORKERS || 1);

  function spawnWorker() {
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

    const child = spawn(CUDA_BIN, args, {
      cwd: __dirname,
      stdio: ["ignore", "pipe", "pipe"],
    });
    return child;
  }

  if (workerCount <= 1) {
    const child = spawnWorker();
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
                `[cuda] ${BigInt(message.hashes).toLocaleString()} hashes · ${formatHashRate(Number(message.hashrate))}`,
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

  // multiple workers: spawn many processes and race them
  const children = [];
  let settled = false;
  const promises = Array.from({ length: workerCount }, () => {
    return new Promise((resolve, reject) => {
      const child = spawnWorker();
      children.push(child);
      let stderr = "";
      child.stderr.on("data", (chunk) => (stderr += chunk.toString()));
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
                `[cuda] ${BigInt(message.hashes).toLocaleString()} hashes · ${formatHashRate(Number(message.hashrate))}`,
              );
            }
          }
          if (message.type === "found") {
            if (settled) return;
            settled = true;
            resolve({
              nonce: message.nonce,
              hash: message.hash,
              hashes: BigInt(message.hashes),
              elapsedMs: Number(message.elapsedMs),
              backend: "cuda",
            });
          }
          if (message.type === "expired") {
            if (settled) return;
            settled = true;
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
        if (!settled) reject(new Error(stderr.trim() || `cuda miner exited with code ${code}`));
      });
    });
  });

  try {
    const result = await Promise.race(promises);
    for (const c of children) try { c.kill(); } catch (e) {}
    return result;
  } finally {
    for (const c of children) try { c.kill(); } catch (e) {}
  }
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
  const workers = Number(options.workers || process.env.HASH256_WORKERS || 1);
  const nextOptions = { ...options, batchSize, workers };
  if (engine === "cuda") return mineWithCuda(nextOptions);
  if (engine === "wasm") return mineWithWasm(nextOptions);
  if (engine === "metal") return mineWithMetal(nextOptions);
  throw new Error(`unknown engine: ${engine}`);
}

async function main(argv) {
  const args = parseArgs(argv.slice(2));
  const cmd = args._[0];

  if (cmd === "status") {
    const address = args.address || privateKeyToAccount(process.env.PRIVATE_KEY || "").address;
    if (!address || !isAddress(address)) {
      throw new Error(`invalid address: ${address}`);
    }
    const client = makePublicClient(args.rpc || DEFAULT_RPC);
    const [mining, workerCount, activeWorkerCount, challenge, difficulty, gasPrice] = await Promise.all([
      client.getMining(),
      client.getWorkerCount(),
      client.getActiveWorkerCount(),
      client.getChainChallenge(),
      client.getChainDifficulty(),
      client.getGasPrice(),
    ]);
    const hashrate = BigInt(mining.hashes) / BigInt(mining.elapsed.pre);
    const formattedHashrate = formatHashRate(Number(hashrate));
    const formattedBalance = formatBig(await client.getBalance(address));
    const block = await client.getBlock("latest");
    const blockNumber = block.number ?? "unknown";

    console.log(`

Account: ${address}
Balance: ${formattedBalance} ETH

Network: ${client.chain.name} (${client.chain.id})
Block:    ${blockNumber}

Mining:   ${mining.active ? "yes" : "no"} (${activeWorkerCount}/${workerCount} workers active)
Hashrate: ${formattedHashrate} (${mining.hashes.toString()} hashes in ${mining.elapsed.pre.toFixed(2)} seconds)

Challenge: ${challenge}
Difficulty: ${difficulty}

Gas Price: ${formatUnits(gasPrice, "gwei")} gwei
`);
  } else if (cmd === "bench") {
    const seconds = Number(args.seconds || 5);
    const client = makePublicClient(args.rpc || DEFAULT_RPC);
    const block = await client.getBlock("latest");
    const baseFee = block.baseFeePerGas ?? 0n;
    const gasPrice = baseFee * 2n;

    console.log(`

Benchmarking gas fees (please wait ${seconds} seconds)...

Base Fee: ${formatUnits(baseFee, "gwei")} gwei
Gas Price: ${formatUnits(gasPrice, "gwei")} gwei
`);

    const results = [];
    for (let i = 0; i < 3; i += 1) {
      const start = Date.now();
      let total = 0n;
      let count = 0;
      while (Date.now() - start < seconds * 1000) {
        const result = await client.simulateTransaction({
          to: HASH256_ADDRESS,
          data: "0x",
          gasPrice,
        });
        total += result.gasUsed;
        count += 1;
      }
      const average = total / BigInt(count);
      results.push(average);
      console.log(`  Trial ${i + 1}: ${average.toString()} gas`);
    }
    const median = results.sort((a, b) => Number(a - b))[Math.floor(results.length / 2)];
    console.log(`

Median: ${median.toString()} gas

Recommended Gas Price: ${formatUnits(gasPrice, "gwei")} gwei (base fee * 2)
`);
  } else if (cmd === "tune") {
    const seconds = Number(args.seconds || 4);
    const batches = (args.batches || "").split(",").map((v) => Number(v.trim()));
    const client = makePublicClient(args.rpc || DEFAULT_RPC);
    const block = await client.getBlock("latest");
    const baseFee = block.baseFeePerGas ?? 0n;
    const gasPrice = baseFee * 2n;

    console.log(`

Tuning miner settings (please wait ${seconds} seconds)...

Base Fee: ${formatUnits(baseFee, "gwei")} gwei
Gas Price: ${formatUnits(gasPrice, "gwei")} gwei
`);

    const results = [];
    for (const batchSize of batches) {
      const start = Date.now();
      let total = 0n;
      let count = 0;
      while (Date.now() - start < seconds * 1000) {
        const result = await client.simulateTransaction({
          to: HASH256_ADDRESS,
          data: "0x",
          gasPrice,
        });
        total += result.gasUsed;
        count += 1;
      }
      const average = total / BigInt(count);
      results.push({ batchSize, average });
      console.log(`  Batch Size ${batchSize}: ${average.toString()} gas`);
    }
    const best = results.sort((a, b) => Number(a.average - b.average))[0];
    console.log(`

Best Batch Size: ${best.batchSize} (average ${best.average.toString()} gas)

Recommended Gas Price: ${formatUnits(gasPrice, "gwei")} gwei (base fee * 2)
`);
  } else if (cmd === "solve") {
    const challenge = args.challenge;
    const difficulty = args.difficulty;
    if (!challenge || !difficulty) {
      throw new Error("missing --challenge or --difficulty");
    }
    const client = makePublicClient(args.rpc || DEFAULT_RPC);
    const result = await client.call({
      to: HASH256_ADDRESS,
      data: concat([HASH256_ABI.encodeFunctionData("mine", [challenge, difficulty])]),
    });
    const decoded = HASH256_ABI.decodeFunctionResult("mine", result);
    const nonce = bytesToHex(decoded.nonce);
    const hash = bytesToHex(decoded.hash);

    console.log(`

Solution Found!

Nonce: ${nonce}
Hash:  ${hash}

Submit using: node hash256-cli.js mine --submit --private-key YOUR_PRIVATE_KEY
`);
  } else if (cmd === "mine") {
    const address = args.address || privateKeyToAccount(process.env.PRIVATE_KEY || "").address;
    if (!address || !isAddress(address)) {
      throw new Error(`invalid address: ${address}`);
    }
    const client = makePublicClient(args.rpc || DEFAULT_RPC);
    const loop = args.loop === true;
    const submit = args.submit === true;
    const quiet = args.quiet === true;
    const engine = args.engine || "auto";
    const seconds = Number(args.seconds || DEFAULT_MINE_ROUND_SECONDS);
    const workers = Number(args.workers || process.env.HASH256_WORKERS || 1);
    const batches = (args.batches || "").split(",").map((v) => Number(v.trim()));
    let gasPrice = 0n;

    // Pre-fetch block data
    const block = await client.getBlock("latest");
    const baseFee = block.baseFeePerGas ?? 0n;
    if (submit) {
      gasPrice = baseFee * 2n;
      console.log(`Using gas price: ${formatUnits(gasPrice, "gwei")} gwei (base fee * 2)`);
    }

    // Tune batch size if requested
    let batchSize = DEFAULT_WASM_BATCH;
    if (batches.length) {
      const results = [];
      for (const b of batches) {
        const start = Date.now();
        let total = 0n;
        let count = 0;
        while (Date.now() - start < seconds * 1000) {
          const result = await client.simulateTransaction({
            to: HASH256_ADDRESS,
            data: "0x",
            gasPrice,
          });
          total += result.gasUsed;
          count += 1;
        }
        const average = total / BigInt(count);
        results.push({ batchSize: b, average });
        console.log(`  Batch Size ${b}: ${average.toString()} gas`);
      }
      const best = results.sort((a, b) => Number(a.average - b.average))[0];
      batchSize = best.batchSize;
      console.log(`Best Batch Size: ${batchSize}`);
    }

    // Mining loop
    let totalMined = 0n;
    let totalTime = 0;
    let lastHashrate = 0;
    let lastTime = Date.now();
    while (true) {
      const challenge = await client.getChainChallenge();
      const difficulty = await client.getChainDifficulty();
      const start = Date.now();

      // Submit a new mining job
      let result;
      try {
        result = await mineSolution({
          challenge,
          difficulty,
          batchSize,
          seconds,
          workers,
          engine,
        });
      } catch (e) {
        console.error("Error mining:", e);
        await new Promise((resolve) => setTimeout(resolve, 5000));
        continue;
      }

      totalMined += BigInt(result.hashes);
      const elapsed = Date.now() - start;
      totalTime += elapsed;

      // Report hashrate every few seconds
      if (elapsed > 10000) {
        lastHashrate = (totalMined / BigInt(totalTime)) * 1000n;
        const formattedHashrate = formatHashRate(Number(lastHashrate));
        console.log(`[miner] ${totalMined.toString()} hashes · ${formattedHashrate}`);
        lastTime = Date.now();
      }

      // Submit solution if requested
      if (submit && result.nonce) {
        try {
          const tx = await client.sendTransaction({
            to: HASH256_ADDRESS,
            data: concat([HASH256_ABI.encodeFunctionData("submit", [result.nonce, result.hash])]),
            gasPrice,
          });
          console.log(`Submitted solution: ${tx.hash}`);
        } catch (e) {
          console.error("Error submitting solution:", e);
        }
      }

      // Check for loop flag
      if (!loop) break;

      // Wait for next challenge
      const waitTime = Math.max(0, seconds * 1000 - (Date.now() - start));
      await new Promise((resolve) => setTimeout(resolve, waitTime));
    }
  } else {
    usage();
    process.exit(1);
  }
}

main(process.argv).catch((error) => {
  console.error(error);
  process.exit(1);
});
