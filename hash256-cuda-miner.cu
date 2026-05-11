// hash256-cuda-miner.cu
// CUDA Keccak-256 GPU miner for HASH256 (NVIDIA GPUs)
// Same CLI interface and JSON protocol as hash256-metal-miner

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Keccak-256 round constants
// ---------------------------------------------------------------------------
__device__ __constant__ uint64_t KECCAK_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL,
};

// ---------------------------------------------------------------------------
// Kernel constants
// ---------------------------------------------------------------------------
static constexpr int ITERATIONS = 16;
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint64_t rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

__device__ __forceinline__ uint64_t bswap64(uint64_t v) {
    v = ((v & 0x00000000FFFFFFFFULL) << 32) | ((v & 0xFFFFFFFF00000000ULL) >> 32);
    v = ((v & 0x0000FFFF0000FFFFULL) << 16) | ((v & 0xFFFF0000FFFF0000ULL) >> 16);
    v = ((v & 0x00FF00FF00FF00FFULL) <<  8) | ((v & 0xFF00FF00FF00FF00ULL) >>  8);
    return v;
}

// ---------------------------------------------------------------------------
// Keccak-f[1600] permutation (25 x uint64_t state)
// ---------------------------------------------------------------------------
__device__ void keccak_f1600(uint64_t s[25]) {
    for (int r = 0; r < 24; r++) {
        // Theta
        uint64_t C0 = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
        uint64_t C1 = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
        uint64_t C2 = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
        uint64_t C3 = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
        uint64_t C4 = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];

        uint64_t D0 = C4 ^ rotl64(C1, 1);
        uint64_t D1 = C0 ^ rotl64(C2, 1);
        uint64_t D2 = C1 ^ rotl64(C3, 1);
        uint64_t D3 = C2 ^ rotl64(C4, 1);
        uint64_t D4 = C3 ^ rotl64(C0, 1);

        // Theta + Rho + Pi
        uint64_t b00 = s[ 0] ^ D0;
        uint64_t b10 = rotl64(s[ 1] ^ D1,  1);
        uint64_t b20 = rotl64(s[ 2] ^ D2, 62);
        uint64_t b05 = rotl64(s[ 3] ^ D3, 28);
        uint64_t b15 = rotl64(s[ 4] ^ D4, 27);
        uint64_t b16 = rotl64(s[ 5] ^ D0, 36);
        uint64_t b01 = rotl64(s[ 6] ^ D1, 44);
        uint64_t b11 = rotl64(s[ 7] ^ D2,  6);
        uint64_t b21 = rotl64(s[ 8] ^ D3, 55);
        uint64_t b06 = rotl64(s[ 9] ^ D4, 20);
        uint64_t b07 = rotl64(s[10] ^ D0,  3);
        uint64_t b17 = rotl64(s[11] ^ D1, 10);
        uint64_t b02 = rotl64(s[12] ^ D2, 43);
        uint64_t b12 = rotl64(s[13] ^ D3, 25);
        uint64_t b22 = rotl64(s[14] ^ D4, 39);
        uint64_t b23 = rotl64(s[15] ^ D0, 41);
        uint64_t b08 = rotl64(s[16] ^ D1, 45);
        uint64_t b18 = rotl64(s[17] ^ D2, 15);
        uint64_t b03 = rotl64(s[18] ^ D3, 21);
        uint64_t b13 = rotl64(s[19] ^ D4,  8);
        uint64_t b14 = rotl64(s[20] ^ D0, 18);
        uint64_t b24 = rotl64(s[21] ^ D1,  2);
        uint64_t b09 = rotl64(s[22] ^ D2, 61);
        uint64_t b19 = rotl64(s[23] ^ D3, 56);
        uint64_t b04 = rotl64(s[24] ^ D4, 14);

        // Chi
        s[ 0] = b00 ^ (~b01 & b02);
        s[ 1] = b01 ^ (~b02 & b03);
        s[ 2] = b02 ^ (~b03 & b04);
        s[ 3] = b03 ^ (~b04 & b00);
        s[ 4] = b04 ^ (~b00 & b01);
        s[ 5] = b05 ^ (~b06 & b07);
        s[ 6] = b06 ^ (~b07 & b08);
        s[ 7] = b07 ^ (~b08 & b09);
        s[ 8] = b08 ^ (~b09 & b05);
        s[ 9] = b09 ^ (~b05 & b06);
        s[10] = b10 ^ (~b11 & b12);
        s[11] = b11 ^ (~b12 & b13);
        s[12] = b12 ^ (~b13 & b14);
        s[13] = b13 ^ (~b14 & b10);
        s[14] = b14 ^ (~b10 & b11);
        s[15] = b15 ^ (~b16 & b17);
        s[16] = b16 ^ (~b17 & b18);
        s[17] = b17 ^ (~b18 & b19);
        s[18] = b18 ^ (~b19 & b15);
        s[19] = b19 ^ (~b15 & b16);
        s[20] = b20 ^ (~b21 & b22);
        s[21] = b21 ^ (~b22 & b23);
        s[22] = b22 ^ (~b23 & b24);
        s[23] = b23 ^ (~b24 & b20);
        s[24] = b24 ^ (~b20 & b21);

        // Iota
        s[0] ^= KECCAK_RC[r];
    }
}

// ---------------------------------------------------------------------------
// GPU uniforms & result
// ---------------------------------------------------------------------------
struct Uniforms {
    uint64_t challenge[4];   // 32 bytes LE for Keccak state loading
    uint64_t difficulty[4];  // 32 bytes BE for comparison
    uint64_t prefix[3];      // 24 bytes LE for Keccak state loading
    uint64_t nonce_base;     // linear counter base
};

struct ResultBuffer {
    unsigned int found;
    unsigned int pad;
    uint64_t nonce_counter;  // the winning counter value
    uint64_t hash[4];        // 32-byte hash as BE uint64s
};

// ---------------------------------------------------------------------------
// Mining kernel
// ---------------------------------------------------------------------------
__global__ void hash256_mine(const Uniforms* __restrict__ u,
                             ResultBuffer* __restrict__ result) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t thread_start = gid * ITERATIONS;

    for (int k = 0; k < ITERATIONS; k++) {
        uint64_t counter = u->nonce_base + thread_start + k;

        // Build Keccak state: 64 bytes input = challenge(32) || nonce(32)
        // nonce = prefix(24) || counter_be(8)
        uint64_t st[25];
        st[0] = u->challenge[0];
        st[1] = u->challenge[1];
        st[2] = u->challenge[2];
        st[3] = u->challenge[3];
        st[4] = u->prefix[0];
        st[5] = u->prefix[1];
        st[6] = u->prefix[2];
        st[7] = bswap64(counter);          // counter as big-endian bytes → LE state
        st[8] = 0x0000000000000001ULL;      // Keccak padding: 0x01 at byte 64
        for (int i = 9; i < 25; i++) st[i] = 0;
        st[16] = 0x8000000000000000ULL;     // Keccak final: 0x80 at byte 135

        keccak_f1600(st);

        // Convert hash to big-endian for comparison
        uint64_t h0 = bswap64(st[0]);
        uint64_t h1 = bswap64(st[1]);
        uint64_t h2 = bswap64(st[2]);
        uint64_t h3 = bswap64(st[3]);

        uint64_t d0 = u->difficulty[0];
        uint64_t d1 = u->difficulty[1];
        uint64_t d2 = u->difficulty[2];
        uint64_t d3 = u->difficulty[3];

        // Compare hash < difficulty (256-bit big-endian)
        bool lt = false;
        if      (h0 < d0) lt = true;
        else if (h0 == d0) {
            if      (h1 < d1) lt = true;
            else if (h1 == d1) {
                if      (h2 < d2) lt = true;
                else if (h2 == d2) {
                    if  (h3 < d3) lt = true;
                }
            }
        }

        if (lt) {
            unsigned int prior = atomicAdd(&result->found, 1u);
            if (prior == 0u) {
                result->nonce_counter = counter;
                result->hash[0] = h0;
                result->hash[1] = h1;
                result->hash[2] = h2;
                result->hash[3] = h3;
            }
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------
static uint64_t nowEpochMs() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

static uint64_t nowSteadyMs() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

static std::vector<uint8_t> parseHex(std::string hex, size_t bytes, const char* name) {
    if (hex.rfind("0x", 0) == 0 || hex.rfind("0X", 0) == 0) hex = hex.substr(2);
    if (hex.size() != bytes * 2) {
        throw std::runtime_error(std::string(name) + " must be " + std::to_string(bytes) + " bytes");
    }
    std::vector<uint8_t> out(bytes);
    for (size_t i = 0; i < bytes; i++) {
        std::string part = hex.substr(i * 2, 2);
        out[i] = static_cast<uint8_t>(std::stoul(part, nullptr, 16));
    }
    return out;
}

// Read 8 bytes as little-endian uint64
static uint64_t le64(const std::vector<uint8_t>& b, size_t off) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)b[off + i] << (i * 8);
    return v;
}

// Read 8 bytes as big-endian uint64
static uint64_t be64(const std::vector<uint8_t>& b, size_t off) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)b[off + i] << ((7 - i) * 8);
    return v;
}

static std::string bytesToHex(const std::vector<uint8_t>& bytes) {
    std::ostringstream out;
    out << "0x" << std::hex << std::setfill('0');
    for (auto b : bytes) out << std::setw(2) << (int)b;
    return out.str();
}

static std::string u64BeToHex(const uint64_t* words, size_t count) {
    std::ostringstream out;
    out << "0x" << std::hex << std::setfill('0');
    for (size_t i = 0; i < count; i++) out << std::setw(16) << words[i];
    return out.str();
}

// Build the full 32-byte nonce hex from prefix + counter
static std::string nonceHex(const std::vector<uint8_t>& prefix, uint64_t counter) {
    std::vector<uint8_t> nonce(32, 0);
    for (size_t i = 0; i < 24; i++) nonce[i] = prefix[i];
    for (int i = 0; i < 8; i++) nonce[24 + i] = (uint8_t)((counter >> ((7 - i) * 8)) & 0xff);
    return bytesToHex(nonce);
}

static std::string getArg(int argc, char** argv, const std::string& key, const std::string& fallback = "") {
    for (int i = 1; i < argc; i++) {
        std::string item(argv[i]);
        if (item == key && i + 1 < argc) return argv[i + 1];
        if (item.rfind(key + "=", 0) == 0) return item.substr(key.size() + 1);
    }
    return fallback;
}

static uint64_t parseU64(const std::string& value, uint64_t fallback) {
    if (value.empty()) return fallback;
    return std::stoull(value);
}

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
    } \
} while(0)

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    try {
        auto challengeBytes = parseHex(getArg(argc, argv, "--challenge"), 32, "challenge");
        auto difficultyBytes = parseHex(getArg(argc, argv, "--difficulty"), 32, "difficulty");
        auto prefixBytes = parseHex(
            getArg(argc, argv, "--prefix", "0x000000000000000000000000000000000000000000000000"),
            24, "prefix");
        uint64_t requestedBatch = parseU64(getArg(argc, argv, "--batch"), 512000000ULL);
        uint64_t progressEveryMs = parseU64(getArg(argc, argv, "--progress-ms"), 1000);
        uint64_t cutoffMs = parseU64(getArg(argc, argv, "--cutoff-ms"), 0);

        // Select GPU device
        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) throw std::runtime_error("No CUDA devices found");
        CUDA_CHECK(cudaSetDevice(0));

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        fprintf(stderr, "CUDA device: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);

        // Prepare uniforms
        Uniforms h_uniforms;
        memset(&h_uniforms, 0, sizeof(h_uniforms));
        for (int i = 0; i < 4; i++) h_uniforms.challenge[i] = le64(challengeBytes, i * 8);
        for (int i = 0; i < 4; i++) h_uniforms.difficulty[i] = be64(difficultyBytes, i * 8);
        for (int i = 0; i < 3; i++) h_uniforms.prefix[i] = le64(prefixBytes, i * 8);
        h_uniforms.nonce_base = 0;

        // Calculate grid dimensions
        uint64_t hashesPerThread = ITERATIONS;
        uint64_t threadsNeeded = (requestedBatch + hashesPerThread - 1) / hashesPerThread;
        uint64_t blocks64 = (threadsNeeded + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        unsigned int numBlocks = (unsigned int)std::min(blocks64, (uint64_t)(1u << 20));
        uint64_t batchHashes = (uint64_t)numBlocks * THREADS_PER_BLOCK * hashesPerThread;

        // Allocate device memory
        Uniforms* d_uniforms;
        ResultBuffer* d_result;
        CUDA_CHECK(cudaMalloc(&d_uniforms, sizeof(Uniforms)));
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(ResultBuffer)));

        ResultBuffer h_result;
        uint64_t totalHashes = 0;
        uint64_t started = nowSteadyMs();
        uint64_t lastProgress = started;

        while (true) {
            if (cutoffMs && nowEpochMs() >= cutoffMs) {
                uint64_t elapsed = nowSteadyMs() - started;
                std::cout << "{\"type\":\"expired\",\"hashes\":\"" << totalHashes
                          << "\",\"elapsedMs\":" << elapsed << "}" << std::endl;
                break;
            }

            // Upload uniforms
            CUDA_CHECK(cudaMemcpy(d_uniforms, &h_uniforms, sizeof(Uniforms), cudaMemcpyHostToDevice));
            memset(&h_result, 0, sizeof(h_result));
            CUDA_CHECK(cudaMemcpy(d_result, &h_result, sizeof(ResultBuffer), cudaMemcpyHostToDevice));

            // Launch kernel
            hash256_mine<<<numBlocks, THREADS_PER_BLOCK>>>(d_uniforms, d_result);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Read result
            CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(ResultBuffer), cudaMemcpyDeviceToHost));
            totalHashes += batchHashes;
            uint64_t elapsed = nowSteadyMs() - started;

            if (h_result.found) {
                std::cout << "{\"type\":\"found\",\"nonce\":\""
                          << nonceHex(prefixBytes, h_result.nonce_counter)
                          << "\",\"hash\":\"" << u64BeToHex(h_result.hash, 4)
                          << "\",\"hashes\":\"" << totalHashes
                          << "\",\"elapsedMs\":" << elapsed << "}" << std::endl;
                break;
            }

            // Advance nonce base
            h_uniforms.nonce_base += batchHashes;

            // Progress report
            uint64_t now = nowSteadyMs();
            if (now - lastProgress >= progressEveryMs) {
                double seconds = std::max(0.001, (double)(now - started) / 1000.0);
                double rate = (double)totalHashes / seconds;
                std::cout << "{\"type\":\"progress\",\"hashes\":\"" << totalHashes
                          << "\",\"hashrate\":" << std::fixed << std::setprecision(2) << rate
                          << ",\"elapsedMs\":" << (now - started) << "}" << std::endl;
                lastProgress = now;
            }
        }

        CUDA_CHECK(cudaFree(d_uniforms));
        CUDA_CHECK(cudaFree(d_result));
        return 0;

    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return 1;
    }
}
