// hash256-cuda-miner.cu
// Multi-GPU CUDA Keccak-256 miner for HASH256 (NVIDIA H100/H200 optimized)
// Async double-buffered streams, pinned memory, per-GPU host threads

#include <atomic>
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
#include <thread>
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
// Kernel constants — tuned for H100 (sm_90, 132 SMs)
// ---------------------------------------------------------------------------
static constexpr int ITERATIONS = 32;
static constexpr int THREADS_PER_BLOCK = 512;
static constexpr int NUM_STREAMS = 2;

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
__device__ __forceinline__ void keccak_f1600(uint64_t s[25]) {
    #pragma unroll
    for (int r = 0; r < 24; r++) {
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

        s[0] ^= KECCAK_RC[r];
    }
}

// ---------------------------------------------------------------------------
// GPU uniforms & result
// ---------------------------------------------------------------------------
struct Uniforms {
    uint64_t challenge[4];
    uint64_t difficulty[4];
    uint64_t prefix[3];
    uint64_t nonce_base;
};

struct ResultBuffer {
    unsigned int found;
    unsigned int pad;
    uint64_t nonce_counter;
    uint64_t hash[4];
};

// ---------------------------------------------------------------------------
// Mining kernel — optimized for H100
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 2)
hash256_mine(const Uniforms* __restrict__ u,
             ResultBuffer* __restrict__ result) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t thread_start = gid * ITERATIONS;

    const uint64_t ch0 = u->challenge[0];
    const uint64_t ch1 = u->challenge[1];
    const uint64_t ch2 = u->challenge[2];
    const uint64_t ch3 = u->challenge[3];
    const uint64_t pr0 = u->prefix[0];
    const uint64_t pr1 = u->prefix[1];
    const uint64_t pr2 = u->prefix[2];
    const uint64_t d0  = u->difficulty[0];
    const uint64_t d1  = u->difficulty[1];
    const uint64_t d2  = u->difficulty[2];
    const uint64_t d3  = u->difficulty[3];
    const uint64_t base = u->nonce_base;

    #pragma unroll 4
    for (int k = 0; k < ITERATIONS; k++) {
        const uint64_t counter = base + thread_start + k;

        uint64_t st[25];
        st[0] = ch0;
        st[1] = ch1;
        st[2] = ch2;
        st[3] = ch3;
        st[4] = pr0;
        st[5] = pr1;
        st[6] = pr2;
        st[7] = bswap64(counter);
        st[8] = 0x0000000000000001ULL;
        st[9]  = 0; st[10] = 0; st[11] = 0;
        st[12] = 0; st[13] = 0; st[14] = 0; st[15] = 0;
        st[16] = 0x8000000000000000ULL;
        st[17] = 0; st[18] = 0; st[19] = 0;
        st[20] = 0; st[21] = 0; st[22] = 0; st[23] = 0; st[24] = 0;

        keccak_f1600(st);

        const uint64_t h0 = bswap64(st[0]);
        const uint64_t h1 = bswap64(st[1]);
        const uint64_t h2 = bswap64(st[2]);
        const uint64_t h3 = bswap64(st[3]);

        const bool lt = (h0 < d0) ||
                        (h0 == d0 && ((h1 < d1) ||
                        (h1 == d1 && ((h2 < d2) ||
                        (h2 == d2 && h3 < d3)))));

        if (lt) {
            const unsigned int prior = atomicAdd(&result->found, 1u);
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
    if (hex.size() != bytes * 2)
        throw std::runtime_error(std::string(name) + " must be " + std::to_string(bytes) + " bytes");
    std::vector<uint8_t> out(bytes);
    for (size_t i = 0; i < bytes; i++) {
        out[i] = static_cast<uint8_t>(std::stoul(hex.substr(i * 2, 2), nullptr, 16));
    }
    return out;
}

static uint64_t le64(const std::vector<uint8_t>& b, size_t off) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)b[off + i] << (i * 8);
    return v;
}

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

static std::string nonceHex(const std::vector<uint8_t>& prefix, uint64_t counter) {
    std::vector<uint8_t> nonce(32, 0);
    for (size_t i = 0; i < 24; i++) nonce[i] = prefix[i];
    for (int i = 0; i < 8; i++) nonce[24 + i] = (uint8_t)((counter >> ((7 - i) * 8)) & 0xff);
    return bytesToHex(nonce);
}

static std::string getArg(int argc, char** argv, const std::string& key,
                           const std::string& fallback = "") {
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

static int parseInt(const std::string& value, int fallback) {
    if (value.empty()) return fallback;
    return std::stoi(value);
}

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
} while(0)

// ---------------------------------------------------------------------------
// Per-GPU mining state (double-buffered streams)
// ---------------------------------------------------------------------------
struct GpuState {
    int device_id;
    int sm_count;

    Uniforms*    d_uniforms[NUM_STREAMS];
    ResultBuffer* d_result[NUM_STREAMS];

    Uniforms*    h_uniforms[NUM_STREAMS];
    ResultBuffer* h_result[NUM_STREAMS];

    cudaStream_t streams[NUM_STREAMS];
    cudaEvent_t  kernel_done[NUM_STREAMS];

    unsigned int numBlocks;
    uint64_t     batchHashes;
    uint64_t     nonceBase;
};

static void initGpu(GpuState& gs, int dev, uint64_t batchPerGpu) {
    gs.device_id = dev;
    CUDA_CHECK(cudaSetDevice(dev));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    gs.sm_count = prop.multiProcessorCount;
    fprintf(stderr, "  GPU %d: %s (%d SMs, %d MB)\n",
            dev, prop.name, gs.sm_count, (int)(prop.totalGlobalMem / (1024 * 1024)));

    uint64_t hashesPerThread = ITERATIONS;
    uint64_t threadsNeeded = (batchPerGpu + hashesPerThread - 1) / hashesPerThread;
    uint64_t blocks64 = (threadsNeeded + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    unsigned int maxBlocks = (unsigned int)gs.sm_count * 24;
    gs.numBlocks = (unsigned int)std::min(blocks64, (uint64_t)maxBlocks);
    gs.batchHashes = (uint64_t)gs.numBlocks * THREADS_PER_BLOCK * hashesPerThread;
    gs.nonceBase = 0;

    fprintf(stderr, "  GPU %d: %u blocks × %d threads × %d iter = %llu hashes/launch\n",
            dev, gs.numBlocks, THREADS_PER_BLOCK, ITERATIONS,
            (unsigned long long)gs.batchHashes);

    for (int s = 0; s < NUM_STREAMS; s++) {
        CUDA_CHECK(cudaMalloc(&gs.d_uniforms[s], sizeof(Uniforms)));
        CUDA_CHECK(cudaMalloc(&gs.d_result[s], sizeof(ResultBuffer)));
        CUDA_CHECK(cudaMallocHost(&gs.h_uniforms[s], sizeof(Uniforms)));
        CUDA_CHECK(cudaMallocHost(&gs.h_result[s], sizeof(ResultBuffer)));
        CUDA_CHECK(cudaStreamCreate(&gs.streams[s]));
        CUDA_CHECK(cudaEventCreate(&gs.kernel_done[s]));
    }
}

static void freeGpu(GpuState& gs) {
    cudaSetDevice(gs.device_id);
    for (int s = 0; s < NUM_STREAMS; s++) {
        cudaEventDestroy(gs.kernel_done[s]);
        cudaStreamDestroy(gs.streams[s]);
        cudaFreeHost(gs.h_uniforms[s]);
        cudaFreeHost(gs.h_result[s]);
        cudaFree(gs.d_uniforms[s]);
        cudaFree(gs.d_result[s]);
    }
}

// ---------------------------------------------------------------------------
// Per-GPU mining thread with double-buffered async pipelines
// ---------------------------------------------------------------------------
struct GpuResult {
    int status;       // 1=found, 0=expired, -1=error
    uint64_t nonce_counter;
    uint64_t hash[4];
    uint64_t totalHashes;
    int gpu_id;
    std::string error;
};

static void mineOnGpu(GpuState& gs,
                       const Uniforms& baseUniforms,
                       int numGpus,
                       uint64_t cutoffMs,
                       uint64_t progressEveryMs,
                       std::atomic<bool>& globalFound,
                       GpuResult& out) {
    CUDA_CHECK(cudaSetDevice(gs.device_id));

    out.gpu_id = gs.device_id;
    out.totalHashes = 0;
    out.status = 0;

    uint64_t started = nowSteadyMs();
    uint64_t lastProgress = started;

    // Launch first batch on stream 0
    {
        memcpy(gs.h_uniforms[0], &baseUniforms, sizeof(Uniforms));
        gs.h_uniforms[0]->nonce_base = gs.nonceBase;
        memset(gs.h_result[0], 0, sizeof(ResultBuffer));

        CUDA_CHECK(cudaMemcpyAsync(gs.d_uniforms[0], gs.h_uniforms[0],
                                    sizeof(Uniforms), cudaMemcpyHostToDevice,
                                    gs.streams[0]));
        CUDA_CHECK(cudaMemcpyAsync(gs.d_result[0], gs.h_result[0],
                                    sizeof(ResultBuffer), cudaMemcpyHostToDevice,
                                    gs.streams[0]));
        hash256_mine<<<gs.numBlocks, THREADS_PER_BLOCK, 0, gs.streams[0]>>>(
            gs.d_uniforms[0], gs.d_result[0]);
        CUDA_CHECK(cudaEventRecord(gs.kernel_done[0], gs.streams[0]));

        gs.nonceBase += (uint64_t)numGpus * gs.batchHashes;
    }

    int cur = 0;

    while (true) {
        if (globalFound.load(std::memory_order_relaxed)) break;
        if (cutoffMs && nowEpochMs() >= cutoffMs) { out.status = 0; break; }

        int next = (cur + 1) % NUM_STREAMS;

        // Prepare & launch next batch (async — overlaps with cur kernel)
        memcpy(gs.h_uniforms[next], &baseUniforms, sizeof(Uniforms));
        gs.h_uniforms[next]->nonce_base = gs.nonceBase;
        memset(gs.h_result[next], 0, sizeof(ResultBuffer));

        CUDA_CHECK(cudaMemcpyAsync(gs.d_uniforms[next], gs.h_uniforms[next],
                                    sizeof(Uniforms), cudaMemcpyHostToDevice,
                                    gs.streams[next]));
        CUDA_CHECK(cudaMemcpyAsync(gs.d_result[next], gs.h_result[next],
                                    sizeof(ResultBuffer), cudaMemcpyHostToDevice,
                                    gs.streams[next]));
        hash256_mine<<<gs.numBlocks, THREADS_PER_BLOCK, 0, gs.streams[next]>>>(
            gs.d_uniforms[next], gs.d_result[next]);
        CUDA_CHECK(cudaEventRecord(gs.kernel_done[next], gs.streams[next]));

        gs.nonceBase += (uint64_t)numGpus * gs.batchHashes;

        // Wait for current stream's kernel, read result
        CUDA_CHECK(cudaEventSynchronize(gs.kernel_done[cur]));
        CUDA_CHECK(cudaMemcpyAsync(gs.h_result[cur], gs.d_result[cur],
                                    sizeof(ResultBuffer), cudaMemcpyDeviceToHost,
                                    gs.streams[cur]));
        CUDA_CHECK(cudaStreamSynchronize(gs.streams[cur]));

        out.totalHashes += gs.batchHashes;

        if (gs.h_result[cur]->found) {
            out.status = 1;
            out.nonce_counter = gs.h_result[cur]->nonce_counter;
            out.hash[0] = gs.h_result[cur]->hash[0];
            out.hash[1] = gs.h_result[cur]->hash[1];
            out.hash[2] = gs.h_result[cur]->hash[2];
            out.hash[3] = gs.h_result[cur]->hash[3];
            globalFound.store(true, std::memory_order_relaxed);
            break;
        }

        uint64_t now = nowSteadyMs();
        if (now - lastProgress >= progressEveryMs) {
            double seconds = std::max(0.001, (double)(now - started) / 1000.0);
            double rate = (double)out.totalHashes / seconds;
            fprintf(stderr, "[gpu%d] %llu hashes · %.2f GH/s\n",
                    gs.device_id, (unsigned long long)out.totalHashes, rate / 1e9);
            lastProgress = now;
        }

        cur = next;
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    try {
        auto challengeBytes = parseHex(getArg(argc, argv, "--challenge"), 32, "challenge");
        auto difficultyBytes = parseHex(getArg(argc, argv, "--difficulty"), 32, "difficulty");
        auto prefixBytes = parseHex(
            getArg(argc, argv, "--prefix",
                   "0x000000000000000000000000000000000000000000000000"),
            24, "prefix");
        uint64_t requestedBatch = parseU64(getArg(argc, argv, "--batch"), 4096000000ULL);
        uint64_t progressEveryMs = parseU64(getArg(argc, argv, "--progress-ms"), 1000);
        uint64_t cutoffMs = parseU64(getArg(argc, argv, "--cutoff-ms"), 0);
        int requestedGpus = parseInt(getArg(argc, argv, "--gpus"), 0);

        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) throw std::runtime_error("No CUDA devices found");

        int numGpus = requestedGpus > 0 ? std::min(requestedGpus, deviceCount) : deviceCount;
        fprintf(stderr, "Using %d GPU(s)\n", numGpus);

        Uniforms baseUniforms;
        memset(&baseUniforms, 0, sizeof(baseUniforms));
        for (int i = 0; i < 4; i++) baseUniforms.challenge[i] = le64(challengeBytes, i * 8);
        for (int i = 0; i < 4; i++) baseUniforms.difficulty[i] = be64(difficultyBytes, i * 8);
        for (int i = 0; i < 3; i++) baseUniforms.prefix[i] = le64(prefixBytes, i * 8);

        uint64_t batchPerGpu = (requestedBatch + numGpus - 1) / numGpus;
        uint64_t align = (uint64_t)THREADS_PER_BLOCK * ITERATIONS;
        batchPerGpu = ((batchPerGpu + align - 1) / align) * align;

        std::vector<GpuState> gpuStates(numGpus);
        fprintf(stderr, "Initializing GPUs...\n");
        for (int g = 0; g < numGpus; g++) {
            initGpu(gpuStates[g], g, batchPerGpu);
            // Partition nonce space: GPU g starts at g * batchHashes
            gpuStates[g].nonceBase = (uint64_t)g * gpuStates[g].batchHashes;
        }

        std::atomic<bool> globalFound{false};
        std::vector<GpuResult> results(numGpus);
        for (int g = 0; g < numGpus; g++) {
            results[g].status = -1;
            results[g].totalHashes = 0;
        }

        uint64_t started = nowSteadyMs();

        std::vector<std::thread> threads;
        threads.reserve(numGpus);
        for (int g = 0; g < numGpus; g++) {
            threads.emplace_back([&, g]() {
                try {
                    mineOnGpu(gpuStates[g], baseUniforms, numGpus, cutoffMs,
                              progressEveryMs, globalFound, results[g]);
                } catch (const std::exception& ex) {
                    results[g].status = -1;
                    results[g].error = ex.what();
                }
            });
        }
        for (auto& t : threads) t.join();

        uint64_t elapsed = nowSteadyMs() - started;

        uint64_t totalHashes = 0;
        int foundGpu = -1;
        for (int g = 0; g < numGpus; g++) {
            totalHashes += results[g].totalHashes;
            if (results[g].status == -1 && !results[g].error.empty())
                fprintf(stderr, "GPU %d error: %s\n", g, results[g].error.c_str());
            if (results[g].status == 1 && foundGpu < 0) foundGpu = g;
        }

        if (foundGpu >= 0) {
            auto& r = results[foundGpu];
            std::cout << "{\"type\":\"found\",\"gpu\":" << foundGpu
                      << ",\"nonce\":\"" << nonceHex(prefixBytes, r.nonce_counter)
                      << "\",\"hash\":\"" << u64BeToHex(r.hash, 4)
                      << "\",\"hashes\":\"" << totalHashes
                      << "\",\"elapsedMs\":" << elapsed << "}" << std::endl;
        } else {
            std::cout << "{\"type\":\"expired\",\"hashes\":\"" << totalHashes
                      << "\",\"elapsedMs\":" << elapsed
                      << ",\"gpus\":" << numGpus << "}" << std::endl;
        }

        for (int g = 0; g < numGpus; g++) freeGpu(gpuStates[g]);
        return (foundGpu >= 0) ? 0 : 1;

    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return 1;
    }
}
