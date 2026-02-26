// SFPU benchmark comparison for AMD GPU via HIP.
// Measures element-wise math throughput for the same operations benchmarked
// on the Tenstorrent Blackhole SFPU.
//
// Build:
//   hipcc --offload-arch=gfx1200 -O3 -o sfpu_bench_hip examples/sfpu_bench_hip.cpp
// Run:
//   ./sfpu_bench_hip

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <chrono>

#define HIP_CHECK(cmd)                                                 \
  do {                                                                 \
    hipError_t e = (cmd);                                              \
    if (e != hipSuccess) {                                             \
      fprintf(stderr, "HIP error %d (%s) at %s:%d\n",                 \
              e, hipGetErrorString(e), __FILE__, __LINE__);            \
      exit(1);                                                         \
    }                                                                  \
  } while (0)

constexpr int N_ELEMS = 1024 * 1024;  // 1M elements
constexpr int N_REPS  = 1000;
constexpr int BLOCK   = 256;
constexpr int GRID    = (N_ELEMS + BLOCK - 1) / BLOCK;

// ---------------------------------------------------------------------------
// Kernels — each applies an operation N_REPS times per element
// ---------------------------------------------------------------------------

__global__ void kern_add(float* __restrict__ data, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = data[i];
  for (int r = 0; r < N_REPS; ++r) x = x + 1.0f;
  data[i] = x;
}

__global__ void kern_mul(float* __restrict__ data, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = data[i];
  for (int r = 0; r < N_REPS; ++r) x = x * 0.99f;
  data[i] = x;
}

__global__ void kern_mad(float* __restrict__ data, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = data[i];
  for (int r = 0; r < N_REPS; ++r) x = x * 0.99f + 0.01f;
  data[i] = x;
}

__global__ void kern_recip(float* __restrict__ data, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = data[i];
  for (int r = 0; r < N_REPS; ++r) x = __frcp_rn(x);
  data[i] = x;
}

__global__ void kern_poly3(float* __restrict__ data, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = data[i];
  for (int r = 0; r < N_REPS; ++r) {
    float v = ((0.1058f * x - 0.7166f) * x + 2.0871f) * x + 1.0f;
    x = v;
  }
  data[i] = x;
}

__global__ void kern_poly7(float* __restrict__ data, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = data[i];
  for (int r = 0; r < N_REPS; ++r) {
    float v = ((((((0.00014f * x + 0.00139f) * x + 0.00833f) * x
                  + 0.04167f) * x + 0.16667f) * x + 0.5f) * x + 1.0f) * x + 1.0f;
    x = v;
  }
  data[i] = x;
}

__global__ void kern_sin(float* __restrict__ data, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = data[i];
  for (int r = 0; r < N_REPS; ++r) x = sinf(x);
  data[i] = x;
}

__global__ void kern_cos(float* __restrict__ data, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = data[i];
  for (int r = 0; r < N_REPS; ++r) x = cosf(x);
  data[i] = x;
}

__global__ void kern_exp(float* __restrict__ data, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = data[i];
  for (int r = 0; r < N_REPS; ++r) x = expf(x);
  data[i] = x;
}

__global__ void kern_log(float* __restrict__ data, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = data[i];
  for (int r = 0; r < N_REPS; ++r) x = logf(x);
  data[i] = x;
}

__global__ void kern_sqrt(float* __restrict__ data, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = data[i];
  for (int r = 0; r < N_REPS; ++r) x = sqrtf(x);
  data[i] = x;
}

__global__ void kern_rsqrt(float* __restrict__ data, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = data[i];
  for (int r = 0; r < N_REPS; ++r) x = rsqrtf(x);
  data[i] = x;
}

__global__ void kern_tanh(float* __restrict__ data, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = data[i];
  for (int r = 0; r < N_REPS; ++r) x = tanhf(x);
  data[i] = x;
}

// ---------------------------------------------------------------------------
// Benchmark harness
// ---------------------------------------------------------------------------

struct BenchResult {
  const char* name;
  const char* desc;
  float ms;
  int flops_per_elem;
};

template <typename Kernel>
BenchResult bench(const char* name, const char* desc, int flops_per_elem,
                  Kernel kernel, float* d_data, int n) {
  // Warmup
  kernel<<<GRID, BLOCK>>>(d_data, n);
  HIP_CHECK(hipDeviceSynchronize());

  // Re-init data (ops may have blown up values)
  std::vector<float> host(n);
  for (int i = 0; i < n; ++i) host[i] = 0.1f + 1.9f * (float(i) / n);
  HIP_CHECK(hipMemcpy(d_data, host.data(), n * sizeof(float), hipMemcpyHostToDevice));

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  HIP_CHECK(hipEventRecord(start));
  kernel<<<GRID, BLOCK>>>(d_data, n);
  HIP_CHECK(hipEventRecord(stop));
  HIP_CHECK(hipEventSynchronize(stop));

  float ms = 0;
  HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));

  // Re-init for next benchmark
  HIP_CHECK(hipMemcpy(d_data, host.data(), n * sizeof(float), hipMemcpyHostToDevice));

  return {name, desc, ms, flops_per_elem};
}

int main() {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  printf("GPU: %s  (%d CUs @ %d MHz, %d MB VRAM)\n",
         prop.name, prop.multiProcessorCount,
         prop.clockRate / 1000, (int)(prop.totalGlobalMem / (1024*1024)));

  float* d_data;
  HIP_CHECK(hipMalloc(&d_data, N_ELEMS * sizeof(float)));

  std::vector<float> host(N_ELEMS);
  for (int i = 0; i < N_ELEMS; ++i) host[i] = 0.1f + 1.9f * (float(i) / N_ELEMS);
  HIP_CHECK(hipMemcpy(d_data, host.data(), N_ELEMS * sizeof(float), hipMemcpyHostToDevice));

  std::vector<BenchResult> results;

  results.push_back(bench("add",    "FP add",               1,  kern_add,    d_data, N_ELEMS));
  results.push_back(bench("mul",    "FP multiply",          1,  kern_mul,    d_data, N_ELEMS));
  results.push_back(bench("mad",    "Fused multiply-add",   2,  kern_mad,    d_data, N_ELEMS));
  results.push_back(bench("recip",  "HW fast reciprocal",   1,  kern_recip,  d_data, N_ELEMS));
  results.push_back(bench("poly3",  "Deg-3 Horner",         6,  kern_poly3,  d_data, N_ELEMS));
  results.push_back(bench("poly7",  "Deg-7 Horner",        14,  kern_poly7,  d_data, N_ELEMS));
  results.push_back(bench("sin",    "sinf() intrinsic",     1,  kern_sin,    d_data, N_ELEMS));
  results.push_back(bench("cos",    "cosf() intrinsic",     1,  kern_cos,    d_data, N_ELEMS));
  results.push_back(bench("exp",    "expf() intrinsic",     1,  kern_exp,    d_data, N_ELEMS));
  results.push_back(bench("log",    "logf() intrinsic",     1,  kern_log,    d_data, N_ELEMS));
  results.push_back(bench("sqrt",   "sqrtf() intrinsic",    1,  kern_sqrt,   d_data, N_ELEMS));
  results.push_back(bench("rsqrt",  "rsqrtf() intrinsic",   1,  kern_rsqrt,  d_data, N_ELEMS));
  results.push_back(bench("tanh",   "tanhf() intrinsic",    1,  kern_tanh,   d_data, N_ELEMS));

  printf("\n==========================================================================\n");
  printf("  HIP Benchmark  (%d elements, %d reps, %s)\n", N_ELEMS, N_REPS, prop.name);
  printf("==========================================================================\n");
  printf("  %-12s %9s %10s %10s  %s\n", "Op", "Time(ms)", "Elem/us", "GFLOPS", "Desc");
  printf("  --------------------------------------------------------------------------\n");

  for (auto& r : results) {
    double total_elems = (double)N_ELEMS * N_REPS;
    double elem_per_us = total_elems / (r.ms * 1000.0);
    double total_flops = total_elems * r.flops_per_elem;
    double gflops = total_flops / (r.ms * 1e6);
    printf("  %-12s %9.3f %10.0f %10.1f  %s\n",
           r.name, r.ms, elem_per_us, gflops, r.desc);
  }

  printf("==========================================================================\n\n");

  HIP_CHECK(hipFree(d_data));
  return 0;
}
