// mmse_apply.cu — CUDA kernel-optimization case study
// ---------------------------------------------------------------------------
// Operation under study: the inner loop of MMSE channel estimation.
// Once the Wiener matrix  W = R_HH (R_HH + sigma^2 I)^-1  is precomputed, every
// coherence interval applies it to a fresh batch of LS estimates:
//
//        H_mmse = W @ H_ls,    W in C^{N x N},   H_ls in C^{N x B}
//
// i.e. a batched complex GEMM. This file implements three variants and times
// them so the naive -> tiled -> vendor optimization journey can be profiled
// with Nsight Compute:
//
//   v0  apply_naive   one thread per output element, no data reuse
//   v1  apply_tiled   classic shared-memory tiled GEMM
//   v2  cuBLAS cgemm  vendor library (Tensor Core path for complex64)
//
// Build:     make                 (or: make ARCH=sm_75 for a T4)
// Correct:   ./mmse_apply --selftest
// Benchmark: ./mmse_apply --bench --N 256 --B 4096 --iters 50
// ---------------------------------------------------------------------------
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>

using cf = cuFloatComplex;  // == float2

#define CUDA_CHECK(x)  do { cudaError_t e=(x); if(e!=cudaSuccess){                 \
  fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));\
  exit(1);} } while(0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){    \
  fprintf(stderr,"cuBLAS error %s:%d: status %d\n",__FILE__,__LINE__,(int)s);       \
  exit(1);} } while(0)

__host__ __device__ __forceinline__ cf cadd(cf a, cf b){
  return make_cuFloatComplex(a.x + b.x, a.y + b.y);
}
__host__ __device__ __forceinline__ cf cmul(cf a, cf b){
  return make_cuFloatComplex(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

// ---- v0: naive. One thread computes Y[n,b] = sum_k W[n,k]*X[k,b]. ----------
// W rereads from global memory for every (n,b); X[k,b] is coalesced across b.
// Low arithmetic intensity as seen by the SM -> memory/latency bound baseline.
__global__ void apply_naive(const cf* __restrict__ W, const cf* __restrict__ X,
                            cf* __restrict__ Y, int N, int B){
  int b = blockIdx.x * blockDim.x + threadIdx.x;  // column (batch index)
  int n = blockIdx.y * blockDim.y + threadIdx.y;  // row    (subcarrier index)
  if(n >= N || b >= B) return;
  cf acc = make_cuFloatComplex(0.f, 0.f);
  for(int k = 0; k < N; ++k)
    acc = cadd(acc, cmul(W[n*N + k], X[k*B + b]));
  Y[n*B + b] = acc;
}

// ---- v1: shared-memory tiled GEMM. -----------------------------------------
// Each TILE x TILE block stages a tile of W and a tile of X into shared memory
// and reuses every loaded value TILE times -> coalesced loads, far higher
// effective bandwidth and SM utilization.
#define TILE 16
__global__ void apply_tiled(const cf* __restrict__ W, const cf* __restrict__ X,
                            cf* __restrict__ Y, int N, int B){
  __shared__ cf sW[TILE][TILE];
  __shared__ cf sX[TILE][TILE];
  int row = blockIdx.y * TILE + threadIdx.y;  // n
  int col = blockIdx.x * TILE + threadIdx.x;  // b
  cf acc = make_cuFloatComplex(0.f, 0.f);
  int ntiles = (N + TILE - 1) / TILE;
  for(int t = 0; t < ntiles; ++t){
    int kW = t*TILE + threadIdx.x;            // column index into W
    int kX = t*TILE + threadIdx.y;            // row index into X
    sW[threadIdx.y][threadIdx.x] =
        (row < N && kW < N) ? W[row*N + kW] : make_cuFloatComplex(0.f, 0.f);
    sX[threadIdx.y][threadIdx.x] =
        (kX < N && col < B) ? X[kX*B + col] : make_cuFloatComplex(0.f, 0.f);
    __syncthreads();
    #pragma unroll
    for(int kk = 0; kk < TILE; ++kk)
      acc = cadd(acc, cmul(sW[threadIdx.y][kk], sX[kk][threadIdx.x]));
    __syncthreads();
  }
  if(row < N && col < B) Y[row*B + col] = acc;
}

// ---- v2: cuBLAS. cublas is column-major, our arrays are row-major. ---------
// Row-major Y(NxB)=W(NxN)*X(NxB)  <=>  column-major  Yt(BxN)=Xt(BxN)*Wt(NxN).
static void apply_cublas(cublasHandle_t h, const cf* dW, const cf* dX, cf* dY,
                         int N, int B){
  cf alpha = make_cuFloatComplex(1.f, 0.f);
  cf beta  = make_cuFloatComplex(0.f, 0.f);
  CUBLAS_CHECK(cublasCgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                           B, N, N, &alpha,
                           dX, B,     // op(A) = Xt  (B x N), lda = B
                           dW, N,     // op(B) = Wt  (N x N), ldb = N
                           &beta, dY, B));  // C = Yt (B x N), ldc = B
}

// ---- double-precision CPU reference (for --selftest) -----------------------
static void ref_cpu(const std::vector<cf>& W, const std::vector<cf>& X,
                    std::vector<cf>& Y, int N, int B){
  for(int n = 0; n < N; ++n)
    for(int b = 0; b < B; ++b){
      double ar = 0.0, ai = 0.0;
      for(int k = 0; k < N; ++k){
        cf w = W[n*N + k], x = X[k*B + b];
        ar += (double)w.x*x.x - (double)w.y*x.y;
        ai += (double)w.x*x.y + (double)w.y*x.x;
      }
      Y[n*B + b] = make_cuFloatComplex((float)ar, (float)ai);
    }
}

static double max_rel_err(const std::vector<cf>& a, const std::vector<cf>& ref){
  double num = 0.0, den = 0.0;
  for(size_t i = 0; i < a.size(); ++i){
    double dr = a[i].x - ref[i].x, di = a[i].y - ref[i].y;
    num += dr*dr + di*di;
    den += (double)ref[i].x*ref[i].x + (double)ref[i].y*ref[i].y;
  }
  return std::sqrt(num / (den + 1e-30));
}

static void fill_random(std::vector<cf>& v, unsigned seed){
  srand(seed);
  for(auto& z : v)
    z = make_cuFloatComplex(2.f*rand()/RAND_MAX - 1.f, 2.f*rand()/RAND_MAX - 1.f);
}

// ---- timing helper ---------------------------------------------------------
template <class F>
static float time_ms(F launch, int iters){
  cudaEvent_t a, b; CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
  launch();                                  // warmup
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(a));
  for(int i = 0; i < iters; ++i) launch();
  CUDA_CHECK(cudaEventRecord(b));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms = 0.f; CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  cudaEventDestroy(a); cudaEventDestroy(b);
  return ms / iters;
}

static void run_selftest(){
  int N = 80, B = 40;                        // small enough for the CPU reference
  std::vector<cf> W(N*N), X(N*B), Yref(N*B), Yh(N*B);
  fill_random(W, 1); fill_random(X, 2);
  ref_cpu(W, X, Yref, N, B);

  cf *dW,*dX,*dY; size_t szW=N*N*sizeof(cf), szX=N*B*sizeof(cf);
  CUDA_CHECK(cudaMalloc(&dW,szW)); CUDA_CHECK(cudaMalloc(&dX,szX)); CUDA_CHECK(cudaMalloc(&dY,szX));
  CUDA_CHECK(cudaMemcpy(dW,W.data(),szW,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dX,X.data(),szX,cudaMemcpyHostToDevice));

  dim3 blk(16,16), grd((B+15)/16,(N+15)/16);
  apply_naive<<<grd,blk>>>(dW,dX,dY,N,B); CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(Yh.data(),dY,szX,cudaMemcpyDeviceToHost));
  printf("  v0 naive   rel.err = %.2e\n", max_rel_err(Yh,Yref));

  apply_tiled<<<grd,blk>>>(dW,dX,dY,N,B); CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(Yh.data(),dY,szX,cudaMemcpyDeviceToHost));
  printf("  v1 tiled   rel.err = %.2e\n", max_rel_err(Yh,Yref));

  cublasHandle_t h; CUBLAS_CHECK(cublasCreate(&h));
  apply_cublas(h,dW,dX,dY,N,B); CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(Yh.data(),dY,szX,cudaMemcpyDeviceToHost));
  printf("  v2 cuBLAS  rel.err = %.2e\n", max_rel_err(Yh,Yref));
  cublasDestroy(h);

  cudaFree(dW); cudaFree(dX); cudaFree(dY);
  puts("  (all variants should be < 1e-3)");
}

static void run_bench(int N, int B, int iters){
  std::vector<cf> W(N*N), X(N*B);
  fill_random(W,1); fill_random(X,2);
  cf *dW,*dX,*dY; size_t szW=N*N*sizeof(cf), szX=(size_t)N*B*sizeof(cf);
  CUDA_CHECK(cudaMalloc(&dW,szW)); CUDA_CHECK(cudaMalloc(&dX,szX)); CUDA_CHECK(cudaMalloc(&dY,szX));
  CUDA_CHECK(cudaMemcpy(dW,W.data(),szW,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dX,X.data(),szX,cudaMemcpyHostToDevice));

  dim3 blk(16,16), grd((B+15)/16,(N+15)/16);
  cublasHandle_t h; CUBLAS_CHECK(cublasCreate(&h));

  float t0 = time_ms([&]{ apply_naive<<<grd,blk>>>(dW,dX,dY,N,B); }, iters);
  float t1 = time_ms([&]{ apply_tiled<<<grd,blk>>>(dW,dX,dY,N,B); }, iters);
  float t2 = time_ms([&]{ apply_cublas(h,dW,dX,dY,N,B); }, iters);
  CUDA_CHECK(cudaGetLastError());

  double flop = 8.0 * (double)N * N * B;     // complex MAC = 8 flop
  auto gflops = [&](float ms){ return flop / (ms*1e-3) / 1e9; };

  printf("\nMMSE-apply GEMM   N=%d  B=%d  iters=%d  (complex64)\n", N, B, iters);
  printf("%-14s %10s %12s %10s\n","variant","ms/call","GFLOP/s","vs naive");
  printf("%-14s %10.4f %12.1f %9.2fx\n","v0 naive", t0, gflops(t0), 1.0);
  printf("%-14s %10.4f %12.1f %9.2fx\n","v1 tiled", t1, gflops(t1), t0/t1);
  printf("%-14s %10.4f %12.1f %9.2fx\n","v2 cuBLAS",t2, gflops(t2), t0/t2);
  printf("CSV,%d,%d,%.4f,%.4f,%.4f,%.1f,%.1f,%.1f\n",
         N,B,t0,t1,t2,gflops(t0),gflops(t1),gflops(t2));

  cublasDestroy(h); cudaFree(dW); cudaFree(dX); cudaFree(dY);
}

int main(int argc, char** argv){
  int N = 256, B = 4096, iters = 50;
  bool bench = false, selftest = false;
  for(int i = 1; i < argc; ++i){
    if(!strcmp(argv[i],"--N") && i+1<argc) N = atoi(argv[++i]);
    else if(!strcmp(argv[i],"--B") && i+1<argc) B = atoi(argv[++i]);
    else if(!strcmp(argv[i],"--iters") && i+1<argc) iters = atoi(argv[++i]);
    else if(!strcmp(argv[i],"--bench")) bench = true;
    else if(!strcmp(argv[i],"--selftest")) selftest = true;
    else { fprintf(stderr,"unknown arg: %s\n",argv[i]); return 2; }
  }
  if(!bench && !selftest) bench = true;       // default action
  if(selftest){ puts("[selftest]"); run_selftest(); }
  if(bench)   run_bench(N, B, iters);
  return 0;
}
