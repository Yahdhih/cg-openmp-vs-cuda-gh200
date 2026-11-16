// cg_cuda.cu : backend CUDA du benchmark.
//
// Versions (1er argument) :
//   cuda          noyaux ecrits a la main, memoire cudaMalloc (HBM)
//   cusparse      SpMV cuSPARSE + dot/axpy cuBLAS (reference vendeur), cudaMalloc
//   cuda-managed  noyaux a la main, cudaMallocManaged (migration de pages)
//   cuda-system   noyaux a la main, malloc() systeme : sur GH200 le GPU lit la
//                 memoire du CPU (LPDDR5X) a travers le lien NVLink-C2C (ATS)
//   cuda-fused    noyaux a la main + fusion (maj x,r + dot ; Jacobi + dot)
//   stream        mesure de la bande passante pratique (copie, triad)
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include "common.h"

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

static const int BS = 256;
static int grid_for(long n) { return (int)std::min<long>((n + BS - 1) / BS, 65535L * 16); }

// ------------------------------------------------------------- noyaux

// SpMV CSR "scalaire" : un thread par ligne. Avec 5 a 7 non-zeros par ligne,
// c'est le bon choix (un warp par ligne laisserait 25 lanes sur 32 inactives).
__global__ void k_spmv(int n, const int* __restrict__ rp, const int* __restrict__ ci,
                       const double* __restrict__ v, const double* __restrict__ x, double* __restrict__ y) {
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < n; r += gridDim.x * blockDim.x) {
        double s = 0.0;
        for (int p = rp[r]; p < rp[r + 1]; p++) s += v[p] * x[ci[p]];
        y[r] = s;
    }
}

// Produit scalaire : boucle "grid-stride", reduction dans le warp par
// __shfl_down_sync, puis entre warps en memoire partagee, puis UNE atomicAdd
// par bloc dans le resultat global.
__global__ void k_dot(int n, const double* __restrict__ x, const double* __restrict__ y, double* out) {
    double s = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) s += x[i] * y[i];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
    __shared__ double w[BS / 32];
    if ((threadIdx.x & 31) == 0) w[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x < 32) {
        s = threadIdx.x < BS / 32 ? w[threadIdx.x] : 0.0;
        for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
        if (threadIdx.x == 0) atomicAdd(out, s);
    }
}

// Reduction d'un bloc puis une atomicAdd : partagee par les noyaux fusionnes.
__device__ void block_sum_atomic(double s, double* out) {
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
    __shared__ double w[BS / 32];
    if ((threadIdx.x & 31) == 0) w[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x < 32) {
        s = threadIdx.x < BS / 32 ? w[threadIdx.x] : 0.0;
        for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
        if (threadIdx.x == 0) atomicAdd(out, s);
    }
}
// x += a p ; r -= a q ; out += r . r   (3 noyaux en 1, r relu depuis un registre)
__global__ void k_upd_dot(int n, double a, const double* __restrict__ p, const double* __restrict__ q,
                          double* __restrict__ x, double* __restrict__ r, double* out) {
    double s = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        x[i] += a * p[i];
        double ri = r[i] - a * q[i];
        r[i] = ri; s += ri * ri;
    }
    block_sum_atomic(s, out);
}
// z = d .* r ; out += r . z   (2 noyaux en 1)
__global__ void k_prec_dot(int n, const double* __restrict__ d, const double* __restrict__ r,
                           double* __restrict__ z, double* out) {
    double s = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        double ri = r[i], zi = d[i] * ri;
        z[i] = zi; s += ri * zi;
    }
    block_sum_atomic(s, out);
}

__global__ void k_axpy(int n, double a, const double* __restrict__ x, double* __restrict__ y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) y[i] += a * x[i];
}
__global__ void k_xpay(int n, const double* __restrict__ x, double a, double* __restrict__ y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) y[i] = x[i] + a * y[i];
}
__global__ void k_pmul(int n, const double* __restrict__ d, const double* __restrict__ r, double* __restrict__ z) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) z[i] = d[i] * r[i];
}
__global__ void k_copy(long n, const double* __restrict__ a, double* __restrict__ b) {
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) b[i] = a[i];
}
__global__ void k_triad(long n, const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ c) {
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) c[i] = a[i] + 3.0 * b[i];
}

// ------------------------------------------------ operations communes

static double* d_scal = nullptr;       // resultat des produits scalaires
static int dot_blocks = 0;

static void c_sync() { CK(cudaDeviceSynchronize()); }
static void c_spmv(const DevMat& A, const double* x, double* y) {
    k_spmv<<<grid_for(A.n), BS>>>(A.n, A.rowptr, A.col, A.val, x, y);
}
static double c_dot(int n, const double* x, const double* y) {
    CK(cudaMemsetAsync(d_scal, 0, sizeof(double)));
    k_dot<<<std::min(grid_for(n), dot_blocks), BS>>>(n, x, y, d_scal);
    double h; CK(cudaMemcpy(&h, d_scal, sizeof(double), cudaMemcpyDeviceToHost));
    return h;
}
static double c_upd_dot(int n, double a, const double* p, const double* q, double* x, double* r) {
    CK(cudaMemsetAsync(d_scal, 0, sizeof(double)));
    k_upd_dot<<<std::min(grid_for(n), dot_blocks), BS>>>(n, a, p, q, x, r, d_scal);
    double h; CK(cudaMemcpy(&h, d_scal, sizeof(double), cudaMemcpyDeviceToHost));
    return h;
}
static double c_prec_dot(int n, const double* d, const double* r, double* z) {
    CK(cudaMemsetAsync(d_scal, 0, sizeof(double)));
    k_prec_dot<<<std::min(grid_for(n), dot_blocks), BS>>>(n, d, r, z, d_scal);
    double h; CK(cudaMemcpy(&h, d_scal, sizeof(double), cudaMemcpyDeviceToHost));
    return h;
}
static void c_axpy(int n, double a, const double* x, double* y) { k_axpy<<<grid_for(n), BS>>>(n, a, x, y); }
static void c_xpay(int n, const double* x, double a, double* y) { k_xpay<<<grid_for(n), BS>>>(n, x, a, y); }
static void c_pmul(int n, const double* d, const double* r, double* z) { k_pmul<<<grid_for(n), BS>>>(n, d, r, z); }

// trois facons d'allouer
static void* a_dev(size_t b) { void* p; CK(cudaMalloc(&p, b)); return p; }
static void  f_dev(void* p) { CK(cudaFree(p)); }
static void  h2d_dev(void* d, const void* h, size_t b) { CK(cudaMemcpy(d, h, b, cudaMemcpyHostToDevice)); }
static void  d2h_dev(void* h, const void* d, size_t b) { CK(cudaMemcpy(h, d, b, cudaMemcpyDeviceToHost)); }
static void* a_man(size_t b) { void* p; CK(cudaMallocManaged(&p, b)); return p; }
static void* a_sys(size_t b) { void* p = nullptr; if (posix_memalign(&p, 4096, b)) exit(1); return p; }
static void  f_sys(void* p) { free(p); }
static void  cpy_host(void* d, const void* s, size_t b) { CK(cudaDeviceSynchronize()); memcpy(d, s, b); }

// ------------------------------------------------- cuSPARSE / cuBLAS

struct SpCtx { cusparseSpMatDescr_t A; cusparseDnVecDescr_t x, y; void* buf; };
static cusparseHandle_t sp_h;
static cublasHandle_t bl_h;

static void sp_setup(DevMat& A) {
    SpCtx* c = new SpCtx;
    cusparseCreateCsr(&c->A, A.n, A.n, A.nnz, A.rowptr, A.col, A.val, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    double *tx, *ty; CK(cudaMalloc(&tx, A.n * sizeof(double))); CK(cudaMalloc(&ty, A.n * sizeof(double)));
    cusparseCreateDnVec(&c->x, A.n, tx, CUDA_R_64F);
    cusparseCreateDnVec(&c->y, A.n, ty, CUDA_R_64F);
    double one = 1.0, zero = 0.0; size_t sz = 0;
    cusparseSpMV_bufferSize(sp_h, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, c->A, c->x, &zero, c->y,
                            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &sz);
    CK(cudaMalloc(&c->buf, std::max<size_t>(sz, 8)));
    A.extra = c;
}
static void sp_spmv(const DevMat& A, const double* x, double* y) {
    SpCtx* c = (SpCtx*)A.extra; double one = 1.0, zero = 0.0;
    cusparseDnVecSetValues(c->x, (void*)x);
    cusparseDnVecSetValues(c->y, (void*)y);
    cusparseSpMV(sp_h, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, c->A, c->x, &zero, c->y, CUDA_R_64F,
                 CUSPARSE_SPMV_ALG_DEFAULT, c->buf);
}
static double bl_dot(int n, const double* x, const double* y) { double r; cublasDdot(bl_h, n, x, 1, y, 1, &r); return r; }
static void bl_axpy(int n, double a, const double* x, double* y) { cublasDaxpy(bl_h, n, &a, x, 1, y, 1); }

// ------------------------------------------------------------ STREAM

static int run_stream(int reps) {
    long n = 1L << 28;                                     // 2 Gio par tableau
    double *a, *b, *c;
    CK(cudaMalloc(&a, n * 8)); CK(cudaMalloc(&b, n * 8)); CK(cudaMalloc(&c, n * 8));
    CK(cudaMemset(a, 0, n * 8)); CK(cudaMemset(b, 0, n * 8));
    int g = 132 * 16;
    k_copy<<<g, BS>>>(n, a, c); k_triad<<<g, BS>>>(n, a, b, c); c_sync();
    for (int rep = 0; rep < reps; rep++) {
        double t0 = now(); for (int i = 0; i < 10; i++) k_copy<<<g, BS>>>(n, a, c); c_sync();
        double t = (now() - t0) / 10;
        printf("stream,none,0,%ld,0,copy,%d,%.6e,%.2f,0,0,0,0\n", n, rep, t, 16.0 * n / t * 1e-9);
        t0 = now(); for (int i = 0; i < 10; i++) k_triad<<<g, BS>>>(n, a, b, c); c_sync();
        t = (now() - t0) / 10;
        printf("stream,none,0,%ld,0,triad,%d,%.6e,%.2f,0,0,0,0\n", n, rep, t, 24.0 * n / t * 1e-9);
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "stream") return run_stream(argc > 2 ? atoi(argv[2]) : 5);
    std::string v = argc > 1 ? argv[1] : "";
    CK(cudaMalloc(&d_scal, sizeof(double)));
    int sms; CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    dot_blocks = sms * 8;

    Ops o{v.c_str(), a_dev, f_dev, h2d_dev, d2h_dev, c_sync, nullptr, c_spmv, c_dot, c_axpy, c_xpay, c_pmul};
    if (v == "cusparse") {
        cusparseCreate(&sp_h); cublasCreate(&bl_h);
        o.mat_setup = sp_setup; o.spmv = sp_spmv; o.dot = bl_dot; o.axpy = bl_axpy;
    } else if (v == "cuda-managed") {
        o.alloc = a_man; o.h2d = cpy_host; o.d2h = cpy_host;
    } else if (v == "cuda-system") {
        o.alloc = a_sys; o.free_ = f_sys; o.h2d = cpy_host; o.d2h = cpy_host;
    } else if (v == "cuda-fused") {
        o.upd_dot = c_upd_dot; o.prec_dot = c_prec_dot;
    } else if (v != "cuda") {
        fprintf(stderr, "version inconnue : %s\n", v.c_str()); return 1;
    }
    return run_main(argc, argv, o);
}
