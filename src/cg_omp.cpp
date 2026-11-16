// cg_omp.cpp : backend OpenMP target (offload GPU) du benchmark.
//
// Versions (1er argument) :
//   omp-naive    code "utilisateur" : tableaux hote mappes une fois par
//                `target enter data`, noyaux en `target teams distribute
//                parallel for` ; le runtime retrouve l'adresse device de
//                chaque pointeur a chaque lancement (table de correspondance)
//   omp-devptr   memoire allouee par omp_target_alloc et passee avec
//                is_device_ptr : plus de recherche dans la table de mapping
//   omp-loop     omp-devptr + construction `target teams loop` (conseillee
//                par NVHPC : le compilateur choisit lui-meme le decoupage)
//   omp-opt      omp-devptr + num_teams / thread_limit fixes (variables
//                OMP_NT, OMP_TL) : decoupage explicite comme en CUDA
//   omp-fused    omp-opt + fusion des noyaux (comme cuda-fused)
//   omp-unified  (binaire compile avec -gpu=mem:unified) malloc() simple,
//                aucune directive de donnees : la memoire unifiee du GH200
#include <omp.h>

#include <map>

#include "common.h"

enum Mode { NAIVE, DEVPTR, LOOP, OPT, UNIFIED };
static Mode mode;
static int TL = 256, NT = 0;          // thread_limit, num_teams (0 = n/TL)
static std::map<void*, size_t> mapped;

// --------------------------------------------------------------- memoire

static void* a_map(size_t b) {        // naive : tableau hote + copie device
    char* c = (char*)malloc(b);
#pragma omp target enter data map(alloc : c[0:b])
    mapped[c] = b; return c;
}
static void f_map(void* p) {
    char* c = (char*)p; size_t b = mapped[p];
#pragma omp target exit data map(delete : c[0:b])
    mapped.erase(p); free(p);
}
static void h2d_map(void* d, const void* h, size_t b) {
    memcpy(d, h, b); char* c = (char*)d;
#pragma omp target update to(c[0:b])
}
static void d2h_map(void* h, const void* d, size_t b) {
    char* c = (char*)d;
#pragma omp target update from(c[0:b])
    memcpy(h, d, b);
}
static int dev() { return omp_get_default_device(); }
static void* a_dev(size_t b) { return omp_target_alloc(b, dev()); }
static void f_dev(void* p) { omp_target_free(p, dev()); }
static void h2d_dev(void* d, const void* h, size_t b) { omp_target_memcpy(d, h, b, 0, 0, dev(), omp_get_initial_device()); }
static void d2h_dev(void* h, const void* d, size_t b) { omp_target_memcpy(h, d, b, 0, 0, omp_get_initial_device(), dev()); }
static void* a_sys(size_t b) { void* p = nullptr; if (posix_memalign(&p, 4096, b)) exit(1); return p; }
static void f_sys(void* p) { free(p); }
static void cpy(void* d, const void* s, size_t b) { memcpy(d, s, b); }
static void o_sync() {}               // les regions target sans nowait sont synchrones

static int nteams(long n) { return NT > 0 ? NT : (int)((n + TL - 1) / TL); }

// ---------------------------------------------------------------- noyaux
// Le corps de chaque boucle est identique dans toutes les versions ; seules
// les directives changent.

static void o_spmv(const DevMat& A, const double* x, double* y) {
    int n = A.n; const int* rp = A.rowptr; const int* ci = A.col; const double* v = A.val;
#define SPMV_BODY                                                         \
    for (int r = 0; r < n; r++) {                                         \
        double s = 0.0;                                                   \
        for (int p = rp[r]; p < rp[r + 1]; p++) s += v[p] * x[ci[p]];     \
        y[r] = s;                                                         \
    }
    switch (mode) {
    case NAIVE:
#pragma omp target teams distribute parallel for
        SPMV_BODY
        break;
    case DEVPTR:
#pragma omp target teams distribute parallel for is_device_ptr(rp, ci, v, x, y)
        SPMV_BODY
        break;
    case LOOP:
#pragma omp target teams loop is_device_ptr(rp, ci, v, x, y)
        SPMV_BODY
        break;
    case OPT: {
        int nt = nteams(n), tl = TL;
#pragma omp target teams distribute parallel for num_teams(nt) thread_limit(tl) is_device_ptr(rp, ci, v, x, y)
        SPMV_BODY
        break; }
    case UNIFIED:
#pragma omp target teams distribute parallel for
        SPMV_BODY
        break;
    }
}

static double o_dot(int n, const double* x, const double* y) {
    double s = 0.0;
#define DOT_BODY for (int i = 0; i < n; i++) s += x[i] * y[i];
    switch (mode) {
    case NAIVE:
#pragma omp target teams distribute parallel for reduction(+ : s)
        DOT_BODY
        break;
    case DEVPTR:
#pragma omp target teams distribute parallel for reduction(+ : s) is_device_ptr(x, y)
        DOT_BODY
        break;
    case LOOP:
#pragma omp target teams loop reduction(+ : s) is_device_ptr(x, y)
        DOT_BODY
        break;
    case OPT: {
        int nt = NT > 0 ? NT : 132 * 8, tl = TL;   // reduction : grille bornee, comme en CUDA
#pragma omp target teams distribute parallel for reduction(+ : s) num_teams(nt) thread_limit(tl) is_device_ptr(x, y)
        DOT_BODY
        break; }
    case UNIFIED:
#pragma omp target teams distribute parallel for reduction(+ : s)
        DOT_BODY
        break;
    }
    return s;
}

// Les trois mises a jour vectorielles partagent la meme forme : une macro
// genere les cinq variantes de directives pour un corps donne.
#define PRAGMA(...) _Pragma(#__VA_ARGS__)
#define VEC_KERNEL(BODY, ...)                                                          \
    switch (mode) {                                                                    \
    case NAIVE:                                                                        \
        PRAGMA(omp target teams distribute parallel for) BODY break;                  \
    case DEVPTR:                                                                       \
        PRAGMA(omp target teams distribute parallel for is_device_ptr(__VA_ARGS__)) BODY break; \
    case LOOP:                                                                         \
        PRAGMA(omp target teams loop is_device_ptr(__VA_ARGS__)) BODY break;          \
    case OPT: {                                                                        \
        int nt = nteams(n), tl = TL;                                                   \
        PRAGMA(omp target teams distribute parallel for num_teams(nt) thread_limit(tl) is_device_ptr(__VA_ARGS__)) BODY break; } \
    case UNIFIED:                                                                      \
        PRAGMA(omp target teams distribute parallel for) BODY break;                  \
    }

static void o_axpy(int n, double a, const double* x, double* y) {
    VEC_KERNEL(for (int i = 0; i < n; i++) y[i] += a * x[i];, x, y)
}
static void o_xpay(int n, const double* x, double a, double* y) {
    VEC_KERNEL(for (int i = 0; i < n; i++) y[i] = x[i] + a * y[i];, x, y)
}
static void o_pmul(int n, const double* d, const double* r, double* z) {
    VEC_KERNEL(for (int i = 0; i < n; i++) z[i] = d[i] * r[i];, d, r, z)
}

// Versions fusionnees (omp-fused) : memes clauses que omp-opt, une seule region
// target pour la mise a jour de x et r et le produit scalaire r.r, une autre
// pour Jacobi et r.z.
static double o_upd_dot(int n, double a, const double* p, const double* q, double* x, double* r) {
    double s = 0.0; int nt = NT > 0 ? NT : 132 * 8, tl = TL;
#pragma omp target teams distribute parallel for reduction(+ : s) num_teams(nt) thread_limit(tl) is_device_ptr(p, q, x, r)
    for (int i = 0; i < n; i++) {
        x[i] += a * p[i];
        double ri = r[i] - a * q[i];
        r[i] = ri; s += ri * ri;
    }
    return s;
}
static double o_prec_dot(int n, const double* d, const double* r, double* z) {
    double s = 0.0; int nt = NT > 0 ? NT : 132 * 8, tl = TL;
#pragma omp target teams distribute parallel for reduction(+ : s) num_teams(nt) thread_limit(tl) is_device_ptr(d, r, z)
    for (int i = 0; i < n; i++) {
        double ri = r[i], zi = d[i] * ri;
        z[i] = zi; s += ri * zi;
    }
    return s;
}

int main(int argc, char** argv) {
    std::string v = argc > 1 ? argv[1] : "";
    if (const char* e = getenv("OMP_TL")) TL = atoi(e);
    if (const char* e = getenv("OMP_NT")) NT = atoi(e);
    Ops o{v.c_str(), a_dev, f_dev, h2d_dev, d2h_dev, o_sync, nullptr, o_spmv, o_dot, o_axpy, o_xpay, o_pmul};
#ifdef OMP_UNIFIED_BUILD
    if (v != "omp-unified") { fprintf(stderr, "ce binaire ne fait que omp-unified\n"); return 1; }
    mode = UNIFIED; o.alloc = a_sys; o.free_ = f_sys; o.h2d = cpy; o.d2h = cpy;
#else
    if (v == "omp-naive") { mode = NAIVE; o.alloc = a_map; o.free_ = f_map; o.h2d = h2d_map; o.d2h = d2h_map; }
    else if (v == "omp-devptr") mode = DEVPTR;
    else if (v == "omp-loop") mode = LOOP;
    else if (v == "omp-opt") mode = OPT;
    else if (v == "omp-fused") { mode = OPT; o.upd_dot = o_upd_dot; o.prec_dot = o_prec_dot; }
    else { fprintf(stderr, "version inconnue : %s\n", v.c_str()); return 1; }
#endif
    return run_main(argc, argv, o);
}
