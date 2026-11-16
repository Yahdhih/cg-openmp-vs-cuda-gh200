// common.h : matrices CSR, generateurs de problemes, gradient conjugue
// preconditionne (Jacobi) et harnais de mesure, communs a tous les backends.
//
// Chaque backend (CUDA, OpenMP target) fournit les memes operations sur des
// pointeurs "device" (struct Ops) ; ce fichier contient tout le reste, pour
// que les versions ne different QUE par la facon d'executer les noyaux.
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------- matrices

struct CSR {                       // matrice creuse, format CSR, cote hote
    int n = 0;
    long nnz = 0;
    std::vector<int> rowptr, col;
    std::vector<double> val;
};

// Poisson 2D, schema a 5 points, conditions de Dirichlet : m x m inconnues.
static CSR poisson2d(int m) {
    CSR A; A.n = m * m; A.rowptr.reserve(A.n + 1); A.rowptr.push_back(0);
    for (int j = 0; j < m; j++)
        for (int i = 0; i < m; i++) {
            int r = j * m + i;
            auto add = [&](int c, double v) { A.col.push_back(c); A.val.push_back(v); };
            if (j > 0) add(r - m, -1.0);
            if (i > 0) add(r - 1, -1.0);
            add(r, 4.0);
            if (i < m - 1) add(r + 1, -1.0);
            if (j < m - 1) add(r + m, -1.0);
            A.rowptr.push_back((int)A.col.size());
        }
    A.nnz = (long)A.col.size();
    return A;
}

// Poisson 3D, schema a 7 points : m x m x m inconnues.
static CSR poisson3d(int m) {
    CSR A; A.n = m * m * m; A.rowptr.reserve(A.n + 1); A.rowptr.push_back(0);
    long mm = (long)m * m;
    for (int k = 0; k < m; k++)
        for (int j = 0; j < m; j++)
            for (int i = 0; i < m; i++) {
                int r = (int)(k * mm + j * m + i);
                auto add = [&](long c, double v) { A.col.push_back((int)c); A.val.push_back(v); };
                if (k > 0) add(r - mm, -1.0);
                if (j > 0) add(r - m, -1.0);
                if (i > 0) add(r - 1, -1.0);
                add(r, 6.0);
                if (i < m - 1) add(r + 1, -1.0);
                if (j < m - 1) add(r + m, -1.0);
                if (k < m - 1) add(r + mm, -1.0);
                A.rowptr.push_back((int)A.col.size());
            }
    A.nnz = (long)A.col.size();
    return A;
}

// -div(k grad u) en 2D, volumes finis, k constant par cellule avec un fort
// contraste (damier de 8 x 8 zones, k = 1 ou k = contrast) : c'est le type
// d'operateur du solveur de pression (coefficient variable, matrice SPD mal
// conditionnee). Coefficient de face = moyenne harmonique des deux cellules.
static CSR varcoef2d(int m, double contrast) {
    std::vector<double> k((size_t)m * m);
    int zone = std::max(1, m / 8);
    for (int j = 0; j < m; j++)
        for (int i = 0; i < m; i++)
            k[(size_t)j * m + i] = (((i / zone) + (j / zone)) % 2 == 0) ? contrast : 1.0;
    auto hm = [](double a, double b) { return 2.0 * a * b / (a + b); };
    CSR A; A.n = m * m; A.rowptr.reserve(A.n + 1); A.rowptr.push_back(0);
    for (int j = 0; j < m; j++)
        for (int i = 0; i < m; i++) {
            int r = j * m + i; double kc = k[r], diag = 0.0;
            double kS = j > 0     ? hm(kc, k[r - m]) : kc;   // bord : Dirichlet
            double kW = i > 0     ? hm(kc, k[r - 1]) : kc;
            double kE = i < m - 1 ? hm(kc, k[r + 1]) : kc;
            double kN = j < m - 1 ? hm(kc, k[r + m]) : kc;
            diag = kS + kW + kE + kN;
            auto add = [&](int c, double v) { A.col.push_back(c); A.val.push_back(v); };
            if (j > 0) add(r - m, -kS);
            if (i > 0) add(r - 1, -kW);
            add(r, diag);
            if (i < m - 1) add(r + 1, -kE);
            if (j < m - 1) add(r + m, -kN);
            A.rowptr.push_back((int)A.col.size());
        }
    A.nnz = (long)A.col.size();
    return A;
}

static CSR make_matrix(const std::string& kind, int m) {
    if (kind == "poisson2d") return poisson2d(m);
    if (kind == "poisson3d") return poisson3d(m);
    if (kind == "varcoef2d") return varcoef2d(m, 1.0e3);
    fprintf(stderr, "matrice inconnue : %s\n", kind.c_str()); exit(1);
}

static void spmv_host(const CSR& A, const double* x, double* y) {
    for (int r = 0; r < A.n; r++) {
        double s = 0.0;
        for (int p = A.rowptr[r]; p < A.rowptr[r + 1]; p++) s += A.val[p] * x[A.col[p]];
        y[r] = s;
    }
}

// ------------------------------------------------- interface des backends

struct DevMat {                    // la meme matrice, cote device
    int n; long nnz;
    int* rowptr; int* col; double* val;
    void* extra;                   // descripteurs propres au backend (cuSPARSE)
};

struct Ops {
    const char* name;
    void* (*alloc)(size_t bytes);
    void  (*free_)(void* p);
    void  (*h2d)(void* d, const void* h, size_t bytes);
    void  (*d2h)(void* h, const void* d, size_t bytes);
    void  (*sync)();
    void  (*mat_setup)(DevMat& A);                       // optionnel (cuSPARSE)
    void  (*spmv)(const DevMat& A, const double* x, double* y);      // y = A x
    double(*dot)(int n, const double* x, const double* y);           // x . y
    void  (*axpy)(int n, double a, const double* x, double* y);      // y += a x
    void  (*xpay)(int n, const double* x, double a, double* y);      // y = x + a y
    void  (*pmul)(int n, const double* d, const double* r, double* z); // z = d .* r
    // noyaux fusionnes (optionnels, nullptr = version separee)
    double(*upd_dot)(int n, double a, const double* p, const double* q, double* x, double* r);
                                   // x += a p ; r -= a q ; renvoie r . r
    double(*prec_dot)(int n, const double* d, const double* r, double* z);
                                   // z = d .* r ; renvoie r . z
};

// Octets deplaces par appel (modele "ideal" : chaque donnee lue ou ecrite une
// fois ; pour la SpMV, x est suppose lu une seule fois grace au cache).
static double bytes_spmv(const DevMat& A) { return 12.0 * A.nnz + 4.0 * (A.n + 1) + 16.0 * A.n; }
static double bytes_dot(int n)  { return 16.0 * n; }
static double bytes_axpy(int n) { return 24.0 * n; }

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------- gradient conjugue

struct CGResult { int iters; double relres, time, true_relres, err_inf; };

// PCG Jacobi, x0 = 0, b = A * 1 (solution exacte connue : le vecteur de 1).
// Par iteration : 1 SpMV, 3 produits scalaires, 3 mises a jour, 1 Jacobi.
static CGResult pcg(const Ops& o, const CSR& Ah, DevMat& A, const double* b, const double* dinv,
                    double tol, int maxit) {
    int n = A.n; size_t vb = (size_t)n * sizeof(double);
    double *x = (double*)o.alloc(vb), *r = (double*)o.alloc(vb), *z = (double*)o.alloc(vb),
           *p = (double*)o.alloc(vb), *q = (double*)o.alloc(vb);
    std::vector<double> zero(n, 0.0);
    o.h2d(x, zero.data(), vb);
    o.h2d(r, zero.data(), vb);
    o.sync();

    double t0 = now();
    o.xpay(n, b, 0.0, r);                       // r = b - A*0 = b
    o.pmul(n, dinv, r, z);                      // z = M^-1 r
    o.h2d(p, zero.data(), vb);
    o.xpay(n, z, 0.0, p);                       // p = z
    double rz = o.dot(n, r, z);
    double bnorm = std::sqrt(o.dot(n, b, b));
    double rnorm = bnorm;
    int k = 0;
    for (; k < maxit; k++) {
        o.spmv(A, p, q);                        // q = A p
        double alpha = rz / o.dot(n, p, q);
        double rr;
        if (o.upd_dot) rr = o.upd_dot(n, alpha, p, q, x, r);   // 1 lancement au lieu de 3
        else {
            o.axpy(n, alpha, p, x);             // x += alpha p
            o.axpy(n, -alpha, q, r);            // r -= alpha q
            rr = o.dot(n, r, r);
        }
        rnorm = std::sqrt(rr);
        if (rnorm / bnorm < tol) { k++; break; }
        double rz2;
        if (o.prec_dot) rz2 = o.prec_dot(n, dinv, r, z);       // 1 lancement au lieu de 2
        else {
            o.pmul(n, dinv, r, z);              // z = M^-1 r
            rz2 = o.dot(n, r, z);
        }
        double beta = rz2 / rz; rz = rz2;
        o.xpay(n, z, beta, p);                  // p = z + beta p
    }
    o.sync();
    double t = now() - t0;

    // verification independante sur l'hote : vrai residu et erreur
    std::vector<double> xh(n), Ax(n), bh(n);
    o.d2h(xh.data(), x, vb); o.d2h(bh.data(), b, vb);
    spmv_host(Ah, xh.data(), Ax.data());
    double res = 0, bb = 0, err = 0;
    for (int i = 0; i < n; i++) {
        res += (bh[i] - Ax[i]) * (bh[i] - Ax[i]); bb += bh[i] * bh[i];
        err = std::max(err, std::fabs(xh[i] - 1.0));
    }
    o.free_(x); o.free_(r); o.free_(z); o.free_(p); o.free_(q);
    return {k, rnorm / bnorm, t, std::sqrt(res / bb), err};
}

// ------------------------------------------------------------ harnais

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t m = v.size() / 2;
    return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

// Usage commun : <exe> <version> <matrice> <m> <reps> [cg|kernels|all]
// Sortie : lignes CSV sur stdout
//   version,matrix,m,n,nnz,test,rep,time_s,gbps,iters,relres,true_relres,err_inf
static int run_main(int argc, char** argv, const Ops& o) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <version> <poisson2d|poisson3d|varcoef2d> <m> <reps> [cg|kernels|all]\n", argv[0]);
        return 1;
    }
    std::string ver = argv[1], kind = argv[2];
    int m = atoi(argv[3]), reps = atoi(argv[4]);
    std::string what = argc > 5 ? argv[5] : "all";

    CSR Ah = make_matrix(kind, m);
    int n = Ah.n;
    std::vector<double> ones(n, 1.0), bh(n), dinvh(n);
    spmv_host(Ah, ones.data(), bh.data());
    for (int r = 0; r < n; r++)
        for (int p = Ah.rowptr[r]; p < Ah.rowptr[r + 1]; p++)
            if (Ah.col[p] == r) dinvh[r] = 1.0 / Ah.val[p];

    DevMat A{n, Ah.nnz, nullptr, nullptr, nullptr, nullptr};
    A.rowptr = (int*)o.alloc((n + 1) * sizeof(int));
    A.col = (int*)o.alloc(Ah.nnz * sizeof(int));
    A.val = (double*)o.alloc(Ah.nnz * sizeof(double));
    o.h2d(A.rowptr, Ah.rowptr.data(), (n + 1) * sizeof(int));
    o.h2d(A.col, Ah.col.data(), Ah.nnz * sizeof(int));
    o.h2d(A.val, Ah.val.data(), Ah.nnz * sizeof(double));
    if (o.mat_setup) o.mat_setup(A);
    size_t vb = (size_t)n * sizeof(double);
    double *b = (double*)o.alloc(vb), *dinv = (double*)o.alloc(vb);
    o.h2d(b, bh.data(), vb); o.h2d(dinv, dinvh.data(), vb);
    o.sync();

    auto line = [&](const char* test, int rep, double t, double gbps, int it, double rr, double trr, double e) {
        printf("%s,%s,%d,%d,%ld,%s,%d,%.6e,%.2f,%d,%.3e,%.3e,%.3e\n", ver.c_str(), kind.c_str(), m, n,
               Ah.nnz, test, rep, t, gbps, it, rr, trr, e);
        fflush(stdout);
    };

    if (what == "kernels" || what == "all") {
        double *x = (double*)o.alloc(vb), *y = (double*)o.alloc(vb);
        o.h2d(x, ones.data(), vb); o.h2d(y, bh.data(), vb);
        const int K = 50;                         // appels par mesure
        volatile double sink = 0;
        for (int w = 0; w < 3; w++) { o.spmv(A, x, y); sink = sink + o.dot(n, x, y); o.axpy(n, 1e-9, x, y); }
        o.sync();
        for (int rep = 0; rep < reps; rep++) {
            double t0 = now();
            for (int i = 0; i < K; i++) o.spmv(A, x, y);
            o.sync();
            double t = (now() - t0) / K;
            line("spmv", rep, t, bytes_spmv(A) / t * 1e-9, 0, 0, 0, 0);

            t0 = now();
            for (int i = 0; i < K; i++) sink = sink + o.dot(n, x, y);
            t = (now() - t0) / K;
            line("dot", rep, t, bytes_dot(n) / t * 1e-9, 0, 0, 0, 0);

            t0 = now();
            for (int i = 0; i < K; i++) o.axpy(n, 1e-9, x, y);
            o.sync();
            t = (now() - t0) / K;
            line("axpy", rep, t, bytes_axpy(n) / t * 1e-9, 0, 0, 0, 0);
        }
        o.free_(x); o.free_(y);
    }

    if (what == "cg" || what == "all") {
        const double tol = 1e-8; const int maxit = 20000;
        pcg(o, Ah, A, b, dinv, tol, maxit);       // echauffement (et migrations)
        for (int rep = 0; rep < reps; rep++) {
            CGResult R = pcg(o, Ah, A, b, dinv, tol, maxit);
            // octets par iteration (modele ideal). Separe : SpMV + 3 dot + 3 axpy-like
            // + Jacobi. Fusionne : SpMV + dot(p,q) + maj x,r (lit p,q,x,r, ecrit x,r)
            // + Jacobi+dot (lit d,r, ecrit z) + xpay.
            double bpi = o.upd_dot ? bytes_spmv(A) + bytes_dot(n) + 48.0 * n + 24.0 * n + bytes_axpy(n)
                                   : bytes_spmv(A) + 3 * bytes_dot(n) + 3 * bytes_axpy(n) + 24.0 * n;
            line("cg", rep, R.time, bpi * R.iters / R.time * 1e-9, R.iters, R.relres, R.true_relres, R.err_inf);
        }
    }
    o.free_(A.rowptr); o.free_(A.col); o.free_(A.val); o.free_(b); o.free_(dinv);
    return 0;
}
