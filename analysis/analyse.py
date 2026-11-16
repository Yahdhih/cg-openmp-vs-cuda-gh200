"""Analyse des mesures : medianes, verification des iterations, graphiques.

Usage : .venv/bin/python analysis/analyse.py results/raw_<job>.csv results/sweep_<job>.csv
Sorties : results/summary.csv, results/summary.md, figures/*.png
"""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PEAK = 4000.0  # Go/s, HBM3 du GH200 96 Go (fiche NVIDIA)
ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
VERS = ["cuda", "cuda-fused", "cusparse", "cuda-managed", "cuda-system",
        "omp-naive", "omp-devptr", "omp-loop", "omp-opt", "omp-fused", "omp-unified"]
STYLE = {"cuda": ("#1b6e5a", "-"), "cuda-fused": ("#0b3d32", "-"), "omp-fused": ("#3d1450", "-"), "cusparse": ("#59a68f", "--"), "cuda-managed": ("#7a9e3d", ":"),
         "cuda-system": ("#a3b86c", "-."), "omp-naive": ("#b5452c", "-"), "omp-devptr": ("#d9822b", "--"),
         "omp-loop": ("#c9a227", ":"), "omp-opt": ("#7d3c98", "-"), "omp-unified": ("#b07cc6", "-.")}

raw, sweep = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None
rows = list(csv.DictReader(open(raw)))

# --- medianes par (version, matrice, m, test)
groups = defaultdict(list)
for r in rows:
    groups[(r["version"], r["matrix"], int(r["m"]), r["test"])].append(r)
summ = {}
for k, rs in groups.items():
    t = statistics.median(float(r["time_s"]) for r in rs)
    g = statistics.median(float(r["gbps"]) for r in rs)
    its = sorted({int(r["iters"]) for r in rs})
    summ[k] = dict(n=int(rs[0]["n"]), nnz=int(rs[0]["nnz"]), time=t, gbps=g, iters=its,
                   true_relres=max(float(r["true_relres"]) for r in rs),
                   err=max(float(r["err_inf"]) for r in rs), nrep=len(rs))

with open(ROOT / "results" / "summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["version", "matrix", "m", "n", "nnz", "test", "median_time_s", "median_gbps",
                "pct_peak", "iters", "max_true_relres", "max_err_inf", "reps"])
    for (v, mat, m, test), s in sorted(summ.items()):
        w.writerow([v, mat, m, s["n"], s["nnz"], test, f"{s['time']:.6e}", f"{s['gbps']:.1f}",
                    f"{100 * s['gbps'] / PEAK:.1f}", "/".join(map(str, s["iters"])),
                    f"{s['true_relres']:.2e}", f"{s['err']:.2e}", s["nrep"]])

cases = sorted({(k[1], k[2]) for k in summ if k[1] != "none"}, key=lambda c: (c[0], c[1]))
md = []

# --- 1. justesse : iterations identiques ?
md.append("## Justesse : nombre d'iterations du CG\n")
md.append("| matrice | m | n | " + " | ".join(VERS) + " |")
md.append("|---" * (3 + len(VERS)) + "|")
mismatch = []
for mat, m in cases:
    its = [summ.get((v, mat, m, "cg"), {}).get("iters", ["-"]) for v in VERS]
    flat = {i for it in its for i in it if i != "-"}
    if len(flat) > 1:
        mismatch.append((mat, m, sorted(flat)))
    n = summ[("cuda", mat, m, "cg")]["n"]
    md.append(f"| {mat} | {m} | {n} | " + " | ".join("/".join(map(str, i)) for i in its) + " |")
md.append("")
md.append("Ecarts d'iterations : " + (", ".join(f"{a} m={b}: {c}" for a, b, c in mismatch) if mismatch else "aucun") + "\n")

# --- 2. CG : temps median et rapport a CUDA
md.append("## CG complet : temps median (s) et rapport au CUDA ecrit a la main\n")
md.append("| matrice | m | iters | " + " | ".join(VERS) + " |")
md.append("|---" * (3 + len(VERS)) + "|")
for mat, m in cases:
    ref = summ[("cuda", mat, m, "cg")]["time"]
    cells = []
    for v in VERS:
        s = summ.get((v, mat, m, "cg"))
        cells.append(f"{s['time']:.3g} ({s['time'] / ref:.2f}x)" if s else "-")
    md.append(f"| {mat} | {m} | {summ[('cuda', mat, m, 'cg')]['iters'][0]} | " + " | ".join(cells) + " |")
md.append("")

# --- 3. noyaux : Go/s medians au plus grand cas de chaque matrice
md.append("## Noyaux : bande passante effective mediane (Go/s, % du pic 4000 Go/s)\n")
for test in ["spmv", "dot", "axpy"]:
    md.append(f"### {test}\n")
    md.append("| matrice | m | " + " | ".join(VERS) + " |")
    md.append("|---" * (2 + len(VERS)) + "|")
    for mat, m in cases:
        cells = []
        for v in VERS:
            s = summ.get((v, mat, m, test))
            cells.append(f"{s['gbps']:.0f} ({100 * s['gbps'] / PEAK:.0f}%)" if s else "-")
        md.append(f"| {mat} | {m} | " + " | ".join(cells) + " |")
    md.append("")

st = {k[3]: s for k, s in summ.items() if k[0] == "stream"}
if st:
    md.append(f"STREAM (CUDA, 2 Gio par tableau) : copy {st['copy']['gbps']:.0f} Go/s, "
              f"triad {st['triad']['gbps']:.0f} Go/s, soit {100 * st['triad']['gbps'] / PEAK:.0f}% du pic theorique.\n")

# --- figures : noyaux en fonction de n (Poisson 2D et 3D)
for mat in ["poisson2d", "poisson3d"]:
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, test in zip(axs, ["spmv", "dot", "axpy"]):
        for v in VERS:
            pts = sorted((summ[k]["n"], summ[k]["gbps"]) for k in summ if k[0] == v and k[1] == mat and k[3] == test)
            if pts:
                c, ls = STYLE[v]
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", color=c, ls=ls, label=v)
        if st:
            ax.axhline(st["triad"]["gbps"], color="grey", lw=0.8, ls=":")
            ax.text(ax.get_xlim()[0] if False else 1, st["triad"]["gbps"], "", fontsize=7)
        ax.axhline(PEAK, color="black", lw=0.8)
        ax.set_xscale("log"); ax.set_title(f"{test} ({mat})"); ax.set_xlabel("n (inconnues)")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("Go/s effectifs (trait noir : pic 4000, pointille : STREAM triad)")
    axs[-1].legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / f"kernels_{mat}.png", dpi=140); plt.close(fig)

# --- figure : CG, rapport de temps a CUDA, par cas
fig, ax = plt.subplots(figsize=(14, 4.8))
wbar = 0.09
for i, v in enumerate(VERS):
    ys = []
    for mat, m in cases:
        s = summ.get((v, mat, m, "cg"))
        ys.append(s["time"] / summ[("cuda", mat, m, "cg")]["time"] if s else 0)
    ax.bar([j + (i - len(VERS) / 2) * wbar for j in range(len(cases))], ys, wbar, color=STYLE[v][0], label=v)
ax.axhline(1, color="black", lw=0.8)
ax.set_xticks(range(len(cases))); ax.set_xticklabels([f"{a}\nm={b}" for a, b in cases], fontsize=8)
ax.set_ylabel("temps CG / temps CUDA (plus bas = mieux)"); ax.set_yscale("log")
ax.legend(ncol=5, fontsize=7); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / "cg_ratio.png", dpi=140); plt.close(fig)

# --- balayage thread_limit
if sweep and Path(sweep).exists():
    sw = list(csv.DictReader(open(sweep)))
    d = defaultdict(list)
    for r in sw:
        d[(int(r["tl"]), r["test"])].append(float(r["gbps"]))
    md.append("## omp-opt : balayage de thread_limit (Poisson 2D, m = 2048), Go/s medians\n")
    md.append("| thread_limit | spmv | dot | axpy |\n|---|---|---|---|")
    for tl in sorted({k[0] for k in d}):
        md.append(f"| {tl} | " + " | ".join(f"{statistics.median(d[(tl, t)]):.0f}" for t in ["spmv", "dot", "axpy"]) + " |")
    md.append("")

(ROOT / "results" / "summary.md").write_text("\n".join(md))
print("\n".join(md))

# --- roofline : noyaux au plus grand cas Poisson 2D (n = 16,8 M)
# intensite arithmetique = flops / octets du modele ideal ; FP64 H100 (GH200) : 67 Tflop/s
# avec Tensor Cores, 34 sans ; les noyaux du CG n'utilisent que les unites FP64 classiques.
FP64 = 34000.0  # Gflop/s
roof_bw = st["triad"]["gbps"] if st else PEAK
big = ("poisson2d", 4096)
fig, ax = plt.subplots(figsize=(7.5, 5))
ai = [10 ** (k / 100) for k in range(-130, -40)]
ax.plot(ai, [min(PEAK * a, FP64) for a in ai], "k-", lw=1, label="pic HBM3 4,0 To/s / FP64 34 Tflop/s")
ax.plot(ai, [min(roof_bw * a, FP64) for a in ai], "k:", lw=1, label=f"STREAM triad mesure {roof_bw:.0f} Go/s")
s0 = summ[("cuda", "poisson2d", 4096, "spmv")]
flops = {"spmv": lambda s: 2.0 * s["nnz"], "dot": lambda s: 2.0 * s["n"], "axpy": lambda s: 2.0 * s["n"]}
octets = {"spmv": lambda s: 12.0 * s["nnz"] + 4.0 * (s["n"] + 1) + 16.0 * s["n"], "dot": lambda s: 16.0 * s["n"],
          "axpy": lambda s: 24.0 * s["n"]}
mk = {"spmv": "o", "dot": "s", "axpy": "^"}
for test in ["spmv", "dot", "axpy"]:
    for v in ["cuda", "omp-devptr", "omp-opt"]:
        s = summ[(v, big[0], big[1], test)]
        I = flops[test](s) / octets[test](s)
        perf = flops[test](s) / s["time"] * 1e-9
        ax.plot(I, perf, mk[test], color=STYLE[v][0], ms=8, label=f"{test} {v}")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.06, 0.3); ax.set_ylim(80, 1500)
ax.set_xlabel("intensite arithmetique (flop / octet)"); ax.set_ylabel("Gflop/s")
ax.set_title("Roofline GH200, noyaux du CG (Poisson 2D, 16,8 M)"); ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=7, ncol=2, loc="upper left")
fig.tight_layout(); fig.savefig(FIG / "roofline.png", dpi=140); plt.close(fig)
