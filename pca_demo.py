"""
pca_demo.py — Samostatná demonštrácia PCA (bez Streamlitu)
===========================================================
Spustenie:
    python3 pca_demo.py              # 2D demo
    python3 pca_demo.py --dim 3      # 3D demo
    python3 pca_demo.py --csv data.csv

Výstup: pca_output.png  (uloží všetky grafy)
"""
from __future__ import annotations

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Matematické jadro PCA (žiadne sklearn, iba NumPy)
# ---------------------------------------------------------------------------

def center(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Krok 1–2: priemer a centrovanie."""
    mu = X.mean(axis=0)
    return X - mu, mu


def covariance(X_c: np.ndarray) -> np.ndarray:
    """Krok 3: kovariančná matica  C = X̃ᵀ X̃ / (n−1)."""
    n = X_c.shape[0]
    return (X_c.T @ X_c) / max(n - 1, 1)


def eigen_sorted(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Krok 4–6: spektrálny rozklad symetrickej matice, zoradenie zostupne,
    deterministické znamienko (najväčšia abs. zložka kladná).
    """
    vals, vecs = np.linalg.eigh(C)          # eigh — garantovane reálne pre sym. matice

    order = np.argsort(vals)[::-1]          # zostupné poradie
    vals  = vals[order]
    vecs  = vecs[:, order]

    for i in range(vecs.shape[1]):          # unifikácia znamienka
        j = np.argmax(np.abs(vecs[:, i]))
        if vecs[j, i] < 0:
            vecs[:, i] *= -1

    return vals, vecs


def energy(vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Krok 7: podiel rozptylu a kumulatívna energia."""
    total = vals.sum()
    ratio = vals / total if total > 0 else np.zeros_like(vals)
    return ratio, np.cumsum(ratio)


def transform(X_c: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Krok 8: projekcia  Y = X̃ · W."""
    return X_c @ W


def reconstruct(Y_k: np.ndarray, W_k: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Spätná rekonštrukcia  X̂ = Y_k · Wₖᵀ + μ."""
    return Y_k @ W_k.T + mu


def pca(X: np.ndarray) -> dict:
    """Celý PCA pipeline — vracia slovník so všetkými medzivýsledkami."""
    X = np.asarray(X, dtype=float)
    X_c, mu   = center(X)
    C         = covariance(X_c)
    vals, W   = eigen_sorted(C)
    ratio, cum = energy(vals)
    Y         = transform(X_c, W)
    return dict(X=X, mu=mu, X_c=X_c, C=C,
                vals=vals, W=W, ratio=ratio, cum=cum, Y=Y)


# ---------------------------------------------------------------------------
# Generátory vzorových dát
# ---------------------------------------------------------------------------

def make_2d(n: int = 200, angle: float = 35.0,
            sx: float = 3.0, sy: float = 0.7,
            seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = np.deg2rad(angle)
    R = np.array([[np.cos(a), -np.sin(a)],
                  [np.sin(a),  np.cos(a)]])
    raw = np.column_stack([rng.standard_normal(n) * sx,
                           rng.standard_normal(n) * sy])
    return raw @ R.T + np.array([3.0, 2.0])


def make_3d(n: int = 300, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cov_true = np.array([[4.0, 1.8, 0.6],
                         [1.8, 1.5, 0.3],
                         [0.6, 0.3, 0.4]])
    L = np.linalg.cholesky(cov_true)
    return rng.standard_normal((n, 3)) @ L.T + np.array([1.0, -2.0, 3.0])


# ---------------------------------------------------------------------------
# Textový výpis výsledkov
# ---------------------------------------------------------------------------

def print_results(r: dict) -> None:
    d = r["vals"].shape[0]
    print("\n" + "=" * 60)
    print("  PCA — výsledky")
    print("=" * 60)
    print(f"  Vzorky n = {r['X'].shape[0]},  dimenzia d = {d}")
    print(f"\n  Vektor stredných hodnôt μ:\n  {r['mu'].round(4)}")
    print("\n  Kovariančná matica C:")
    for row in r["C"].round(4):
        print("  ", row)
    print("\n  Transformačné vektory (stĺpce = hlavné komponenty):")
    for i in range(d):
        vec_str = "  ".join(f"{v:+.4f}" for v in r["W"][:, i])
        ev = r["vals"][i]
        pct = r["ratio"][i] * 100
        cum = r["cum"][i] * 100
        print(f"    PC{i+1}: [{vec_str}]  λ={ev:.4f}  "
              f"var={pct:.2f}%  cum={cum:.2f}%")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Vizualizácia — 2D
# ---------------------------------------------------------------------------

def plot_2d(r: dict, save_path: str = "pca_output.png") -> None:
    X, mu, X_c, W, vals, Y = r["X"], r["mu"], r["X_c"], r["W"], r["vals"], r["Y"]
    scale = 2.5
    colors_pc = ["#C44E52", "#55A868"]

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle("PCA — 2D demonštrácia", fontsize=14, fontweight="bold")

    # ── (A) Pôvodné dáta + osi PC ──────────────────────────────────────────
    ax = fig.add_subplot(2, 3, 1)
    ax.scatter(X[:, 0], X[:, 1], alpha=0.55, s=25, color="#4C72B0",
               edgecolor="white", lw=0.4, label="dáta")
    ax.scatter(*mu, color="black", s=90, marker="x", zorder=5, label="μ")
    for i, col in enumerate(colors_pc):
        v = W[:, i]
        length = np.sqrt(max(vals[i], 1e-12)) * scale
        ax.annotate("", xy=mu + v * length, xytext=mu,
                    arrowprops=dict(arrowstyle="->", color=col, lw=2.5))
        ax.text(*(mu + v * length * 1.12), f"PC{i+1}",
                color=col, fontsize=11, fontweight="bold")
    ax.set_title("(A)  Pôvodné dáta + PCA osi")
    ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
    ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    ax.set_aspect("equal", "datalim"); ax.grid(True, alpha=0.3); ax.legend()

    # ── (B) Centrované dáta + osi cez 0 ────────────────────────────────────
    ax = fig.add_subplot(2, 3, 2)
    ax.scatter(X_c[:, 0], X_c[:, 1], alpha=0.55, s=25, color="#4C72B0",
               edgecolor="white", lw=0.4)
    for i, col in enumerate(colors_pc):
        v = W[:, i]
        length = np.sqrt(max(vals[i], 1e-12)) * scale
        ax.annotate("", xy=v * length, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=col, lw=2.5))
        ax.text(*(v * length * 1.12), f"PC{i+1}",
                color=col, fontsize=11, fontweight="bold")
    ax.scatter(0, 0, color="black", s=90, marker="x", zorder=5)
    ax.set_title("(B)  Centrované dáta  X̃ = X − μ")
    ax.set_xlabel("x₁ − μ₁"); ax.set_ylabel("x₂ − μ₂")
    ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    ax.set_aspect("equal", "datalim"); ax.grid(True, alpha=0.3)

    # ── (C) Po PCA transformácii (dekorovaný priestor) ─────────────────────
    ax = fig.add_subplot(2, 3, 3)
    ax.scatter(Y[:, 0], Y[:, 1], alpha=0.55, s=25, color="#8172B2",
               edgecolor="white", lw=0.4)
    ax.axhline(0, color=colors_pc[1], lw=1.8, alpha=0.7, label="PC2 = 0")
    ax.axvline(0, color=colors_pc[0], lw=1.8, alpha=0.7, label="PC1 = 0")
    ax.set_title("(C)  Po transformácii  Y = X̃ · W\n(dekorovaný, os = hlavné komponenty)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_aspect("equal", "datalim"); ax.grid(True, alpha=0.3); ax.legend()

    # ── (D) 1D projekcia + rekonštrukcia ───────────────────────────────────
    W1 = W[:, :1]
    Y1 = X_c @ W1
    X_rec = Y1 @ W1.T + mu

    ax = fig.add_subplot(2, 3, 4)
    ax.scatter(X[:, 0], X[:, 1], alpha=0.35, s=22, color="#4C72B0",
               label="pôvodné", edgecolor="white", lw=0.3)
    ax.scatter(X_rec[:, 0], X_rec[:, 1], alpha=0.75, s=22, color="#C44E52",
               label=f"rekonštrukcia k=1 ({r['cum'][0]*100:.1f}% energie)",
               edgecolor="white", lw=0.3)
    for i in range(len(X)):
        ax.plot([X[i, 0], X_rec[i, 0]], [X[i, 1], X_rec[i, 1]],
                color="gray", lw=0.4, alpha=0.35)
    ax.set_title("(D)  Rekonštrukcia z 1 komponentu")
    ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
    ax.set_aspect("equal", "datalim"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # ── (E) Scree plot / variancie ─────────────────────────────────────────
    ax = fig.add_subplot(2, 3, 5)
    d = len(vals)
    pc_labels = [f"PC{i+1}" for i in range(d)]
    bars = ax.bar(pc_labels, vals, color=colors_pc[:d], zorder=3)
    for bar, v, pct in zip(bars, vals, r["ratio"]):
        ax.text(bar.get_x() + bar.get_width() / 2, v,
                f"λ={v:.3f}\n{pct*100:.1f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("vlastné číslo λ  (rozptyl)")
    ax.set_title("(E)  Scree plot — rozptyl na komponent")
    ax.grid(True, alpha=0.3, axis="y")

    # ── (F) Kumulatívna energia ────────────────────────────────────────────
    ax = fig.add_subplot(2, 3, 6)
    ax.plot(range(1, d + 1), r["cum"] * 100, "o-",
            color="#C44E52", lw=2, ms=8, zorder=4)
    ax.fill_between(range(1, d + 1), r["cum"] * 100, alpha=0.15, color="#C44E52")
    ax.axhline(95, ls="--", color="gray", alpha=0.6, label="95 %")
    ax.axhline(99, ls=":",  color="gray", alpha=0.6, label="99 %")
    ax.set_xticks(range(1, d + 1)); ax.set_xticklabels(pc_labels)
    ax.set_ylim(0, 105)
    ax.set_ylabel("kumulatívna energia [%]")
    ax.set_title("(F)  Kumulatívna energia")
    ax.grid(True, alpha=0.3); ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"  Grafy uložené: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Vizualizácia — 3D
# ---------------------------------------------------------------------------

def plot_3d(r: dict, save_path: str = "pca_output.png") -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    X, mu, W, vals, Y = r["X"], r["mu"], r["W"], r["vals"], r["Y"]
    d = vals.shape[0]
    colors_pc = ["#C44E52", "#55A868", "#8172B2"]
    scale = 2.5

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle("PCA — 3D demonštrácia", fontsize=14, fontweight="bold")

    # ── (A) 3D pôvodné dáta + osi ─────────────────────────────────────────
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=0.4, s=14, color="#4C72B0")
    for i, col in enumerate(colors_pc):
        v = W[:, i]
        length = np.sqrt(max(vals[i], 1e-12)) * scale
        end = mu + v * length
        ax1.plot([mu[0], end[0]], [mu[1], end[1]], [mu[2], end[2]],
                 color=col, lw=2.5)
        ax1.text(*end, f"  PC{i+1}", color=col, fontweight="bold", fontsize=9)
    ax1.set_title("(A)  Pôvodné 3D dáta + PCA osi")
    ax1.set_xlabel("x₁"); ax1.set_ylabel("x₂"); ax1.set_zlabel("x₃")

    # ── (B) Projekcia PC1 × PC2 ───────────────────────────────────────────
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.scatter(Y[:, 0], Y[:, 1], alpha=0.55, s=18, color="#8172B2",
                edgecolor="white", lw=0.3)
    ax2.axhline(0, color="gray", lw=0.5); ax2.axvline(0, color="gray", lw=0.5)
    ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")
    ax2.set_title(
        f"(B)  Projekcia PC1 × PC2\n"
        f"({r['cum'][1]*100:.1f} % energie z 2 komp.)"
    )
    ax2.set_aspect("equal", "datalim"); ax2.grid(True, alpha=0.3)

    # ── (C) Scree + kumulatívna energia ───────────────────────────────────
    ax3 = fig.add_subplot(1, 3, 3)
    pc_labels = [f"PC{i+1}" for i in range(d)]
    bars = ax3.bar(pc_labels, vals, color=colors_pc[:d], zorder=3)
    for bar, v, pct in zip(bars, vals, r["ratio"]):
        ax3.text(bar.get_x() + bar.get_width() / 2, v,
                 f"λ={v:.2f}\n{pct*100:.1f}%",
                 ha="center", va="bottom", fontsize=8)
    ax3_r = ax3.twinx()
    ax3_r.plot(range(d), r["cum"] * 100, "o-",
               color="black", lw=1.5, ms=6, label="kum. energia")
    ax3_r.axhline(95, ls="--", color="gray", alpha=0.5)
    ax3_r.set_ylim(0, 105); ax3_r.set_ylabel("kumulatívna energia [%]")
    ax3.set_title("(C)  Scree plot")
    ax3.set_ylabel("vlastné číslo λ"); ax3.grid(True, alpha=0.3, axis="y")
    ax3_r.legend(loc="center right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"  Grafy uložené: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Vstupný bod
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PCA demo — čistý výpočet")
    parser.add_argument("--dim",  type=int, choices=[2, 3], default=2,
                        help="Dimenzia syntetických dát (2 alebo 3)")
    parser.add_argument("--csv",  type=str, default=None,
                        help="Vstupný CSV súbor (bez hlavičky alebo s ňou)")
    parser.add_argument("--n",    type=int, default=200,
                        help="Počet vzoriek (pre syntetické dáta)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--out",  type=str, default="pca_output.png",
                        help="Výstupný PNG súbor")
    args = parser.parse_args()

    if args.csv:
        try:
            data = np.loadtxt(args.csv, delimiter=",", skiprows=1)
        except Exception:
            data = np.loadtxt(args.csv, delimiter=",")
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        print(f"  Načítané dáta: {data.shape[0]} vzoriek, {data.shape[1]} znakov")
        X = data
        dim = data.shape[1]
    elif args.dim == 3:
        X   = make_3d(n=args.n, seed=args.seed)
        dim = 3
        print(f"  Syntetické 3D dáta: {args.n} vzoriek")
    else:
        X   = make_2d(n=args.n, seed=args.seed)
        dim = 2
        print(f"  Syntetické 2D dáta: {args.n} vzoriek, rotácia 35°")

    r = pca(X)
    print_results(r)

    if dim == 2:
        plot_2d(r, save_path=args.out)
    elif dim == 3:
        plot_3d(r, save_path=args.out)
    else:
        print("  Vizualizácia len pre 2D a 3D dáta.")


if __name__ == "__main__":
    main()
