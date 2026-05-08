"""
pca_app.py — Streamlit UI pre PCA kalkulačku
=============================================
Spustenie:
    streamlit run pca_app.py

Celá PCA matematika je v pca_core.py.
"""
from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from pca_core import manual_pca, reduce_dimension

# ----------------------------------------------------------------------------
# Konfigurácia stránky
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="PCA — Analýza Hlavných Komponentov",
    page_icon="📊",
    layout="wide",
)

# ============================================================================
# UI — HLAVIČKA + VSTUPNÉ DÁTA (sidebar)
# ============================================================================
st.title("📊 PCA — Analýza Hlavných Komponentov")
st.caption(
    "Interaktívny výpočet PCA s **kompletnými medzivýpočtami** "
    "(stredná hodnota, kovariančná matica, vlastné čísla, transformačné vektory, "
    "variancie, energia, redukcia dimenzie a rekonštrukcia)."
)

st.sidebar.header("📥 Vstupné dáta")
data_source = st.sidebar.radio(
    "Zdroj dát:",
    ["Vzorka 2D (rotovaný oblak)", "Vzorka 3D (korelovaný)", "Vlastné CSV", "Ručné zadanie"],
)

standardize = st.sidebar.checkbox(
    "Štandardizovať (z-score) pred PCA",
    value=False,
    help="Užitočné, ak majú znaky rôzne jednotky/škály. PCA je citlivá na škálu.",
)

X_raw: np.ndarray | None = None

if data_source == "Vzorka 2D (rotovaný oblak)":
    seed = st.sidebar.number_input("Random seed", 0, 9999, 42)
    n = st.sidebar.slider("Počet vzoriek", 20, 1000, 150)
    angle = np.deg2rad(st.sidebar.slider("Otočenie [°]", 0, 180, 30))
    sx = st.sidebar.slider("Rozptyl pozdĺž hlavnej osi", 0.5, 5.0, 3.0)
    sy = st.sidebar.slider("Rozptyl pozdĺž vedľajšej osi", 0.1, 3.0, 0.8)
    rng = np.random.default_rng(int(seed))
    base = np.column_stack([rng.standard_normal(n) * sx, rng.standard_normal(n) * sy])
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    X_raw = base @ R.T + np.array([2.0, 1.0])

elif data_source == "Vzorka 3D (korelovaný)":
    seed = st.sidebar.number_input("Random seed", 0, 9999, 7)
    n = st.sidebar.slider("Počet vzoriek", 30, 1000, 200)
    rng = np.random.default_rng(int(seed))
    cov_true = np.array([[3.0, 1.0, 0.5], [1.0, 1.5, 0.3], [0.5, 0.3, 0.4]])
    L = np.linalg.cholesky(cov_true)
    X_raw = rng.standard_normal((n, 3)) @ L.T + np.array([1.0, -2.0, 3.0])

elif data_source == "Vlastné CSV":
    uploaded = st.sidebar.file_uploader("Nahraj CSV", type=["csv", "txt"])
    has_header = st.sidebar.checkbox("Súbor má hlavičku", value=True)
    sep = st.sidebar.selectbox("Oddeľovač", [",", ";", "\\t", " "], index=0)
    if uploaded is not None:
        try:
            real_sep = "\t" if sep == "\\t" else sep
            df = pd.read_csv(uploaded, header=0 if has_header else None, sep=real_sep)
            df = df.select_dtypes(include=[np.number]).dropna()
            X_raw = df.values.astype(float)
        except Exception as e:  # noqa: BLE001
            st.sidebar.error(f"Chyba pri načítaní: {e}")
            st.stop()
    else:
        st.info("⬅️ Nahraj CSV (riadky = vzorky, stĺpce = znaky).")
        st.stop()

elif data_source == "Ručné zadanie":
    default = "1.0, 2.0\n2.0, 3.5\n3.0, 4.5\n4.5, 6.0\n5.0, 7.0\n6.0, 8.5\n7.5, 9.0"
    text = st.sidebar.text_area(
        "Dáta (čísla oddelené čiarkou, riadky = vzorky):", default, height=220
    )
    try:
        rows = [
            [float(v) for v in line.replace(";", ",").split(",")]
            for line in text.strip().splitlines()
            if line.strip()
        ]
        X_raw = np.array(rows, dtype=float)
    except Exception as e:  # noqa: BLE001
        st.sidebar.error(f"Chybný formát: {e}")
        st.stop()

if X_raw is None or X_raw.size == 0:
    st.warning("Žiadne dáta na výpočet.")
    st.stop()
if X_raw.ndim != 2 or X_raw.shape[0] < 2:
    st.error("PCA vyžaduje aspoň 2 vzorky a 2D maticu.")
    st.stop()

# Voliteľná štandardizácia
if standardize:
    sd = X_raw.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    X = (X_raw - X_raw.mean(axis=0)) / sd
else:
    X = X_raw.copy()

# ============================================================================
# VÝPOČET (delegovaný do pca_core)
# ============================================================================
pca = manual_pca(X)
feature_names = [f"x{i + 1}" for i in range(pca["n_features"])]
pc_names = [f"PC{i + 1}" for i in range(pca["n_features"])]

# Súhrnné metriky
m1, m2, m3, m4 = st.columns(4)
m1.metric("Vzorky n", pca["n_samples"])
m2.metric("Znaky d", pca["n_features"])
m3.metric("Σ vlast. čísel", f"{pca['eigenvalues'].sum():.4f}")
m4.metric(
    "Energia 2 hl. komp.",
    f"{pca['cumulative_energy'][min(1, pca['n_features'] - 1)] * 100:.2f} %",
)

# ============================================================================
# TABS
# ============================================================================
tab_data, tab_steps, tab_components, tab_reduction, tab_viz = st.tabs(
    ["📋 Dáta", "🧮 Medzivýpočty", "🎯 Komponenty", "📉 Redukcia dimenzie", "📈 Vizualizácia"]
)

# ----- Tab 1 — Dáta ---------------------------------------------------------
with tab_data:
    st.subheader("Vstupná matica X")
    st.write(
        f"Tvar: **{pca['n_samples']} × {pca['n_features']}**"
        + ("  (po štandardizácii z-score)" if standardize else "")
    )
    df_X = pd.DataFrame(pca["X"], columns=feature_names)
    df_X.index.name = "vzorka"
    st.dataframe(df_X.style.format("{:.4f}"), use_container_width=True, height=300)

    csv_buf = io.StringIO()
    df_X.to_csv(csv_buf)
    st.download_button(
        "⬇️ Stiahnuť spracované dáta (CSV)", csv_buf.getvalue(),
        file_name="vstup.csv", mime="text/csv",
    )

# ----- Tab 2 — Medzivýpočty -------------------------------------------------
with tab_steps:
    st.markdown(
        "Postup PCA krok za krokom — všetky výpočty sú robené ručne cez NumPy "
        "(matematické vzorce, žiadne sklearn)."
    )

    st.subheader("Krok 1 — Vektor stredných hodnôt μ")
    st.latex(r"\boldsymbol{\mu} = \frac{1}{n}\sum_{i=1}^{n}\mathbf{x}_i")
    st.dataframe(
        pd.DataFrame([pca["mean"]], columns=feature_names, index=["μ"]).style.format("{:.4f}"),
        use_container_width=True,
    )

    st.subheader("Krok 2 — Centrované dáta X̃ = X − μ")
    df_c = pd.DataFrame(pca["X_centered"], columns=feature_names)
    df_c.index.name = "vzorka"
    st.dataframe(df_c.style.format("{:.4f}"), use_container_width=True, height=260)

    st.subheader("Krok 3 — Kovariančná matica C")
    st.latex(r"\mathbf{C} = \frac{1}{n-1}\, \tilde{\mathbf{X}}^{\top} \tilde{\mathbf{X}}")
    df_cov = pd.DataFrame(pca["cov"], index=feature_names, columns=feature_names)
    st.dataframe(
        df_cov.style.format("{:.4f}").background_gradient(cmap="RdBu_r", axis=None),
        use_container_width=True,
    )

    st.subheader("Krok 4 — Spektrálny rozklad C")
    st.latex(
        r"\mathbf{C}\,\mathbf{v}_i = \lambda_i \mathbf{v}_i,"
        r"\qquad \lambda_1 \geq \lambda_2 \geq \dots \geq \lambda_d"
    )
    st.markdown("**Vlastné čísla λ a podiely rozptylu:**")
    df_eig = pd.DataFrame(
        {
            "vlastné číslo λ": pca["eigenvalues"],
            "podiel rozptylu": pca["explained_variance_ratio"],
            "kumulatívna energia": pca["cumulative_energy"],
        },
        index=pc_names,
    )
    st.dataframe(
        df_eig.style.format(
            {"vlastné číslo λ": "{:.6f}", "podiel rozptylu": "{:.2%}", "kumulatívna energia": "{:.2%}"}
        ),
        use_container_width=True,
    )

    st.markdown("**Vlastné vektory (stĺpce = transformačné vektory W):**")
    df_W = pd.DataFrame(pca["eigenvectors"], index=feature_names, columns=pc_names)
    st.dataframe(
        df_W.style.format("{:.4f}").background_gradient(cmap="RdBu_r", axis=None),
        use_container_width=True,
    )

    with st.expander("🔍 Overenie: WᵀW ≈ I (ortonormálnosť)"):
        ortho = pca["eigenvectors"].T @ pca["eigenvectors"]
        st.dataframe(
            pd.DataFrame(ortho, index=pc_names, columns=pc_names).style.format("{:.4f}"),
            use_container_width=True,
        )

# ----- Tab 3 — Komponenty ---------------------------------------------------
with tab_components:
    st.subheader("Variancie a energia")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(pc_names, pca["eigenvalues"], color="#4C72B0")
    axes[0].set_ylabel("vlastné číslo λ (rozptyl)")
    axes[0].set_title("Rozptyl pripadajúci na jednotlivé PC")
    axes[0].grid(True, alpha=0.3)
    for i, v in enumerate(pca["eigenvalues"]):
        axes[0].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    axes[1].plot(
        range(1, pca["n_features"] + 1), pca["cumulative_energy"] * 100,
        "o-", color="#C44E52", lw=2, ms=8,
    )
    axes[1].axhline(95, ls="--", color="gray", alpha=0.6, label="95 %")
    axes[1].axhline(99, ls=":",  color="gray", alpha=0.6, label="99 %")
    axes[1].set_xlabel("počet komponentov")
    axes[1].set_ylabel("kumulatívna energia [%]")
    axes[1].set_title("Kumulatívna energia (scree-like)")
    axes[1].set_xticks(range(1, pca["n_features"] + 1))
    axes[1].set_ylim(0, 105)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Transformovaná matica Y = X̃ · W")
    df_Y = pd.DataFrame(pca["Y"], columns=pc_names)
    df_Y.index.name = "vzorka"
    st.dataframe(df_Y.style.format("{:.4f}"), use_container_width=True, height=320)

    with st.expander("🔍 Overenie: kovariancia Y by mala byť diagonálna"):
        cov_Y = (pca["Y"].T @ pca["Y"]) / max(pca["n_samples"] - 1, 1)
        st.dataframe(
            pd.DataFrame(cov_Y, index=pc_names, columns=pc_names)
            .style.format("{:.4f}")
            .background_gradient(cmap="RdBu_r", axis=None),
            use_container_width=True,
        )
        st.caption("Diagonálne prvky = vlastné čísla λᵢ, mimo-diagonálne ≈ 0 (dekorelácia ✔).")

# ----- Tab 4 — Redukcia dimenzie --------------------------------------------
with tab_reduction:
    st.subheader("Redukcia dimenzie")
    max_k = pca["n_features"]
    mode = st.radio(
        "Spôsob výberu k:",
        ["Pevný počet komponentov", "Cieľová zachovaná energia"],
        horizontal=True,
    )

    if mode == "Pevný počet komponentov":
        k = st.slider("Počet zachovaných komponentov k", 1, max_k, min(2, max_k))
    else:
        target = st.slider("Cieľová zachovaná energia [%]", 50.0, 100.0, 95.0, 0.5)
        k = int(np.searchsorted(pca["cumulative_energy"] * 100, target) + 1)
        k = max(1, min(k, max_k))
        st.info(
            f"Pre energiu ≥ {target:.1f} % stačí **k = {k}** "
            f"(reálne: {pca['cumulative_energy'][k - 1] * 100:.2f} %)."
        )

    red = reduce_dimension(pca, k)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("k", k)
    c2.metric("Zachovaná energia", f"{red['energy_kept'] * 100:.2f} %")
    c3.metric("Stratená energia", f"{(1 - red['energy_kept']) * 100:.2f} %")
    c4.metric("Σ rekonštrukčná chyba", f"{red['reconstruction_error']:.4f}")

    st.markdown(f"**Transformačná matica $W_k$ ({pca['n_features']} × {k}):**")
    st.dataframe(
        pd.DataFrame(red["W_k"], index=feature_names, columns=[f"PC{i + 1}" for i in range(k)])
        .style.format("{:.4f}")
        .background_gradient(cmap="RdBu_r", axis=None),
        use_container_width=True,
    )

    st.markdown(f"**Redukované dáta $Y_k = \\tilde X \\cdot W_k$ ({pca['n_samples']} × {k}):**")
    df_Yk = pd.DataFrame(red["Y_k"], columns=[f"PC{i + 1}" for i in range(k)])
    df_Yk.index.name = "vzorka"
    st.dataframe(df_Yk.style.format("{:.4f}"), use_container_width=True, height=260)

    st.markdown("**Spätná rekonštrukcia v pôvodnom priestore:**")
    st.latex(r"\hat{\mathbf{X}} = Y_k\, W_k^{\top} + \boldsymbol{\mu}")
    df_Xrec = pd.DataFrame(red["X_reconstructed"], columns=feature_names)
    df_Xrec.index.name = "vzorka"
    st.dataframe(df_Xrec.style.format("{:.4f}"), use_container_width=True, height=260)

    out = io.StringIO()
    df_Yk.to_csv(out)
    st.download_button(
        "⬇️ Stiahnuť redukované dáta (CSV)", out.getvalue(),
        file_name=f"pca_redukcia_k{k}.csv", mime="text/csv",
    )

# ----- Tab 5 — Vizualizácia -------------------------------------------------
with tab_viz:
    st.subheader("Vizualizácia")

    if pca["n_features"] == 2:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        scale = 2.5

        # (a) Pôvodné dáta + PCA osi
        ax = axes[0]
        ax.scatter(pca["X"][:, 0], pca["X"][:, 1], alpha=0.6, s=30,
                   color="#4C72B0", label="dáta", edgecolor="white", lw=0.5)
        mu = pca["mean"]
        for i, color in enumerate(["#C44E52", "#55A868"]):
            v = pca["eigenvectors"][:, i]
            length = np.sqrt(max(pca["eigenvalues"][i], 1e-12)) * scale
            ax.annotate("", xy=mu + v * length, xytext=mu,
                        arrowprops=dict(arrowstyle="->", color=color, lw=2.5))
            ax.text(*(mu + v * length * 1.1), f"PC{i + 1}",
                    color=color, fontsize=12, fontweight="bold")
        ax.scatter(*mu, color="black", s=80, marker="x", label="μ")
        ax.set_title("(a) Pôvodné dáta + PCA osi")
        ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
        ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
        ax.set_aspect("equal", "datalim")
        ax.grid(True, alpha=0.3); ax.legend(loc="best")

        # (b) Centrované dáta + osi cez 0
        ax = axes[1]
        ax.scatter(pca["X_centered"][:, 0], pca["X_centered"][:, 1],
                   alpha=0.6, s=30, color="#4C72B0", edgecolor="white", lw=0.5)
        for i, color in enumerate(["#C44E52", "#55A868"]):
            v = pca["eigenvectors"][:, i]
            length = np.sqrt(max(pca["eigenvalues"][i], 1e-12)) * scale
            ax.annotate("", xy=v * length, xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color=color, lw=2.5))
            ax.text(*(v * length * 1.1), f"PC{i + 1}",
                    color=color, fontsize=12, fontweight="bold")
        ax.set_title("(b) Centrované dáta")
        ax.set_xlabel("x₁ − μ₁"); ax.set_ylabel("x₂ − μ₂")
        ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
        ax.set_aspect("equal", "datalim"); ax.grid(True, alpha=0.3)

        # (c) Po PCA transformácii
        ax = axes[2]
        ax.scatter(pca["Y"][:, 0], pca["Y"][:, 1], alpha=0.6, s=30,
                   color="#8172B2", edgecolor="white", lw=0.5)
        ax.axhline(0, color="#55A868", lw=2, alpha=0.7, label="PC2 = 0")
        ax.axvline(0, color="#C44E52", lw=2, alpha=0.7, label="PC1 = 0")
        ax.set_title("(c) Po PCA transformácii")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_aspect("equal", "datalim")
        ax.legend(loc="best"); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        if k == 1:
            st.markdown("##### Pri k = 1 — projekcia dát na priamku PC1")
            fig2, ax = plt.subplots(figsize=(11, 2.6))
            ax.scatter(red["Y_k"][:, 0], np.zeros_like(red["Y_k"][:, 0]),
                       alpha=0.6, s=40, color="#8172B2", edgecolor="white", lw=0.5)
            ax.axhline(0, color="gray", lw=0.6)
            ax.set_yticks([]); ax.set_xlabel("PC1")
            ax.set_title("1D projekcia (k = 1)"); ax.grid(True, alpha=0.3)
            plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)

            st.markdown("##### Rekonštrukcia v pôvodnom priestore")
            fig3, ax = plt.subplots(figsize=(7, 7))
            ax.scatter(pca["X"][:, 0], pca["X"][:, 1],
                       alpha=0.4, s=30, color="#4C72B0", label="pôvodné")
            ax.scatter(red["X_reconstructed"][:, 0], red["X_reconstructed"][:, 1],
                       alpha=0.8, s=30, color="#C44E52", label="rekonštrukcia")
            for i in range(pca["n_samples"]):
                ax.plot(
                    [pca["X"][i, 0], red["X_reconstructed"][i, 0]],
                    [pca["X"][i, 1], red["X_reconstructed"][i, 1]],
                    color="gray", lw=0.5, alpha=0.4,
                )
            ax.set_aspect("equal", "datalim")
            ax.set_title(
                f"Rekonštrukcia z k = {k} komponentov "
                f"(zachovaná energia {red['energy_kept'] * 100:.1f} %)"
            )
            ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
            ax.legend(); ax.grid(True, alpha=0.3)
            plt.tight_layout(); st.pyplot(fig3); plt.close(fig3)

    elif pca["n_features"] == 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(15, 6))
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.scatter(pca["X"][:, 0], pca["X"][:, 1], pca["X"][:, 2],
                    alpha=0.5, s=18, color="#4C72B0")
        mu = pca["mean"]
        scale = 2.5
        for i, color in enumerate(["#C44E52", "#55A868", "#8172B2"]):
            v = pca["eigenvectors"][:, i]
            length = np.sqrt(max(pca["eigenvalues"][i], 1e-12)) * scale
            end = mu + v * length
            ax1.plot([mu[0], end[0]], [mu[1], end[1]], [mu[2], end[2]], color=color, lw=2.5)
            ax1.text(end[0], end[1], end[2], f"PC{i + 1}", color=color, fontweight="bold")
        ax1.set_title("Pôvodné 3D dáta + PCA osi")
        ax1.set_xlabel("x₁"); ax1.set_ylabel("x₂"); ax1.set_zlabel("x₃")

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.scatter(pca["Y"][:, 0], pca["Y"][:, 1],
                    alpha=0.6, s=22, color="#8172B2", edgecolor="white", lw=0.5)
        ax2.axhline(0, color="gray", lw=0.5); ax2.axvline(0, color="gray", lw=0.5)
        ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")
        ax2.set_title("Projekcia do roviny PC1–PC2")
        ax2.grid(True, alpha=0.3); ax2.set_aspect("equal", "datalim")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    else:
        st.info(
            f"Vstup má {pca['n_features']} dimenzií — zobrazujem 2D projekciu na PC1×PC2 "
            "a heatmapu prvých vlastných vektorov."
        )
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].scatter(pca["Y"][:, 0], pca["Y"][:, 1],
                        alpha=0.6, s=22, color="#8172B2", edgecolor="white", lw=0.5)
        axes[0].axhline(0, color="gray", lw=0.5); axes[0].axvline(0, color="gray", lw=0.5)
        axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
        axes[0].set_title("Projekcia na prvé dva PC")
        axes[0].grid(True, alpha=0.3); axes[0].set_aspect("equal", "datalim")

        k_show = min(pca["n_features"], 6)
        im = axes[1].imshow(pca["eigenvectors"][:, :k_show], cmap="RdBu_r",
                            aspect="auto", vmin=-1, vmax=1)
        axes[1].set_xticks(range(k_show))
        axes[1].set_xticklabels([f"PC{i + 1}" for i in range(k_show)])
        axes[1].set_yticks(range(pca["n_features"]))
        axes[1].set_yticklabels(feature_names)
        axes[1].set_title("Loadings (vlastné vektory)")
        plt.colorbar(im, ax=axes[1])
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

st.divider()
st.caption(
    "PCA je tu vypočítaná **úplne ručne** cez NumPy: priemer → centrovanie → "
    "kovariančná matica → vlastné čísla/vektory (eigh) → zoradenie → "
    "transformácia → redukcia + rekonštrukcia. Žiadne sklearn."
)
