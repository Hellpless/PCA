"""
pca_core.py — Manuálny výpočet PCA (žiadne sklearn, žiadne UI)
===============================================================
Použitie samostatne:
    from pca_core import manual_pca, reduce_dimension

    result = manual_pca(X)          # X: np.ndarray (n_samples, n_features)
    red    = reduce_dimension(result, k=2)
"""
from __future__ import annotations

import numpy as np


def manual_pca(X: np.ndarray) -> dict:
    """
    Krok-za-krokom PCA bez sklearn.

    Kroky
    -----
    1. Vektor stredných hodnôt μ
    2. Centrovanie  X̃ = X − μ
    3. Kovariančná matica  C = (1/(n−1)) X̃ᵀ X̃
    4. Spektrálny rozklad  C v = λ v  (numpy.linalg.eigh)
    5. Zoradenie vlastných čísel zostupne
    6. Zjednotenie znamienka (najväčšia abs. zložka každého vektora kladná)
    7. Variančný podiel a kumulatívna energia
    8. Transformácia  Y = X̃ · W

    Parametre
    ---------
    X : np.ndarray, tvar (n_samples, n_features)

    Výstup (slovník)
    ----------------
    X               vstupná matica (skopírovaná ako float64)
    n_samples       počet vzoriek n
    n_features      počet znakov d
    mean            vektor μ, tvar (d,)
    X_centered      centrované dáta X̃, tvar (n, d)
    cov             kovariančná matica C, tvar (d, d)
    eigenvalues     vlastné čísla λ zostupne, tvar (d,)
    eigenvectors    vlastné vektory W ako stĺpce, tvar (d, d)
    explained_variance_ratio  λᵢ / Σλ, tvar (d,)
    cumulative_energy         kumulatívny súčet podielov, tvar (d,)
    Y               transformované dáta Y = X̃·W, tvar (n, d)
    """
    X = np.asarray(X, dtype=float)
    n_samples, n_features = X.shape

    # --- Krok 1 ---
    mean = X.mean(axis=0)

    # --- Krok 2 ---
    X_centered = X - mean

    # --- Krok 3 ---
    cov = (X_centered.T @ X_centered) / max(n_samples - 1, 1)

    # --- Krok 4 ---
    # eigh je numericky stabilné pre symetrické matice; vracia reálne hodnoty
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # --- Krok 5 ---
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # --- Krok 6 ---
    # Zabezpečíme deterministické znamienko: najväčšia absolútna zložka kladná
    for i in range(eigenvectors.shape[1]):
        j = np.argmax(np.abs(eigenvectors[:, i]))
        if eigenvectors[j, i] < 0:
            eigenvectors[:, i] *= -1

    # --- Krok 7 ---
    total_var = eigenvalues.sum()
    if total_var > 0:
        explained_variance_ratio = eigenvalues / total_var
    else:
        explained_variance_ratio = np.zeros_like(eigenvalues)
    cumulative_energy = np.cumsum(explained_variance_ratio)

    # --- Krok 8 ---
    Y = X_centered @ eigenvectors

    return {
        "X": X,
        "n_samples": n_samples,
        "n_features": n_features,
        "mean": mean,
        "X_centered": X_centered,
        "cov": cov,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_energy": cumulative_energy,
        "Y": Y,
    }


def reduce_dimension(pca: dict, k: int) -> dict:
    """
    Redukcia dimenzie na k hlavných komponentov + spätná rekonštrukcia.

    Parametre
    ---------
    pca : výstup funkcie manual_pca()
    k   : počet zachovaných komponentov (1 ≤ k ≤ n_features)

    Výstup (slovník)
    ----------------
    k                            zvolený počet komponentov
    W_k                          transformačná matica, tvar (d, k)
    Y_k                          redukované dáta, tvar (n, k)
    X_reconstructed              rekonštrukcia v pôvodnom priestore, tvar (n, d)
    reconstruction_error         Σ||xᵢ − x̂ᵢ||²  (skalár)
    reconstruction_error_per_sample  priemer na vzorku
    energy_kept                  kumulatívna energia pre k komponentov
    """
    if not (1 <= k <= pca["n_features"]):
        raise ValueError(f"k musí byť v rozsahu 1 … {pca['n_features']}, dostali sme {k}")

    W_k = pca["eigenvectors"][:, :k]
    Y_k = pca["X_centered"] @ W_k
    X_reconstructed = Y_k @ W_k.T + pca["mean"]

    err = float(np.sum((pca["X"] - X_reconstructed) ** 2))
    energy_kept = float(pca["cumulative_energy"][k - 1])

    return {
        "k": k,
        "W_k": W_k,
        "Y_k": Y_k,
        "X_reconstructed": X_reconstructed,
        "reconstruction_error": err,
        "reconstruction_error_per_sample": err / pca["n_samples"],
        "energy_kept": energy_kept,
    }
