"""Pure numpy/scipy/sklearn geometry used by analyze.py. No torch, no I/O.

Conventions: feature matrices are [n, D] (rows = inputs). Subspace bases are
[D, k] with orthonormal columns. Displacements are pairwise: D = H_corr - H_clean
for the same input indices.
"""
import numpy as np
from scipy.linalg import subspace_angles


def nonconstant_mask(X):
    """True for feature dims with non-zero variance (e.g. SNN channels that fire)."""
    return X.var(axis=0) > 0


def sorted_eig(G):
    """Eigen-decomposition of a symmetric PSD matrix, descending, negatives clipped."""
    w, V = np.linalg.eigh(G)
    order = np.argsort(w)[::-1]
    return np.clip(w[order], 0, None), V[:, order]


def displacement_spectrum(D, center=False):
    """Eigenvalues/vectors of D^T D (uncentered) or of the centered scatter.

    The right singular vectors of D are the eigenvectors of D^T D, and the
    eigenvalues are the squared singular values = energy per direction.
    """
    D = np.asarray(D, dtype=np.float64)
    if center:
        D = D - D.mean(axis=0, keepdims=True)
    return sorted_eig(D.T @ D)


def k_for_energy(evals, threshold=0.9):
    """Smallest k such that the top-k eigenvalues hold >= threshold of the total."""
    total = evals.sum()
    if total <= 0:
        return np.nan
    cum = np.cumsum(evals) / total
    return int(np.argmax(cum >= threshold - 1e-12) + 1)


def participation_ratio(evals):
    s2 = np.square(evals).sum()
    return float(evals.sum() ** 2 / s2) if s2 > 0 else np.nan


def captured_energy(D, U):
    """Fraction of the total (pooled) energy of D lying in span(U)."""
    D = np.asarray(D, dtype=np.float64)
    tot = np.square(D).sum()
    if tot <= 0:
        return np.nan
    return float(np.square(D @ U).sum() / tot)


def random_subspace(dim, k, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((dim, k)))
    return Q


def orthonormalize(A):
    Q, _ = np.linalg.qr(np.asarray(A, dtype=np.float64))
    return Q


def principal_angles_deg(A, B):
    """Principal angles between span(A) and span(B), ascending, in degrees."""
    return np.sort(np.degrees(subspace_angles(A, B)))


def energy_fraction(D, Q, min_norm=1e-12):
    """Mean over rows of ||P_Q d||^2 / ||d||^2 (rows with ~zero norm excluded).

    Returns (mean_fraction, n_used).
    """
    D = np.asarray(D, dtype=np.float64)
    n2 = np.square(D).sum(axis=1)
    keep = n2 > min_norm ** 2
    if not keep.any():
        return np.nan, 0
    frac = np.square(D[keep] @ Q).sum(axis=1) / n2[keep]
    return float(frac.mean()), int(keep.sum())


def relative_magnitude(H_clean, H_corr, min_norm=1e-8):
    """Mean of ||h_corr - h_clean|| / ||h_clean||. Returns (mean, n_used)."""
    D = np.asarray(H_corr, dtype=np.float64) - np.asarray(H_clean, dtype=np.float64)
    hn = np.linalg.norm(np.asarray(H_clean, dtype=np.float64), axis=1)
    keep = hn > min_norm
    if not keep.any():
        return np.nan, 0
    return float((np.linalg.norm(D[keep], axis=1) / hn[keep]).mean()), int(keep.sum())


def lda_subspace(X, y, n_components=9):
    """Orthonormal basis of the shrinkage-LDA discriminant directions.

    The columns of scalings_ are the directions w whose projections w.h the LDA
    reads out; they are not orthogonal in feature space, so we orthonormalise
    their span. Returns (W [D, n_components], fitted lda).
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
    lda.fit(np.asarray(X, dtype=np.float64), y)
    return orthonormalize(lda.scalings_[:, :n_components]), lda


def project_out(H, m, U):
    """h' = m + (I - U U^T)(h - m), applied row-wise."""
    H = np.asarray(H, dtype=np.float64)
    C = H - m
    return m + C - (C @ U) @ U.T


def clean_pca_basis(H_clean):
    """Principal directions of clean feature variation (centered), descending."""
    _, V = displacement_spectrum(H_clean, center=True)
    return V
