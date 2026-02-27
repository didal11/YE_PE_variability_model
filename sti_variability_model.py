# sti_variability_model.py
# Python 3.12+

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

# ---- STI 공정 변수(6개) ----
STI_X_KEYS: Tuple[str, ...] = (
    "D_STI",     # trench depth [nm]
    "W_STI",     # STI width [nm]
    "R_CORNER",  # corner radius [nm]
    "SIG_CH",    # channel stress proxy [MPa] (or arbitrary unit)
    "T_LINER",   # liner oxide thickness [nm]
    "DH_STI",    # STI height variation [nm]
)

# ---- SPICE 주입 파라미터(예: 6개) ----
P_KEYS: Tuple[str, ...] = (
    "DVTH",      # [V]
    "DMU",       # [-]  (mu_eff = mu0*(1+DMU))
    "DCJD",      # [F]
    "DCJS",      # [F]
    "DLNIRD",    # [-]  (Irev_D = Irev0 * exp(DLNIRD))
    "DLNIRS",    # [-]
)

@dataclass(frozen=True)
class GaussianSpec:
    mu: np.ndarray      # (6,)
    Sigma: np.ndarray   # (6,6)

    def validate(self) -> None:
        n = len(STI_X_KEYS)
        if self.mu.shape != (n,):
            raise ValueError(f"mu shape must be ({n},)")
        if self.Sigma.shape != (n, n):
            raise ValueError(f"Sigma shape must be ({n},{n})")
        if not np.allclose(self.Sigma, self.Sigma.T, atol=1e-12):
            raise ValueError("Sigma must be symmetric")
        eig = np.linalg.eigvalsh(self.Sigma)
        if np.min(eig) < -1e-9:
            raise ValueError(f"Sigma must be PSD. min eigen={np.min(eig)}")

@dataclass(frozen=True)
class LinearMap:
    # p = A (x - mu_x) + b
    A: np.ndarray  # (k,6)
    b: np.ndarray  # (k,)

    def validate(self) -> None:
        k = len(P_KEYS)
        n = len(STI_X_KEYS)
        if self.A.shape != (k, n):
            raise ValueError(f"A shape must be ({k},{n})")
        if self.b.shape != (k,):
            raise ValueError(f"b shape must be ({k},)")

def make_Sigma_from_sigma_and_corr(sigma: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """Sigma = diag(sigma) * corr * diag(sigma)"""
    D = np.diag(sigma)
    return D @ corr @ D

def covariance_to_sigma_and_corr(Sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sigma = np.sqrt(np.clip(np.diag(Sigma), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = Sigma / np.outer(sigma, sigma)
    corr[np.isnan(corr)] = 0.0
    np.fill_diagonal(corr, 1.0)
    return sigma, corr

def propagate_covariance(spec_x: GaussianSpec, map_px: LinearMap) -> Tuple[np.ndarray, np.ndarray]:
    """mu_p, Sigma_p"""
    spec_x.validate()
    map_px.validate()
    mu_p = map_px.b.copy()
    Sigma_p = map_px.A @ spec_x.Sigma @ map_px.A.T
    return mu_p, Sigma_p

def sample_mvnormal(mu: np.ndarray, Sigma: np.ndarray, n: int, seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean=mu, cov=Sigma, size=n, method="svd")

def write_spice_param_include(filepath: str, p_vec: np.ndarray, header: str = "generated") -> None:
    """Create .inc with .param DVTH=... DMU=..."""
    if p_vec.shape != (len(P_KEYS),):
        raise ValueError(f"p_vec must be shape ({len(P_KEYS)},)")
    parts = [f"{k}={float(v):.12g}" for k, v in zip(P_KEYS, p_vec)]
    content = f"* {header}\n.param " + " ".join(parts) + "\n"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

def default_linear_map() -> LinearMap:
    """빈 매핑(전부 0)"""
    A = np.zeros((len(P_KEYS), len(STI_X_KEYS)), dtype=float)
    b = np.zeros((len(P_KEYS),), dtype=float)
    return LinearMap(A=A, b=b)