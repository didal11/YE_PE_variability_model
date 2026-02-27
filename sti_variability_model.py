# sti_variability_model.py
# Python 3.12+

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

# ---- Tier1 latent process drivers z ----
Z_KEYS: Tuple[str, ...] = (
    "L_GATE_ELEC",      # Leff proxy
    "EOT",
    "CH_DOSE",          # channel implant dose
    "CH_RP",            # channel implant projected range
    "ACT_EFF",          # activation efficiency
    "SD_LDIFF",         # S/D lateral diffusion length
    "RSD_BASE",         # baseline Rsd process component
    "CH_STRESS",        # channel stress (longitudinal)
    "R_CONTACT_BASE",   # contact resistance process component
    "R_VIA_BASE",       # via resistance process component
    "RHO_METAL",        # metal resistivity
    "R_LINE_PER_L",     # line resistance per unit length
    "C_LINE_PER_L",     # line capacitance per unit length
    "RDF_DENSITY",      # random dopant fluctuation density
    "LER_AMP",          # line edge roughness amplitude
)

# ---- Intermediate process state x ----
X_KEYS: Tuple[str, ...] = (
    "LEFF",
    "EOT",
    "NCH_EFF",          # effective channel doping proxy
    "XJ_EFF",           # effective junction depth proxy
    "ACT_EFF",
    "LDIFF",
    "RSD",
    "STRESS",
    "RCONTACT",
    "RVIA",
    "RHO_METAL",
    "RWIRE_PER_L",
    "CWIRE_PER_L",
    "RDF",
    "LER",
)

# ---- Injection parameters p (device + interconnect) ----
P_KEYS: Tuple[str, ...] = (
    "DVTH_N", "DVTH_P",
    "DBETA_N", "DBETA_P",
    "DDIBL_N", "DDIBL_P",
    "DSS_N", "DSS_P",
    "DRSD_N", "DRSD_P",
    "DCJD_N", "DCJD_P",
    "DCJS_N", "DCJS_P",
    "DLNIREVD_N", "DLNIREVD_P",
    "DLNIREVS_N", "DLNIREVS_P",
    "DRCONTACT",
    "DRWIRE", "DCWIRE",
)


@dataclass(frozen=True)
class GaussianSpec:
    mu: np.ndarray
    Sigma: np.ndarray


def _validate_gaussian(spec: GaussianSpec, keys: Tuple[str, ...], name: str) -> None:
    n = len(keys)
    if spec.mu.shape != (n,):
        raise ValueError(f"{name}.mu shape must be ({n},)")
    if spec.Sigma.shape != (n, n):
        raise ValueError(f"{name}.Sigma shape must be ({n},{n})")
    if not np.allclose(spec.Sigma, spec.Sigma.T, atol=1e-12):
        raise ValueError(f"{name}.Sigma must be symmetric")
    eig = np.linalg.eigvalsh(spec.Sigma)
    if np.min(eig) < -1e-9:
        raise ValueError(f"{name}.Sigma must be PSD. min eigen={np.min(eig)}")


def make_Sigma_from_sigma_and_corr(sigma: np.ndarray, corr: np.ndarray) -> np.ndarray:
    D = np.diag(sigma)
    return D @ corr @ D


def covariance_to_sigma_and_corr(Sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sigma = np.sqrt(np.clip(np.diag(Sigma), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = Sigma / np.outer(sigma, sigma)
    corr[np.isnan(corr)] = 0.0
    np.fill_diagonal(corr, 1.0)
    return sigma, corr


def propagate_linear(spec_in: GaussianSpec, A: np.ndarray, b: np.ndarray, in_keys: Tuple[str, ...], out_keys: Tuple[str, ...]) -> Tuple[np.ndarray, np.ndarray]:
    _validate_gaussian(spec_in, in_keys, "spec_in")
    if A.shape != (len(out_keys), len(in_keys)):
        raise ValueError(f"A shape must be ({len(out_keys)},{len(in_keys)})")
    if b.shape != (len(out_keys),):
        raise ValueError(f"b shape must be ({len(out_keys)},)")
    mu_out = A @ spec_in.mu + b
    Sigma_out = A @ spec_in.Sigma @ A.T
    return mu_out, Sigma_out


def default_z_corr() -> np.ndarray:
    corr = np.eye(len(Z_KEYS), dtype=float)
    zi = {k: i for i, k in enumerate(Z_KEYS)}

    pairs = {
        ("L_GATE_ELEC", "LER_AMP"): 0.55,
        ("CH_DOSE", "CH_RP"): -0.35,
        ("CH_DOSE", "ACT_EFF"): 0.30,
        ("CH_RP", "SD_LDIFF"): 0.40,
        ("ACT_EFF", "SD_LDIFF"): 0.45,
        ("RSD_BASE", "R_CONTACT_BASE"): 0.40,
        ("R_CONTACT_BASE", "R_VIA_BASE"): 0.35,
        ("RHO_METAL", "R_LINE_PER_L"): 0.75,
        ("R_LINE_PER_L", "C_LINE_PER_L"): 0.25,
        ("RDF_DENSITY", "LER_AMP"): 0.20,
    }
    for (a, b), v in pairs.items():
        i, j = zi[a], zi[b]
        corr[i, j] = v
        corr[j, i] = v
    return corr


def default_B_z_to_x() -> np.ndarray:
    B = np.zeros((len(X_KEYS), len(Z_KEYS)), dtype=float)
    xi = {k: i for i, k in enumerate(X_KEYS)}
    zi = {k: i for i, k in enumerate(Z_KEYS)}

    # primary couplings
    B[xi["LEFF"], zi["L_GATE_ELEC"]] = 1.0
    B[xi["LEFF"], zi["LER_AMP"]] = -0.35

    B[xi["EOT"], zi["EOT"]] = 1.0

    B[xi["NCH_EFF"], zi["CH_DOSE"]] = 0.85
    B[xi["NCH_EFF"], zi["CH_RP"]] = -0.25
    B[xi["NCH_EFF"], zi["ACT_EFF"]] = 0.30

    B[xi["XJ_EFF"], zi["CH_RP"]] = 0.75
    B[xi["XJ_EFF"], zi["SD_LDIFF"]] = 0.40

    B[xi["ACT_EFF"], zi["ACT_EFF"]] = 1.0

    B[xi["LDIFF"], zi["SD_LDIFF"]] = 0.90

    B[xi["RSD"], zi["RSD_BASE"]] = 0.85
    B[xi["RSD"], zi["ACT_EFF"]] = -0.30
    B[xi["RSD"], zi["SD_LDIFF"]] = -0.20

    B[xi["STRESS"], zi["CH_STRESS"]] = 1.0

    B[xi["RCONTACT"], zi["R_CONTACT_BASE"]] = 0.95
    B[xi["RCONTACT"], zi["R_VIA_BASE"]] = 0.20

    B[xi["RVIA"], zi["R_VIA_BASE"]] = 0.95

    B[xi["RHO_METAL"], zi["RHO_METAL"]] = 1.0
    B[xi["RWIRE_PER_L"], zi["R_LINE_PER_L"]] = 0.90
    B[xi["RWIRE_PER_L"], zi["RHO_METAL"]] = 0.35

    B[xi["CWIRE_PER_L"], zi["C_LINE_PER_L"]] = 0.95
    B[xi["CWIRE_PER_L"], zi["L_GATE_ELEC"]] = -0.10

    B[xi["RDF"], zi["RDF_DENSITY"]] = 1.0
    B[xi["LER"], zi["LER_AMP"]] = 1.0

    return B


def default_A_x_to_p() -> np.ndarray:
    A = np.zeros((len(P_KEYS), len(X_KEYS)), dtype=float)
    xi = {k: i for i, k in enumerate(X_KEYS)}
    pi = {k: i for i, k in enumerate(P_KEYS)}

    # Vth
    A[pi["DVTH_N"], xi["LEFF"]] = -0.9e-3
    A[pi["DVTH_N"], xi["EOT"]] = +1.2e-3
    A[pi["DVTH_N"], xi["NCH_EFF"]] = +0.8e-3
    A[pi["DVTH_N"], xi["RDF"]] = +0.5e-3

    A[pi["DVTH_P"], xi["LEFF"]] = +0.8e-3
    A[pi["DVTH_P"], xi["EOT"]] = -1.1e-3
    A[pi["DVTH_P"], xi["NCH_EFF"]] = -0.7e-3
    A[pi["DVTH_P"], xi["RDF"]] = -0.4e-3

    # beta / mobility
    A[pi["DBETA_N"], xi["STRESS"]] = +6.0e-3
    A[pi["DBETA_N"], xi["LEFF"]] = -2.0e-3
    A[pi["DBETA_N"], xi["RSD"]] = -4.0e-3

    A[pi["DBETA_P"], xi["STRESS"]] = -4.5e-3
    A[pi["DBETA_P"], xi["LEFF"]] = +1.5e-3
    A[pi["DBETA_P"], xi["RSD"]] = -3.5e-3

    # DIBL / SS
    A[pi["DDIBL_N"], xi["LEFF"]] = -2.5e-3
    A[pi["DDIBL_N"], xi["XJ_EFF"]] = +1.6e-3
    A[pi["DDIBL_N"], xi["EOT"]] = +0.7e-3

    A[pi["DDIBL_P"], xi["LEFF"]] = +2.3e-3
    A[pi["DDIBL_P"], xi["XJ_EFF"]] = -1.4e-3
    A[pi["DDIBL_P"], xi["EOT"]] = -0.6e-3

    A[pi["DSS_N"], xi["EOT"]] = +1.3e-3
    A[pi["DSS_N"], xi["RDF"]] = +0.7e-3
    A[pi["DSS_N"], xi["LEFF"]] = -0.9e-3

    A[pi["DSS_P"], xi["EOT"]] = -1.1e-3
    A[pi["DSS_P"], xi["RDF"]] = -0.6e-3
    A[pi["DSS_P"], xi["LEFF"]] = +0.8e-3

    # Rsd
    A[pi["DRSD_N"], xi["RSD"]] = +1.0
    A[pi["DRSD_N"], xi["RCONTACT"]] = +0.20
    A[pi["DRSD_N"], xi["ACT_EFF"]] = -0.10

    A[pi["DRSD_P"], xi["RSD"]] = +0.95
    A[pi["DRSD_P"], xi["RCONTACT"]] = +0.25
    A[pi["DRSD_P"], xi["ACT_EFF"]] = -0.08

    # Cj
    A[pi["DCJD_N"], xi["XJ_EFF"]] = +2.0e-3
    A[pi["DCJD_N"], xi["EOT"]] = +0.5e-3
    A[pi["DCJD_P"], xi["XJ_EFF"]] = +1.8e-3
    A[pi["DCJD_P"], xi["EOT"]] = +0.4e-3

    A[pi["DCJS_N"], xi["XJ_EFF"]] = +1.9e-3
    A[pi["DCJS_N"], xi["LEFF"]] = -0.3e-3
    A[pi["DCJS_P"], xi["XJ_EFF"]] = +1.7e-3
    A[pi["DCJS_P"], xi["LEFF"]] = +0.3e-3

    # reverse leakage logs
    A[pi["DLNIREVD_N"], xi["XJ_EFF"]] = +8.0e-3
    A[pi["DLNIREVD_N"], xi["EOT"]] = +2.5e-3
    A[pi["DLNIREVD_N"], xi["RDF"]] = +2.0e-3

    A[pi["DLNIREVD_P"], xi["XJ_EFF"]] = +7.0e-3
    A[pi["DLNIREVD_P"], xi["EOT"]] = +2.0e-3
    A[pi["DLNIREVD_P"], xi["RDF"]] = +1.7e-3

    A[pi["DLNIREVS_N"], xi["XJ_EFF"]] = +7.5e-3
    A[pi["DLNIREVS_N"], xi["EOT"]] = +2.2e-3

    A[pi["DLNIREVS_P"], xi["XJ_EFF"]] = +6.8e-3
    A[pi["DLNIREVS_P"], xi["EOT"]] = +1.9e-3

    # BEOL/contact
    A[pi["DRCONTACT"], xi["RCONTACT"]] = +1.0
    A[pi["DRCONTACT"], xi["RVIA"]] = +0.25

    A[pi["DRWIRE"], xi["RWIRE_PER_L"]] = +1.0
    A[pi["DRWIRE"], xi["RHO_METAL"]] = +0.35

    A[pi["DCWIRE"], xi["CWIRE_PER_L"]] = +1.0
    A[pi["DCWIRE"], xi["LEFF"]] = -0.08

    return A


def default_mu_z() -> np.ndarray:
    return np.zeros((len(Z_KEYS),), dtype=float)


def default_sigma_z() -> np.ndarray:
    return np.array([
        1.0, 0.8, 1.0, 0.9, 0.8, 0.9, 0.7, 0.8, 0.7, 0.7, 0.6, 0.7, 0.7, 1.0, 1.0
    ], dtype=float)


def default_mu_eps_x() -> np.ndarray:
    return np.zeros((len(X_KEYS),), dtype=float)


def default_sigma_eps_x() -> np.ndarray:
    return np.array([
        0.20, 0.15, 0.25, 0.20, 0.15, 0.20, 0.20, 0.15, 0.15, 0.15, 0.10, 0.12, 0.12, 0.20, 0.20
    ], dtype=float)


def default_corr_eps_x() -> np.ndarray:
    corr = np.eye(len(X_KEYS), dtype=float)
    xi = {k: i for i, k in enumerate(X_KEYS)}
    pairs = {
        ("LEFF", "LER"): 0.25,
        ("RSD", "RCONTACT"): 0.20,
        ("RCONTACT", "RVIA"): 0.20,
        ("RWIRE_PER_L", "CWIRE_PER_L"): 0.10,
    }
    for (a, b), v in pairs.items():
        i, j = xi[a], xi[b]
        corr[i, j] = v
        corr[j, i] = v
    return corr


def build_x_from_z(spec_z: GaussianSpec, B: np.ndarray, spec_eps_x: GaussianSpec) -> Tuple[np.ndarray, np.ndarray]:
    _validate_gaussian(spec_z, Z_KEYS, "spec_z")
    _validate_gaussian(spec_eps_x, X_KEYS, "spec_eps_x")
    if B.shape != (len(X_KEYS), len(Z_KEYS)):
        raise ValueError(f"B shape must be ({len(X_KEYS)},{len(Z_KEYS)})")

    mu_x = B @ spec_z.mu + spec_eps_x.mu
    Sigma_x = B @ spec_z.Sigma @ B.T + spec_eps_x.Sigma
    return mu_x, Sigma_x


def write_spice_param_include(filepath: str, p_vec: np.ndarray, header: str = "generated") -> None:
    if p_vec.shape != (len(P_KEYS),):
        raise ValueError(f"p_vec must be shape ({len(P_KEYS)},)")
    parts = [f"{k}={float(v):.12g}" for k, v in zip(P_KEYS, p_vec)]
    content = f"* {header}\n.param " + " ".join(parts) + "\n"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


def sample_mvnormal(mu: np.ndarray, Sigma: np.ndarray, n: int, seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean=mu, cov=Sigma, size=n, method="svd")
