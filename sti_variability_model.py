# sti_variability_model.py
# Python 3.12+

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

# ---- Tier1 latent process drivers z (fixed 16, as requested) ----
Z_KEYS: Tuple[str, ...] = (
    "CD_G",          # global CD bias
    "OVL_X",         # overlay X
    "OVL_Y",         # overlay Y
    "LER",           # line-edge roughness amplitude
    "ETCH_RATE",     # etch rate / over-etch depth tendency
    "ETCH_PROFILE",  # sidewall angle / corner profile tendency
    "EOT",           # gate dielectric EOT
    "ILD_THK",       # non-gate dielectric thickness
    "DOSE",          # implant dose
    "RP",            # projected range (implant depth)
    "THERM",         # thermal budget / diffusion
    "CMP",           # CMP dishing / erosion tendency
    "STRESS",        # mechanical stress
    "DEFECT",        # defect / trap density tendency
    "RCONT",         # contact resistance tendency
    "RINT",          # interconnect resistance tendency
    "CINT",          # interconnect capacitance tendency
)

# ---- Intermediate process state x ----
X_KEYS: Tuple[str, ...] = (
    "LEFF",
    "EOT_EFF",
    "NCH_EFF",
    "XJ_EFF",
    "LDIFF",
    "RSD_EFF",
    "STRESS_EFF",
    "RCONTACT_EFF",
    "RWIRE_EFF",
    "CWIRE_EFF",
    "R_CORNER_EFF",
    "T_LINER_EFF",
    "DH_STI_EFF",
    "DEFECT_EFF",
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


def propagate_linear(
    spec_in: GaussianSpec,
    A: np.ndarray,
    b: np.ndarray,
    in_keys: Tuple[str, ...],
    out_keys: Tuple[str, ...],
) -> Tuple[np.ndarray, np.ndarray]:
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

    # physically plausible non-zero defaults
    pairs = {
        ("CD_G", "LER"): 0.45,
        ("OVL_X", "OVL_Y"): 0.25,
        ("ETCH_RATE", "ETCH_PROFILE"): 0.35,
        ("EOT", "ILD_THK"): 0.20,
        ("DOSE", "RP"): -0.35,
        ("DOSE", "THERM"): 0.20,
        ("RP", "THERM"): 0.40,
        ("CMP", "ILD_THK"): 0.30,
        ("CMP", "STRESS"): 0.20,
        ("STRESS", "CD_G"): 0.15,
        ("RCONT", "RINT"): 0.25,
        ("RINT", "CINT"): 0.15,
        ("DEFECT", "LER"): 0.25,
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

    # litho / geometry
    B[xi["LEFF"], zi["CD_G"]] = 0.90
    B[xi["LEFF"], zi["LER"]] = -0.35
    B[xi["LEFF"], zi["OVL_X"]] = -0.10

    # dielectric
    B[xi["EOT_EFF"], zi["EOT"]] = 1.00
    B[xi["EOT_EFF"], zi["ILD_THK"]] = 0.10

    # channel / junction
    B[xi["NCH_EFF"], zi["DOSE"]] = 0.85
    B[xi["NCH_EFF"], zi["RP"]] = -0.20
    B[xi["NCH_EFF"], zi["THERM"]] = 0.25

    B[xi["XJ_EFF"], zi["RP"]] = 0.75
    B[xi["XJ_EFF"], zi["THERM"]] = 0.45
    B[xi["XJ_EFF"], zi["ETCH_RATE"]] = 0.10

    B[xi["LDIFF"], zi["THERM"]] = 0.85
    B[xi["LDIFF"], zi["DOSE"]] = 0.10

    # resistance / stress / interconnect
    B[xi["RSD_EFF"], zi["DOSE"]] = -0.20
    B[xi["RSD_EFF"], zi["THERM"]] = -0.20
    B[xi["RSD_EFF"], zi["RCONT"]] = 0.30

    B[xi["STRESS_EFF"], zi["STRESS"]] = 1.00
    B[xi["STRESS_EFF"], zi["CMP"]] = 0.15

    B[xi["RCONTACT_EFF"], zi["RCONT"]] = 0.95
    B[xi["RCONTACT_EFF"], zi["OVL_X"]] = 0.10
    B[xi["RCONTACT_EFF"], zi["OVL_Y"]] = 0.10

    B[xi["RWIRE_EFF"], zi["RINT"]] = 0.95
    B[xi["RWIRE_EFF"], zi["CD_G"]] = -0.15

    B[xi["CWIRE_EFF"], zi["CINT"]] = 0.95
    B[xi["CWIRE_EFF"], zi["ILD_THK"]] = -0.20

    # STI-style effective knobs
    B[xi["R_CORNER_EFF"], zi["ETCH_PROFILE"]] = 0.90
    B[xi["R_CORNER_EFF"], zi["ETCH_RATE"]] = -0.20

    B[xi["T_LINER_EFF"], zi["ILD_THK"]] = 0.90
    B[xi["T_LINER_EFF"], zi["CMP"]] = -0.10

    B[xi["DH_STI_EFF"], zi["CMP"]] = 0.85
    B[xi["DH_STI_EFF"], zi["ETCH_RATE"]] = 0.15

    B[xi["DEFECT_EFF"], zi["DEFECT"]] = 1.00

    return B


def default_A_x_to_p() -> np.ndarray:
    A = np.zeros((len(P_KEYS), len(X_KEYS)), dtype=float)
    xi = {k: i for i, k in enumerate(X_KEYS)}
    pi = {k: i for i, k in enumerate(P_KEYS)}

    # Vth family (Tier1->x->VTH path emphasized)
    A[pi["DVTH_N"], xi["LEFF"]] = -0.9e-3
    A[pi["DVTH_N"], xi["EOT_EFF"]] = +1.2e-3
    A[pi["DVTH_N"], xi["NCH_EFF"]] = +0.9e-3
    A[pi["DVTH_N"], xi["DEFECT_EFF"]] = +0.3e-3

    A[pi["DVTH_P"], xi["LEFF"]] = +0.8e-3
    A[pi["DVTH_P"], xi["EOT_EFF"]] = -1.1e-3
    A[pi["DVTH_P"], xi["NCH_EFF"]] = -0.8e-3
    A[pi["DVTH_P"], xi["DEFECT_EFF"]] = -0.3e-3

    # beta/mobility
    A[pi["DBETA_N"], xi["STRESS_EFF"]] = +6.0e-3
    A[pi["DBETA_N"], xi["LEFF"]] = -2.0e-3
    A[pi["DBETA_N"], xi["RSD_EFF"]] = -4.0e-3

    A[pi["DBETA_P"], xi["STRESS_EFF"]] = -4.5e-3
    A[pi["DBETA_P"], xi["LEFF"]] = +1.5e-3
    A[pi["DBETA_P"], xi["RSD_EFF"]] = -3.5e-3

    # DIBL / SS
    A[pi["DDIBL_N"], xi["LEFF"]] = -2.4e-3
    A[pi["DDIBL_N"], xi["XJ_EFF"]] = +1.5e-3
    A[pi["DDIBL_N"], xi["EOT_EFF"]] = +0.7e-3

    A[pi["DDIBL_P"], xi["LEFF"]] = +2.2e-3
    A[pi["DDIBL_P"], xi["XJ_EFF"]] = -1.4e-3
    A[pi["DDIBL_P"], xi["EOT_EFF"]] = -0.6e-3

    A[pi["DSS_N"], xi["EOT_EFF"]] = +1.2e-3
    A[pi["DSS_N"], xi["DEFECT_EFF"]] = +0.8e-3
    A[pi["DSS_N"], xi["LEFF"]] = -0.8e-3

    A[pi["DSS_P"], xi["EOT_EFF"]] = -1.1e-3
    A[pi["DSS_P"], xi["DEFECT_EFF"]] = -0.7e-3
    A[pi["DSS_P"], xi["LEFF"]] = +0.8e-3

    # Rsd / Cj / leakage
    A[pi["DRSD_N"], xi["RSD_EFF"]] = 1.00
    A[pi["DRSD_N"], xi["RCONTACT_EFF"]] = 0.25
    A[pi["DRSD_P"], xi["RSD_EFF"]] = 0.95
    A[pi["DRSD_P"], xi["RCONTACT_EFF"]] = 0.25

    A[pi["DCJD_N"], xi["XJ_EFF"]] = +2.0e-3
    A[pi["DCJD_N"], xi["R_CORNER_EFF"]] = +0.3e-3
    A[pi["DCJD_P"], xi["XJ_EFF"]] = +1.8e-3
    A[pi["DCJD_P"], xi["R_CORNER_EFF"]] = +0.3e-3

    A[pi["DCJS_N"], xi["XJ_EFF"]] = +1.9e-3
    A[pi["DCJS_N"], xi["T_LINER_EFF"]] = -0.2e-3
    A[pi["DCJS_P"], xi["XJ_EFF"]] = +1.7e-3
    A[pi["DCJS_P"], xi["T_LINER_EFF"]] = -0.2e-3

    A[pi["DLNIREVD_N"], xi["XJ_EFF"]] = +8.0e-3
    A[pi["DLNIREVD_N"], xi["DEFECT_EFF"]] = +3.0e-3
    A[pi["DLNIREVD_P"], xi["XJ_EFF"]] = +7.0e-3
    A[pi["DLNIREVD_P"], xi["DEFECT_EFF"]] = +2.6e-3

    A[pi["DLNIREVS_N"], xi["XJ_EFF"]] = +7.4e-3
    A[pi["DLNIREVS_N"], xi["DEFECT_EFF"]] = +2.5e-3
    A[pi["DLNIREVS_P"], xi["XJ_EFF"]] = +6.8e-3
    A[pi["DLNIREVS_P"], xi["DEFECT_EFF"]] = +2.2e-3

    # interconnect/contact
    A[pi["DRCONTACT"], xi["RCONTACT_EFF"]] = +1.0
    A[pi["DRWIRE"], xi["RWIRE_EFF"]] = +1.0
    A[pi["DCWIRE"], xi["CWIRE_EFF"]] = +1.0

    return A


def default_mu_z() -> np.ndarray:
    return np.zeros((len(Z_KEYS),), dtype=float)


def default_sigma_z() -> np.ndarray:
    return np.array(
        [1.0, 0.8, 0.8, 1.0, 0.8, 0.8, 0.8, 0.7, 1.0, 0.9, 0.8, 0.7, 0.8, 0.9, 0.7, 0.7, 0.7],
        dtype=float,
    )


def default_mu_eps_x() -> np.ndarray:
    return np.zeros((len(X_KEYS),), dtype=float)


def default_sigma_eps_x() -> np.ndarray:
    return np.array([0.20, 0.15, 0.25, 0.20, 0.20, 0.20, 0.15, 0.15, 0.12, 0.12, 0.15, 0.12, 0.12, 0.20], dtype=float)


def default_corr_eps_x() -> np.ndarray:
    corr = np.eye(len(X_KEYS), dtype=float)
    xi = {k: i for i, k in enumerate(X_KEYS)}
    pairs = {
        ("LEFF", "R_CORNER_EFF"): 0.20,
        ("RSD_EFF", "RCONTACT_EFF"): 0.20,
        ("RWIRE_EFF", "CWIRE_EFF"): 0.10,
        ("T_LINER_EFF", "DH_STI_EFF"): 0.15,
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
