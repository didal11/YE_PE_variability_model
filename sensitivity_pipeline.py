#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

METRICS_ORDER = [
    "Vth_lowVds", "Vth_highVds", "DIBL", "SS_longL", "SS_shortL", "Ioff_ref", "Voff_index", "n_eff", "Body_sens_Vth",
    "Idlin_Vdd", "Ion_Vdd", "Ion_over_Idlin", "gm_max", "Vg_at_gm_max", "gm_over_Id_weak", "gm_over_Id_mod", "Vdsat_knee",
    "gds_sat", "ro_sat", "gds_Vds_sens", "CLM_index", "Ron_lin_Vdd", "Ron_lin_Vddm", "Early_like",
    "Vth_rolloff_L", "SS_rolloff_L", "Ion_rolloff_L_ratio", "Vth_narrowW", "SS_narrowW", "Ron_narrowW_ratio",
    "GIDL_index", "Junc_leak_index",
]

PARAMS_150 = [
    "a0", "ags", "alpha0", "alpha1", "at", "b0", "b1", "beta0", "cdscd", "cgdo", "cgso", "cjs", "cjswgs", "cjsws", "dlc", "drout", "dsub", "dwc", "eta0", "etab", "k1", "k2", "keta", "kt1", "kt2", "la0", "lags", "lalpha0", "lalpha1", "lat", "lb0", "lb1", "lbeta0", "lcdscd", "ldrout", "ldsub", "leta0", "letab", "lint", "lk1", "lk2", "lketa", "lkt1", "lkt2", "lnfactor", "lpclm", "lpdiblc1", "lpdiblc2", "lpdiblcb", "lpscbe1", "lu0", "lua", "lua1", "lub", "lub1", "luc", "luc1", "lute", "lvoff", "lvsat", "lvth0", "nfactor", "pa0", "pags", "palpha0", "palpha1", "pat", "pb0", "pb1", "pbeta0", "pcdscd", "pclm", "pdiblc1", "pdiblc2", "pdiblcb", "pdrout", "pdsub", "peta0", "petab", "pk1", "pk2", "pketa", "pkt1", "pkt2", "pnfactor", "ppclm", "ppdiblc1", "ppdiblc2", "ppdiblcb", "ppscbe1", "pscbe1", "pu0", "pua", "pua1", "pub", "pub1", "puc", "puc1", "pute", "pvoff", "pvsat", "pvth0", "toxe", "u0", "ua", "ua1", "ub", "ub1", "uc", "uc1", "ute", "voff", "vsat", "vth0", "wa0", "wags", "walpha0", "walpha1", "wat", "wb0", "wb1", "wbeta0", "wcdscd", "wdrout", "wdsub", "weta0", "wetab", "wint", "wk1", "wk2", "wketa", "wkt1", "wkt2", "wnfactor", "wpclm", "wpdiblc1", "wpdiblc2", "wpdiblcb", "wpscbe1", "wu0", "wua", "wua1", "wub", "wub1", "wuc", "wuc1", "wute", "wvoff", "wvsat", "wvth0"
]

@dataclass
class BiasConfig:
    vdd: float = 1.8
    vds_low: float = 0.05
    temp: float = 25.0
    l_long: float = 1.0e-6
    l_short: float = 0.15e-6
    w_wide: float = 2.0e-6
    w_narrow: float = 0.3e-6
    vbs_for_body: Tuple[float, ...] = (0.0, -0.2, -0.4, -0.6)
    vg_min: float = -0.4
    vg_max: float = 1.8
    vg_step: float = 0.01
    vd_step: float = 0.01
    vdd_minus_delta: float = 1.6
    icrit: float = 1e-7
    gidl_vg_start: float = -0.4
    gidl_vg_stop: float = 0.0
    gidl_vg_step: float = 0.02

FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
ASSIGN_RE = re.compile(r"(\b[a-zA-Z_]\w*\b)\s*=\s*([^\s]+)")


def parse_num(token: str) -> float:
    m = FLOAT_RE.search(token)
    if not m:
        raise ValueError(f"숫자 파싱 실패: {token}")
    return float(m.group(0))


def get_param_values(model_text: str, param: str) -> List[float]:
    vals = []
    for m in ASSIGN_RE.finditer(model_text):
        if m.group(1).lower() == param.lower():
            vals.append(parse_num(m.group(2)))
    if not vals:
        raise ValueError(f"파라미터 '{param}'를 찾지 못했습니다.")
    return vals


def patch_param_alpha(tt_text: str, corner_text: str, param: str, alpha: float) -> str:
    tt_vals = get_param_values(tt_text, param)
    c_vals = get_param_values(corner_text, param)
    if len(c_vals) < 1:
        raise ValueError(f"{param} corner occurrence가 없습니다.")

    idx = 0
    def repl(m: re.Match[str]) -> str:
        nonlocal idx
        name = m.group(1)
        val = m.group(2)
        if name.lower() != param.lower():
            return m.group(0)
        cval = c_vals[idx] if idx < len(c_vals) else tt_vals[idx]
        newv = tt_vals[idx] + alpha * (cval - tt_vals[idx])
        idx += 1
        return f"{name} = {newv:.12g}"

    patched = ASSIGN_RE.sub(repl, tt_text)
    if idx != len(tt_vals):
        raise ValueError(f"{param} 치환 카운트 오류: {idx} != {len(tt_vals)}")
    return patched


def inject_missing_monte_params(model_text: str) -> str:
    slope_names = sorted(set(re.findall(r"(sky130_fd_pr__nfet_01v8__\w*slope\w*)", model_text)))
    lines = [".param MC_MM_SWITCH = 0"] + [f".param {name} = 0.0" for name in slope_names]
    return "\n".join(lines) + "\n" + model_text


def run_ngspice(netlist: str, cwd: Path) -> None:
    print(f"[ngspice] start: {cwd / netlist}", flush=True)
    p = subprocess.run(["ngspice", "-b", "-o", "ngspice.log", netlist], cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ngspice 실패: {p.stderr}\n{p.stdout}")
    print(f"[ngspice] done : {cwd / netlist}", flush=True)


def load_wrdata(path: Path) -> pd.DataFrame:
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    data = {}
    if arr.shape[1] >= 3 and arr.shape[1] % 2 == 1:
        scale = arr[:, 0]
        for i in range(arr.shape[1] - 1):
            data[f"x_vec{i}"] = scale
            data[f"y_vec{i}"] = arr[:, i + 1]
        return pd.DataFrame(data)
    if arr.shape[1] % 2 == 0:
        nvec = arr.shape[1] // 2
        for i in range(nvec):
            data[f"x_vec{i}"] = arr[:, 2 * i]
            data[f"y_vec{i}"] = arr[:, 2 * i + 1]
        return pd.DataFrame(data)
    raise ValueError(f"wrdata 출력 형식 오류: {path}")


def interp_x_for_y(x: np.ndarray, y: np.ndarray, target: float) -> float:
    idx = np.where(np.diff(np.sign(y - target)) != 0)[0]
    if len(idx) == 0:
        raise ValueError("교차점을 찾지 못했습니다.")
    i = idx[0]
    x0, x1 = x[i], x[i + 1]
    y0, y1 = y[i], y[i + 1]
    return x0 + (target - y0) * (x1 - x0) / (y1 - y0)


def calc_ss(vg: np.ndarray, id_abs: np.ndarray) -> float:
    mask = id_abs > 1e-18
    vg2 = vg[mask]
    lid = np.log10(id_abs[mask])
    return 1.0 / np.max(np.gradient(lid, vg2))


def deriv(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.gradient(y, x)


def interp_y_for_x_sorted(x: np.ndarray, y: np.ndarray, target_x: float) -> float:
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]
    uniq_x, idx = np.unique(xs, return_index=True)
    uniq_y = ys[idx]
    if target_x < uniq_x[0] or target_x > uniq_x[-1]:
        raise ValueError(f"보간 타겟 {target_x}가 데이터 범위를 벗어났습니다.")
    return float(np.interp(target_x, uniq_x, uniq_y))


def simulate_metrics(model_path: Path, cfg: BiasConfig, workdir: Path) -> Dict[str, float]:
    body_sweeps = []
    for i, vb in enumerate(cfg.vbs_for_body):
        body_sweeps += [f"alter Vb {vb}", f"dc Vg {cfg.vg_min} {cfg.vg_max} {cfg.vg_step}", f"wrdata idvg_body_{i}.dat v(g) i(Vd)"]
    deck = f"""
.include '{model_path.resolve().as_posix()}'
.option temp={cfg.temp}
.param Llong={cfg.l_long} Lshort={cfg.l_short} Wwide={cfg.w_wide} Wnarrow={cfg.w_narrow}
Vd d 0 0
Vg g 0 0
Vs s 0 0
Vb b 0 0
X1 d g s b sky130_fd_pr__nfet_01v8 l={{Llong}} w={{Wwide}}
.control
set wr_singlescale
alter Vd {cfg.vds_low}
dc Vg {cfg.vg_min} {cfg.vg_max} {cfg.vg_step}
wrdata idvg_low.dat v(g) i(Vd)
alter Vd {cfg.vdd}
dc Vg {cfg.vg_min} {cfg.vg_max} {cfg.vg_step}
wrdata idvg_high.dat v(g) i(Vd)
altermod @x1[sky130_fd_pr__nfet_01v8] l={cfg.l_short}
dc Vg {cfg.vg_min} {cfg.vg_max} {cfg.vg_step}
wrdata idvg_short_high.dat v(g) i(Vd)
altermod @x1[sky130_fd_pr__nfet_01v8] l={cfg.l_long}
altermod @x1[sky130_fd_pr__nfet_01v8] w={cfg.w_narrow}
dc Vg {cfg.vg_min} {cfg.vg_max} {cfg.vg_step}
wrdata idvg_narrow_high.dat v(g) i(Vd)
altermod @x1[sky130_fd_pr__nfet_01v8] w={cfg.w_wide}
alter Vg {cfg.vdd}
dc Vd 0 {cfg.vdd} {cfg.vd_step}
wrdata idvd_vdd.dat v(d) i(Vd)
altermod @x1[sky130_fd_pr__nfet_01v8] l={cfg.l_short}
altermod @x1[sky130_fd_pr__nfet_01v8] w={cfg.w_wide}
dc Vd 0 {cfg.vdd} {cfg.vd_step}
wrdata idvd_short_vdd.dat v(d) i(Vd)
altermod @x1[sky130_fd_pr__nfet_01v8] l={cfg.l_long}
altermod @x1[sky130_fd_pr__nfet_01v8] w={cfg.w_narrow}
dc Vd 0 {cfg.vdd} {cfg.vd_step}
wrdata idvd_narrow_vdd.dat v(d) i(Vd)
altermod @x1[sky130_fd_pr__nfet_01v8] l={cfg.l_long}
altermod @x1[sky130_fd_pr__nfet_01v8] w={cfg.w_wide}
alter Vg {cfg.vdd_minus_delta}
dc Vd 0 {cfg.vds_low} {cfg.vd_step}
wrdata idvd_vddm_lin.dat v(d) i(Vd)
alter Vd {cfg.vdd}
{'\n'.join(body_sweeps)}
alter Vb 0
dc Vg {cfg.gidl_vg_start} {cfg.gidl_vg_stop} {cfg.gidl_vg_step}
wrdata gidl_vg.dat v(g) i(Vd)
alter Vg 0
alter Vs 0
alter Vb 0
dc Vd 0 -0.5 -0.05
wrdata junc.dat v(d) i(Vd)
quit
.endc
.end
"""
    (workdir / "sim.sp").write_text(deck)
    run_ngspice("sim.sp", workdir)

    def read_xy(fname: str):
        df = load_wrdata(workdir / fname)
        return df["x_vec0"].to_numpy(), -df["y_vec1"].to_numpy()

    vg_l, id_l = read_xy("idvg_low.dat")
    vg_h, id_h = read_xy("idvg_high.dat")
    vg_sh, id_sh = read_xy("idvg_short_high.dat")
    vg_n, id_n = read_xy("idvg_narrow_high.dat")
    vd, idvd = read_xy("idvd_vdd.dat")
    vd_s, idvd_s = read_xy("idvd_short_vdd.dat")
    vd_nw, idvd_nw = read_xy("idvd_narrow_vdd.dat")
    vd_m, idvd_m = read_xy("idvd_vddm_lin.dat")
    vd_j, id_j = read_xy("junc.dat")

    icrit_long = cfg.icrit * (cfg.w_wide / cfg.l_long)
    icrit_short = cfg.icrit * (cfg.w_wide / cfg.l_short)
    vth_low = interp_x_for_y(vg_l, id_l, icrit_long)
    vth_high = interp_x_for_y(vg_h, id_h, icrit_long)
    ss_long = calc_ss(vg_h, np.abs(id_h))
    ss_short = calc_ss(vg_sh, np.abs(id_sh))
    ioff = float(np.interp(0.0, vg_h, id_h))
    voff_index = float(np.log10(max(abs(ioff), 1e-30)))
    n_eff = ss_long / (math.log(10) * 0.02585)

    vths = [interp_x_for_y(read_xy(f"idvg_body_{i}.dat")[0], read_xy(f"idvg_body_{i}.dat")[1], icrit_long) for i,_ in enumerate(cfg.vbs_for_body)]
    body_sens = np.polyfit(np.array(cfg.vbs_for_body), np.array(vths), 1)[0]

    idlin = float(np.interp(cfg.vds_low, vd, idvd)); ion = float(np.interp(cfg.vdd, vd, idvd))
    gm = deriv(vg_h, id_h); gm_id = gm / np.clip(id_h, 1e-30, None)
    gds = deriv(vd, idvd); gds_sat = float(np.interp(cfg.vdd, vd, gds))

    vg_gidl, gidl_i = read_xy("gidl_vg.dat")
    return {
        "Vth_lowVds": vth_low,
        "Vth_highVds": vth_high,
        "DIBL": (vth_high - vth_low) / (cfg.vdd - cfg.vds_low),
        "SS_longL": ss_long,
        "SS_shortL": ss_short,
        "Ioff_ref": ioff,
        "Voff_index": voff_index,
        "n_eff": n_eff,
        "Body_sens_Vth": float(body_sens),
        "Idlin_Vdd": idlin,
        "Ion_Vdd": ion,
        "Ion_over_Idlin": ion / idlin,
        "gm_max": float(np.max(gm)),
        "Vg_at_gm_max": float(vg_h[np.argmax(gm)]),
        "gm_over_Id_weak": interp_y_for_x_sorted(id_h, gm_id, icrit_long),
        "gm_over_Id_mod": interp_y_for_x_sorted(id_h, gm_id, 10 * icrit_long),
        "Vdsat_knee": float(vd[np.where(gds < 0.1 * gds[0])[0][0]]),
        "gds_sat": gds_sat,
        "ro_sat": 1.0 / gds_sat,
        "gds_Vds_sens": (float(np.interp(cfg.vdd, vd, gds)) - float(np.interp(0.8 * cfg.vdd, vd, gds))) / (0.2 * cfg.vdd),
        "CLM_index": (ion - float(np.interp(0.8 * cfg.vdd, vd, idvd))) / ion,
        "Ron_lin_Vdd": cfg.vds_low / idlin,
        "Ron_lin_Vddm": cfg.vds_low / float(np.interp(cfg.vds_low, vd_m, idvd_m)),
        "Early_like": (1.0 / gds_sat) * ion,
        "Vth_rolloff_L": interp_x_for_y(vg_sh, id_sh, icrit_short) - vth_high,
        "SS_rolloff_L": ss_short - ss_long,
        "Ion_rolloff_L_ratio": float(np.interp(cfg.vdd, vd_s, idvd_s)) / ion,
        "Vth_narrowW": interp_x_for_y(vg_n, id_n, cfg.icrit * (cfg.w_narrow / cfg.l_long)) - vth_high,
        "SS_narrowW": calc_ss(vg_n, np.abs(id_n)) - ss_long,
        "Ron_narrowW_ratio": (cfg.vds_low / float(np.interp(cfg.vds_low, vd_nw, idvd_nw))) / (cfg.vds_low / idlin),
        "GIDL_index": float(np.interp(cfg.gidl_vg_start, vg_gidl, gidl_i)),
        "Junc_leak_index": float(np.interp(-0.5, vd_j, id_j)),
    }


def five_point_forward(f0: float, f1: float, f2: float, f3: float, f4: float, h: float) -> float:
    return (-25 * f0 + 48 * f1 - 36 * f2 + 16 * f3 - 3 * f4) / (12 * h)


def make_scope_doc(path: Path) -> None:
    with path.open("w") as f:
        f.write("# 확정 항목\n\n## 32 Metrics\n")
        for m in METRICS_ORDER:
            f.write(f"- {m}\n")
        f.write("\n## 150 Parameter Items\n")
        for p in PARAMS_150:
            f.write(f"- {p}\n")


def run_sensitivity(tt_text: str, ss_text: str, ff_text: str, cfg: BiasConfig, outdir: Path) -> pd.DataFrame:
    rows = []
    print("[progress] baseline(TT) simulation", flush=True)
    with tempfile.TemporaryDirectory() as td:
        base_model = Path(td) / "tt.spice"
        base_model.write_text(tt_text)
        base_metrics = simulate_metrics(base_model, cfg, Path(td))

    total = len(PARAMS_150)
    for idx_param, param in enumerate(PARAMS_150, start=1):
        print(f"[progress] param {idx_param}/{total}: {param}", flush=True)
        row = {"param": param}
        for m in METRICS_ORDER:
            row[f"{m}_sens_SS_5pt"] = np.nan
            row[f"{m}_sens_FF_5pt"] = np.nan
            row[f"{m}_sens_avg_5pt"] = np.nan

        for corner_name, corner_text in (("SS", ss_text), ("FF", ff_text)):
            print(f"[progress]   corner={corner_name}", flush=True)
            metrics_samples = [base_metrics]
            for alpha in (0.25, 0.5, 0.75, 1.0):
                print(f"[progress]     alpha={alpha}", flush=True)
                patched = patch_param_alpha(tt_text, corner_text, param, alpha)
                with tempfile.TemporaryDirectory() as td:
                    mp = Path(td) / f"{param}_{corner_name}_{alpha}.spice"
                    mp.write_text(patched)
                    metrics_samples.append(simulate_metrics(mp, cfg, Path(td)))
            for m in METRICS_ORDER:
                dfdalpha = five_point_forward(
                    metrics_samples[0][m], metrics_samples[1][m], metrics_samples[2][m], metrics_samples[3][m], metrics_samples[4][m], 0.25
                )
                row[f"{m}_sens_{corner_name}_5pt"] = dfdalpha

        for m in METRICS_ORDER:
            v = [row[f"{m}_sens_SS_5pt"], row[f"{m}_sens_FF_5pt"]]
            row[f"{m}_sens_avg_5pt"] = float(np.mean(v))
        rows.append(row)

    df = pd.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "sensitivity_matrix_5pt.csv", index=False)
    (outdir / "tt_metrics.json").write_text(json.dumps(base_metrics, indent=2))
    make_scope_doc(outdir / "scope_32_metrics_150_params.md")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spice-zip", type=Path, default=Path("spice.zip"))
    ap.add_argument("--outdir", type=Path, default=Path("out_sensitivity"))
    ap.add_argument("--tt-file", default="sky130_fd_pr__nfet_01v8__tt.pm3.spice")
    ap.add_argument("--ss-file", default="sky130_fd_pr__nfet_01v8__ss.pm3.spice")
    ap.add_argument("--ff-file", default="sky130_fd_pr__nfet_01v8__ff.pm3.spice")
    args = ap.parse_args()

    with zipfile.ZipFile(args.spice_zip) as z:
        tt_text = z.read(args.tt_file).decode("utf-8", errors="ignore")
        ss_text = z.read(args.ss_file).decode("utf-8", errors="ignore")
        ff_text = z.read(args.ff_file).decode("utf-8", errors="ignore")

    tt_text = inject_missing_monte_params(tt_text)
    ss_text = inject_missing_monte_params(ss_text)
    ff_text = inject_missing_monte_params(ff_text)
    df = run_sensitivity(tt_text, ss_text, ff_text, BiasConfig(), args.outdir)
    print(df[["param", f"{METRICS_ORDER[0]}_sens_avg_5pt"]].head())


if __name__ == "__main__":
    main()
