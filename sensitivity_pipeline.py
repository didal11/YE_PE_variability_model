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
    "RO_freq", "INV_tpHL", "INV_tpLH", "INV_tr", "INV_tf", "STBY_power", "SRAM_hold_SNM", "SRAM_read_margin",
    "Ion_n", "Ion_p", "Idlin_n", "Idlin_p", "Ron_n_Vdd", "Ron_n_Vddm", "Ron_p_Vdd", "Ron_p_Vddm",
    "Ioff_n", "Ioff_p", "ro_n", "ro_p", "CLM_index_n", "CLM_index_p", "GIDL_n", "GIDL_p",
    "Junc_leak_n", "Junc_leak_p", "Ig_off_n", "Ig_off_p",
    "Ion_rolloff_L_ratio_n", "Ron_narrowW_ratio_n", "Ion_short_narrow_ratio_n", "AX_PD_ratio_Ion", "PD_PU_ratio_Ion",
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
    delta_vg: float = 0.2
    icrit: float = 1e-7
    gidl_vg_start: float = -0.4
    gidl_vg_stop: float = 0.0
    gidl_vg_step: float = 0.02


@dataclass
class ProgressTracker:
    total_units: int
    completed_units: int = 0

    def pct(self) -> float:
        if self.total_units <= 0:
            return 100.0
        return 100.0 * self.completed_units / self.total_units

    def log(self, msg: str) -> None:
        print(f"[progress][overall {self.completed_units}/{self.total_units} {self.pct():5.1f}%] {msg}", flush=True)

    def advance(self, msg: str) -> None:
        self.completed_units += 1
        self.log(msg)

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






def get_all_params(model_text: str) -> List[str]:
    seen = set()
    out = []
    for m in ASSIGN_RE.finditer(model_text):
        k = m.group(1).lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


def get_delta_params(tt_text: str, ss_text: str, ff_text: str) -> List[str]:
    common = sorted(set(get_all_params(tt_text)) & set(get_all_params(ss_text)) & set(get_all_params(ff_text)))
    delta = []
    for param in common:
        try:
            tv = get_param_values(tt_text, param)
            sv = get_param_values(ss_text, param)
            fv = get_param_values(ff_text, param)
        except ValueError:
            continue
        if not (tv == sv == fv):
            delta.append(param)
    return delta

def get_tnom_celsius(model_text: str, fallback_c: float = 27.0) -> float:
    try:
        vals = get_param_values(model_text, "tnom")
        return float(vals[0])
    except ValueError:
        return float(fallback_c)

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
    slope_names = sorted(set(re.findall(r"(sky130_fd_pr__\w+__\w*slope\w*)", model_text)))
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


def calc_ss(vg: np.ndarray, id_abs: np.ndarray, id_target: float) -> float:
    lo = id_target / 100.0
    hi = id_target / 10.0
    mask = (id_abs >= lo) & (id_abs <= hi)
    if np.count_nonzero(mask) < 3:
        raise ValueError("SS 추출용 전류창 데이터가 부족합니다.")
    vg_w = vg[mask]
    logid_w = np.log10(np.clip(id_abs[mask], 1e-30, None))
    slope, _ = np.polyfit(vg_w, logid_w, 1)
    if slope <= 0:
        raise ValueError("SS 추출 slope가 비정상입니다.")
    return float(1.0 / slope)


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




def extract_vdsat_knee(vd: np.ndarray, idvd: np.ndarray) -> float:
    fit_mask = (vd >= 0.01) & (vd <= 0.1)
    if np.count_nonzero(fit_mask) < 3:
        raise ValueError("Vdsat_knee 선형영역 데이터 부족")
    a, b = np.polyfit(vd[fit_mask], idvd[fit_mask], 1)
    id_lin = a * vd + b
    chk_mask = vd >= 0.1
    err = np.abs(idvd[chk_mask] - id_lin[chk_mask]) / np.clip(np.abs(id_lin[chk_mask]), 1e-30, None)
    idx = np.where(err > 0.1)[0]
    if len(idx) == 0:
        raise ValueError("Vdsat_knee 추출 실패")
    return float(vd[chk_mask][idx[0]])


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
alter Vd {cfg.vdd}
dc Vg {cfg.vg_min} {cfg.vg_max} {cfg.vg_step}
wrdata idvg_short_high.dat v(g) i(Vd)
alter Vd {cfg.vds_low}
dc Vg {cfg.vg_min} {cfg.vg_max} {cfg.vg_step}
wrdata idvg_short_low.dat v(g) i(Vd)
altermod @x1[sky130_fd_pr__nfet_01v8] l={cfg.l_long}
altermod @x1[sky130_fd_pr__nfet_01v8] w={cfg.w_narrow}
alter Vd {cfg.vdd}
dc Vg {cfg.vg_min} {cfg.vg_max} {cfg.vg_step}
wrdata idvg_narrow_high.dat v(g) i(Vd)
alter Vd {cfg.vds_low}
dc Vg {cfg.vg_min} {cfg.vg_max} {cfg.vg_step}
wrdata idvg_narrow_low.dat v(g) i(Vd)
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
altermod @x1[sky130_fd_pr__nfet_01v8] l={cfg.l_short}
altermod @x1[sky130_fd_pr__nfet_01v8] w={cfg.w_narrow}
dc Vd 0 {cfg.vdd} {cfg.vd_step}
wrdata idvd_short_narrow_vdd.dat v(d) i(Vd)
altermod @x1[sky130_fd_pr__nfet_01v8] l={cfg.l_long}
altermod @x1[sky130_fd_pr__nfet_01v8] w={cfg.w_wide}
alter Vg {cfg.vdd - cfg.delta_vg}
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
dc Vd 0 0.5 0.05
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
    vg_sh_low, id_sh_low = read_xy("idvg_short_low.dat")
    vg_n, id_n = read_xy("idvg_narrow_high.dat")
    vg_n_low, id_n_low = read_xy("idvg_narrow_low.dat")
    vd, idvd = read_xy("idvd_vdd.dat")
    vd_s, idvd_s = read_xy("idvd_short_vdd.dat")
    vd_nw, idvd_nw = read_xy("idvd_narrow_vdd.dat")
    vd_sn, idvd_sn = read_xy("idvd_short_narrow_vdd.dat")
    vd_m, idvd_m = read_xy("idvd_vddm_lin.dat")
    vd_j, id_j = read_xy("junc.dat")

    icrit_long = cfg.icrit * (cfg.w_wide / cfg.l_long)
    icrit_short = cfg.icrit * (cfg.w_wide / cfg.l_short)
    vth_low = interp_x_for_y(vg_l, id_l, icrit_long)
    vth_high = interp_x_for_y(vg_h, id_h, icrit_long)
    ss_long = calc_ss(vg_l, np.abs(id_l), icrit_long)
    ss_short = calc_ss(vg_sh_low, np.abs(id_sh_low), icrit_short)
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
        "DIBL": (vth_low - vth_high) / (cfg.vdd - cfg.vds_low),
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
        "Vdsat_knee": extract_vdsat_knee(vd, idvd),
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
        "SS_narrowW": calc_ss(vg_n_low, np.abs(id_n_low), cfg.icrit * (cfg.w_narrow / cfg.l_long)) - ss_long,
        "Ron_narrowW_ratio": (cfg.vds_low / float(np.interp(cfg.vds_low, vd_nw, idvd_nw))) / (cfg.vds_low / idlin),
        "Ion_short_narrow_ratio": float(np.interp(cfg.vdd, vd_sn, idvd_sn)) / ion,
        "GIDL_index": float(np.interp(cfg.gidl_vg_start, vg_gidl, gidl_i)),
        "Junc_leak_index": float(np.interp(0.5, vd_j, id_j)),
    }




def simulate_metrics_pfet(model_path: Path, cfg: BiasConfig, workdir: Path) -> Dict[str, float]:
    body_sweeps = []
    for i, vb in enumerate((cfg.vdd, cfg.vdd - 0.2, cfg.vdd - 0.4, cfg.vdd - 0.6)):
        body_sweeps += [f"alter Vb {vb}", f"dc Vg {cfg.vg_min} {cfg.vg_max} {cfg.vg_step}", f"wrdata idvg_body_{i}.dat v(g) i(Vd)"]
    deck = f"""
.include '{model_path.resolve().as_posix()}'
.option temp={cfg.temp}
.param Llong={cfg.l_long} Lshort={cfg.l_short} Wwide={cfg.w_wide} Wnarrow={cfg.w_narrow}
Vd d 0 0
Vg g 0 0
Vs s 0 {cfg.vdd}
Vb b 0 {cfg.vdd}
X1 d g s b sky130_fd_pr__pfet_01v8 l={{Llong}} w={{Wwide}}
.control
set wr_singlescale
alter Vd {cfg.vdd - cfg.vds_low}
alter Vg 0
dc Vg {cfg.vg_min} {cfg.vg_max} {cfg.vg_step}
wrdata idvg_low.dat v(g) i(Vd)
alter Vd 0
dc Vg {cfg.vg_min} {cfg.vg_max} {cfg.vg_step}
wrdata idvg_high.dat v(g) i(Vd)
altermod @x1[sky130_fd_pr__pfet_01v8] l={cfg.l_short}
alter Vd 0
dc Vg {cfg.vg_min} {cfg.vg_max} {cfg.vg_step}
wrdata idvg_short_high.dat v(g) i(Vd)
alter Vd {cfg.vdd - cfg.vds_low}
dc Vg {cfg.vg_min} {cfg.vg_max} {cfg.vg_step}
wrdata idvg_short_low.dat v(g) i(Vd)
altermod @x1[sky130_fd_pr__pfet_01v8] l={cfg.l_long}
altermod @x1[sky130_fd_pr__pfet_01v8] w={cfg.w_narrow}
alter Vd 0
dc Vg {cfg.vg_min} {cfg.vg_max} {cfg.vg_step}
wrdata idvg_narrow_high.dat v(g) i(Vd)
alter Vd {cfg.vdd - cfg.vds_low}
dc Vg {cfg.vg_min} {cfg.vg_max} {cfg.vg_step}
wrdata idvg_narrow_low.dat v(g) i(Vd)
altermod @x1[sky130_fd_pr__pfet_01v8] w={cfg.w_wide}
alter Vg 0
dc Vd {cfg.vdd} 0 {-cfg.vd_step}
wrdata idvd_vdd.dat v(d) i(Vd)
altermod @x1[sky130_fd_pr__pfet_01v8] l={cfg.l_short}
altermod @x1[sky130_fd_pr__pfet_01v8] w={cfg.w_wide}
dc Vd {cfg.vdd} 0 {-cfg.vd_step}
wrdata idvd_short_vdd.dat v(d) i(Vd)
altermod @x1[sky130_fd_pr__pfet_01v8] l={cfg.l_long}
altermod @x1[sky130_fd_pr__pfet_01v8] w={cfg.w_narrow}
dc Vd {cfg.vdd} 0 {-cfg.vd_step}
wrdata idvd_narrow_vdd.dat v(d) i(Vd)
altermod @x1[sky130_fd_pr__pfet_01v8] l={cfg.l_long}
altermod @x1[sky130_fd_pr__pfet_01v8] w={cfg.w_wide}
alter Vg {cfg.delta_vg}
dc Vd {cfg.vdd} {cfg.vdd-cfg.vds_low} {-cfg.vd_step}
wrdata idvd_vddm_lin.dat v(d) i(Vd)
alter Vd 0
{chr(10).join(body_sweeps)}
alter Vb {cfg.vdd}
dc Vg {cfg.vdd} {cfg.vdd+0.4} {cfg.gidl_vg_step}
wrdata gidl_vg.dat v(g) i(Vd)
alter Vg {cfg.vdd}
alter Vs {cfg.vdd}
alter Vb {cfg.vdd}
dc Vd {cfg.vdd} {cfg.vdd-0.5} -0.05
wrdata junc.dat v(d) i(Vd)
quit
.endc
.end
"""
    (workdir / "sim.sp").write_text(deck)
    run_ngspice("sim.sp", workdir)

    def read_xy(fname: str):
        df = load_wrdata(workdir / fname)
        return df["x_vec0"].to_numpy(), np.abs(df["y_vec1"].to_numpy())

    vg_l, id_l = read_xy("idvg_low.dat")
    vg_h, id_h = read_xy("idvg_high.dat")
    vg_sh, id_sh = read_xy("idvg_short_high.dat")
    vg_sh_low, id_sh_low = read_xy("idvg_short_low.dat")
    vg_n, id_n = read_xy("idvg_narrow_high.dat")
    vg_n_low, id_n_low = read_xy("idvg_narrow_low.dat")
    vd, idvd = read_xy("idvd_vdd.dat")
    vd_s, idvd_s = read_xy("idvd_short_vdd.dat")
    vd_nw, idvd_nw = read_xy("idvd_narrow_vdd.dat")
    vd_m, idvd_m = read_xy("idvd_vddm_lin.dat")
    vd_j, id_j = read_xy("junc.dat")

    icrit_long = cfg.icrit * (cfg.w_wide / cfg.l_long)
    icrit_short = cfg.icrit * (cfg.w_wide / cfg.l_short)
    vth_low = interp_x_for_y(vg_l, id_l, icrit_long)
    vth_high = interp_x_for_y(vg_h, id_h, icrit_long)
    ss_long = calc_ss(vg_l, np.abs(id_l), icrit_long)
    ss_short = calc_ss(vg_sh_low, np.abs(id_sh_low), icrit_short)
    ioff = float(np.interp(cfg.vdd, vg_h, id_h))
    voff_index = float(np.log10(max(abs(ioff), 1e-30)))
    n_eff = ss_long / (math.log(10) * 0.02585)

    vths = [interp_x_for_y(read_xy(f"idvg_body_{i}.dat")[0], read_xy(f"idvg_body_{i}.dat")[1], icrit_long) for i,_ in enumerate((0,1,2,3))]
    body_sens = np.polyfit(np.array((cfg.vdd, cfg.vdd - 0.2, cfg.vdd - 0.4, cfg.vdd - 0.6)), np.array(vths), 1)[0]

    idlin = float(np.interp(cfg.vdd - cfg.vds_low, vd, idvd)); ion = float(np.interp(0.0, vd, idvd))
    gm = deriv(vg_h, id_h); gm_id = gm / np.clip(id_h, 1e-30, None)
    gds = deriv(vd, idvd); gds_sat = float(np.interp(0.0, vd, np.abs(gds)))

    vg_gidl, gidl_i = read_xy("gidl_vg.dat")
    return {
        "Vth_lowVds": vth_low,
        "Vth_highVds": vth_high,
        "DIBL": (vth_low - vth_high) / (cfg.vdd - cfg.vds_low),
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
        "Vdsat_knee": extract_vdsat_knee(vd[::-1], idvd[::-1]),
        "gds_sat": gds_sat,
        "ro_sat": 1.0 / gds_sat,
        "gds_Vds_sens": (float(np.interp(0.0, vd, np.abs(gds))) - float(np.interp(0.2 * cfg.vdd, vd, np.abs(gds)))) / (0.2 * cfg.vdd),
        "CLM_index": (ion - float(np.interp(0.2 * cfg.vdd, vd, idvd))) / ion,
        "Ron_lin_Vdd": cfg.vds_low / idlin,
        "Ron_lin_Vddm": cfg.vds_low / float(np.interp(cfg.vdd - cfg.vds_low, vd_m, idvd_m)),
        "Early_like": (1.0 / gds_sat) * ion,
        "Vth_rolloff_L": interp_x_for_y(vg_sh, id_sh, icrit_short) - vth_high,
        "SS_rolloff_L": ss_short - ss_long,
        "Ion_rolloff_L_ratio": float(np.interp(0.0, vd_s, idvd_s)) / ion,
        "Vth_narrowW": interp_x_for_y(vg_n, id_n, cfg.icrit * (cfg.w_narrow / cfg.l_long)) - vth_high,
        "SS_narrowW": calc_ss(vg_n_low, np.abs(id_n_low), cfg.icrit * (cfg.w_narrow / cfg.l_long)) - ss_long,
        "Ron_narrowW_ratio": (cfg.vds_low / float(np.interp(cfg.vdd - cfg.vds_low, vd_nw, idvd_nw))) / (cfg.vds_low / idlin),
        "GIDL_index": float(np.interp(cfg.vdd + 0.4, vg_gidl, gidl_i)),
        "Junc_leak_index": float(np.interp(cfg.vdd - 0.5, vd_j, id_j)),
    }


def simulate_circuit_metrics(model_path: Path, cfg: BiasConfig, workdir: Path) -> Dict[str, float]:
    inv_deck = f"""
.include '{model_path.resolve().as_posix()}'
.option temp={cfg.temp}
VDD vdd 0 {cfg.vdd}
VIN in 0 PULSE(0 {cfg.vdd} 0 20p 20p 200p 400p)
MP out in vdd vdd sky130_fd_pr__pfet_01v8 l={cfg.l_short} w={cfg.w_wide}
MN out in 0 0 sky130_fd_pr__nfet_01v8 l={cfg.l_short} w={cfg.w_wide}
CLOAD out 0 5f
.tran 1p 2n
.measure tran INV_tpHL trig v(in) val={0.5*cfg.vdd} rise=1 targ v(out) val={0.5*cfg.vdd} fall=1
.measure tran INV_tpLH trig v(in) val={0.5*cfg.vdd} fall=1 targ v(out) val={0.5*cfg.vdd} rise=1
.measure tran INV_tr trig v(out) val={0.1*cfg.vdd} rise=1 targ v(out) val={0.9*cfg.vdd} rise=1
.measure tran INV_tf trig v(out) val={0.9*cfg.vdd} fall=1 targ v(out) val={0.1*cfg.vdd} fall=1
.end
"""
    (workdir / "inv.sp").write_text(inv_deck)
    run_ngspice("inv.sp", workdir)
    inv_log = (workdir / "ngspice.log").read_text(errors="ignore")

    def meas(log: str, key: str) -> float:
        m = re.search(rf"{re.escape(key)}\s*=\s*([-+0-9.eE]+)", log, re.IGNORECASE)
        if not m:
            raise ValueError(f"measure {key} not found")
        return float(m.group(1))

    ro_nodes = [f"n{i}" for i in range(11)]
    inv_chain = []
    for i in range(11):
        inv_chain.append(f"MP{i} {ro_nodes[(i+1)%11]} {ro_nodes[i]} vdd vdd sky130_fd_pr__pfet_01v8 l={cfg.l_short} w={cfg.w_wide}")
        inv_chain.append(f"MN{i} {ro_nodes[(i+1)%11]} {ro_nodes[i]} 0 0 sky130_fd_pr__nfet_01v8 l={cfg.l_short} w={cfg.w_wide}")
    ro_deck = f"""
.include '{model_path.resolve().as_posix()}'
.option temp={cfg.temp}
VDD vdd 0 {cfg.vdd}
{chr(10).join(inv_chain)}
.ic v(n0)=0
.control
tran 1p 8n
wrdata ro_wave.dat time v(n0)
quit
.endc
.end
"""
    (workdir / "ro.sp").write_text(ro_deck)
    run_ngspice("ro.sp", workdir)
    arr = np.loadtxt(workdir / "ro_wave.dat")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    t = arr[:, 0]
    v = arr[:, 1]
    mask = (t >= 2e-9) & (t <= 8e-9)
    tw = t[mask]
    vw = v[mask]
    thr = 0.5 * cfg.vdd
    idx = np.where((vw[:-1] < thr) & (vw[1:] >= thr))[0]
    if len(idx) < 11:
        raise ValueError("RO edge count 부족")
    crossings = []
    for i in idx[:11]:
        t0, t1 = tw[i], tw[i + 1]
        v0, v1 = vw[i], vw[i + 1]
        crossings.append(t0 + (thr - v0) * (t1 - t0) / (v1 - v0))
    periods = np.diff(np.array(crossings[:11]))
    ro_freq = float(1.0 / np.mean(periods[:10]))

    return {
        "RO_freq": ro_freq,
        "INV_tpHL": meas(inv_log, "INV_tpHL"),
        "INV_tpLH": meas(inv_log, "INV_tpLH"),
        "INV_tr": meas(inv_log, "INV_tr"),
        "INV_tf": meas(inv_log, "INV_tf"),
    }


def _parse_meas(log_text: str, key: str) -> float:
    m = re.search(rf"{re.escape(key)}\s*=\s*([-+0-9.eE]+)", log_text, re.IGNORECASE)
    if not m:
        raise ValueError(f"measure {key} not found")
    return float(m.group(1))


def simulate_gate_off_metrics(model_path: Path, cfg: BiasConfig, workdir: Path) -> Dict[str, float]:
    deck = f"""
.include '{model_path.resolve().as_posix()}'
.option temp={cfg.temp}
* NMOS gate off leakage
Vdn dn 0 {cfg.vdd}
Vgn gn 0 0
Vsn sn 0 0
Vbn bn 0 0
MN dn gn sn bn sky130_fd_pr__nfet_01v8 l={cfg.l_long} w={cfg.w_wide}
* PMOS gate off leakage
Vdp dp 0 0
Vgp gp 0 {cfg.vdd}
Vsp sp 0 {cfg.vdd}
Vbp bp 0 {cfg.vdd}
MP dp gp sp bp sky130_fd_pr__pfet_01v8 l={cfg.l_long} w={cfg.w_wide}
.op
.measure op Ig_off_n param='abs(i(Vgn))'
.measure op Ig_off_p param='abs(i(Vgp))'
.end
"""
    (workdir / "ig_off.sp").write_text(deck)
    run_ngspice("ig_off.sp", workdir)
    log = (workdir / "ngspice.log").read_text(errors="ignore")
    return {
        "Ig_off_n": _parse_meas(log, "Ig_off_n"),
        "Ig_off_p": _parse_meas(log, "Ig_off_p"),
    }


def simulate_sram_metrics(model_path: Path, cfg: BiasConfig, workdir: Path) -> Dict[str, float]:
    # SRAM 6T (AX/PD/PU): AX=2.7e-7, PD=4.5e-7, PU=2.1e-7, L=1.5e-7
    ax_w, pd_w, pu_w, lmin = 2.7e-7, 4.5e-7, 2.1e-7, 1.5e-7

    hold_tran = f"""
.include '{model_path.resolve().as_posix()}'
.option temp={cfg.temp}
VDD vdd 0 {cfg.vdd}
VWL wl 0 0
VBL bl 0 {cfg.vdd}
VBLB blb 0 {cfg.vdd}
MP1 q qb vdd vdd sky130_fd_pr__pfet_01v8 l={lmin} w={pu_w}
MP2 qb q vdd vdd sky130_fd_pr__pfet_01v8 l={lmin} w={pu_w}
MN1 q qb 0 0 sky130_fd_pr__nfet_01v8 l={lmin} w={pd_w}
MN2 qb q 0 0 sky130_fd_pr__nfet_01v8 l={lmin} w={pd_w}
MAX1 q wl bl 0 sky130_fd_pr__nfet_01v8 l={lmin} w={ax_w}
MAX2 qb wl blb 0 sky130_fd_pr__nfet_01v8 l={lmin} w={ax_w}
.ic v(q)={cfg.vdd} v(qb)=0
.tran 1p 2n
.measure tran IAVG avg i(VDD) from=0.5n to=2n
.measure tran STBY_power param='{cfg.vdd}*abs(IAVG)'
.end
"""
    (workdir / "sram_hold_tran.sp").write_text(hold_tran)
    run_ngspice("sram_hold_tran.sp", workdir)
    hold_log = (workdir / "ngspice.log").read_text(errors="ignore")

    hold_dc = f"""
.include '{model_path.resolve().as_posix()}'
.option temp={cfg.temp}
VDD vdd 0 {cfg.vdd}
VWL wl 0 0
VBL bl 0 {cfg.vdd}
VBLB blb 0 {cfg.vdd}
VFORCE qb 0 0
MP1 q qb vdd vdd sky130_fd_pr__pfet_01v8 l={lmin} w={pu_w}
MN1 q qb 0 0 sky130_fd_pr__nfet_01v8 l={lmin} w={pd_w}
MAX1 q wl bl 0 sky130_fd_pr__nfet_01v8 l={lmin} w={ax_w}
.control
set wr_singlescale
dc VFORCE 0 {cfg.vdd} 0.01
wrdata sram_hold_vtc.dat v(qb) v(q)
quit
.endc
.end
"""
    (workdir / "sram_hold_dc.sp").write_text(hold_dc)
    run_ngspice("sram_hold_dc.sp", workdir)
    hdf = load_wrdata(workdir / "sram_hold_vtc.dat")
    hqb = hdf["x_vec0"].to_numpy()
    hq = hdf["y_vec1"].to_numpy()
    hold_snm = float(np.min(np.abs(hq - hqb)) / 2.0)

    read_dc = f"""
.include '{model_path.resolve().as_posix()}'
.option temp={cfg.temp}
VDD vdd 0 {cfg.vdd}
VWL wl 0 {cfg.vdd}
VBL bl 0 {cfg.vdd - 0.05}
VBLB blb 0 {cfg.vdd}
VFORCE qb 0 0
MP1 q qb vdd vdd sky130_fd_pr__pfet_01v8 l={lmin} w={pu_w}
MN1 q qb 0 0 sky130_fd_pr__nfet_01v8 l={lmin} w={pd_w}
MAX1 q wl bl 0 sky130_fd_pr__nfet_01v8 l={lmin} w={ax_w}
.control
set wr_singlescale
dc VFORCE 0 {cfg.vdd} 0.01
wrdata sram_read_vtc.dat v(qb) v(q)
quit
.endc
.end
"""
    (workdir / "sram_read_dc.sp").write_text(read_dc)
    run_ngspice("sram_read_dc.sp", workdir)
    rdf = load_wrdata(workdir / "sram_read_vtc.dat")
    rqb = rdf["x_vec0"].to_numpy()
    rq = rdf["y_vec1"].to_numpy()
    read_snm = float(np.min(np.abs(rq - rqb)) / 2.0)

    return {
        "STBY_power": _parse_meas(hold_log, "STBY_power"),
        "SRAM_hold_SNM": hold_snm,
        "SRAM_read_margin": read_snm,
    }


def simulate_sram_ion_ratios(model_path: Path, cfg: BiasConfig, workdir: Path) -> Dict[str, float]:
    ax_w, pd_w, pu_w, lmin = 2.7e-7, 4.5e-7, 2.1e-7, 1.5e-7
    deck = f"""
.include '{model_path.resolve().as_posix()}'
.option temp={cfg.temp}
* AX (nMOS)
Vdax dax 0 {cfg.vdd}
Vgax gax 0 {cfg.vdd}
Vsax sax 0 0
Vbax bax 0 0
MAX dax gax sax bax sky130_fd_pr__nfet_01v8 l={lmin} w={ax_w}
* PD (nMOS)
Vdpd dpd 0 {cfg.vdd}
Vgpd gpd 0 {cfg.vdd}
Vspd spd 0 0
Vbpd bpd 0 0
MPD dpd gpd spd bpd sky130_fd_pr__nfet_01v8 l={lmin} w={pd_w}
* PU (pMOS)
Vdpu dpu 0 0
Vgpu gpu 0 0
Vspu spu 0 {cfg.vdd}
Vbpu bpu 0 {cfg.vdd}
MPU dpu gpu spu bpu sky130_fd_pr__pfet_01v8 l={lmin} w={pu_w}
.op
.measure op IAX param='abs(i(Vdax))'
.measure op IPD param='abs(i(Vdpd))'
.measure op IPU param='abs(i(Vdpu))'
.measure op AX_PD_ratio_Ion param='IAX/IPD'
.measure op PD_PU_ratio_Ion param='IPD/IPU'
.end
"""
    (workdir / "sram_ratio.sp").write_text(deck)
    run_ngspice("sram_ratio.sp", workdir)
    log = (workdir / "ngspice.log").read_text(errors="ignore")
    return {
        "AX_PD_ratio_Ion": _parse_meas(log, "AX_PD_ratio_Ion"),
        "PD_PU_ratio_Ion": _parse_meas(log, "PD_PU_ratio_Ion"),
    }



def collect_all_metrics(model_path: Path, cfg: BiasConfig, workdir: Path) -> Dict[str, float]:
    n_dir = workdir / "n"
    p_dir = workdir / "p"
    c_dir = workdir / "c"
    g_dir = workdir / "g"
    s_dir = workdir / "s"
    sr_dir = workdir / "sr"
    n_dir.mkdir(parents=True, exist_ok=True)
    p_dir.mkdir(parents=True, exist_ok=True)
    c_dir.mkdir(parents=True, exist_ok=True)
    g_dir.mkdir(parents=True, exist_ok=True)
    s_dir.mkdir(parents=True, exist_ok=True)
    sr_dir.mkdir(parents=True, exist_ok=True)

    n = simulate_metrics(model_path, cfg, n_dir)
    p = simulate_metrics_pfet(model_path, cfg, p_dir)
    c = simulate_circuit_metrics(model_path, cfg, c_dir)
    g = simulate_gate_off_metrics(model_path, cfg, g_dir)
    s = simulate_sram_metrics(model_path, cfg, s_dir)
    sr = simulate_sram_ion_ratios(model_path, cfg, sr_dir)

    m = {k: np.nan for k in METRICS_ORDER}
    m.update({
        "RO_freq": c.get("RO_freq", np.nan),
        "INV_tpHL": c.get("INV_tpHL", np.nan),
        "INV_tpLH": c.get("INV_tpLH", np.nan),
        "INV_tr": c.get("INV_tr", np.nan),
        "INV_tf": c.get("INV_tf", np.nan),
        "Ion_n": n.get("Ion_Vdd", np.nan),
        "Ion_p": p.get("Ion_Vdd", np.nan),
        "Idlin_n": n.get("Idlin_Vdd", np.nan),
        "Idlin_p": p.get("Idlin_Vdd", np.nan),
        "Ron_n_Vdd": n.get("Ron_lin_Vdd", np.nan),
        "Ron_n_Vddm": n.get("Ron_lin_Vddm", np.nan),
        "Ron_p_Vdd": p.get("Ron_lin_Vdd", np.nan),
        "Ron_p_Vddm": p.get("Ron_lin_Vddm", np.nan),
        "Ioff_n": n.get("Ioff_ref", np.nan),
        "Ioff_p": p.get("Ioff_ref", np.nan),
        "ro_n": n.get("ro_sat", np.nan),
        "ro_p": p.get("ro_sat", np.nan),
        "CLM_index_n": n.get("CLM_index", np.nan),
        "CLM_index_p": p.get("CLM_index", np.nan),
        "GIDL_n": n.get("GIDL_index", np.nan),
        "GIDL_p": p.get("GIDL_index", np.nan),
        "Junc_leak_n": n.get("Junc_leak_index", np.nan),
        "Junc_leak_p": p.get("Junc_leak_index", np.nan),
        "Ig_off_n": g.get("Ig_off_n", np.nan),
        "Ig_off_p": g.get("Ig_off_p", np.nan),
        "Ion_rolloff_L_ratio_n": n.get("Ion_rolloff_L_ratio", np.nan),
        "Ron_narrowW_ratio_n": n.get("Ron_narrowW_ratio", np.nan),
        "Ion_short_narrow_ratio_n": n.get("Ion_short_narrow_ratio", np.nan),
        "STBY_power": s.get("STBY_power", np.nan),
        "SRAM_hold_SNM": s.get("SRAM_hold_SNM", np.nan),
        "SRAM_read_margin": s.get("SRAM_read_margin", np.nan),
        "AX_PD_ratio_Ion": sr.get("AX_PD_ratio_Ion", np.nan),
        "PD_PU_ratio_Ion": sr.get("PD_PU_ratio_Ion", np.nan),
    })
    return m


def five_point_central(fm1: float, fm05: float, fp05: float, fp1: float, h: float = 0.5) -> float:
    """5-point central derivative around alpha=0 with samples at -1, -0.5, 0.5, 1.0."""
    return (fm1 - 8 * fm05 + 8 * fp05 - fp1) / (12 * h)


def make_scope_doc(path: Path) -> None:
    with path.open("w") as f:
        f.write("# v2 확정 항목\n\n## Tier1+Tier2 Metrics (simulation-extracted)\n")
        for m in METRICS_ORDER:
            f.write(f"- {m}\n")
        f.write("\n## Tier3 Sweep Parameters\n")
        f.write("- device별 TT/SS/FF 코너 비교에서 delta가 존재하는 파라미터만 자동 선택\n")
        f.write("- alpha levels: -1.0, -0.5, 0.0, 0.5, 1.0\n")


def run_sensitivity_for_device(
    device_name: str,
    tt_text: str,
    ss_text: str,
    ff_text: str,
    other_tt_text: str,
    cfg: BiasConfig,
    outdir: Path,
    progress: ProgressTracker,
) -> pd.DataFrame:
    rows = []
    raw_rows: List[Dict[str, float | str]] = []
    baseline_rows: List[Dict[str, float | str]] = []
    delta_params = get_delta_params(tt_text, ss_text, ff_text)
    progress.log(f"{device_name} baseline(TT) simulation start")
    with tempfile.TemporaryDirectory() as td:
        base_model = Path(td) / f"{device_name}_tt.spice"
        base_model.write_text(tt_text + "\n" + other_tt_text)
        base_metrics = collect_all_metrics(base_model, cfg, Path(td))
    baseline_rows.append({"device": device_name, "corner": "TT", "alpha": 0.0, **base_metrics})
    progress.advance(f"{device_name} baseline(TT) simulation done")

    total_params = len(delta_params)
    for idx_param, param in enumerate(delta_params, start=1):
        param_pct = 100.0 * idx_param / max(total_params, 1)
        print(f"[progress][{device_name} item {idx_param}/{total_params} {param_pct:5.1f}%] param={param}", flush=True)
        row = {"param": param}
        for m in METRICS_ORDER:
            row[f"{m}_sens_SS_5pt"] = np.nan
            row[f"{m}_sens_FF_5pt"] = np.nan
            row[f"{m}_sens_avg_5pt"] = np.nan

        tt_vals = get_param_values(tt_text, param)
        p0 = float(np.mean(tt_vals))
        corners = (("SS", ss_text), ("FF", ff_text))
        for idx_corner, (corner_name, corner_text) in enumerate(corners, start=1):
            print(f"[progress][{device_name} item {idx_param}/{total_params}] corner {idx_corner}/{len(corners)}={corner_name}", flush=True)
            c_vals = get_param_values(corner_text, param)
            n = min(len(tt_vals), len(c_vals))
            if n == 0:
                raise ValueError(f"{param} corner 값이 없습니다.")
            dp = float(np.mean(np.array(c_vals[:n]) - np.array(tt_vals[:n])))

            alpha_points = (-1.0, -0.5, 0.0, 0.5, 1.0)
            metric_at_alpha: Dict[float, Dict[str, float]] = {}
            for idx_alpha, alpha in enumerate(alpha_points, start=1):
                print(
                    f"[progress][{device_name} item {idx_param}/{total_params}] corner={corner_name} alpha {idx_alpha}/{len(alpha_points)}={alpha}",
                    flush=True,
                )
                if alpha == 0.0:
                    metric_at_alpha[alpha] = dict(base_metrics)
                else:
                    patched = patch_param_alpha(tt_text, corner_text, param, alpha)
                    with tempfile.TemporaryDirectory() as td:
                        mp = Path(td) / f"{device_name}_{param}_{corner_name}_{alpha}.spice"
                        mp.write_text(patched + "\n" + other_tt_text)
                        metric_at_alpha[alpha] = collect_all_metrics(mp, cfg, Path(td))
                raw_rows.append(
                    {
                        "device": device_name,
                        "param": param,
                        "corner": corner_name,
                        "alpha": alpha,
                        **metric_at_alpha[alpha],
                    }
                )
                progress.advance(f"{device_name} param={param} corner={corner_name} alpha={alpha} done")

            for m in METRICS_ORDER:
                dfdalpha = five_point_central(
                    metric_at_alpha[-1.0][m],
                    metric_at_alpha[-0.5][m],
                    metric_at_alpha[0.5][m],
                    metric_at_alpha[1.0][m],
                )
                m0 = base_metrics[m]
                if dp == 0 or m0 == 0:
                    row[f"{m}_sens_{corner_name}_5pt"] = np.nan
                else:
                    dmdp = dfdalpha / dp
                    row[f"{m}_sens_{corner_name}_5pt"] = (p0 / m0) * dmdp

        for m in METRICS_ORDER:
            v = [row[f"{m}_sens_SS_5pt"], row[f"{m}_sens_FF_5pt"]]
            row[f"{m}_sens_avg_5pt"] = float(np.mean(v))
        rows.append(row)

    df = pd.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / f"sensitivity_matrix_central5_{device_name}.csv", index=False)
    (outdir / f"tt_metrics_{device_name}.json").write_text(json.dumps(base_metrics, indent=2))
    pd.DataFrame(raw_rows).to_csv(outdir / f"metrics_raw_{device_name}.csv", index=False)
    pd.DataFrame(baseline_rows).to_csv(outdir / f"metrics_baseline_{device_name}.csv", index=False)
    return df




def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spice-zip", type=Path, default=Path("spice.zip"))
    ap.add_argument("--outdir", type=Path, default=Path("out_sensitivity"))
    ap.add_argument("--tt-file", default="sky130_fd_pr__nfet_01v8__tt.pm3.spice")
    ap.add_argument("--ss-file", default="sky130_fd_pr__nfet_01v8__ss.pm3.spice")
    ap.add_argument("--ff-file", default="sky130_fd_pr__nfet_01v8__ff.pm3.spice")
    ap.add_argument("--tt-file-p", default="sky130_fd_pr__pfet_01v8__tt.pm3.spice")
    ap.add_argument("--ss-file-p", default="sky130_fd_pr__pfet_01v8__ss.pm3.spice")
    ap.add_argument("--ff-file-p", default="sky130_fd_pr__pfet_01v8__ff.pm3.spice")
    args = ap.parse_args()

    with zipfile.ZipFile(args.spice_zip) as z:
        n_tt_text = z.read(args.tt_file).decode("utf-8", errors="ignore")
        n_ss_text = z.read(args.ss_file).decode("utf-8", errors="ignore")
        n_ff_text = z.read(args.ff_file).decode("utf-8", errors="ignore")
        p_tt_text = z.read(args.tt_file_p).decode("utf-8", errors="ignore")
        p_ss_text = z.read(args.ss_file_p).decode("utf-8", errors="ignore")
        p_ff_text = z.read(args.ff_file_p).decode("utf-8", errors="ignore")

    n_tt_text = inject_missing_monte_params(n_tt_text)
    n_ss_text = inject_missing_monte_params(n_ss_text)
    n_ff_text = inject_missing_monte_params(n_ff_text)
    p_tt_text = inject_missing_monte_params(p_tt_text)
    p_ss_text = inject_missing_monte_params(p_ss_text)
    p_ff_text = inject_missing_monte_params(p_ff_text)
    cfg = BiasConfig()
    cfg.temp = get_tnom_celsius(n_tt_text, fallback_c=27.0)
    print(f"[config] temp set from TNOM: {cfg.temp}C", flush=True)

    n_params = get_delta_params(n_tt_text, n_ss_text, n_ff_text)
    p_params = get_delta_params(p_tt_text, p_ss_text, p_ff_text)
    total_units = 2 + (len(n_params) * 2 * 5) + (len(p_params) * 2 * 5)
    progress = ProgressTracker(total_units=total_units)

    n_df = run_sensitivity_for_device("nfet", n_tt_text, n_ss_text, n_ff_text, p_tt_text, cfg, args.outdir, progress)
    p_df = run_sensitivity_for_device("pfet", p_tt_text, p_ss_text, p_ff_text, n_tt_text, cfg, args.outdir, progress)
    make_scope_doc(args.outdir / "scope_32_metrics_150_params.md")

    print(n_df[["param", f"{METRICS_ORDER[0]}_sens_avg_5pt"]].head())
    print(p_df[["param", f"{METRICS_ORDER[0]}_sens_avg_5pt"]].head())


if __name__ == "__main__":
    main()
