#!/usr/bin/env python3
import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cma
import numpy as np

FIT_PARAMS = ["vth0", "u0", "vsat", "k1", "voff"]


@dataclass
class ModelCorner:
    name: str
    text: str
    values: Dict[str, float]


def parse_first_param_value(model_text: str, param: str) -> float:
    pat = re.compile(rf"^\+\s*{re.escape(param)}\s*=\s*([^\s]+)", re.MULTILINE)
    m = pat.search(model_text)
    if not m:
        raise ValueError(f"Cannot find parameter '{param}'")
    expr = m.group(1).strip("{}")
    # expression may contain mismatch term for some params. take leading numeric token.
    num = re.match(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", expr)
    if not num:
        raise ValueError(f"Cannot parse numeric value for '{param}' from '{expr}'")
    return float(num.group(0))




def strip_mismatch_terms(model_text: str) -> str:
    model_text = re.sub(r"\+\s*MC_MM_SWITCH\*AGAUSS\(0,1\.0,1\)\*\([^}]+\)", "", model_text)
    model_text = re.sub(r"\+\s*mc_mm_switch\*agauss\(0,1\.0,1\)\*\([^}]+\)", "", model_text)
    return model_text
def load_corner(path: Path, name: str) -> ModelCorner:
    text = strip_mismatch_terms(path.read_text())
    values = {k: parse_first_param_value(text, k) for k in FIT_PARAMS}
    return ModelCorner(name=name, text=text, values=values)


def replace_param(model_text: str, param: str, new_value: float) -> str:
    pat = re.compile(rf"^(\+\s*{re.escape(param)}\s*=\s*)([^\n]+)$", re.MULTILINE)

    def repl(m: re.Match) -> str:
        rhs = m.group(2)
        rhs_new = re.sub(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", f"{new_value:.8e}", rhs, count=1)
        return m.group(1) + rhs_new

    out, n = pat.subn(repl, model_text, count=1)
    if n != 1:
        raise ValueError(f"Failed to replace '{param}'")
    return out


def write_iv_netlist(model_file: Path, device: str, out_csv: Path) -> str:
    return f"""
.param MC_MM_SWITCH=0
.include '{model_file.as_posix()}'

Vd d 0 1.8
Vg g 0 1.8
Vs s 0 0
Vb b 0 0

X1 d g s b {device} l=0.15u w=1u

.control
set wr_vecnames
set wr_singlescale
dc Vd 0 1.8 0.02
wrdata {out_csv.as_posix()} -I(Vd)
quit
.endc
.end
""".strip() + "\n"


def run_ngspice(netlist: Path) -> None:
    subprocess.run(["ngspice", "-b", "-o", str(netlist.with_suffix('.log')), str(netlist)], check=True)


def load_iv_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, skiprows=1)
    x = data[:, 0]
    y = data[:, 1]
    return x, y


def objective(x: np.ndarray, tt_text: str, names: List[str], bounds: List[Tuple[float, float]],
              workdir: Path, device: str, target_iv: Tuple[np.ndarray, np.ndarray], idx: int) -> float:
    clipped = np.array([np.clip(v, lo, hi) for v, (lo, hi) in zip(x, bounds)])
    text = tt_text
    for n, v in zip(names, clipped):
        text = replace_param(text, n, float(v))

    model = workdir / f"cand_{idx}.pm3.spice"
    model.write_text(text)
    out_csv = workdir / f"cand_{idx}.csv"
    netlist = workdir / f"cand_{idx}.sp"
    netlist.write_text(write_iv_netlist(model, device, out_csv))
    run_ngspice(netlist)

    vg, id_ = load_iv_csv(out_csv)
    tvg, tid = target_iv
    if vg.shape != tvg.shape:
        return 1e9
    err = np.sqrt(np.mean((np.log10(np.abs(id_) + 1e-15) - np.log10(np.abs(tid) + 1e-15)) ** 2))
    return float(err)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit 5 key PM3 params using CMA-ES + ngspice")
    ap.add_argument("--workdir", default="extracted_models")
    ap.add_argument("--device", required=True)
    ap.add_argument("--target-corner", choices=["ss", "ff"], default="ss")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    run_dir = workdir / f"work_{args.device}"
    run_dir.mkdir(exist_ok=True)

    tt = load_corner(workdir / f"{args.device}__tt.pm3.spice", "tt")
    ss = load_corner(workdir / f"{args.device}__ss.pm3.spice", "ss")
    ff = load_corner(workdir / f"{args.device}__ff.pm3.spice", "ff")
    target = ss if args.target_corner == "ss" else ff

    bounds = []
    for k in FIT_PARAMS:
        lo = min(ss.values[k], ff.values[k])
        hi = max(ss.values[k], ff.values[k])
        if np.isclose(lo, hi):
            eps = max(abs(lo) * 1e-3, 1e-9)
            lo, hi = lo - eps, hi + eps
        bounds.append((lo, hi))
    x0 = np.array([np.clip(tt.values[k], bounds[i][0], bounds[i][1]) for i, k in enumerate(FIT_PARAMS)], dtype=float)

    target_model = run_dir / f"target_{target.name}.pm3.spice"
    target_model.write_text(target.text)
    target_csv = run_dir / f"target_{target.name}.csv"
    target_sp = run_dir / f"target_{target.name}.sp"
    target_sp.write_text(write_iv_netlist(target_model, args.device, target_csv))
    run_ngspice(target_sp)
    target_iv = load_iv_csv(target_csv)

    es = cma.CMAEvolutionStrategy(x0, 0.1, {"bounds": [[b[0] for b in bounds], [b[1] for b in bounds]], "maxiter": args.iters})
    counter = 0
    while not es.stop():
        xs = es.ask()
        fs = []
        for x in xs:
            fs.append(objective(np.array(x), tt.text, FIT_PARAMS, bounds, run_dir, args.device, target_iv, counter))
            counter += 1
        es.tell(xs, fs)
        es.disp()

    result = es.result
    best = np.array(result.xbest)
    best_text = tt.text
    for n, v in zip(FIT_PARAMS, best):
        best_text = replace_param(best_text, n, float(v))

    out_model = run_dir / f"{args.device}__tt_fitted.pm3.spice"
    out_model.write_text(best_text)
    print("Best objective:", result.fbest)
    print("Best params:")
    for n, v in zip(FIT_PARAMS, best):
        print(f"  {n} = {v:.8e}")
    print(f"Saved fitted model: {out_model}")


if __name__ == "__main__":
    main()
