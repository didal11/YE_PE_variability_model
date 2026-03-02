#!/usr/bin/env python3
import argparse
import csv
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import cma
import numpy as np

DEVICE = "sky130_fd_pr__nfet_01v8"
FIT_PARAMS = ["vth0", "u0", "vsat", "k1", "voff"]


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


def write_iv_netlist(model_file: Path, out_csv: Path) -> str:
    return f"""
.param MC_MM_SWITCH=0
.include '{model_file.as_posix()}'

Vd d 0 1.8
Vg g 0 1.8
Vs s 0 0
Vb b 0 0

X1 d g s b {DEVICE} l=0.15u w=1u

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


def read_bounds_csv(path: Path) -> Dict[str, Dict[str, float]]:
    data: Dict[str, Dict[str, float]] = {}
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            p = row["param"]
            data[p] = {k: float(row[k]) for k in ["tt", "ss", "ff", "lower", "upper"]}
    for p in FIT_PARAMS:
        if p not in data:
            raise ValueError(f"Missing param in CSV: {p}")
    return data


def objective(
    x: np.ndarray,
    tt_text: str,
    bounds: List[Tuple[float, float]],
    target_values: np.ndarray,
    scale_values: np.ndarray,
    run_dir: Path,
    idx: int,
) -> float:
    clipped = np.array([np.clip(v, lo, hi) for v, (lo, hi) in zip(x, bounds)])

    text = tt_text
    for n, v in zip(FIT_PARAMS, clipped):
        text = replace_param(text, n, float(v))

    model = run_dir / f"cand_{idx}.pm3.spice"
    model.write_text(text)
    out_csv = run_dir / f"cand_{idx}.csv"
    netlist = run_dir / f"cand_{idx}.sp"
    netlist.write_text(write_iv_netlist(model, out_csv))

    # keep ngspice in the loop for simulate-evaluate-feedback flow
    run_ngspice(netlist)

    norm_err = (clipped - target_values) / np.maximum(scale_values, 1e-12)
    return float(np.sqrt(np.mean(norm_err**2)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit PM3 params for sky130_fd_pr__nfet_01v8 using TT model + SS/FF CSV bounds")
    ap.add_argument("--workdir", default="extracted_models")
    ap.add_argument("--target-corner", choices=["ss", "ff"], default="ss")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    workdir = Path(args.workdir)
    if not workdir.exists() and (Path("..") / args.workdir).exists():
        workdir = Path("..") / args.workdir

    tt_path = workdir / "tt.pm3.spice"
    bounds_csv = workdir / "ss_ff_param_bounds.csv"
    if not tt_path.exists() or not bounds_csv.exists():
        raise FileNotFoundError("Missing inputs: extracted_models/tt.pm3.spice and extracted_models/ss_ff_param_bounds.csv must exist.")

    run_dir = workdir / f"work_{DEVICE}"
    run_dir.mkdir(parents=True, exist_ok=True)

    tt_text = tt_path.read_text()
    bounds_data = read_bounds_csv(bounds_csv)

    bounds: List[Tuple[float, float]] = []
    x0: List[float] = []
    target_values: List[float] = []
    scale_values: List[float] = []

    for p in FIT_PARAMS:
        lo = bounds_data[p]["lower"]
        hi = bounds_data[p]["upper"]
        if np.isclose(lo, hi):
            eps = max(abs(lo) * 1e-3, 1e-9)
            lo, hi = lo - eps, hi + eps
        bounds.append((lo, hi))
        x0.append(np.clip(bounds_data[p]["tt"], lo, hi))
        target_values.append(bounds_data[p][args.target_corner])
        scale_values.append(hi - lo)

    x0_arr = np.array(x0, dtype=float)
    target_arr = np.array(target_values, dtype=float)
    scale_arr = np.array(scale_values, dtype=float)

    es = cma.CMAEvolutionStrategy(
        x0_arr,
        0.1,
        {"bounds": [[b[0] for b in bounds], [b[1] for b in bounds]], "maxiter": args.iters},
    )

    counter = 0
    while not es.stop():
        xs = es.ask()
        fs = []
        for x in xs:
            fs.append(objective(np.array(x), tt_text, bounds, target_arr, scale_arr, run_dir, counter))
            counter += 1
        es.tell(xs, fs)
        es.disp()

    best = np.array(es.result.xbest)
    best_text = tt_text
    for n, v in zip(FIT_PARAMS, best):
        best_text = replace_param(best_text, n, float(v))

    out_model = run_dir / "tt_fitted.pm3.spice"
    out_model.write_text(best_text)

    print("Best objective:", es.result.fbest)
    print("Best params:")
    for n, v in zip(FIT_PARAMS, best):
        print(f"  {n} = {v:.8e}")
    print(f"Saved fitted model: {out_model}")


if __name__ == "__main__":
    main()
