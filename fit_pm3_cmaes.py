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
DEFAULT_FIT_PARAMS = ["vth0", "u0", "vsat", "k1", "voff"]


def parse_all_tt_params(model_text: str) -> List[str]:
    pat = re.compile(r"^\+\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=", re.MULTILINE)
    seen = set()
    params = []
    for name in pat.findall(model_text):
        if name not in seen:
            seen.add(name)
            params.append(name)
    return params


def rank_params(params: List[str]) -> List[str]:
    primary_order = [
        "vth0", "u0", "vsat", "k1", "voff", "k2", "eta0", "nfactor", "dvt0", "dvt1", "dvt2", "rdsw",
        "ua", "ub", "uc", "pclm", "pdiblc1", "pdiblc2", "a0", "ags",
    ]
    primary_rank = {p: i for i, p in enumerate(primary_order)}

    def score(name: str) -> Tuple[int, int, str]:
        if name in primary_rank:
            return (0, primary_rank[name], name)
        mobility_like = int(any(k in name for k in ["u", "vsat", "ua", "ub", "uc"]))
        vth_like = int(any(k in name for k in ["vth", "voff", "dvt", "eta", "nfac"]))
        channel_like = int(any(k in name for k in ["pclm", "pdibl", "rds", "lambda"]))
        class_rank = - (mobility_like + vth_like + channel_like)
        return (1, class_rank, name)

    return sorted(params, key=score)


def select_params_tk(sorted_params: List[str], default_selected: List[str], available_bounds: set[str]) -> List[str]:
    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.title("PM3 Parameter Selection (Tk)")
    root.geometry("560x720")

    info = tk.Label(
        root,
        text="중요도 순 파라미터 목록입니다. 체크된 항목만 피팅됩니다.\n(빨간 항목은 ss_ff_param_bounds.csv에 없어 선택 불가)",
        justify="left",
        anchor="w",
    )
    info.pack(fill="x", padx=8, pady=8)

    frame = tk.Frame(root)
    frame.pack(fill="both", expand=True, padx=8, pady=4)

    canvas = tk.Canvas(frame)
    scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    inner = tk.Frame(canvas)

    inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    checks: Dict[str, tk.BooleanVar] = {}
    for i, p in enumerate(sorted_params, 1):
        v = tk.BooleanVar(value=(p in default_selected))
        checks[p] = v
        enabled = p in available_bounds
        cb = tk.Checkbutton(inner, text=f"{i:03d}. {p}", variable=v, anchor="w", justify="left")
        if not enabled:
            cb.configure(state="disabled", fg="red")
            v.set(False)
        cb.pack(fill="x", padx=4, pady=1)

    selected: List[str] = []

    def on_ok() -> None:
        selected.extend([p for p, v in checks.items() if v.get()])
        if not selected:
            messagebox.showerror("Error", "최소 1개 파라미터를 선택하세요.")
            selected.clear()
            return
        root.destroy()

    def on_cancel() -> None:
        root.destroy()

    btns = tk.Frame(root)
    btns.pack(fill="x", padx=8, pady=8)
    tk.Button(btns, text="확인", command=on_ok).pack(side="left")
    tk.Button(btns, text="취소", command=on_cancel).pack(side="left", padx=8)

    root.mainloop()
    return selected


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
    return data


def objective(
    x: np.ndarray,
    tt_text: str,
    fit_params: List[str],
    bounds: List[Tuple[float, float]],
    target_values: np.ndarray,
    scale_values: np.ndarray,
    run_dir: Path,
    idx: int,
) -> float:
    clipped = np.array([np.clip(v, lo, hi) for v, (lo, hi) in zip(x, bounds)])

    text = tt_text
    for n, v in zip(fit_params, clipped):
        text = replace_param(text, n, float(v))

    model = run_dir / f"cand_{idx}.pm3.spice"
    model.write_text(text)
    out_csv = run_dir / f"cand_{idx}.csv"
    netlist = run_dir / f"cand_{idx}.sp"
    netlist.write_text(write_iv_netlist(model, out_csv))

    run_ngspice(netlist)

    norm_err = (clipped - target_values) / np.maximum(scale_values, 1e-12)
    return float(np.sqrt(np.mean(norm_err**2)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit PM3 params for sky130_fd_pr__nfet_01v8 using TT model + SS/FF CSV bounds")
    ap.add_argument("--workdir", default=".")
    ap.add_argument("--target-corner", choices=["ss", "ff"], default="ss")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--params", default=",".join(DEFAULT_FIT_PARAMS), help="Comma-separated params to fit")
    ap.add_argument("--ui", action="store_true", help="Open Tk UI for parameter checkbox selection")
    args = ap.parse_args()

    workdir = Path(args.workdir)

    tt_path = workdir / "tt.pm3.spice"
    bounds_csv = workdir / "ss_ff_param_bounds.csv"
    if not tt_path.exists() or not bounds_csv.exists():
        raise FileNotFoundError("Missing inputs: tt.pm3.spice and ss_ff_param_bounds.csv must exist in --workdir.")

    run_dir = workdir / f"work_{DEVICE}"
    run_dir.mkdir(parents=True, exist_ok=True)

    tt_text = tt_path.read_text()
    all_params = rank_params(parse_all_tt_params(tt_text))
    bounds_data = read_bounds_csv(bounds_csv)
    available_bounds = set(bounds_data.keys())

    if args.ui:
        try:
            fit_params = select_params_tk(all_params, DEFAULT_FIT_PARAMS, available_bounds)
        except Exception as e:
            raise RuntimeError("Tk UI failed to open. Check desktop/display environment.") from e
        if not fit_params:
            raise RuntimeError("No parameters selected in UI.")
    else:
        fit_params = [p.strip() for p in args.params.split(",") if p.strip()]

    missing = [p for p in fit_params if p not in available_bounds]
    if missing:
        raise ValueError(f"Selected params not present in ss_ff_param_bounds.csv: {missing}")

    bounds: List[Tuple[float, float]] = []
    x0: List[float] = []
    target_values: List[float] = []
    scale_values: List[float] = []

    for p in fit_params:
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
            fs.append(objective(np.array(x), tt_text, fit_params, bounds, target_arr, scale_arr, run_dir, counter))
            counter += 1
        es.tell(xs, fs)
        es.disp()

    best = np.array(es.result.xbest)
    best_text = tt_text
    for n, v in zip(fit_params, best):
        best_text = replace_param(best_text, n, float(v))

    out_model = run_dir / "tt_fitted.pm3.spice"
    out_model.write_text(best_text)

    print("Best objective:", es.result.fbest)
    print("Selected params:", ", ".join(fit_params))
    print("Best params:")
    for n, v in zip(fit_params, best):
        print(f"  {n} = {v:.8e}")
    print(f"Saved fitted model: {out_model}")


if __name__ == "__main__":
    main()
