#!/usr/bin/env python3
import argparse
import csv
import re
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cma
import numpy as np

DEVICE = "sky130_fd_pr__nfet_01v8"
DEFAULT_FIT_PARAMS = ["vth0", "u0", "vsat", "k1", "voff"]
DEFAULT_RAW_ZIP = "skywater-pdk-sky130-raw-data-main.zip"


@dataclass
class RawDataset:
    key: str
    idvg_path: str
    idvd_path: str | None


def discover_raw_datasets(raw_zip: Path) -> List[RawDataset]:
    if not raw_zip.exists():
        return []

    idvg_pat = re.compile(
        r".*/cells/nfet_01v8/sky130_fd_pr__nfet_01v8_(w.+\(.*_IDVG(?:_D\d+)?\))\.mdm$"
    )
    idvd_pat = re.compile(
        r".*/cells/nfet_01v8/sky130_fd_pr__nfet_01v8_(w.+\(.*_IDVD(?:_D\d+)?\))\.mdm$"
    )

    idvg_map: Dict[str, str] = {}
    idvd_map: Dict[str, str] = {}

    with zipfile.ZipFile(raw_zip, "r") as zf:
        for n in zf.namelist():
            m = idvg_pat.match(n)
            if m:
                key = m.group(1).replace("_IDVG", "")
                idvg_map[key] = n
                continue
            m = idvd_pat.match(n)
            if m:
                key = m.group(1).replace("_IDVD", "")
                idvd_map[key] = n

    keys = sorted(set(idvg_map.keys()) | set(idvd_map.keys()))
    out: List[RawDataset] = []
    for k in keys:
        if k in idvg_map:  # 최소 IDVG 있는 항목만 사용
            out.append(RawDataset(key=k, idvg_path=idvg_map[k], idvd_path=idvd_map.get(k)))
    return out


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
        class_rank = -(mobility_like + vth_like + channel_like)
        return (1, class_rank, name)

    return sorted(params, key=score)


def select_with_tk(
    sorted_params: List[str],
    default_selected: List[str],
    available_bounds: set[str],
    raw_datasets: List[RawDataset],
) -> Tuple[List[str], List[str], int]:
    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.title("PM3 Fit Selection")
    root.geometry("980x760")

    top = tk.Label(
        root,
        text="좌측: 피팅 파라미터 선택 / 우측: 로우데이터(mdm) 선택 / 하단: 데이터별 반복 횟수",
        anchor="w",
        justify="left",
    )
    top.pack(fill="x", padx=8, pady=8)

    cols = tk.Frame(root)
    cols.pack(fill="both", expand=True, padx=8, pady=4)

    # params panel
    p_frame = tk.LabelFrame(cols, text="파라미터(중요도 순)")
    p_frame.pack(side="left", fill="both", expand=True, padx=4)
    p_canvas = tk.Canvas(p_frame)
    p_scroll = tk.Scrollbar(p_frame, orient="vertical", command=p_canvas.yview)
    p_inner = tk.Frame(p_canvas)
    p_inner.bind("<Configure>", lambda e: p_canvas.configure(scrollregion=p_canvas.bbox("all")))
    p_canvas.create_window((0, 0), window=p_inner, anchor="nw")
    p_canvas.configure(yscrollcommand=p_scroll.set)
    p_canvas.pack(side="left", fill="both", expand=True)
    p_scroll.pack(side="right", fill="y")

    p_vars: Dict[str, tk.BooleanVar] = {}
    for i, p in enumerate(sorted_params, 1):
        v = tk.BooleanVar(value=(p in default_selected and p in available_bounds))
        p_vars[p] = v
        cb = tk.Checkbutton(p_inner, text=f"{i:03d}. {p}", variable=v, anchor="w")
        if p not in available_bounds:
            cb.configure(state="disabled", fg="red")
        cb.pack(fill="x", padx=3, pady=1)

    # raw data panel
    d_frame = tk.LabelFrame(cols, text=f"로우데이터 목록 ({len(raw_datasets)}개)")
    d_frame.pack(side="left", fill="both", expand=True, padx=4)
    d_canvas = tk.Canvas(d_frame)
    d_scroll = tk.Scrollbar(d_frame, orient="vertical", command=d_canvas.yview)
    d_inner = tk.Frame(d_canvas)
    d_inner.bind("<Configure>", lambda e: d_canvas.configure(scrollregion=d_canvas.bbox("all")))
    d_canvas.create_window((0, 0), window=d_inner, anchor="nw")
    d_canvas.configure(yscrollcommand=d_scroll.set)
    d_canvas.pack(side="left", fill="both", expand=True)
    d_scroll.pack(side="right", fill="y")

    d_vars: Dict[str, tk.BooleanVar] = {}
    for i, d in enumerate(raw_datasets, 1):
        v = tk.BooleanVar(value=(i == 1))
        d_vars[d.key] = v
        tk.Checkbutton(d_inner, text=f"{i:03d}. {d.key}", variable=v, anchor="w").pack(fill="x", padx=3, pady=1)

    bottom = tk.Frame(root)
    bottom.pack(fill="x", padx=8, pady=8)
    tk.Label(bottom, text="데이터별 반복 횟수:").pack(side="left")
    runs_var = tk.StringVar(value="1")
    tk.Entry(bottom, textvariable=runs_var, width=8).pack(side="left", padx=6)

    selected_params: List[str] = []
    selected_raw: List[str] = []
    repeats = 1

    def on_ok() -> None:
        nonlocal repeats
        selected_params.extend([p for p, v in p_vars.items() if v.get()])
        selected_raw.extend([k for k, v in d_vars.items() if v.get()])
        if not selected_params:
            messagebox.showerror("Error", "최소 1개 파라미터 선택 필요")
            selected_params.clear()
            return
        if not selected_raw:
            messagebox.showerror("Error", "최소 1개 로우데이터 선택 필요")
            selected_raw.clear()
            return
        try:
            repeats = max(1, int(runs_var.get()))
        except Exception:
            messagebox.showerror("Error", "반복 횟수는 정수 >=1")
            return
        root.destroy()

    def on_cancel() -> None:
        root.destroy()

    btn = tk.Frame(root)
    btn.pack(fill="x", padx=8, pady=8)
    tk.Button(btn, text="확인", command=on_ok).pack(side="left")
    tk.Button(btn, text="취소", command=on_cancel).pack(side="left", padx=8)

    root.mainloop()
    return selected_params, selected_raw, repeats


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


def optimize_single_param(
    tt_text: str,
    fit_param: str,
    bound: Tuple[float, float],
    target_value: float,
    scale_value: float,
    run_dir: Path,
    budget: int,
) -> Tuple[float, float]:
    lo, hi = bound
    n_points = max(7, budget * 4)
    xs = np.linspace(lo, hi, n_points)
    best_x = float(xs[0])
    best_f = float("inf")
    idx = 0
    for x in xs:
        f = objective(
            np.array([x], dtype=float),
            tt_text,
            [fit_param],
            [bound],
            np.array([target_value], dtype=float),
            np.array([max(scale_value, 1e-12)], dtype=float),
            run_dir,
            idx,
        )
        idx += 1
        if f < best_f:
            best_f = f
            best_x = float(x)
    return best_x, best_f


def run_fit_once(
    tt_text: str,
    fit_params: List[str],
    bounds_data: Dict[str, Dict[str, float]],
    target_corner: str,
    run_dir: Path,
    iters: int,
) -> Tuple[np.ndarray, float, str]:
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
        target_values.append(bounds_data[p][target_corner])
        scale_values.append(hi - lo)

    x0_arr = np.array(x0, dtype=float)
    target_arr = np.array(target_values, dtype=float)
    scale_arr = np.array(scale_values, dtype=float)

    if len(fit_params) == 1:
        best_x, best_f = optimize_single_param(
            tt_text=tt_text,
            fit_param=fit_params[0],
            bound=bounds[0],
            target_value=float(target_arr[0]),
            scale_value=float(scale_arr[0]),
            run_dir=run_dir,
            budget=iters,
        )
        best = np.array([best_x], dtype=float)
        return best, best_f, "grid-1d"

    es = cma.CMAEvolutionStrategy(
        x0_arr,
        0.1,
        {"bounds": [[b[0] for b in bounds], [b[1] for b in bounds]], "maxiter": iters},
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

    return np.array(es.result.xbest), float(es.result.fbest), "cma-es"


def main() -> None:
    ap = argparse.ArgumentParser(description="PM3 fitter with Tk selection for params + raw datasets")
    ap.add_argument("--workdir", default=".")
    ap.add_argument("--target-corner", choices=["ss", "ff"], default="ss")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--params", default=",".join(DEFAULT_FIT_PARAMS), help="Comma-separated params to fit")
    ap.add_argument("--ui", action="store_true", help="Open Tk UI for parameter/raw-data selection")
    ap.add_argument("--raw-zip", default=DEFAULT_RAW_ZIP)
    ap.add_argument("--raw-select", default="", help="Comma-separated raw dataset keys")
    ap.add_argument("--runs-per-dataset", type=int, default=1)
    args = ap.parse_args()

    workdir = Path(args.workdir)
    tt_path = workdir / "tt.pm3.spice"
    bounds_csv = workdir / "ss_ff_param_bounds.csv"
    if not tt_path.exists() or not bounds_csv.exists():
        raise FileNotFoundError("Missing inputs: tt.pm3.spice and ss_ff_param_bounds.csv must exist in --workdir.")

    raw_sets = discover_raw_datasets(Path(args.raw_zip))
    print(f"Discovered raw datasets: {len(raw_sets)}")

    tt_text = tt_path.read_text()
    all_params = rank_params(parse_all_tt_params(tt_text))
    bounds_data = read_bounds_csv(bounds_csv)
    available_bounds = set(bounds_data.keys())

    if args.ui:
        try:
            fit_params, selected_raw, runs_per_ds = select_with_tk(all_params, DEFAULT_FIT_PARAMS, available_bounds, raw_sets)
        except Exception as e:
            raise RuntimeError("Tk UI failed to open. Check desktop/display environment.") from e
        if not fit_params:
            raise RuntimeError("No parameters selected.")
        if not selected_raw:
            raise RuntimeError("No raw datasets selected.")
    else:
        fit_params = [p.strip() for p in args.params.split(",") if p.strip()]
        runs_per_ds = max(1, args.runs_per_dataset)
        if args.raw_select.strip():
            selected_raw = [s.strip() for s in args.raw_select.split(",") if s.strip()]
        else:
            selected_raw = [raw_sets[0].key] if raw_sets else ["NO_RAW_DATA"]

    missing = [p for p in fit_params if p not in available_bounds]
    if missing:
        raise ValueError(f"Selected params not present in ss_ff_param_bounds.csv: {missing}")

    raw_lookup = {d.key: d for d in raw_sets}
    for key in selected_raw:
        if key != "NO_RAW_DATA" and key not in raw_lookup:
            raise ValueError(f"Unknown raw dataset key: {key}")

    print("Selected params:", ", ".join(fit_params))
    print("Selected raw datasets:", ", ".join(selected_raw))
    print("Runs per dataset:", runs_per_ds)

    summary_rows = []
    for ds_key in selected_raw:
        for rep in range(1, runs_per_ds + 1):
            run_tag = f"{ds_key}_run{rep}"
            safe_tag = re.sub(r"[^a-zA-Z0-9_.-]+", "_", run_tag)
            run_dir = workdir / f"work_{DEVICE}" / safe_tag
            run_dir.mkdir(parents=True, exist_ok=True)

            best, best_obj, method = run_fit_once(
                tt_text=tt_text,
                fit_params=fit_params,
                bounds_data=bounds_data,
                target_corner=args.target_corner,
                run_dir=run_dir,
                iters=args.iters,
            )

            best_text = tt_text
            for n, v in zip(fit_params, best):
                best_text = replace_param(best_text, n, float(v))
            out_model = run_dir / "tt_fitted.pm3.spice"
            out_model.write_text(best_text)

            summary_rows.append((ds_key, rep, method, best_obj, out_model.as_posix()))
            print(f"[{ds_key} #{rep}] objective={best_obj:.6e} method={method} saved={out_model}")

    print("\n=== Fit summary ===")
    for ds_key, rep, method, obj, path in summary_rows:
        print(f"{ds_key}, run={rep}, method={method}, objective={obj:.6e}, model={path}")


if __name__ == "__main__":
    main()
