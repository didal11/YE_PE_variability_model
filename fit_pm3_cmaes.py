#!/usr/bin/env python3
import argparse
import csv
import re
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cma
import numpy as np

DEVICE = "sky130_fd_pr__nfet_01v8"
DEFAULT_FIT_PARAMS = [
    "vth0", "u0", "nfactor", "k2", "toxe",
    "wint", "lint", "ua", "ub", "a0", "ags",
    "dlc", "dwc", "cgso", "cgdo", "cjs", "cjsws", "cjswgs",
]
DEFAULT_RAW_ZIP = "skywater-pdk-sky130-raw-data-main.zip"


@dataclass
class RawDataset:
    key: str
    idvg_path: str
    idvd_path: str | None


@dataclass
class MdmCurve:
    sweep_name: str
    sweep: np.ndarray
    current: np.ndarray
    vd: float
    vg: float
    vb: float
    vs: float


@dataclass
class DeviceGeom:
    w: str
    l: str
    m: int


ProgressCallback = Callable[[int, List[MdmCurve], List[Tuple[np.ndarray, np.ndarray]], float], None]


def discover_raw_datasets(raw_zip: Path) -> List[RawDataset]:
    if not raw_zip.exists():
        return []
    idvg_pat = re.compile(r".*/cells/nfet_01v8/sky130_fd_pr__nfet_01v8_(w.+\(.*_IDVG(?:_D\d+)?\))\.mdm$")
    idvd_pat = re.compile(r".*/cells/nfet_01v8/sky130_fd_pr__nfet_01v8_(w.+\(.*_IDVD(?:_D\d+)?\))\.mdm$")
    idvg_map, idvd_map = {}, {}
    with zipfile.ZipFile(raw_zip, "r") as zf:
        for n in zf.namelist():
            m = idvg_pat.match(n)
            if m:
                idvg_map[m.group(1).replace("_IDVG", "")] = n
                continue
            m = idvd_pat.match(n)
            if m:
                idvd_map[m.group(1).replace("_IDVD", "")] = n
    out = []
    for k in sorted(set(idvg_map) | set(idvd_map)):
        if k in idvg_map:
            out.append(RawDataset(k, idvg_map[k], idvd_map.get(k)))
    return out


def parse_all_tt_params(model_text: str) -> List[str]:
    pat = re.compile(r"^\+\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=", re.MULTILINE)
    seen, params = set(), []
    for name in pat.findall(model_text):
        if name not in seen:
            seen.add(name)
            params.append(name)
    return params


def rank_params(params: List[str]) -> List[str]:
    pri = ["vth0", "u0", "vsat", "k1", "voff", "k2", "eta0", "nfactor", "dvt0", "dvt1", "dvt2", "rdsw"]
    pidx = {p: i for i, p in enumerate(pri)}

    def score(n: str) -> Tuple[int, int, str]:
        return (0, pidx[n], n) if n in pidx else (1, 0, n)

    return sorted(params, key=score)




def select_fit_candidates(bounds_data: Dict[str, Dict[str, float]]) -> List[str]:
    candidates = []
    for p in DEFAULT_FIT_PARAMS:
        if p not in bounds_data:
            continue
        d = bounds_data[p]
        if d["tt"] == d["ss"] == d["ff"]:
            continue
        candidates.append(p)
    return rank_params(candidates)

def select_with_tk(sorted_params: List[str], default_selected: List[str], available_bounds: set[str], raw_datasets: List[RawDataset]) -> Tuple[List[str], List[str], int]:
    import tkinter as tk
    from tkinter import messagebox

    def build_scrollable_checklist(parent: tk.Widget, title: str) -> tuple[tk.LabelFrame, tk.Frame]:
        frame = tk.LabelFrame(parent, text=title)
        frame.pack(side="left", fill="both", expand=True, padx=4)

        canvas = tk.Canvas(frame, highlightthickness=0)
        scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        items = tk.Frame(canvas)

        items.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window_id, width=e.width))
        window_id = canvas.create_window((0, 0), window=items, anchor="nw")

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        return frame, items

    root = tk.Tk()
    root.title("PM3 Fit Selection")
    root.geometry("980x760")

    tk.Label(root, text="좌: 파라미터 / 우: 로우데이터 / 하단: 데이터별 반복 횟수", anchor="w").pack(fill="x", padx=8, pady=8)
    cols = tk.Frame(root)
    cols.pack(fill="both", expand=True, padx=8, pady=4)

    _, p_items = build_scrollable_checklist(cols, "파라미터")
    _, d_items = build_scrollable_checklist(cols, f"로우데이터 ({len(raw_datasets)}개)")

    p_vars, d_vars = {}, {}
    for i, p in enumerate(sorted_params, 1):
        v = tk.BooleanVar(value=(p in default_selected and p in available_bounds))
        cb = tk.Checkbutton(p_items, text=f"{i:03d}. {p}", variable=v, anchor="w")
        if p not in available_bounds:
            cb.configure(state="disabled", fg="red")
        cb.pack(fill="x")
        p_vars[p] = v
    for i, d in enumerate(raw_datasets, 1):
        v = tk.BooleanVar(value=(i == 1))
        tk.Checkbutton(d_items, text=f"{i:03d}. {d.key}", variable=v, anchor="w").pack(fill="x")
        d_vars[d.key] = v

    btm = tk.Frame(root)
    btm.pack(fill="x", padx=8, pady=8)
    tk.Label(btm, text="Runs per dataset:").pack(side="left")
    runs_var = tk.StringVar(value="1")
    tk.Entry(btm, textvariable=runs_var, width=8).pack(side="left", padx=4)

    picked_p, picked_d, runs = [], [], 1

    def ok() -> None:
        nonlocal runs
        picked_p.extend([k for k, v in p_vars.items() if v.get()])
        picked_d.extend([k for k, v in d_vars.items() if v.get()])
        if not picked_p:
            messagebox.showerror("Error", "파라미터 최소 1개 선택")
            picked_p.clear()
            return
        if not picked_d:
            messagebox.showerror("Error", "로우데이터 최소 1개 선택")
            picked_d.clear()
            return
        runs = max(1, int(runs_var.get()))
        root.destroy()

    tk.Button(root, text="확인", command=ok).pack(side="left", padx=8, pady=8)
    tk.Button(root, text="취소", command=root.destroy).pack(side="left", padx=8, pady=8)
    root.mainloop()
    return picked_p, picked_d, runs


def replace_param(model_text: str, param: str, new_value: float) -> str:
    pat = re.compile(rf"^(\+\s*{re.escape(param)}\s*=\s*)([^\n]+)$", re.MULTILINE)
    out, n = pat.subn(lambda m: m.group(1) + re.sub(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", f"{new_value:.8e}", m.group(2), count=1), model_text, count=1)
    if n != 1:
        raise ValueError(f"Failed to replace '{param}'")
    return out


def read_bounds_csv(path: Path) -> Dict[str, Dict[str, float]]:
    data = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            p = row["param"]
            data[p] = {k: float(row[k]) for k in ["tt", "ss", "ff", "lower", "upper"]}
    return data


def parse_mdm_curves_from_zip(raw_zip: Path, mdm_path: str, sweep_name: str) -> List[MdmCurve]:
    with zipfile.ZipFile(raw_zip, "r") as zf:
        txt = zf.read(mdm_path).decode(errors="ignore")
    blocks = txt.split("BEGIN_DB")
    curves = []
    for b in blocks[1:]:
        section = b.split("END_DB")[0]
        vd = vg = vb = vs = 0.0
        for line in section.splitlines():
            m = re.match(r"\s*ICCAP_VAR\s+(\w+)\s+([-+0-9.eE]+)", line)
            if m:
                name, val = m.group(1), float(m.group(2))
                if name == "VD": vd = val
                elif name == "VG": vg = val
                elif name == "VB": vb = val
                elif name == "VS": vs = val
        header_tokens = None
        for ln in section.splitlines():
            sl = ln.strip()
            if sl.startswith("#"):
                header_tokens = sl.lstrip("#").split()
                break

        if header_tokens is None:
            raise ValueError(f"Missing data header in {mdm_path}")

        if sweep_name not in header_tokens or "ID" not in header_tokens:
            raise ValueError(f"Required columns missing in {mdm_path}: header={header_tokens}")

        x_idx = header_tokens.index(sweep_name)
        i_idx = header_tokens.index("ID")

        lines = [ln.strip() for ln in section.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        data = []
        for ln in lines:
            parts = ln.split()
            if len(parts) > max(x_idx, i_idx):
                try:
                    x = float(parts[x_idx]); i = float(parts[i_idx])
                    data.append((x, i))
                except Exception:
                    pass
        if data:
            arr = np.array(data)
            curves.append(MdmCurve(sweep_name, arr[:, 0], arr[:, 1], vd, vg, vb, vs))
    if not curves:
        raise ValueError(f"No curve parsed from {mdm_path}")
    curves.sort(key=lambda c: c.vd)
    return curves


def load_dataset_curves(raw_zip: Path, ds: RawDataset) -> List[MdmCurve]:
    curves = parse_mdm_curves_from_zip(raw_zip, ds.idvg_path, "VG")
    if ds.idvd_path:
        curves.extend(parse_mdm_curves_from_zip(raw_zip, ds.idvd_path, "VD"))
    return curves


def parse_device_geom(dataset_key: str) -> DeviceGeom:
    m = re.search(r"w([0-9]+(?:p[0-9]+)?)u_l([0-9]+(?:p[0-9]+)?)u_m([0-9]+)", dataset_key)
    if not m:
        raise ValueError(f"Failed to parse geometry from dataset key: {dataset_key}")

    def fmt(tok: str) -> str:
        return f"{tok.replace('p', '.')}u"

    w = fmt(m.group(1))
    l = fmt(m.group(2))
    return DeviceGeom(w=w, l=l, m=int(m.group(3)))


def write_curve_netlist(model_file: Path, out_csv: Path, curve: MdmCurve, geom: DeviceGeom) -> str:
    sweep_start, sweep_stop = float(curve.sweep.min()), float(curve.sweep.max())
    step = float(np.median(np.diff(np.unique(curve.sweep)))) if len(np.unique(curve.sweep)) > 1 else 0.05
    if step <= 0:
        step = 0.05

    if curve.sweep_name == "VG":
        v_d = curve.vd
        v_g = sweep_start
        dc_line = f"dc Vg {sweep_start} {sweep_stop} {step}"
        wr_line = f"wrdata {out_csv.as_posix()} V(g) -I(Vd)"
    elif curve.sweep_name == "VD":
        v_d = sweep_start
        v_g = curve.vg
        dc_line = f"dc Vd {sweep_start} {sweep_stop} {step}"
        wr_line = f"wrdata {out_csv.as_posix()} V(d) -I(Vd)"
    else:
        raise ValueError(f"Unsupported sweep type: {curve.sweep_name}")

    return f"""
.param MC_MM_SWITCH=0
.include '{model_file.as_posix()}'
Vd d 0 {v_d}
Vg g 0 {v_g}
Vs s 0 {curve.vs}
Vb b 0 {curve.vb}
X1 d g s b {DEVICE} l={geom.l} w={geom.w} m={geom.m}
.control
set wr_vecnames
set wr_singlescale
{dc_line}
{wr_line}
quit
.endc
.end
""".strip() + "\n"


def run_ngspice(netlist: Path) -> None:
    subprocess.run(["ngspice", "-b", "-o", str(netlist.with_suffix('.log')), str(netlist)], check=True)


def load_wrdata(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    d = np.loadtxt(path, skiprows=1)
    return d[:, 0], d[:, 1]


def curve_error(meas_x: np.ndarray, meas_i: np.ndarray, sim_x: np.ndarray, sim_i: np.ndarray) -> float:
    order = np.argsort(sim_x)
    sx, si = sim_x[order], sim_i[order]
    si_interp = np.interp(meas_x, sx, si)
    return float(np.sqrt(np.mean((np.log10(np.abs(si_interp) + 1e-15) - np.log10(np.abs(meas_i) + 1e-15)) ** 2)))


def objective(x: np.ndarray, tt_text: str, fit_params: List[str], bounds: List[Tuple[float, float]], run_dir: Path, idx: int, curves: List[MdmCurve], geom: DeviceGeom, fixed_params: Dict[str, float], progress_cb: Optional[ProgressCallback] = None) -> float:
    clipped = np.array([np.clip(v, lo, hi) for v, (lo, hi) in zip(x, bounds)])
    text = tt_text
    for n, v in fixed_params.items():
        text = replace_param(text, n, float(v))
    for n, v in zip(fit_params, clipped):
        text = replace_param(text, n, float(v))
    model = run_dir / f"cand_{idx}.pm3.spice"
    model.write_text(text)
    errs = []
    sim_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    for cidx, curve in enumerate(curves):
        out_csv = run_dir / f"cand_{idx}_curve_{cidx}.csv"
        netlist = run_dir / f"cand_{idx}_curve_{cidx}.sp"
        netlist.write_text(write_curve_netlist(model, out_csv, curve, geom))
        run_ngspice(netlist)
        sim_x, sim_i = load_wrdata(out_csv)
        sim_pairs.append((sim_x, sim_i))
        errs.append(curve_error(curve.sweep, curve.current, sim_x, sim_i))
    mean_err = float(np.mean(errs))
    if progress_cb is not None:
        progress_cb(idx, curves, sim_pairs, mean_err)
    return mean_err


def run_fit_once(tt_text: str, fit_params: List[str], bounds_data: Dict[str, Dict[str, float]], run_dir: Path, iters: int, curves: List[MdmCurve], geom: DeviceGeom, progress_cb: Optional[ProgressCallback] = None) -> Tuple[np.ndarray, float, str]:
    n_idvg = sum(1 for c in curves if c.sweep_name == "VG")
    n_idvd = sum(1 for c in curves if c.sweep_name == "VD")
    print(f"[{run_dir.name}] geometry w={geom.w}, l={geom.l}, m={geom.m}, biases(total={len(curves)}, idvg={n_idvg}, idvd={n_idvd})")
    bounds, x0, free_params = [], [], []
    fixed_params: Dict[str, float] = {}
    for p in fit_params:
        lo, hi = bounds_data[p]["lower"], bounds_data[p]["upper"]
        tt = np.clip(bounds_data[p]["tt"], lo, hi)
        if np.isclose(lo, hi):
            fixed_params[p] = float(tt)
            continue
        free_params.append(p)
        bounds.append((lo, hi))
        x0.append(tt)

    print(f"[{run_dir.name}] free params: {free_params}, fixed params: {sorted(fixed_params.keys())}")

    if not free_params:
        f = objective(np.array([]), tt_text, [], [], run_dir, 0, curves, geom, fixed_params, progress_cb)
        return np.array([fixed_params[p] for p in fit_params], dtype=float), f, "fixed-params"

    if len(free_params) == 1:
        lo, hi = bounds[0]
        xs = np.linspace(lo, hi, max(7, iters * 4))
        bf, bx = float("inf"), xs[0]
        for i, x in enumerate(xs):
            f = objective(np.array([x]), tt_text, free_params, bounds, run_dir, i, curves, geom, fixed_params, progress_cb)
            if f < bf:
                bf, bx = f, x
        best_map = {**fixed_params, free_params[0]: float(bx)}
        return np.array([best_map[p] for p in fit_params], dtype=float), bf, "grid-1d"

    spans = [hi - lo for lo, hi in bounds]
    sigma0 = max(min(spans) * 0.25, 1e-6)
    es = cma.CMAEvolutionStrategy(np.array(x0, dtype=float), sigma0, {"bounds": [[b[0] for b in bounds], [b[1] for b in bounds]], "maxiter": iters})
    idx = 0
    while not es.stop():
        xs = es.ask()
        fs = []
        for x in xs:
            fs.append(objective(np.array(x), tt_text, free_params, bounds, run_dir, idx, curves, geom, fixed_params, progress_cb))
            idx += 1
        es.tell(xs, fs)
        es.disp()
    best_map = {**fixed_params}
    best_map.update({n: float(v) for n, v in zip(free_params, es.result.xbest)})
    return np.array([best_map[p] for p in fit_params], dtype=float), float(es.result.fbest), "cma-es"


def format_param_values(names: List[str], values: np.ndarray) -> str:
    return ", ".join(f"{n}={float(v):.8e}" for n, v in zip(names, values))


def create_live_plotter(curves: List[MdmCurve]):
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    groups = {
        "IDVG": [c for c in curves if c.sweep_name == "VG"],
        "IDVD": [c for c in curves if c.sweep_name == "VD"],
    }

    windows = []

    def build_window(title: str, grp_curves: List[MdmCurve]):
        win = tk.Tk()
        win.title(title)
        win.geometry("900x620")

        status_var = tk.StringVar(value="waiting for first candidate...")
        tk.Label(win, textvariable=status_var, anchor="w").pack(fill="x", padx=8, pady=4)

        fig = Figure(figsize=(8.5, 5.5), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_yscale("log")
        ax.set_xlabel("Sweep voltage (V)")
        ax.set_ylabel("|Id| (A)")
        ax.grid(True, which="both", alpha=0.2)

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill="both", expand=True)

        sim_lines = []
        for i, curve in enumerate(grp_curves):
            color = f"C{i % 10}"
            ax.plot(curve.sweep, np.abs(curve.current) + 1e-15, color=color, linewidth=2.0, alpha=0.75,
                    label=("meas" if i == 0 else None))
            (sl,) = ax.plot(curve.sweep, np.abs(curve.current) + 1e-15, color=color, linewidth=1.0, linestyle="--", alpha=0.9,
                            label=("sim" if i == 0 else None))
            sim_lines.append(sl)
        ax.legend(loc="best", fontsize=8)
        canvas.draw()

        windows.append({"root": win, "status": status_var, "canvas": canvas, "lines": sim_lines, "curves": grp_curves})

    if groups["IDVG"]:
        build_window("PM3 Fitting Live Plot - IDVG", groups["IDVG"])
    if groups["IDVD"]:
        build_window("PM3 Fitting Live Plot - IDVD", groups["IDVD"])

    def update(idx: int, cb_curves: List[MdmCurve], sim_pairs: List[Tuple[np.ndarray, np.ndarray]], err: float) -> None:
        sim_map = {id(c): pair for c, pair in zip(cb_curves, sim_pairs)}
        for w in windows:
            for line, curve in zip(w["lines"], w["curves"]):
                sx, si = sim_map[id(curve)]
                line.set_data(sx, np.abs(si) + 1e-15)
            ax = w["canvas"].figure.axes[0]
            ax.relim()
            ax.autoscale_view()
            w["status"].set(f"candidate={idx}, objective={err:.6e}")
            w["canvas"].draw_idle()
            w["root"].update_idletasks()
            w["root"].update()

    return update, windows


def wait_for_plot_windows(plot_windows: List[dict]) -> None:
    alive = [w for w in plot_windows if w["root"].winfo_exists()]
    while alive:
        for w in alive:
            w["root"].update_idletasks()
            w["root"].update()
        alive = [w for w in plot_windows if w["root"].winfo_exists()]


def main() -> None:
    ap = argparse.ArgumentParser(description="PM3 fitter with real raw-data error")
    ap.add_argument("--workdir", default=".")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--params", default=",".join(DEFAULT_FIT_PARAMS))
    ap.add_argument("--ui", action="store_true")
    ap.add_argument("--raw-zip", default=DEFAULT_RAW_ZIP)
    ap.add_argument("--raw-select", default="")
    ap.add_argument("--runs-per-dataset", type=int, default=1)
    args = ap.parse_args()

    workdir = Path(args.workdir)
    tt_path, bounds_csv = workdir / "tt.pm3.spice", workdir / "ss_ff_param_bounds.csv"
    if not tt_path.exists() or not bounds_csv.exists():
        raise FileNotFoundError("Missing inputs: tt.pm3.spice and ss_ff_param_bounds.csv must exist in --workdir.")

    raw_sets = discover_raw_datasets(Path(args.raw_zip))
    print(f"Discovered raw datasets: {len(raw_sets)}")

    tt_text = tt_path.read_text()
    bounds_data = read_bounds_csv(bounds_csv)
    fit_candidates = select_fit_candidates(bounds_data)

    if args.ui:
        fit_params, selected_raw, runs_per_ds = select_with_tk(fit_candidates, DEFAULT_FIT_PARAMS, set(fit_candidates), raw_sets)
    else:
        fit_params = [p.strip() for p in args.params.split(",") if p.strip()]
        selected_raw = [s.strip() for s in args.raw_select.split(",") if s.strip()] if args.raw_select else ([raw_sets[0].key] if raw_sets else [])
        runs_per_ds = max(1, args.runs_per_dataset)

    if not fit_params:
        raise RuntimeError("No parameters selected.")
    if not selected_raw:
        raise RuntimeError("No raw datasets selected.")

    missing = [p for p in fit_params if p not in bounds_data]
    if missing:
        raise ValueError(f"Selected params not present in ss_ff_param_bounds.csv: {missing}")

    unsupported = [p for p in fit_params if p not in fit_candidates]
    if unsupported:
        raise ValueError(f"Only variable params are allowed for fitting: {unsupported}")

    raw_lookup = {d.key: d for d in raw_sets}
    summary_rows = []
    for ds_key in selected_raw:
        if ds_key not in raw_lookup:
            raise ValueError(f"Unknown raw dataset key: {ds_key}")
        geom = parse_device_geom(ds_key)
        curves = load_dataset_curves(Path(args.raw_zip), raw_lookup[ds_key])
        for rep in range(1, runs_per_ds + 1):
            tag = re.sub(r"[^a-zA-Z0-9_.-]+", "_", f"{ds_key}_run{rep}")
            run_dir = workdir / f"work_{DEVICE}" / tag
            run_dir.mkdir(parents=True, exist_ok=True)
            progress_cb = None
            plot_windows: List[dict] = []
            if args.ui:
                progress_cb, plot_windows = create_live_plotter(curves)
            best, best_obj, method = run_fit_once(tt_text, fit_params, bounds_data, run_dir, args.iters, curves, geom, progress_cb)
            if args.ui and plot_windows:
                print("Fitting finished. Close plot windows to continue.")
                wait_for_plot_windows(plot_windows)
            best_text = tt_text
            for n, v in zip(fit_params, best):
                best_text = replace_param(best_text, n, float(v))
            out_model = run_dir / "tt_fitted.pm3.spice"
            out_model.write_text(best_text)
            fit_values = format_param_values(fit_params, best)
            summary_rows.append((ds_key, rep, method, best_obj, out_model.as_posix(), fit_values))
            print(f"[{ds_key} #{rep}] objective={best_obj:.6e} method={method} params: {fit_values} saved={out_model}")

    print("\n=== Fit summary ===")
    for ds_key, rep, method, obj, path, fit_values in summary_rows:
        print(f"{ds_key}, run={rep}, method={method}, objective={obj:.6e}, params: {fit_values}, model={path}")


if __name__ == "__main__":
    main()
