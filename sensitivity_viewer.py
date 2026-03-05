#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import tkinter as tk
from tkinter import ttk

METRIC_GROUPS = {
    "RO": ["RO_freq"],
    "INV": ["INV_tpHL", "INV_tpLH", "INV_tr", "INV_tf"],
    "SRAM": ["STBY_power", "SRAM_hold_SNM", "SRAM_read_margin"],
}


@dataclass
class DeviceData:
    name: str
    norm_df: pd.DataFrame
    score_df: pd.DataFrame
    cause_df: pd.DataFrame


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_device_data(outdir: Path, device_name: str) -> DeviceData:
    return DeviceData(
        name=device_name,
        norm_df=_read_csv_if_exists(outdir / f"sensitivity_vector_normalized_{device_name}.csv"),
        score_df=_read_csv_if_exists(outdir / f"corner_integrated_scores_{device_name}.csv"),
        cause_df=_read_csv_if_exists(outdir / f"cause_impact_map_{device_name}.csv"),
    )


def draw_bars(canvas: tk.Canvas, labels: list[str], values: list[float]) -> None:
    canvas.delete("all")
    if not labels:
        canvas.create_text(16, 16, anchor="nw", text="데이터 없음", fill="#666")
        return
    w = max(canvas.winfo_width(), 600)
    bar_h = 22
    gap = 8
    left = 190
    top = 10
    max_v = max(abs(v) for v in values) if values else 1.0
    max_v = max(max_v, 1e-12)
    for i, (lab, val) in enumerate(zip(labels, values)):
        y0 = top + i * (bar_h + gap)
        y1 = y0 + bar_h
        x1 = left + (w - left - 20) * (abs(val) / max_v)
        color = "#4e79a7" if val >= 0 else "#e15759"
        canvas.create_text(8, (y0 + y1) / 2, anchor="w", text=lab)
        canvas.create_rectangle(left, y0, x1, y1, fill=color, outline="")
        canvas.create_text(x1 + 6, (y0 + y1) / 2, anchor="w", text=f"{val:.3g}")
    canvas.configure(scrollregion=(0, 0, w, top + len(labels) * (bar_h + gap) + 20))


class AnalysisPane(ttk.Frame):
    def __init__(self, master: tk.Misc, data: DeviceData, metrics: Iterable[str] | None = None) -> None:
        super().__init__(master)
        self.data = data
        self.allowed_metrics = list(metrics) if metrics else None

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        self.tab_3 = ttk.Frame(nb)
        self.tab_2 = ttk.Frame(nb)
        nb.add(self.tab_3, text="3차 기준")
        nb.add(self.tab_2, text="2차 기준")

        self._build_3rd_tab()
        self._build_2nd_tab()

    def _metric_candidates(self) -> list[str]:
        if self.data.score_df.empty:
            return []
        ms = sorted(self.data.score_df["metric"].dropna().unique().tolist())
        if self.allowed_metrics is None:
            return ms
        return [m for m in ms if m in self.allowed_metrics]

    def _build_3rd_tab(self) -> None:
        top = ttk.Frame(self.tab_3)
        top.pack(fill="x", padx=8, pady=6)
        ttk.Label(top, text="Metric").pack(side="left")
        self.metric_var_3 = tk.StringVar()
        metric_box = ttk.Combobox(top, textvariable=self.metric_var_3, state="readonly", width=22)
        metric_box["values"] = self._metric_candidates()
        if metric_box["values"]:
            self.metric_var_3.set(metric_box["values"][0])
        metric_box.pack(side="left", padx=8)
        metric_box.bind("<<ComboboxSelected>>", lambda _e: self.refresh_3rd())

        self.canvas3 = tk.Canvas(self.tab_3, height=420)
        self.canvas3.pack(fill="both", expand=True, padx=8, pady=8)
        self.canvas3.bind("<Configure>", lambda _e: self.refresh_3rd())
        self.refresh_3rd()

    def _build_2nd_tab(self) -> None:
        top = ttk.Frame(self.tab_2)
        top.pack(fill="x", padx=8, pady=6)
        ttk.Label(top, text="Impact Metric").pack(side="left")
        self.metric_var_2 = tk.StringVar()
        metric_box = ttk.Combobox(top, textvariable=self.metric_var_2, state="readonly", width=22)
        metric_box["values"] = self._metric_candidates()
        if metric_box["values"]:
            self.metric_var_2.set(metric_box["values"][0])
        metric_box.pack(side="left", padx=8)

        ttk.Label(top, text="Param").pack(side="left")
        self.param_var_2 = tk.StringVar(value="ALL")
        param_box = ttk.Combobox(top, textvariable=self.param_var_2, state="readonly", width=18)
        params = ["ALL"]
        if not self.data.cause_df.empty and "param" in self.data.cause_df.columns:
            params += sorted(self.data.cause_df["param"].dropna().unique().tolist())
        param_box["values"] = params
        param_box.pack(side="left", padx=8)

        metric_box.bind("<<ComboboxSelected>>", lambda _e: self.refresh_2nd())
        param_box.bind("<<ComboboxSelected>>", lambda _e: self.refresh_2nd())

        self.canvas2 = tk.Canvas(self.tab_2, height=420)
        self.canvas2.pack(fill="both", expand=True, padx=8, pady=8)
        self.canvas2.bind("<Configure>", lambda _e: self.refresh_2nd())
        self.refresh_2nd()

    def refresh_3rd(self) -> None:
        if self.data.score_df.empty:
            draw_bars(self.canvas3, [], [])
            return
        metric = self.metric_var_3.get()
        if not metric:
            draw_bars(self.canvas3, [], [])
            return
        df = self.data.score_df[self.data.score_df["metric"] == metric].copy()
        if df.empty:
            draw_bars(self.canvas3, [], [])
            return
        df = df.sort_values("worst_abs_sens", ascending=False).head(20)
        labels = [f"{p} (A={a:.2g}, N={n:.2g})" for p, a, n in zip(df["param"], df["ss_ff_asym"], df["nonlinearity_index"])]
        values = df["worst_abs_sens"].fillna(0.0).tolist()
        draw_bars(self.canvas3, labels, values)

    def refresh_2nd(self) -> None:
        if self.data.cause_df.empty:
            draw_bars(self.canvas2, [], [])
            return
        metric = self.metric_var_2.get()
        param = self.param_var_2.get()
        df = self.data.cause_df.copy()
        if metric:
            df = df[df["impact_metric"] == metric]
        if param and param != "ALL":
            df = df[df["param"] == param]
        if df.empty:
            draw_bars(self.canvas2, [], [])
            return
        agg = df.groupby("cause_metric", as_index=False)["corr"].mean()
        agg = agg.sort_values("corr", key=lambda s: s.abs(), ascending=False).head(20)
        labels = agg["cause_metric"].tolist()
        values = agg["corr"].fillna(0.0).tolist()
        draw_bars(self.canvas2, labels, values)


class App(tk.Tk):
    def __init__(self, outdir: Path) -> None:
        super().__init__()
        self.title("Sensitivity Viewer")
        self.geometry("1250x820")

        n_data = load_device_data(outdir, "nfet")
        p_data = load_device_data(outdir, "pfet")

        top_nb = ttk.Notebook(self)
        top_nb.pack(fill="both", expand=True)

        top_nb.add(AnalysisPane(top_nb, n_data), text="NFET")
        top_nb.add(AnalysisPane(top_nb, p_data), text="PFET")

        circuit_tab = ttk.Frame(top_nb)
        top_nb.add(circuit_tab, text="Circuit")

        c_nb = ttk.Notebook(circuit_tab)
        c_nb.pack(fill="both", expand=True)
        c_nb.add(AnalysisPane(c_nb, n_data, METRIC_GROUPS["RO"]), text="RO")
        c_nb.add(AnalysisPane(c_nb, n_data, METRIC_GROUPS["INV"]), text="INV")
        c_nb.add(AnalysisPane(c_nb, n_data, METRIC_GROUPS["SRAM"]), text="SRAM")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("out_sensitivity"))
    ap.add_argument("--dry-run", action="store_true", help="CSV load summary only")
    args = ap.parse_args()

    n = load_device_data(args.outdir, "nfet")
    p = load_device_data(args.outdir, "pfet")
    if args.dry_run:
        print("nfet:", len(n.norm_df), len(n.score_df), len(n.cause_df))
        print("pfet:", len(p.norm_df), len(p.score_df), len(p.cause_df))
        return

    app = App(args.outdir)
    app.mainloop()


if __name__ == "__main__":
    main()
