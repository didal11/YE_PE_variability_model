\
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tk_rootcause_ui.py
==================
- base+overlay 방식: my_tt_base_corner.spice + rootcause_overlay.spice를 합쳐 출력 코너 생성
- Streamlit 대신 Tkinter(Tk) UI
- 출력:
  - 코너 1개 생성: base + overlay + UI 값을 합쳐 출력 코너 생성
  - MC 생성: corner_runs/에 run별 코너 파일 저장
- 옵션:
  - mm 스위치(MC_MM_SWITCH)
  - n/p 계수 분리(NP_SPLIT, a_*_n, a_*_p)
  - Root-cause 분포(mean%, sigma%) + MC 샘플링
  - (선택) ngspice 배치 실행 및 logs/ 저장

사용:
  python tk_rootcause_ui.py
(이 파일과 my_tt_base_corner.spice/rootcause_overlay.spice를 같은 폴더에 두는 게 가장 간단)
"""

import random, re, subprocess
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

BEGIN = "* === ROOTCAUSE_UI_BEGIN ==="
END   = "* === ROOTCAUSE_UI_END ==="

def pct_to_frac(x_pct: float) -> float:
    return x_pct/100.0

def replace_param_in_block(block: str, name: str, value: str) -> str:
    pat = re.compile(rf"(^\.param\s+{re.escape(name)}\s*=\s*)(.+)$", re.MULTILINE)
    if not pat.search(block):
        return block + f"\n.param {name} = {value}\n"
    return pat.sub(rf"\g<1>{value}", block, count=1)

def split_rootcause_block(txt: str) -> tuple[str,str,str]:
    if BEGIN not in txt or END not in txt:
        raise ValueError("overlay 파일에 ROOTCAUSE_UI_BEGIN/END 마커가 없어. rootcause_overlay.spice를 확인해줘.")
    pre, rest = txt.split(BEGIN, 1)
    mid, post = rest.split(END, 1)
    return pre, mid, post

def render_corner(base_path: Path, overlay_path: Path, out_path: Path, values: dict) -> None:
    base_txt = base_path.read_text(encoding="utf-8", errors="ignore").rstrip() + "\n\n"
    overlay_txt = overlay_path.read_text(encoding="utf-8", errors="ignore")
    pre, inner, post = split_rootcause_block(overlay_txt)
    for k,v in values.items():
        inner = replace_param_in_block(inner, k, v)
    out_txt = base_txt + pre + BEGIN + inner + END + post
    out_path.write_text(out_txt, encoding="utf-8")

def run_ngspice(top: Path, workdir: Path, run_id: int) -> int:
    logdir = workdir/"logs"
    logdir.mkdir(exist_ok=True)
    logfile = logdir/f"run_{run_id:04d}.log"
    cmd = ["ngspice", "-b", str(top)]
    p = subprocess.run(cmd, cwd=str(workdir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    logfile.write_text(p.stdout, encoding="utf-8")
    return p.returncode

@dataclass
class Dist:
    mu_pct: float
    sig_pct: float

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TT Root-cause Corner Builder (Tk)")
        self.geometry("1120x740")

        self.base_path = tk.StringVar(value="my_tt_base_corner.spice")
        self.overlay_path = tk.StringVar(value="rootcause_overlay.spice")
        self.out_path = tk.StringVar(value="my_tt_rootcause_corner.spice")
        self.top_path = tk.StringVar(value="nmos_test.spice")

        self.mm_on = tk.BooleanVar(value=True)

        self.runs = tk.IntVar(value=200)
        self.seed = tk.IntVar(value=1)
        self.run_sim = tk.BooleanVar(value=False)

        self.dist = {
            "X_cd": Dist(0.0, 1.0),
            "X_damage": Dist(0.0, 3.0),
            "X_eot": Dist(0.0, 1.0),
            "X_act": Dist(0.0, 1.0),
            "X_rc": Dist(0.0, 2.0),
        }

        self._build_ui()

    def _build_ui(self):
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="both", expand=True)

        filefrm = ttk.LabelFrame(frm, text="파일", padding=10)
        filefrm.pack(fill="x")

        ttk.Label(filefrm, text="Base corner file").grid(row=0, column=0, sticky="w")
        ttk.Entry(filefrm, textvariable=self.base_path, width=70).grid(row=0, column=1, sticky="we", padx=6)
        ttk.Button(filefrm, text="Browse", command=self._browse_base).grid(row=0, column=2)

        ttk.Label(filefrm, text="Root-cause overlay file").grid(row=1, column=0, sticky="w")
        ttk.Entry(filefrm, textvariable=self.overlay_path, width=70).grid(row=1, column=1, sticky="we", padx=6)
        ttk.Button(filefrm, text="Browse", command=self._browse_overlay).grid(row=1, column=2)

        ttk.Label(filefrm, text="Output corner file").grid(row=2, column=0, sticky="w")
        ttk.Entry(filefrm, textvariable=self.out_path, width=70).grid(row=2, column=1, sticky="we", padx=6)
        ttk.Button(filefrm, text="Browse", command=self._browse_output).grid(row=2, column=2)

        ttk.Label(filefrm, text="Top netlist (ngspice 실행 시)").grid(row=3, column=0, sticky="w")
        ttk.Entry(filefrm, textvariable=self.top_path, width=70).grid(row=3, column=1, sticky="we", padx=6)
        ttk.Button(filefrm, text="Browse", command=self._browse_top).grid(row=3, column=2)

        filefrm.columnconfigure(1, weight=1)

        swfrm = ttk.LabelFrame(frm, text="스위치 / MC", padding=10)
        swfrm.pack(fill="x", pady=10)

        ttk.Checkbutton(swfrm, text="mm(로컬 mismatch) ON", variable=self.mm_on).grid(row=0, column=0, sticky="w")

        ttk.Label(swfrm, text="runs").grid(row=1, column=0, sticky="w")
        ttk.Entry(swfrm, textvariable=self.runs, width=10).grid(row=1, column=0, sticky="e", padx=6)
        ttk.Label(swfrm, text="seed").grid(row=1, column=1, sticky="w")
        ttk.Entry(swfrm, textvariable=self.seed, width=10).grid(row=1, column=1, sticky="e", padx=6)

        ttk.Checkbutton(swfrm, text="MC 생성 후 ngspice도 실행(로그 저장)", variable=self.run_sim).grid(row=1, column=2, sticky="w", padx=20)

        main = ttk.Frame(frm)
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main); left.pack(side="left", fill="both", expand=True, padx=(0,8))
        right = ttk.Frame(main); right.pack(side="left", fill="both", expand=True, padx=(8,0))

        distfrm = ttk.LabelFrame(left, text="Root-cause 분포 (mean%, sigma%)", padding=10)
        distfrm.pack(fill="both", expand=True)

        self.dist_vars = {}
        r=0
        for k,d in self.dist.items():
            mu = tk.DoubleVar(value=d.mu_pct)
            sg = tk.DoubleVar(value=d.sig_pct)
            self.dist_vars[k] = (mu, sg)
            ttk.Label(distfrm, text=k).grid(row=r, column=0, sticky="w")
            ttk.Label(distfrm, text="mean%").grid(row=r, column=1, sticky="e")
            ttk.Entry(distfrm, textvariable=mu, width=10).grid(row=r, column=2, padx=4)
            ttk.Label(distfrm, text="sigma%").grid(row=r, column=3, sticky="e")
            ttk.Entry(distfrm, textvariable=sg, width=10).grid(row=r, column=4, padx=4)
            r += 1
        placeholder = ttk.LabelFrame(right, text="Reserved (k_* UI 예정)", padding=10)
        placeholder.pack(fill="both", expand=True)
        ttk.Label(
            placeholder,
            text="민감도 입력 영역은 비워두었습니다.\n추후 다른 UI를 이 영역에 추가할 예정입니다.",
            justify="left",
        ).pack(anchor="w")


        btnfrm = ttk.Frame(frm); btnfrm.pack(fill="x", pady=10)
        ttk.Button(btnfrm, text="코너 1개 생성(현재 mean 값)", command=self.make_one).pack(side="left")
        ttk.Button(btnfrm, text="MC 생성(코너 run별 저장)", command=self.make_mc).pack(side="left", padx=10)
        ttk.Button(btnfrm, text="도움말", command=self._help).pack(side="right")

        self.status = tk.StringVar(value="준비됨")
        ttk.Label(frm, textvariable=self.status).pack(fill="x")

    def _browse_base(self):
        p = filedialog.askopenfilename(title="Select base corner spice", filetypes=[("SPICE","*.spice"),("All","*.*")])
        if p: self.base_path.set(p)

    def _browse_overlay(self):
        p = filedialog.askopenfilename(title="Select overlay spice", filetypes=[("SPICE","*.spice"),("All","*.*")])
        if p: self.overlay_path.set(p)

    def _browse_output(self):
        p = filedialog.asksaveasfilename(title="Select output corner spice", defaultextension=".spice", filetypes=[("SPICE","*.spice"),("All","*.*")])
        if p: self.out_path.set(p)

    def _browse_top(self):
        p = filedialog.askopenfilename(title="Select top netlist", filetypes=[("SPICE","*.spice"),("All","*.*")])
        if p: self.top_path.set(p)

    def _collect_values(self, sample: dict) -> dict:
        vals = {
            "MC_MM_SWITCH": "1" if self.mm_on.get() else "0",
            "X_cd": f"{sample['X_cd']:.12g}",
            "X_damage": f"{sample['X_damage']:.12g}",
            "X_eot": f"{sample['X_eot']:.12g}",
            "X_act": f"{sample['X_act']:.12g}",
            "X_rc": f"{sample['X_rc']:.12g}",
        }
        return vals

    def make_one(self):
        try:
            base = Path(self.base_path.get()).resolve()
            overlay = Path(self.overlay_path.get()).resolve()
            out_corner = Path(self.out_path.get()).resolve()
            if not base.exists():
                messagebox.showerror("오류", f"base 파일 없음: {base}")
                return
            if not overlay.exists():
                messagebox.showerror("오류", f"overlay 파일 없음: {overlay}")
                return
            sample = {k: pct_to_frac(self.dist_vars[k][0].get()) for k in self.dist_vars}
            render_corner(base, overlay, out_corner, self._collect_values(sample))
            self.status.set(f"코너 생성 완료: {out_corner}")
            messagebox.showinfo("완료", f"코너 생성 완료:\n{out_corner}")
        except Exception as e:
            messagebox.showerror("오류", str(e))

    def make_mc(self):
        try:
            base = Path(self.base_path.get()).resolve()
            overlay = Path(self.overlay_path.get()).resolve()
            out_corner = Path(self.out_path.get()).resolve()
            if not base.exists():
                messagebox.showerror("오류", f"base 파일 없음: {base}")
                return
            if not overlay.exists():
                messagebox.showerror("오류", f"overlay 파일 없음: {overlay}")
                return
            workdir = out_corner.parent
            runs = int(self.runs.get())
            random.seed(int(self.seed.get()))

            dist = {k:(pct_to_frac(mu.get()), pct_to_frac(sg.get())) for k,(mu,sg) in self.dist_vars.items()}

            top = Path(self.top_path.get()).resolve()
            if self.run_sim.get() and (not top.exists()):
                messagebox.showerror("오류", f"top netlist 없음: {top}")
                return

            outdir = workdir/"corner_runs"; outdir.mkdir(exist_ok=True)
            csv_path = workdir/"mc_rootcause_inputs.csv"
            rows=[]
            for r in range(runs):
                sample = {k: random.gauss(*dist[k]) for k in dist}
                vals = self._collect_values(sample)

                run_corner = outdir/f"my_tt_rootcause_corner_run{r:04d}.spice"
                render_corner(base, overlay, run_corner, vals)

                rc=0
                if self.run_sim.get():
                    tmp_top = workdir/f".__tmp_top_{r:04d}.spice"
                    top_txt = top.read_text(encoding="utf-8", errors="ignore")
                    tmp_top.write_text(f'.include "{run_corner.as_posix()}"\n\n' + top_txt, encoding="utf-8")
                    rc = run_ngspice(tmp_top, workdir, r)
                    try: tmp_top.unlink()
                    except Exception: pass
                rows.append((r,rc,sample))

            with csv_path.open("w", encoding="utf-8") as f:
                f.write("run,rc,X_cd,X_damage,X_eot,X_act,X_rc\n")
                for r,rc,s in rows:
                    f.write(f"{r},{rc},{s['X_cd']},{s['X_damage']},{s['X_eot']},{s['X_act']},{s['X_rc']}\n")

            self.status.set(f"MC 완료: {outdir} / {csv_path}")
            messagebox.showinfo("완료", f"MC 완료\ncorner_runs/에 run별 코너 저장\nCSV: {csv_path}\n(선택) logs/에 ngspice 로그")
        except Exception as e:
            messagebox.showerror("오류", str(e))

    def _help(self):
        messagebox.showinfo(
            "도움말",
            "1) base(my_tt_base_corner.spice) + overlay(rootcause_overlay.spice)를 합쳐 코너를 만듭니다.\n"
            "2) UI는 overlay의 ROOTCAUSE_UI_BEGIN/END 사이 .param만 갱신합니다.\n"
            "3) 코너 1개 생성은 Output corner file 경로에 저장합니다.\n"
            "4) MC 생성은 corner_runs/에 run별 코너를 따로 저장합니다.\n"
            "5) ngspice 실행 옵션을 켜면 logs/에 run별 로그를 남깁니다.\n"
        )

if __name__ == "__main__":
    App().mainloop()
