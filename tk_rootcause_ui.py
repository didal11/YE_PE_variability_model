\
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tk_rootcause_ui.py
==================
- spice.zip 불필요: self-contained my_tt_rootcause_corner.spice(템플릿)만 수정/생성
- Streamlit 대신 Tkinter(Tk) UI
- 출력:
  - 코너 1개 생성: my_tt_rootcause_corner.spice 덮어쓰기
  - MC 생성: corner_runs/에 run별 코너 파일 저장
- 옵션:
  - mm 스위치(MC_MM_SWITCH)
  - n/p 계수 분리(NP_SPLIT, a_*_n, a_*_p)
  - Root-cause 분포(mean%, sigma%) + MC 샘플링
  - (선택) ngspice 배치 실행 및 logs/ 저장

사용:
  python tk_rootcause_ui.py
(이 파일과 my_tt_rootcause_corner.spice를 같은 폴더에 두는 게 가장 간단)
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
    return pat.sub(rf"\1{value}", block, count=1)

def load_corner(path: Path) -> tuple[str,str,str]:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    if BEGIN not in txt or END not in txt:
        raise ValueError("코너 파일에 ROOTCAUSE_UI_BEGIN/END 마커가 없어. 제공된 템플릿을 사용해줘.")
    pre, rest = txt.split(BEGIN, 1)
    mid, post = rest.split(END, 1)
    return pre, mid, post

def update_corner_tokens(path: Path, values: dict) -> None:
    pre, inner, post = load_corner(path)
    for k,v in values.items():
        inner = replace_param_in_block(inner, k, v)
    path.write_text(pre + BEGIN + inner + END + post, encoding="utf-8")

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

        self.corner_path = tk.StringVar(value="my_tt_rootcause_corner.spice")
        self.top_path = tk.StringVar(value="nmos_test.spice")

        self.mm_on = tk.BooleanVar(value=True)
        self.np_split = tk.BooleanVar(value=False)

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

        self.sens_defaults = {
            "a_dlc":"1e-9","a_dwc":"0.0","a_toxe":"1e-2","a_vth":"2e-2",
            "a_u0":"5e-4","a_nf":"2e-2","a_voff":"1e-2","a_rdsw":"1e2"
        }

        self._build_ui()

    def _build_ui(self):
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="both", expand=True)

        filefrm = ttk.LabelFrame(frm, text="파일", padding=10)
        filefrm.pack(fill="x")

        ttk.Label(filefrm, text="Corner file").grid(row=0, column=0, sticky="w")
        ttk.Entry(filefrm, textvariable=self.corner_path, width=70).grid(row=0, column=1, sticky="we", padx=6)
        ttk.Button(filefrm, text="Browse", command=self._browse_corner).grid(row=0, column=2)

        ttk.Label(filefrm, text="Top netlist (ngspice 실행 시)").grid(row=1, column=0, sticky="w")
        ttk.Entry(filefrm, textvariable=self.top_path, width=70).grid(row=1, column=1, sticky="we", padx=6)
        ttk.Button(filefrm, text="Browse", command=self._browse_top).grid(row=1, column=2)

        filefrm.columnconfigure(1, weight=1)

        swfrm = ttk.LabelFrame(frm, text="스위치 / MC", padding=10)
        swfrm.pack(fill="x", pady=10)

        ttk.Checkbutton(swfrm, text="mm(로컬 mismatch) ON", variable=self.mm_on).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(swfrm, text="n/p 계수 분리(NP_SPLIT)", variable=self.np_split).grid(row=0, column=1, sticky="w", padx=20)

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

        sensfrm = ttk.LabelFrame(right, text="민감도 a_* (파인튜닝)", padding=10)
        sensfrm.pack(fill="both", expand=True)

        labels = ["a_dlc","a_dwc","a_toxe","a_vth","a_u0","a_nf","a_voff","a_rdsw"]

        ttk.Label(sensfrm, text="기본 a_* (NP_SPLIT=0일 때)").grid(row=0, column=0, columnspan=4, sticky="w")
        self.sens_vars={}
        for i,k in enumerate(labels):
            v=tk.StringVar(value=self.sens_defaults[k])
            self.sens_vars[k]=v
            ttk.Label(sensfrm, text=k).grid(row=1+i//2, column=(i%2)*2, sticky="e", pady=2)
            ttk.Entry(sensfrm, textvariable=v, width=14).grid(row=1+i//2, column=(i%2)*2+1, padx=6, pady=2, sticky="w")

        npfrm = ttk.LabelFrame(right, text="n/p 분리 계수 (NP_SPLIT=1일 때)", padding=10)
        npfrm.pack(fill="both", expand=True, pady=10)
        self.sensn_vars={}; self.sensp_vars={}
        ttk.Label(npfrm, text="nMOS").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(npfrm, text="pMOS").grid(row=0, column=2, columnspan=2, sticky="w")
        for i,k in enumerate(labels):
            vn=tk.StringVar(value=self.sens_defaults[k])
            vp=tk.StringVar(value=self.sens_defaults[k])
            self.sensn_vars[k]=vn; self.sensp_vars[k]=vp
            ttk.Label(npfrm, text=k).grid(row=1+i, column=0, sticky="e", pady=1)
            ttk.Entry(npfrm, textvariable=vn, width=14).grid(row=1+i, column=1, padx=6, pady=1, sticky="w")
            ttk.Label(npfrm, text=k).grid(row=1+i, column=2, sticky="e", pady=1)
            ttk.Entry(npfrm, textvariable=vp, width=14).grid(row=1+i, column=3, padx=6, pady=1, sticky="w")

        btnfrm = ttk.Frame(frm); btnfrm.pack(fill="x", pady=10)
        ttk.Button(btnfrm, text="코너 1개 생성(현재 mean 값)", command=self.make_one).pack(side="left")
        ttk.Button(btnfrm, text="MC 생성(코너 run별 저장)", command=self.make_mc).pack(side="left", padx=10)
        ttk.Button(btnfrm, text="도움말", command=self._help).pack(side="right")

        self.status = tk.StringVar(value="준비됨")
        ttk.Label(frm, textvariable=self.status).pack(fill="x")

    def _browse_corner(self):
        p = filedialog.askopenfilename(title="Select corner spice", filetypes=[("SPICE","*.spice"),("All","*.*")])
        if p: self.corner_path.set(p)

    def _browse_top(self):
        p = filedialog.askopenfilename(title="Select top netlist", filetypes=[("SPICE","*.spice"),("All","*.*")])
        if p: self.top_path.set(p)

    def _collect_values(self, sample: dict) -> dict:
        vals = {
            "MC_MM_SWITCH": "1" if self.mm_on.get() else "0",
            "NP_SPLIT": "1" if self.np_split.get() else "0",
            "X_cd": f"{sample['X_cd']:.12g}",
            "X_damage": f"{sample['X_damage']:.12g}",
            "X_eot": f"{sample['X_eot']:.12g}",
            "X_act": f"{sample['X_act']:.12g}",
            "X_rc": f"{sample['X_rc']:.12g}",
        }
        for k,v in self.sens_vars.items():
            vals[k]=v.get().strip()
        if self.np_split.get():
            for k,v in self.sensn_vars.items():
                vals[k+"_n"]=v.get().strip()
            for k,v in self.sensp_vars.items():
                vals[k+"_p"]=v.get().strip()
        else:
            for k in self.sensn_vars.keys():
                vals[k+"_n"]="0"
                vals[k+"_p"]="0"
        return vals

    def make_one(self):
        try:
            corner = Path(self.corner_path.get()).resolve()
            if not corner.exists():
                messagebox.showerror("오류", f"corner 파일 없음: {corner}")
                return
            sample = {k: pct_to_frac(self.dist_vars[k][0].get()) for k in self.dist_vars}
            update_corner_tokens(corner, self._collect_values(sample))
            self.status.set(f"코너 업데이트 완료: {corner}")
            messagebox.showinfo("완료", f"코너 업데이트 완료:\n{corner}")
        except Exception as e:
            messagebox.showerror("오류", str(e))

    def make_mc(self):
        try:
            corner = Path(self.corner_path.get()).resolve()
            if not corner.exists():
                messagebox.showerror("오류", f"corner 파일 없음: {corner}")
                return
            workdir = corner.parent
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
                run_corner.write_text(corner.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
                update_corner_tokens(run_corner, vals)

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
            "1) my_tt_rootcause_corner.spice는 self-contained입니다(외부 include 필요 없음).\n"
            "2) UI는 ROOTCAUSE_UI_BEGIN/END 사이의 .param만 갱신합니다.\n"
            "3) MC 생성은 corner_runs/에 run별 코너를 따로 저장합니다.\n"
            "4) ngspice 실행 옵션을 켜면 logs/에 run별 로그를 남깁니다.\n"
        )

if __name__ == "__main__":
    App().mainloop()
