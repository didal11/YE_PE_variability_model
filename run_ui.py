# run_ui.py
# Python 3.12+
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np

import sti_variability_model as sti


def fill_example_A(A: np.ndarray) -> None:
    """예시 계수: 나중에 네가 논문/피팅으로 교체할 자리"""
    # index helper
    xi = {k: i for i, k in enumerate(sti.STI_X_KEYS)}
    pi = {k: i for i, k in enumerate(sti.P_KEYS)}

    A[:] = 0.0
    # DVTH [V]
    A[pi["DVTH"], xi["W_STI"]]  = 2.0e-4   # V/nm
    A[pi["DVTH"], xi["SIG_CH"]] = 1.0e-6   # V/MPa
    A[pi["DVTH"], xi["DH_STI"]] = 5.0e-4   # V/nm

    # DMU [-]
    A[pi["DMU"], xi["SIG_CH"]]  = -5.0e-4  # 1/MPa
    A[pi["DMU"], xi["W_STI"]]   =  1.0e-3  # 1/nm

    # DCJD/DCJS [F]
    A[pi["DCJD"], xi["R_CORNER"]] = -2.0e-18  # F/nm
    A[pi["DCJS"], xi["R_CORNER"]] = -2.0e-18  # F/nm

    # DLNIR* [-]
    A[pi["DLNIRD"], xi["R_CORNER"]] = -0.02   # 1/nm
    A[pi["DLNIRD"], xi["T_LINER"]]  = +1.0    # 1/nm
    A[pi["DLNIRS"], xi["R_CORNER"]] = -0.02
    A[pi["DLNIRS"], xi["T_LINER"]]  = +1.0


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("STI Variability → SPICE Parameter Covariance")
        self.geometry("1100x700")

        self.mu_entries = {}
        self.sigma_entries = {}

        self.A_entries = [[None]*len(sti.STI_X_KEYS) for _ in sti.P_KEYS]

        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self, padding=10)
        main.pack(fill="both", expand=True)

        # ---- 좌측: 입력 ----
        left = ttk.Frame(main)
        left.pack(side="left", fill="y")

        # mu/sigma 입력
        box1 = ttk.LabelFrame(left, text="STI 공정변수 입력 (μ, σ)  — 단위는 네 정의 그대로")
        box1.pack(fill="x", pady=5)

        grid = ttk.Frame(box1)
        grid.pack()

        ttk.Label(grid, text="변수").grid(row=0, column=0, padx=5, pady=2)
        ttk.Label(grid, text="μ").grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(grid, text="σ (1σ)").grid(row=0, column=2, padx=5, pady=2)

        # 기본값
        defaults_mu = {"D_STI":300.0, "W_STI":200.0, "R_CORNER":30.0, "SIG_CH":0.0, "T_LINER":5.0, "DH_STI":0.0}
        defaults_sigma = {"D_STI":3.0, "W_STI":2.0, "R_CORNER":1.5, "SIG_CH":20.0, "T_LINER":0.2, "DH_STI":1.0}

        for r, k in enumerate(sti.STI_X_KEYS, start=1):
            ttk.Label(grid, text=k).grid(row=r, column=0, sticky="w", padx=5, pady=2)
            e_mu = ttk.Entry(grid, width=12)
            e_mu.insert(0, str(defaults_mu.get(k, 0.0)))
            e_mu.grid(row=r, column=1, padx=5, pady=2)
            self.mu_entries[k] = e_mu

            e_si = ttk.Entry(grid, width=12)
            e_si.insert(0, str(defaults_sigma.get(k, 1.0)))
            e_si.grid(row=r, column=2, padx=5, pady=2)
            self.sigma_entries[k] = e_si

        # 상관 옵션(간단 버전: 독립)
        self.use_independent = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            box1,
            text="(현재 버전) 입력 변수들을 독립으로 가정 (corr=I)",
            variable=self.use_independent
        ).pack(anchor="w", padx=5, pady=5)

        # A 매핑 입력
        box2 = ttk.LabelFrame(left, text="STI x → SPICE 주입 파라미터 p 매핑 (선형) : p = A(x-μ)+b")
        box2.pack(fill="both", expand=True, pady=5)

        btnrow = ttk.Frame(box2)
        btnrow.pack(fill="x", pady=3)
        ttk.Button(btnrow, text="A 예시값 채우기", command=self._on_fill_A).pack(side="left", padx=5)
        ttk.Button(btnrow, text="A 모두 0", command=self._on_zero_A).pack(side="left", padx=5)

        canvas = tk.Canvas(box2, height=260)
        canvas.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(box2, orient="vertical", command=canvas.yview)
        scroll.pack(side="right", fill="y")
        canvas.configure(yscrollcommand=scroll.set)

        inner = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=inner, anchor="nw")

        # 헤더
        ttk.Label(inner, text="p \\ x").grid(row=0, column=0, padx=4, pady=2, sticky="w")
        for c, xk in enumerate(sti.STI_X_KEYS, start=1):
            ttk.Label(inner, text=xk).grid(row=0, column=c, padx=4, pady=2)

        for r, pk in enumerate(sti.P_KEYS, start=1):
            ttk.Label(inner, text=pk).grid(row=r, column=0, padx=4, pady=2, sticky="w")
            for c, xk in enumerate(sti.STI_X_KEYS, start=1):
                e = ttk.Entry(inner, width=10)
                e.insert(0, "0")
                e.grid(row=r, column=c, padx=3, pady=2)
                self.A_entries[r-1][c-1] = e

        def _on_configure(_):
            canvas.configure(scrollregion=canvas.bbox("all"))
        inner.bind("<Configure>", _on_configure)

        # ---- 우측: 출력 ----
        right = ttk.Frame(main)
        right.pack(side="right", fill="both", expand=True, padx=(10,0))

        outbox = ttk.LabelFrame(right, text="출력 (Σp, σp, corr_p)")
        outbox.pack(fill="both", expand=True)

        self.output = tk.Text(outbox, wrap="none")
        self.output.pack(fill="both", expand=True)

        # 하단 버튼
        bot = ttk.Frame(right)
        bot.pack(fill="x", pady=8)

        ttk.Button(bot, text="계산", command=self._on_calc).pack(side="left", padx=5)
        ttk.Button(bot, text="샘플 1개 생성 + .inc 저장", command=self._on_save_sample).pack(side="left", padx=5)
        ttk.Button(bot, text="출력 지우기", command=lambda: self.output.delete("1.0", "end")).pack(side="right", padx=5)

    def _read_mu_sigma(self):
        mu = []
        sig = []
        for k in sti.STI_X_KEYS:
            try:
                mu.append(float(self.mu_entries[k].get()))
                sig.append(float(self.sigma_entries[k].get()))
            except ValueError:
                raise ValueError(f"{k}의 μ/σ 입력이 숫자가 아님")
        mu = np.array(mu, dtype=float)
        sig = np.array(sig, dtype=float)
        if np.any(sig < 0):
            raise ValueError("σ는 음수일 수 없음")
        return mu, sig

    def _read_A(self):
        A = np.zeros((len(sti.P_KEYS), len(sti.STI_X_KEYS)), dtype=float)
        for r in range(len(sti.P_KEYS)):
            for c in range(len(sti.STI_X_KEYS)):
                try:
                    A[r, c] = float(self.A_entries[r][c].get())
                except ValueError:
                    raise ValueError(f"A[{sti.P_KEYS[r]},{sti.STI_X_KEYS[c]}]가 숫자가 아님")
        return A

    def _on_fill_A(self):
        A = np.zeros((len(sti.P_KEYS), len(sti.STI_X_KEYS)), dtype=float)
        fill_example_A(A)
        for r in range(len(sti.P_KEYS)):
            for c in range(len(sti.STI_X_KEYS)):
                self.A_entries[r][c].delete(0, "end")
                self.A_entries[r][c].insert(0, f"{A[r,c]:.6g}")

    def _on_zero_A(self):
        for r in range(len(sti.P_KEYS)):
            for c in range(len(sti.STI_X_KEYS)):
                self.A_entries[r][c].delete(0, "end")
                self.A_entries[r][c].insert(0, "0")

    def _on_calc(self):
        try:
            mu_x, sigma_x = self._read_mu_sigma()

            if self.use_independent.get():
                corr_x = np.eye(len(sti.STI_X_KEYS), dtype=float)
            else:
                corr_x = np.eye(len(sti.STI_X_KEYS), dtype=float)  # (확장 예정)
            Sigma_x = sti.make_Sigma_from_sigma_and_corr(sigma_x, corr_x)

            A = self._read_A()
            spec_x = sti.GaussianSpec(mu=mu_x, Sigma=Sigma_x)
            lm = sti.LinearMap(A=A, b=np.zeros((len(sti.P_KEYS),), dtype=float))

            mu_p, Sigma_p = sti.propagate_covariance(spec_x, lm)
            sigma_p, corr_p = sti.covariance_to_sigma_and_corr(Sigma_p)

            self.output.insert("end", "=== 입력 x ===\n")
            self.output.insert("end", f"keys: {sti.STI_X_KEYS}\n")
            self.output.insert("end", f"mu_x: {mu_x}\n")
            self.output.insert("end", f"sigma_x: {sigma_x}\n\n")

            self.output.insert("end", "=== 출력 p (SPICE 주입 파라미터) ===\n")
            self.output.insert("end", f"keys: {sti.P_KEYS}\n")
            self.output.insert("end", f"mu_p (오프셋): {mu_p}\n\n")

            self.output.insert("end", "=== Sigma_p (공분산, 교차기여 포함) ===\n")
            self.output.insert("end", f"{Sigma_p}\n\n")

            self.output.insert("end", "=== sigma_p (1σ) ===\n")
            for k, s in zip(sti.P_KEYS, sigma_p):
                self.output.insert("end", f"{k}: {s:.6g}\n")
            self.output.insert("end", "\n=== corr_p ===\n")
            self.output.insert("end", f"{corr_p}\n\n")

        except Exception as e:
            messagebox.showerror("오류", str(e))

    def _on_save_sample(self):
        try:
            # 먼저 계산 수행해서 Sigma_p 필요
            mu_x, sigma_x = self._read_mu_sigma()
            corr_x = np.eye(len(sti.STI_X_KEYS), dtype=float)
            Sigma_x = sti.make_Sigma_from_sigma_and_corr(sigma_x, corr_x)

            A = self._read_A()
            spec_x = sti.GaussianSpec(mu=mu_x, Sigma=Sigma_x)
            lm = sti.LinearMap(A=A, b=np.zeros((len(sti.P_KEYS),), dtype=float))

            mu_p, Sigma_p = sti.propagate_covariance(spec_x, lm)

            # 샘플 1개
            p = sti.sample_mvnormal(mu=mu_p, Sigma=Sigma_p, n=1, seed=None)[0]

            # 저장 경로
            path = filedialog.asksaveasfilename(
                defaultextension=".inc",
                filetypes=[("SPICE include", "*.inc"), ("All files", "*.*")]
            )
            if not path:
                return

            sti.write_spice_param_include(path, p, header="sti sample params")
            messagebox.showinfo("저장 완료", f"저장됨: {path}")

        except Exception as e:
            messagebox.showerror("오류", str(e))


if __name__ == "__main__":
    app = App()
    app.mainloop()