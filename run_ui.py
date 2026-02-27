# run_ui.py
# Python 3.12+
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np

import sti_variability_model as model


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Tier1 z → x → p Variability Propagation")
        self.geometry("1300x820")

        self.mu_z_entries = {}
        self.sigma_z_entries = {}
        self.corr_z_entries = [[None] * len(model.Z_KEYS) for _ in model.Z_KEYS]

        self._build_ui()
        self._fill_defaults()

    def _build_ui(self):
        main = ttk.Frame(self, padding=10)
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=False)

        right = ttk.Frame(main)
        right.pack(side="right", fill="both", expand=True, padx=(10, 0))

        self._build_z_input(left)
        self._build_output(right)

    def _build_z_input(self, parent):
        z_box = ttk.LabelFrame(parent, text="Tier1 z 입력 (평균 시프트 μz, 표준편차 σz, 상관계수 corr_z)")
        z_box.pack(fill="both", expand=True)

        top_grid = ttk.Frame(z_box)
        top_grid.pack(fill="x", padx=6, pady=6)

        ttk.Label(top_grid, text="z key").grid(row=0, column=0, padx=4, pady=2)
        ttk.Label(top_grid, text="μz").grid(row=0, column=1, padx=4, pady=2)
        ttk.Label(top_grid, text="σz").grid(row=0, column=2, padx=4, pady=2)

        for r, k in enumerate(model.Z_KEYS, start=1):
            ttk.Label(top_grid, text=k).grid(row=r, column=0, padx=4, pady=1, sticky="w")
            em = ttk.Entry(top_grid, width=8)
            es = ttk.Entry(top_grid, width=8)
            em.grid(row=r, column=1, padx=4, pady=1)
            es.grid(row=r, column=2, padx=4, pady=1)
            self.mu_z_entries[k] = em
            self.sigma_z_entries[k] = es

        corr_box = ttk.LabelFrame(z_box, text="corr_z (편집 가능, 대각=1 권장)")
        corr_box.pack(fill="both", expand=True, padx=6, pady=6)

        canvas = tk.Canvas(corr_box, height=300)
        canvas.pack(side="left", fill="both", expand=True)
        yscroll = ttk.Scrollbar(corr_box, orient="vertical", command=canvas.yview)
        yscroll.pack(side="right", fill="y")
        xscroll = ttk.Scrollbar(corr_box, orient="horizontal", command=canvas.xview)
        xscroll.pack(side="bottom", fill="x")
        canvas.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)

        inner = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=inner, anchor="nw")

        ttk.Label(inner, text="corr").grid(row=0, column=0, padx=2, pady=2)
        for c, key in enumerate(model.Z_KEYS, start=1):
            ttk.Label(inner, text=key, width=10).grid(row=0, column=c, padx=1, pady=1)
            ttk.Label(inner, text=key, width=10).grid(row=c, column=0, padx=1, pady=1, sticky="w")

        for r in range(len(model.Z_KEYS)):
            for c in range(len(model.Z_KEYS)):
                e = ttk.Entry(inner, width=6)
                e.grid(row=r + 1, column=c + 1, padx=1, pady=1)
                self.corr_z_entries[r][c] = e

        def _on_cfg(_):
            canvas.configure(scrollregion=canvas.bbox("all"))

        inner.bind("<Configure>", _on_cfg)

    def _build_output(self, parent):
        out_box = ttk.LabelFrame(parent, text="결과")
        out_box.pack(fill="both", expand=True)

        self.output = tk.Text(out_box, wrap="none")
        self.output.pack(fill="both", expand=True)

        btn = ttk.Frame(parent)
        btn.pack(fill="x", pady=8)
        ttk.Button(btn, text="기본값 다시 채우기", command=self._fill_defaults).pack(side="left", padx=5)
        ttk.Button(btn, text="계산", command=self._on_calc).pack(side="left", padx=5)
        ttk.Button(btn, text="p 샘플 1개 + .inc 저장", command=self._on_save_sample).pack(side="left", padx=5)
        ttk.Button(btn, text="출력 지우기", command=lambda: self.output.delete("1.0", "end")).pack(side="right", padx=5)

    def _fill_defaults(self):
        mu_z = model.default_mu_z()
        sigma_z = model.default_sigma_z()
        corr_z = model.default_z_corr()

        for i, k in enumerate(model.Z_KEYS):
            self.mu_z_entries[k].delete(0, "end")
            self.mu_z_entries[k].insert(0, f"{mu_z[i]:.6g}")
            self.sigma_z_entries[k].delete(0, "end")
            self.sigma_z_entries[k].insert(0, f"{sigma_z[i]:.6g}")

        for r in range(len(model.Z_KEYS)):
            for c in range(len(model.Z_KEYS)):
                self.corr_z_entries[r][c].delete(0, "end")
                self.corr_z_entries[r][c].insert(0, f"{corr_z[r, c]:.4g}")

    def _read_z_spec(self):
        mu_z = []
        sigma_z = []
        for k in model.Z_KEYS:
            try:
                mu_z.append(float(self.mu_z_entries[k].get()))
                sigma_z.append(float(self.sigma_z_entries[k].get()))
            except ValueError:
                raise ValueError(f"{k}의 μz/σz 입력이 숫자가 아님")

        mu_z = np.array(mu_z, dtype=float)
        sigma_z = np.array(sigma_z, dtype=float)
        if np.any(sigma_z < 0):
            raise ValueError("σz는 음수일 수 없음")

        n = len(model.Z_KEYS)
        corr_z = np.zeros((n, n), dtype=float)
        for r in range(n):
            for c in range(n):
                try:
                    corr_z[r, c] = float(self.corr_z_entries[r][c].get())
                except ValueError:
                    raise ValueError("corr_z 입력 중 숫자가 아닌 값이 있음")

        if not np.allclose(corr_z, corr_z.T, atol=1e-10):
            raise ValueError("corr_z는 대칭이어야 함")
        if not np.allclose(np.diag(corr_z), np.ones(n), atol=1e-8):
            raise ValueError("corr_z 대각성분은 1이어야 함")

        Sigma_z = model.make_Sigma_from_sigma_and_corr(sigma_z, corr_z)
        return model.GaussianSpec(mu=mu_z, Sigma=Sigma_z), sigma_z, corr_z

    def _compute_all(self):
        spec_z, sigma_z, corr_z = self._read_z_spec()

        B = model.default_B_z_to_x()
        A = model.default_A_x_to_p()

        mu_eps_x = model.default_mu_eps_x()
        sigma_eps_x = model.default_sigma_eps_x()
        corr_eps_x = model.default_corr_eps_x()
        Sigma_eps_x = model.make_Sigma_from_sigma_and_corr(sigma_eps_x, corr_eps_x)
        spec_eps_x = model.GaussianSpec(mu=mu_eps_x, Sigma=Sigma_eps_x)

        mu_x, Sigma_x = model.build_x_from_z(spec_z, B, spec_eps_x)
        sigma_x, corr_x = model.covariance_to_sigma_and_corr(Sigma_x)

        spec_x = model.GaussianSpec(mu=mu_x, Sigma=Sigma_x)
        mu_p, Sigma_p = model.propagate_linear(spec_x, A=A, b=np.zeros((len(model.P_KEYS),), dtype=float), in_keys=model.X_KEYS, out_keys=model.P_KEYS)
        sigma_p, corr_p = model.covariance_to_sigma_and_corr(Sigma_p)

        return {
            "mu_z": spec_z.mu,
            "sigma_z": sigma_z,
            "corr_z": corr_z,
            "mu_x": mu_x,
            "sigma_x": sigma_x,
            "corr_x": corr_x,
            "mu_p": mu_p,
            "sigma_p": sigma_p,
            "corr_p": corr_p,
            "Sigma_p": Sigma_p,
        }

    def _on_calc(self):
        try:
            r = self._compute_all()
            self.output.insert("end", "=== 입력 z ===\n")
            self.output.insert("end", f"keys: {model.Z_KEYS}\n")
            self.output.insert("end", f"mu_z: {r['mu_z']}\n")
            self.output.insert("end", f"sigma_z: {r['sigma_z']}\n")
            self.output.insert("end", f"corr_z:\n{r['corr_z']}\n\n")

            self.output.insert("end", "=== 계산된 x (평균 시프트 / 표준편차) ===\n")
            for k, m, s in zip(model.X_KEYS, r["mu_x"], r["sigma_x"]):
                self.output.insert("end", f"{k:14s}  mu={m: .6g}  sigma={s: .6g}\n")
            self.output.insert("end", f"\ncorr_x:\n{r['corr_x']}\n\n")

            self.output.insert("end", "=== 계산된 p (평균 시프트 / 표준편차) ===\n")
            for k, m, s in zip(model.P_KEYS, r["mu_p"], r["sigma_p"]):
                self.output.insert("end", f"{k:14s}  mu={m: .6g}  sigma={s: .6g}\n")
            self.output.insert("end", f"\nSigma_p:\n{r['Sigma_p']}\n")
            self.output.insert("end", f"\ncorr_p:\n{r['corr_p']}\n\n")

        except Exception as e:
            messagebox.showerror("오류", str(e))

    def _on_save_sample(self):
        try:
            r = self._compute_all()
            p = model.sample_mvnormal(mu=r["mu_p"], Sigma=model.make_Sigma_from_sigma_and_corr(r["sigma_p"], r["corr_p"]), n=1, seed=None)[0]
            path = filedialog.asksaveasfilename(
                defaultextension=".inc",
                filetypes=[("SPICE include", "*.inc"), ("All files", "*.*")],
            )
            if not path:
                return
            model.write_spice_param_include(path, p, header="tier1 z->x->p sample")
            messagebox.showinfo("저장 완료", f"저장됨: {path}")
        except Exception as e:
            messagebox.showerror("오류", str(e))


if __name__ == "__main__":
    app = App()
    app.mainloop()
