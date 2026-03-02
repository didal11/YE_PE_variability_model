#!/usr/bin/env python3
import argparse
import csv
import re
import zipfile
from pathlib import Path

DEVICE = "sky130_fd_pr__nfet_01v8"
FIT_PARAMS = ["vth0", "u0", "vsat", "k1", "voff"]


def strip_mismatch_terms(model_text: str) -> str:
    model_text = re.sub(r"\+\s*MC_MM_SWITCH\*AGAUSS\(0,1\.0,1\)\*\([^}]+\)", "", model_text)
    model_text = re.sub(r"\+\s*mc_mm_switch\*agauss\(0,1\.0,1\)\*\([^}]+\)", "", model_text)
    return model_text


def parse_first_param_value(model_text: str, param: str) -> float:
    pat = re.compile(rf"^\+\s*{re.escape(param)}\s*=\s*([^\s]+)", re.MULTILINE)
    m = pat.search(model_text)
    if not m:
        raise ValueError(f"Cannot find parameter '{param}'")
    expr = m.group(1).strip("{}")
    num = re.match(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", expr)
    if not num:
        raise ValueError(f"Cannot parse numeric value for '{param}' from '{expr}'")
    return float(num.group(0))


def main() -> None:
    p = argparse.ArgumentParser(description="Extract flattened TT PM3 and SS/FF numeric bounds CSV for sky130_fd_pr__nfet_01v8")
    p.add_argument("--zip", default="spice.zip", help="Path to spice.zip")
    p.add_argument("--out", default="extracted_models", help="Output directory")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tt_name = f"{DEVICE}__tt.pm3.spice"
    ss_name = f"{DEVICE}__ss.pm3.spice"
    ff_name = f"{DEVICE}__ff.pm3.spice"

    with zipfile.ZipFile(args.zip, "r") as zf:
        names = set(zf.namelist())
        for n in [tt_name, ss_name, ff_name]:
            if n not in names:
                raise FileNotFoundError(f"Missing file in zip: {n}")

        tt_text = strip_mismatch_terms(zf.read(tt_name).decode())
        ss_text = strip_mismatch_terms(zf.read(ss_name).decode())
        ff_text = strip_mismatch_terms(zf.read(ff_name).decode())

    # flattened standalone TT model file for fitting
    tt_out = out / "tt.pm3.spice"
    tt_out.write_text(tt_text)

    rows = []
    for param in FIT_PARAMS:
        tt = parse_first_param_value(tt_text, param)
        ss = parse_first_param_value(ss_text, param)
        ff = parse_first_param_value(ff_text, param)
        lower = min(ss, ff)
        upper = max(ss, ff)
        rows.append({
            "param": param,
            "tt": tt,
            "ss": ss,
            "ff": ff,
            "lower": lower,
            "upper": upper,
        })

    csv_out = out / "ss_ff_param_bounds.csv"
    with csv_out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["param", "tt", "ss", "ff", "lower", "upper"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"written: {tt_out}")
    print(f"written: {csv_out}")


if __name__ == "__main__":
    main()
