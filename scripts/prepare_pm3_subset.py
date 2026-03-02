#!/usr/bin/env python3
import argparse
import zipfile
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Extract only TT/SS/FF PM3 model files for a device")
    p.add_argument("--zip", required=True, help="Path to spice.zip")
    p.add_argument("--device", required=True, help="e.g. sky130_fd_pr__nfet_01v8")
    p.add_argument("--out", default="extracted_models", help="Output directory")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    wanted = [f"{args.device}__tt.pm3.spice", f"{args.device}__ss.pm3.spice", f"{args.device}__ff.pm3.spice"]

    with zipfile.ZipFile(args.zip, "r") as zf:
        names = set(zf.namelist())
        missing = [w for w in wanted if w not in names]
        if missing:
            raise FileNotFoundError(f"Missing files in zip: {missing}")

        for w in wanted:
            target = out / Path(w).name
            target.write_bytes(zf.read(w))
            print(f"copied: {w} -> {target}")


if __name__ == "__main__":
    main()
