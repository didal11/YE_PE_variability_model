# ngspice + CMA-ES 기반 PM3 피팅 가이드 (고정 디바이스: sky130_fd_pr__nfet_01v8)

요청사항대로 이제 흐름은 **sky130_fd_pr__nfet_01v8 전용**입니다.

이 레포에는 요청대로 아래 2개 산출물을 직접 포함했습니다.

- `extracted_models/tt.pm3.spice`
- `extracted_models/ss_ff_param_bounds.csv`

- `tt.pm3.spice` 하나만 바깥 폴더(`extracted_models/`)에 평탄화해서 저장
- `ss`, `ff`는 모델 파일을 쓰지 않고 **파라미터 값만 CSV**로 추출

## 포함된 파일

- `extracted_models/tt.pm3.spice`
- `extracted_models/ss_ff_param_bounds.csv`

CSV 컬럼: `param, tt, ss, ff, lower, upper`

## 설치

- ngspice: https://ngspice.sourceforge.io/download.html
- CMA-ES(python `cma`): https://pypi.org/project/cma/

```bash
python3 -m pip install -r requirements.txt
```

## 실행

```bash
python3 scripts/fit_pm3_cmaes.py --workdir extracted_models --target-corner ss --iters 20
```

## 피팅 파라미터(5개)

- `vth0`
- `u0`
- `vsat`
- `k1`
- `voff`

> 미스매치 항은 제외합니다.
