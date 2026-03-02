# ngspice + CMA-ES 기반 PM3 피팅 가이드 (고정 디바이스: sky130_fd_pr__nfet_01v8)

요청사항대로 이제 흐름은 **sky130_fd_pr__nfet_01v8 전용**입니다.

이 레포에는 요청대로 아래 2개 산출물을 직접 포함했습니다.

- `tt.pm3.spice`
- `ss_ff_param_bounds.csv`

- `tt.pm3.spice` 단일 평탄화 모델 사용
- `ss`, `ff`는 모델 파일 대신 **파라미터 값 CSV** 사용

## 포함된 파일

- `tt.pm3.spice`
- `ss_ff_param_bounds.csv`

CSV 컬럼: `param, tt, ss, ff, lower, upper`

현재 CSV는 5개가 아니라 **tt.pm3에 있는 파라미터 전체(추출 가능한 항목 전부)**를 담도록 갱신했습니다.

## 설치

- ngspice: https://ngspice.sourceforge.io/download.html
- CMA-ES(python `cma`): https://pypi.org/project/cma/

```bash
python3 -m pip install -r requirements.txt
```

## 실행

```bash
python3 fit_pm3_cmaes.py --workdir . --iters 20

# Tk UI로 파라미터 체크박스 선택 후 피팅
python3 fit_pm3_cmaes.py --workdir . --iters 20 --ui

# CLI로 로우데이터 2개 선택 + 데이터당 2회 반복
python3 fit_pm3_cmaes.py --workdir . --iters 20 --raw-select "w0p36u_l0p15u_m1(8701_9_10),w0p39u_l0p15u_m1(8701_11_12)" --runs-per-dataset 2
```

## 피팅 파라미터(5개)

- `vth0`
- `u0`
- `vsat`
- `k1`
- `voff`

> 미스매치 항은 제외합니다.


## UI 기능
- 파라미터를 1개만 선택해도 동작하도록 처리되어 있습니다(1D는 CMA-ES 대신 단일 변수 스캔으로 최적화).
- `--ui` 옵션 사용 시 `tt.pm3.spice`의 파라미터를 중요도 순으로 나열한 Tk 체크박스 창이 열립니다.
- 체크된 파라미터만 피팅 대상으로 사용됩니다.
- 현재 `ss_ff_param_bounds.csv`에 있는 파라미터만 선택 가능합니다.


## 로우데이터 선택
- `--raw-zip skywater-pdk-sky130-raw-data-main.zip`에서 `nfet_01v8`의 MDM(IDVG 중심) 목록을 자동 탐색합니다.
- `--ui`에서 파라미터 체크박스 + 로우데이터 체크박스를 같이 선택할 수 있습니다.
- 각 로우데이터마다 반복 횟수(`Runs per dataset`)를 지정해 개별 피팅 결과를 생성합니다.
- CLI에서는 `--raw-select "<dataset_key1>,<dataset_key2>" --runs-per-dataset N`으로 지정할 수 있습니다.


## 실제 피팅 오차 계산
- 이제 objective는 선택한 MDM(IDVG) 로우데이터의 측정 ID 곡선과 ngspice 시뮬레이션 곡선의 로그 오차(RMSE)로 계산합니다.
- `clipped 파라미터 vs SS/FF target 값` 비교는 제거되었습니다.
- `ss_ff_param_bounds.csv`는 초기값/상하한 용도로만 사용됩니다.
