# ngspice + CMA-ES 기반 PM3 피팅 가이드

이 저장소에서는 `spice.zip` 안의 PM3 모델을 꺼내서, **TT를 초기값**으로 두고 **SS/FF를 하한/상한 경계**로 사용하는 피팅 흐름을 바로 실행할 수 있게 준비했습니다.

## 1) 지금 바로 필요한 것

- Python 3.10+
- ngspice (CLI)
- Python 패키지: `cma`, `numpy`, `pandas`

### 설치 링크
- ngspice 다운로드: https://ngspice.sourceforge.io/download.html
- CMA-ES 패키지(PyPI `cma`): https://pypi.org/project/cma/

## 2) 빠른 시작

```bash
python3 scripts/prepare_pm3_subset.py \
  --zip spice.zip \
  --device sky130_fd_pr__nfet_01v8 \
  --out extracted_models

python3 -m pip install -r requirements.txt

python3 scripts/fit_pm3_cmaes.py \
  --workdir extracted_models \
  --device sky130_fd_pr__nfet_01v8 \
  --target-corner ss
```

## 3) 구현된 5개 핵심 피팅 파라미터

기본값(초기값)은 `tt.pm3`에서 읽고, 각 파라미터별 하한/상한은 `ss.pm3`와 `ff.pm3` 값의 min/max로 자동 설정됩니다.

- `vth0`
- `u0`
- `vsat`
- `k1`
- `voff`

> 미스매치 항은 피팅 대상에서 제외했습니다. 시뮬레이션 시 `MC_MM_SWITCH=0`으로 고정합니다.

## 4) 피팅 루프

스크립트는 아래 순서로 동작합니다.

1. TT/SS/FF에서 5개 파라미터 값 추출
2. CMA-ES로 새 후보 파라미터 생성
3. 후보값을 `tt` 기반 모델에 반영
4. ngspice로 Id-Vd 시뮬레이션 (Vg=1.8V 고정, Vd sweep)
5. 타겟 코너(기본 SS) 시뮬레이션과 오차 계산
6. 오차를 CMA-ES에 피드백

## 5) 폴더 출력물

- `extracted_models/<device>__tt.pm3.spice`
- `extracted_models/<device>__ss.pm3.spice`
- `extracted_models/<device>__ff.pm3.spice`
- `extracted_models/work_<device>/` (피팅 중간 산출물)

