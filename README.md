# ESS 배터리 수명 예측

초기 100 사이클 데이터만으로 리튬이온(LFP/흑연) 셀의 **EOL 수명(cycle_life, SOH 80% 도달 사이클)** 을 예측하는 회귀 모델. ESS 에서 배터리 교체 비용이 CAPEX 의 30~40% 를 차지하므로, 열화가 시작되기 전에 셀별 교체 시점을 예측하는 것이 목적이다.

## 프로젝트 개요
- 데이터셋 : MIT-Stanford Battery Dataset (Severson et al., Nature Energy 2019)
- 학습 데이터 : Batch 1 (2017-05-12)
- 평가 데이터 : Batch 2 (2018-02-20)  ·  *(추가 과제 Batch 3 (2018-04-12) 는 별도 진행 예정)*
- 태스크 : **Regression** (Cycle Life 예측), 타깃 `log10(cycle_life)`, 지표 MAPE
- 정제 후 120셀 : Batch1 41 / Batch2 39 / Batch3 40

## 파일 구조
```
├── data/                     # 정제 메타·셀 단위 피처 CSV (원본 .mat 는 미포함, data/README.md 참고)
├── notebooks/
│   ├── 01_EDA.ipynb                  # 정제 + EDA Q1~Q5 (Day 1)
│   ├── 02_feature_engineering.ipynb  # 피처 선별 근거, 배치 간 일관성
│   └── 03_modeling.ipynb             # 검증 설계 변경 근거, 모델 비교, 성능, 오류 분석
├── src/
│   ├── preprocess.py         # .mat 추출(HDF5) + 5단계 정제
│   ├── features.py           # ΔQ(V) / 용량 / 온도 / 저항 / 충전정책 피처 (cycle ≤ 100)
│   ├── validation.py         # 정책 단위 Hold-out·CV, 외삽 검증
│   └── train.py              # 모델 비교 → 선택 → Valid/Test 평가 → results/
├── results/                  # model_performance.csv, cv_comparison.csv, stage1|2/, figures/
├── requirements.txt
└── README.md
```

## 환경 설정
```bash
git clone https://github.com/<팀명>/ess-battery-project
cd ess-battery-project
pip install -r requirements.txt
python -m src.train --eval-batches 2        # data/ 의 CSV 만으로 재현 (약 30초)
```

## EDA
- **Cycle Life 분포**
  - 전체 392 ~ 1,935 (왜도 0.96 → log10 후 -0.03). 장수명(>1,000) 26.7%, 단수명(<500) 23.3%
  - **핵심 발견 : 배치 간 분포가 크게 다르다.** Batch 2 의 28/39 셀이 Train(Batch 1) 최솟값(534)보다 짧고, Batch 3 는 Train 최댓값을 넘는 셀이 많다 → Train 이 커버하지 못하는 구간이 Test 에 존재
- **열화 곡선**
  - 초기에는 거의 평탄하다가 수명의 약 77% 지점에서 Knee, 이후 감소 속도가 약 10배로 빨라짐 (세 배치 공통)
  - **핵심 발견 : 초기 100 사이클의 용량 변화(중앙값 약 1.7 mAh, 정격의 0.15%)만으로는 장/단수명이 구분되지 않는다** → 방전 곡선 *형태*가 필요
- **ΔQ(V) 곡선 분석**
  - Cycle 100 − Cycle 10 의 ΔQ(V) : 단수명 셀은 변화 폭이 크고, 장/단수명 차이는 3.0 ~ 3.3V 구간에서 가장 뚜렷
  - **핵심 발견 : log10 Var(ΔQ) ↔ log10 수명 r = −0.85 (세 배치 −0.80 ~ −0.90), 모든 배치에서 같은 방향**
- **충전 속도(C-rate)와 수명**
  - 평균 C-rate 가 높을수록 짧아지는 경향이 있으나 같은 C-rate 에서도 편차가 크고, 같은 정책 안의 셀 간 편차도 존재
  - **핵심 발견 : 충전 조건은 단독 설명변수가 못 된다.** `-newstructure` 표기 셀은 같은 정책에서도 1.6 ~ 2.2배 오래 가며(예: 5.6C(26%)-4.5C 중앙값 446 → 997), Batch 1 에는 없다
- **(추가) 피처 상관과 배치 일관성** — 상관 상위는 ΔQ 계열이며 **ΔQ 계열만 세 배치에서 부호가 일관**. 충전시간·용량·저항·온도 계열은 배치마다 부호가 뒤집힌다 (`notebooks/02_feature_engineering.ipynb`)

## Modeling

### 피처 엔지니어링 전략
모든 피처는 cycle ≤ 100 만 사용 (Knee, 수명으로 정규화한 값 등 타깃 의존 값 제외). 상세: `notebooks/02_feature_engineering.ipynb`

| 구분 | 피처 | 근거 |
|---|---|---|
| 핵심 | `dq_log_var` | 상관 최상위, 세 배치 일관, 물리적 해석 가능 |
| ΔQ 형태 | `dq_log_abs_min`, `dq_skew`, `dq_kurt` | 분산이 못 잡는 곡선 형태 보완 |
| 초기 용량 | `QD_2`, `QD_max_minus_2` | 논문 Discharge 모델 피처 (단, 배치 간 값 범위 이동 큼 → 오류 분석 참고) |
| 운전/상태 | `chargetime_avg_2_6`, `Tavg_mean_2_100`, `IR_min_2_100` | 후보. 배치 간 부호가 뒤집혀 일반화 위험 |
| 제외 | `QD_intercept_2_100`, `t80_min`, `C_avg`, `-newstructure` 플래그, Knee 계열 | 공선성(VIF 매우 큼) / `chargetime` 과 중복 / Train 에 없는 값 / 100 사이클에 알 수 없음 |

### 검증 설계 : 정책(충전 프로토콜) 단위 분리
- Batch 1 → **Train 32셀 / Valid(Hold-out) 9셀**. 같은 정책의 셀이 Train 과 Valid 에 함께 들어가지 않도록 **정책 단위**로 분리 (Valid 5개 정책)
- Train CV 도 정책 단위 Repeated 5-Fold × 10. Batch 2 는 모델 선택에 쓰지 않고 선택 후 최종 평가에만 사용
- **Day 1 계획에서 바꾼 이유** : Day 1 에는 수명 층화 Hold-out + 셀 단위 랜덤 5-Fold 를 계획했다. 그런데 Batch 1 은 정책 22개 × 셀 1~3개 구조라 랜덤 분할은 같은 정책의 형제 셀을 학습에서 보여 준다. 운전 조건 피처(`chargetime`, `Tavg` 등)는 같은 정책이면 값이 거의 같아 모델이 새 셀이 아니라 *이미 본 정책*을 맞히게 된다.
  같은 모델·피처로 비교하자 **27개 조합 모두 정책 단위 CV 가 랜덤 CV 보다 나빴고 평균 4.0%p 차이**가 났다 (`results/figures/cv_random_vs_group.png`). 랜덤 분할은 낙관적이므로 정책 단위로 변경했다.

### 모델 선택 및 근거
- **후보 모델** : Linear, Ridge, ElasticNet, PLS(2), Gaussian Process Regression, GBM(max_depth=2) × 피처셋 5개 = 27개 조합 (논문 Variance 선형 모델 = 베이스라인)
- **최종 모델** : `ΔQ 4개 + QD_2 + QD_max_minus_2` 피처, **Gaussian Process Regression** (RBF + WhiteKernel, 표준화)
- **선택 이유** : Train 정책 단위 CV MAPE 최소(8.9%). 소표본(32셀)에서 비선형 관계를 규제된 형태로 맞추고, 공선성이 큰 ΔQ 피처에서 선형 계열(Linear/Ridge/ElasticNet 16~20%)보다 안정적
- **개선 단계(1차 결과 확인 후 추가)** : Test 에서 큰 격차가 나온 뒤, Batch 1 안에서만 **외삽 CV**(수명이 가장 짧은 정책 20·25·30% 를 검증으로)를 설계하고 `(정책 단위 CV + 외삽 CV)/2` 로 재선택했다 (선택 규칙은 외삽 CV 값을 보기 전에 확정). **같은 모델이 다시 선택되어 최종 모델은 바뀌지 않았다.** "선형 계열이 외삽에 더 강할 것"이라는 가설은 Batch 1 외삽 CV 에서 뒷받침되지 않음 (선형 계열 외삽 CV 16.4~35%, 대부분 19% 이상 vs GPR 14.6%).

## 성능 결과
지표 : MAPE (%). Gap 은 Train-Valid = Valid − Train, Valid-Test = Test − Valid, Target-Test = Test − 9.1.

| 구분 | MAPE (%) | 비고 |
|---|---|---|
| Train (Batch 1 CV) | 8.9 | 정책 단위 Repeated 5-Fold × 10, SD 3.4 |
| Valid (Batch 1 Hold-out) | 6.9 | 9셀. 30개 시드 평균 7.6 ± 2.0 (4.1 ~ 11.4) |
| Test (Batch 2) | **38.5** | 39셀, Batch 1 전체(41셀)로 재학습. Train(32셀)만으로 학습 시 37.3 |
| Gap (Train-Valid) | −2.0 | (+) : 과적합 의심 → 과적합 징후 없음 |
| Gap (Valid-Test) | +31.6 | (+) : 배치간 일반화 저하 의심 → **크게 저하** |
| Gap (Target-Test) | +29.4 | Target : 원논문 9.1% |

참고 (모델 선택에는 사용하지 않음) : 논문 Variance 베이스라인(`dq_log_var` 선형)은 CV 15.3 / Valid 11.4 / **Test 36.0**.
전체 비교표 : `results/cv_comparison.csv`, 단계별 결과 : `results/stage1/`, `results/stage2/`

## 오류 분석
- **분포 이동 + 외삽이 주원인.** 모델의 Batch 2 예측은 **92% 셀에서 실제보다 길게** 나왔고, 예측값 최솟값이 525 로 Train 최솟값(534) 아래로 내려가지 못한다. 반면 Batch 2 기존 구조 셀 30개는 모두 392 ~ 514 사이클이다.
- 실제 수명이 450 미만인 셀(15개) MAPE 45.9%, 450~550 (15개) 43.7%, 800 초과 (7개) 11.2%. 가장 크게 틀린 셀 : b2c6 (3.6C(9%)-5C, 실제 393 → 예측 684, 74%), b2c29, b2c11, b2c0 (모두 단수명, 고속 충전 정책).
- **구조 효과와 범위 효과가 겹쳐 있다.** `-newstructure` 9셀은 MAPE 17.4% 로 낮지만, 이 셀들이 Train 범위 안에 있는 셀과 같아서 둘을 분리해 말할 수 없다.
- **초기 용량 계열 피처의 배치 의존성.** `QD_2` 는 Batch 2 셀 22/39, `QD_max_minus_2` 는 24/39 가 Batch 1 값 범위 밖이고 배치별 상관 부호도 뒤집힌다. 최종 모델이 이 피처를 포함하므로 일반화에 불리했을 가능성이 있다 *(가설. 피처 제거 실험은 Test 를 다시 보게 되어 미실시)*.
- **개선 방향(미시도)** : 배치 간 offset 이 없는 피처만 사용(ΔQ 계열 중심), 초기 용량 값을 셀 자체 기준으로 정규화, 예측 구간(GPR) 제공, 새 배치 도입 시 소수 셀로 재보정.

## ESS 도메인 해석
- **활용 가능한 의사결정** : 같은 제조 로트·유사 조건의 셀에 대해 초기 100 사이클만으로 단수명 셀을 조기 선별해 교체·예비품 계획 우선순위를 정하는 데 쓸 수 있다 (Batch 1 내 새 충전 정책 검증 MAPE 약 7~9%).
- **한계** : 새로운 배치·구조(Batch 2)에서는 MAPE 38.5% 로, 이 모델의 수명 예측값으로 교체 시점을 확정할 수 없다. 특히 오차가 **실제보다 길게 예측하는 방향**(92%)으로 쏠려, 열화가 빠른 셀을 정상으로 오판할 위험이 있어 ESS 안전 관점에서 불리한 방향이다.
- **실 배포를 위해 필요한 것** : 로트별 재보정(소수 셀 실측으로 보정), 예측 구간 기반 보수적 의사결정, 배치 간 일관된 피처(방전 곡선 형태) 중심 설계, 고해상도 방전 전압–용량 데이터를 BMS 가 저장하는 체계.

## 참고문헌
- Severson et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 4, 383–391.

## 팀 구성
- 강지훈 : 데이터 전처리, EDA, 피처 엔지니어링, 성능 평가
- 윤서진 : EDA, 피처 엔지니어링, 모델 개발, 보고서 작성
