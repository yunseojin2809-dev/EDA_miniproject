"""최종 모델(1차 선택)의 예측을 SHAP 으로 설명하는 사후 진단. 모델 선택·재학습에는 사용하지 않는다.

선형 모델이므로 SHAP 값은 (표준화된 계수 × (값 − 배경 평균)) 과 같고, shap.LinearExplainer 로 계산해 직접 계산값과 대조한다.
기여도 단위는 log10(cycle_life) 이며, 10**기여도 − 1 로 '수명을 몇 % 늘리거나 줄이는가' 로 읽을 수 있다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .train import FEATURE_SETS, MODELS

FINAL = ("D6 dQ4+QD2+QDmax", "ElasticNet")       # results/stage1 의 최종 모델과 동일


def fit_final(b1: pd.DataFrame):
    """Batch 1 전체로 최종 모델 재학습 (src.train 의 Test 평가용 학습과 같은 절차)"""
    fs, mn = FINAL; cols = FEATURE_SETS[fs]
    d = b1.dropna(subset=cols)
    return MODELS[mn]().fit(d[cols].values, d.log_life.values), cols, d


def shap_table(model, cols, bg: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """셀 × 피처 SHAP 값. 배경은 학습(Batch 1) 분포."""
    import shap
    sc, en = model.steps[0][1], model.steps[-1][1]
    xs_bg, xs = sc.transform(bg[cols].values), sc.transform(data[cols].values)
    phi = shap.LinearExplainer((en.coef_, en.intercept_), xs_bg).shap_values(xs)
    manual = en.coef_ * (xs - xs_bg.mean(0))                       # 선형 모델의 정확한 SHAP
    assert np.allclose(phi, manual, atol=1e-8)
    out = pd.DataFrame(phi, columns=cols, index=data.cell_id.values)
    out["base"] = float(en.predict(xs_bg).mean())                   # 배경 평균 예측 (log10)
    out["pred_log"] = en.predict(xs)
    return out
