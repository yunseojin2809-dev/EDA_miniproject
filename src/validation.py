"""충전 프로토콜 단위 분리 — 같은 충전 프로토콜의 셀이 Train/Valid 양쪽에 들어가는 누수를 차단."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def mape(y_log, p_log) -> float:
    """log10 타깃 → 사이클로 역변환 후 MAPE(%)."""
    yt, yp = 10 ** np.asarray(y_log, float), 10 ** np.ravel(p_log)
    return float(np.mean(np.abs(yp - yt) / yt) * 100)


def group_holdout(d: pd.DataFrame, seed: int = 42, target_frac: float = 0.2, tries: int = 5000):
    """충전 프로토콜 단위 Hold-out (셀 비율 ≈20%).
    조건: Valid 충전 프로토콜이 Train 에 전혀 없을 것 + Valid 수명이 Train 수명 분포(25~75%)를 걸칠 것."""
    rng = np.random.default_rng(seed)
    pols, n = d.policy.unique(), len(d)
    want = round(n * target_frac)
    for _ in range(tries):
        chosen, cnt = [], 0
        for p in rng.permutation(pols):
            c = int((d.policy == p).sum())
            if cnt + c <= want + 1:
                chosen.append(p); cnt += c
            if cnt >= want:
                break
        if not (want - 1 <= cnt <= want + 1):
            continue
        va, tr = d[d.policy.isin(chosen)], d[~d.policy.isin(chosen)]
        q25, q75 = tr.cycle_life.quantile([.25, .75])
        if va.cycle_life.min() <= q25 and va.cycle_life.max() >= q75:
            assert set(tr.policy).isdisjoint(va.policy)
            return tr.index.values, va.index.values, chosen
    raise RuntimeError("조건을 만족하는 충전 프로토콜 단위 Hold-out 을 찾지 못함")


def repeated_group_folds(groups, n_splits: int = 5, n_repeats: int = 10, seed: int = 0):
    """충전 프로토콜 단위 Repeated K-Fold: 충전 프로토콜(그룹)을 무작위로 섞어 폴드별 셀 수가 비슷하게 배분."""
    rng = np.random.default_rng(seed)
    g = np.asarray(groups); ug = np.unique(g)
    for _ in range(n_repeats):
        folds, sizes = [[] for _ in range(n_splits)], [0] * n_splits
        for p in rng.permutation(ug):
            k = int(np.argmin(sizes)); folds[k].append(p); sizes[k] += int((g == p).sum())
        for k in range(n_splits):
            te = np.where(np.isin(g, folds[k]))[0]
            yield np.setdiff1d(np.arange(len(g)), te), te


def random_folds(n: int, n_splits: int = 5, n_repeats: int = 10, seed: int = 0):
    """(비교용) Day 1 설계의 셀 단위 랜덤 Repeated K-Fold."""
    for r in range(n_repeats):
        for tr, te in KFold(n_splits, shuffle=True, random_state=seed + r).split(np.arange(n)):
            yield tr, te


def extrapolation_splits(d: pd.DataFrame, fracs=(0.2, 0.25, 0.3)):
    """외삽 검증: 수명이 가장 짧은 충전 프로토콜들을 검증으로, 나머지(더 긴 수명)로 학습.
    → 'Train 수명 범위 밖(더 짧은 쪽)' 예측 능력을 Batch 1 안에서만 평가 (Batch 2 라벨 미사용)."""
    med = d.groupby("policy").cycle_life.median().sort_values()
    for f in fracs:
        want, chosen, cnt = round(len(d) * f), [], 0
        for p in med.index:
            if cnt >= want:
                break
            chosen.append(p); cnt += int((d.policy == p).sum())
        va = np.where(d.policy.isin(chosen))[0]
        yield np.setdiff1d(np.arange(len(d)), va), va
