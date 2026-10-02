"""셀 단위 피처 생성 (모든 피처는 cycle <= 100 데이터만 사용 → 타깃 누수 없음)."""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from scipy import stats

from .preprocess import CYC_HIGH, CYC_LOW, NOMINAL_CAP, QI_HIGH, QI_LOW, V_GRID

POLICY_RE = re.compile(r"([\d.]+)C\(([\d.]+)%\)-([\d.]+)C")
DQ_FEATS = ["dq_log_var", "dq_log_abs_min", "dq_log_abs_mean", "dq_skew", "dq_kurt"]
POL_FEATS = ["C1", "Q1", "C2", "C_avg", "t80_min"]


def delta_q_features(dq: np.ndarray) -> dict:
    """ΔQ100-10(V) 곡선 → 요약 통계량 (논문 Variance / Discharge 모델 피처)."""
    dq = dq[np.isfinite(dq)]
    return {"dq_min": dq.min(), "dq_mean": dq.mean(), "dq_var": dq.var(),
            "dq_skew": stats.skew(dq), "dq_kurt": stats.kurtosis(dq),
            "dq_log_var": np.log10(dq.var() + 1e-12),
            "dq_log_abs_min": np.log10(abs(dq.min()) + 1e-12),
            "dq_log_abs_mean": np.log10(abs(dq.mean()) + 1e-12),
            "dq_v_at_min": V_GRID[np.argmin(dq)]}


def parse_policy(s: str) -> dict:
    """'5.4C(40%)-3.6C' → C1=5.4, Q1=40, C2=3.6"""
    m = POLICY_RE.search(str(s).replace(" ", ""))
    if not m:
        return dict(C1=np.nan, Q1=np.nan, C2=np.nan)
    return dict(C1=float(m.group(1)), Q1=float(m.group(2)), C2=float(m.group(3)))


def clean_qd(sdf: pd.DataFrame) -> pd.Series:
    q = sdf["QD"].copy()
    q[(q < 0.5 * NOMINAL_CAP) | (q > 1.2 * NOMINAL_CAP)] = np.nan
    return q.rolling(5, center=True, min_periods=1).median().interpolate(limit_direction="both")


def lin_fit(s: pd.Series, lo: int, hi: int):
    s = s.loc[lo:hi].dropna()
    if len(s) < 3:
        return np.nan, np.nan
    return tuple(np.polyfit(s.index.values.astype(float), s.values, 1))


def at(s: pd.Series, c: int):
    s = s.dropna()
    return s.iloc[np.argmin(np.abs(s.index.values - c))] if len(s) else np.nan


def early_features(cid: str, sdf: pd.DataFrame) -> dict:
    qd = clean_qd(sdf).loc[2:CYC_HIGH]
    f = {"cell_id": cid, "QD_2": at(qd, 2)}
    f["QD_max_minus_2"] = qd.max() - f["QD_2"]
    f["QD_100_minus_2"] = at(qd, CYC_HIGH) - f["QD_2"]
    f["QD_slope_2_100"], f["QD_intercept_2_100"] = lin_fit(qd, 2, CYC_HIGH)
    f["QD_slope_91_100"], _ = lin_fit(qd, 91, CYC_HIGH)
    if "chargetime" in sdf: f["chargetime_avg_2_6"] = sdf["chargetime"].loc[2:6].mean()
    if "Tavg" in sdf:       f["Tavg_mean_2_100"] = sdf["Tavg"].loc[2:CYC_HIGH].mean()
    if "Tmax" in sdf:       f["Tmax_max_2_100"] = sdf["Tmax"].loc[2:CYC_HIGH].max()
    if "Tmin" in sdf:       f["Tmin_min_2_100"] = sdf["Tmin"].loc[2:CYC_HIGH].min()
    if "IR" in sdf:
        ir = sdf["IR"].loc[2:CYC_HIGH].replace(0, np.nan)
        f["IR_min_2_100"] = ir.min()
        f["IR_diff_100_2"] = at(ir, CYC_HIGH) - at(ir, 2)
    return f


def build_features(cells: dict, SUMMARY: dict, meta: pd.DataFrame) -> pd.DataFrame:
    """meta(셀 목록) → 셀 단위 피처 테이블 (1 row = 1 cell)."""
    dq_rows = []
    for cid in meta.cell_id:
        q_hi, q_lo = cells[cid]["qdlin"].get(QI_HIGH), cells[cid]["qdlin"].get(QI_LOW)
        if q_hi is None or q_lo is None:
            continue
        dq_rows.append({"cell_id": cid, **delta_q_features(np.asarray(q_hi, float) - np.asarray(q_lo, float))})
    base = meta.merge(pd.DataFrame(dq_rows), on="cell_id", how="left")

    pol = pd.DataFrame([parse_policy(p) for p in base.policy])
    pol["t80_min"] = 60 * (pol.Q1 / 100 / pol.C1 + (80 - pol.Q1).clip(lower=0) / 100 / pol.C2)  # 0→80% 충전시간(분)
    pol["C_avg"] = 0.8 * 60 / pol.t80_min                                                        # 0→80% 평균 C-rate
    base = base.join(pol)

    feat = pd.DataFrame([early_features(c, SUMMARY[c]) for c in base.cell_id])
    cols = ["cell_id", "batch", "cycle_life", "log_life", "long_life"] + DQ_FEATS + POL_FEATS
    return base[cols].merge(feat, on="cell_id")


def main():
    import argparse
    from pathlib import Path
    from .preprocess import clean_cells, load_cells
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="..", help="archive/, cache/ 가 있는 폴더 (Mini project)")
    ap.add_argument("--out", default="data")
    a = ap.parse_args(); base = Path(a.base)
    cells = load_cells(base / "archive", base / "cache" / "cells_extracted.pkl")
    cells, SUMMARY, meta, attr, merges = clean_cells(cells)
    out = Path(a.out); out.mkdir(exist_ok=True, parents=True)
    meta.to_csv(out / "meta_clean.csv", index=False); attr.to_csv(out / "attrition.csv"); merges.to_csv(out / "merge_check.csv", index=False)
    build_features(cells, SUMMARY, meta).to_csv(out / "features_cell_level.csv", index=False)
    print("saved:", out / "meta_clean.csv", out / "features_cell_level.csv")


if __name__ == "__main__":
    main()
