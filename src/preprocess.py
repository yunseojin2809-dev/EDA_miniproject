"""MAT 추출 + 정제.  (Day 1 eda_day1.ipynb 3~4단계와 동일한 로직)

사용:
    python -m src.preprocess --base "<Mini project 경로>"
        → <base>/cache/cells_extracted.pkl (없으면 .mat 에서 추출), data/meta_clean.csv 생성
"""
from __future__ import annotations

import argparse, pickle, re, time
from pathlib import Path

import numpy as np
import pandas as pd

NOMINAL_CAP, EOL_CAP = 1.1, 0.88          # A123 APR18650M1A 정격 1.1Ah, SOH 80% = 0.88Ah
CYC_LOW, CYC_HIGH = 10, 100               # ΔQ(V) = Qdlin[100] - Qdlin[10]
KEEP_CYCLES = 121                         # 사이클별 곡선은 앞 121개(index 0..120)만 추출
V_GRID = np.linspace(3.6, 2.0, 1000)
LIFE_THRESHOLD = 550

MAT_GLOB = {1: "2017-05-12*errorcorrect.mat",      # varcharge(2018-04-03)는 과제 범위 밖
            2: "2018-02-20*errorcorrect.mat",
            3: "2018-04-12*errorcorrect.mat"}
SUMMARY_MAP = {"IR": "IR", "QC": "QCharge", "QD": "QDischarge", "Tavg": "Tavg", "Tmin": "Tmin",
               "Tmax": "Tmax", "chargetime": "chargetime", "cycle": "cycle"}

# 원 논문(Severson 2019) 공개 전처리 규칙 — 병합은 '검증을 통과할 때만' 적용
PUBLISHED = dict(
    merge_pairs=[("b1c0", "b2c7", 662), ("b1c1", "b2c8", 981), ("b1c2", "b2c9", 1060),
                 ("b1c3", "b2c15", 208), ("b1c4", "b2c16", 482)],
    drop_unfinished_b1=["b1c8", "b1c10", "b1c12", "b1c13", "b1c22"],
    drop_noisy_b3=["b3c2", "b3c23", "b3c32", "b3c37", "b3c42", "b3c43"],
)


# ----------------------------------------------------------------------------- MAT 추출 (HDF5 v7.3)
def _vec(ds):
    return np.ravel(np.asarray(ds[()], dtype=float))


def _scalar(f, ref):
    ds = f[ref]
    if ds.attrs.get("MATLAB_empty", 0):
        return np.nan
    a = _vec(ds)
    return float(a[0]) if a.size else np.nan


def _text(f, ref):
    return "".join(chr(int(c)) for c in np.ravel(f[ref][()]))


def extract_mat(path, b, keep=KEEP_CYCLES):
    import h5py
    out = {}
    with h5py.File(path, "r") as f:
        B = f["batch"]
        pol_key = "policy_readable" if "policy_readable" in B else "policy"
        n = B["summary"].shape[0]
        t0 = time.time()
        for i in range(n):
            cid = f"b{b}c{i}"
            try:
                s = f[B["summary"][i, 0]]
                summary = {k: _vec(s[v]) for k, v in SUMMARY_MAP.items() if v in s}
                cy = f[B["cycles"][i, 0]]
                J = cy["Qdlin"].shape[0] if "Qdlin" in cy else 0
                curves = {nm: {} for nm in ("Qdlin", "Tdlin", "dQdV")}
                for j in range(min(J, keep)):
                    for nm in curves:
                        if nm in cy:
                            curves[nm][j] = _vec(f[cy[nm][j, 0]]).astype(np.float32)
                out[cid] = dict(batch=b, cycle_life=_scalar(f, B["cycle_life"][i, 0]),
                                charge_policy=_text(f, B[pol_key][i, 0]), summary=summary,
                                qdlin=curves["Qdlin"], tdlin=curves["Tdlin"], dqdv=curves["dQdV"],
                                n_cycles_total=int(J))
            except Exception as e:
                print(f"  [WARN] {cid} 추출 실패: {type(e).__name__}: {e}")
            if (i + 1) % 10 == 0 or i + 1 == n:
                print(f"  Batch{b}: {i + 1}/{n} cells ({time.time() - t0:.0f}s)")
    return out


def load_cells(archive: Path, cache: Path):
    """캐시가 있으면 로드, 없으면 .mat 에서 추출 후 저장 (3GB 파일 재로딩 방지)."""
    if cache.exists():
        print(f"[cache] {cache} 로드")
        return pickle.load(open(cache, "rb"))
    cells = {}
    for b, pat in MAT_GLOB.items():
        hits = sorted(Path(archive).glob(pat))
        assert len(hits) == 1, f"Batch{b}: '{pat}' 에 해당하는 파일이 {len(hits)}개"
        print(f"Batch{b}: {hits[0].name}")
        cells.update(extract_mat(hits[0], b))
    cache.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(cells, open(cache, "wb"))
    return cells


# ----------------------------------------------------------------------------- 정제
_RULES = {"QD": lambda x: (x < .5 * NOMINAL_CAP) | (x > 1.2 * NOMINAL_CAP),
          "QC": lambda x: (x < .5 * NOMINAL_CAP) | (x > 1.2 * NOMINAL_CAP),
          "IR": lambda x: (x <= 0) | (x > 0.05),
          "chargetime": lambda x: (x <= 0) | (x > 60),
          "Tavg": lambda x: (x < 10) | (x > 70), "Tmax": lambda x: (x < 10) | (x > 70),
          "Tmin": lambda x: (x < 10) | (x > 70)}


def clean_summary(s: dict) -> tuple[pd.DataFrame, dict]:
    """사이클 정렬·중복 제거 → 물리 범위 밖 값 NaN → 선형보간(연속 5개까지)."""
    n = min(len(v) for v in s.values())
    df = pd.DataFrame({k: v[:n] for k, v in s.items()})
    df["cycle"] = df["cycle"].round().astype(int)
    df = df.drop_duplicates("cycle").sort_values("cycle").set_index("cycle")
    log = {}
    for col, rule in _RULES.items():
        if col in df:
            bad = rule(df[col]); log[f"bad_{col}"] = int(bad.sum()); df.loc[bad, col] = np.nan
    return df.interpolate(limit=5, limit_direction="both"), log


def clean_cells(cells: dict, force_merge: bool = False, verbose: bool = True):
    """정제 5단계. 반환: cells, SUMMARY(dict of DataFrame), meta, attrition(DataFrame), merge_table
       ① cycle_life 결측 제거 ② 이어측정 병합(검증형) ③ summary 정제 ④ 논문 제거 목록 ⑤ 초기 윈도우 무결성"""
    attr = []

    def snap(step):
        attr.append(pd.Series({b: sum(c["batch"] == b for c in cells.values()) for b in (1, 2, 3)}, name=step))

    snap("0. 추출 직후")
    for k in [k for k, v in cells.items() if not np.isfinite(v["cycle_life"])]:
        del cells[k]
    snap("1. life 결측 제거")

    mrows = []
    for k1, k2, add in PUBLISHED["merge_pairs"]:
        if k1 not in cells or k2 not in cells:
            mrows.append(dict(b1=k1, b2=k2, status="셀 없음")); continue
        n2 = len(cells[k2]["summary"]["cycle"])
        verified = (n2 == add)
        do = verified or force_merge
        mrows.append(dict(b1=k1, b2=k2, policy_b1=cells[k1]["charge_policy"], policy_b2=cells[k2]["charge_policy"],
                          b2_cycles=n2, published_add=add, verified=verified, merged=do))
        if do:
            s1, s2 = cells[k1]["summary"], cells[k2]["summary"]; n1 = len(s1["cycle"])
            for key in s1:
                if key in s2:
                    s1[key] = np.hstack([s1[key], s2[key] + (n1 if key == "cycle" else 0)])
            cells[k1]["cycle_life"] += add
            del cells[k2]
    snap("2. 이어측정 병합")

    SUMMARY, rows = {}, []
    for cid, c in cells.items():
        SUMMARY[cid], log = clean_summary(c["summary"]); rows.append({"cid": cid, **log})
    prep_log = pd.DataFrame(rows).set_index("cid")

    for c in [c for c in PUBLISHED["drop_unfinished_b1"] + PUBLISHED["drop_noisy_b3"] if c in cells]:
        del cells[c]; SUMMARY.pop(c, None)
    snap("4. 논문 제거 목록")

    bad_ids = []
    for cid, c in cells.items():
        ok = (SUMMARY[cid].index.max() >= CYC_HIGH) and all(
            k in c["qdlin"] and c["qdlin"][k].size == len(V_GRID) and np.isfinite(c["qdlin"][k]).all()
            for k in (CYC_LOW, CYC_HIGH))
        if not ok:
            bad_ids.append(cid)
    for cid in bad_ids:
        del cells[cid]; SUMMARY.pop(cid)
    snap("5. 초기윈도우 무결성")

    attr_df = pd.DataFrame(attr).T
    attr_df.loc["합계"] = attr_df.sum()
    meta = pd.DataFrame([{"cell_id": cid, "batch": c["batch"], "cycle_life": float(c["cycle_life"]),
                          "policy": c["charge_policy"].strip()} for cid, c in cells.items()])
    meta = meta.sort_values(["batch", "cycle_life"]).reset_index(drop=True)
    meta["role"] = meta["batch"].map({1: "Train", 2: "Test-1", 3: "Test-2"})
    meta["new_structure"] = meta["policy"].str.contains("newstructure")
    meta["log_life"] = np.log10(meta["cycle_life"])
    meta["long_life"] = (meta["cycle_life"] >= LIFE_THRESHOLD).astype(int)
    if verbose:
        print(attr_df); print(meta.groupby("batch").size().to_dict(), "총", len(meta))
    return cells, SUMMARY, meta, attr_df, pd.DataFrame(mrows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=".", help="archive/, cache/ 가 있는 폴더 (Mini project)")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    base = Path(a.base)
    cells = load_cells(base / "archive", base / "cache" / "cells_extracted.pkl")
    cells, SUMMARY, meta, attr, merges = clean_cells(cells)
    Path(a.out).mkdir(exist_ok=True, parents=True)
    meta.to_csv(Path(a.out) / "meta_clean.csv", index=False)
    attr.to_csv(Path(a.out) / "attrition.csv"); merges.to_csv(Path(a.out) / "merge_check.csv", index=False)


if __name__ == "__main__":
    main()
