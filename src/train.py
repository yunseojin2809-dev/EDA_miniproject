"""모델 비교 → 선택(Train 정책단위 CV 기준) → Valid / Test 평가 → results/ 저장.

    python -m src.train --eval-batches 2          # 필수 (Batch 2)
    python -m src.train --eval-batches 2 3        # Batch 3 추가 시

원칙: Batch 2·3 은 모델 선택에 사용하지 않고, 확정된 모델로 최종 1회만 평가한다.
"""
from __future__ import annotations

import argparse, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .validation import extrapolation_splits, group_holdout, mape, random_folds, repeated_group_folds

warnings.filterwarnings("ignore")
SEED = 42
TARGET_MAPE = 9.1          # 원논문 회귀 성능 (초기 100 사이클, MAPE %)
BASELINE = ("V1 var", "Linear")

MODELS = {
    "Linear":     lambda: make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge":      lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 30))),
    "ElasticNet": lambda: make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[.1, .5, .9, .99], cv=3, max_iter=50000, random_state=SEED)),
    "PLS(2)":     lambda: make_pipeline(StandardScaler(), PLSRegression(n_components=2)),
    "GPR":        lambda: make_pipeline(StandardScaler(), GaussianProcessRegressor(ConstantKernel() * RBF(3.0) + WhiteKernel(), normalize_y=True, random_state=SEED)),
    "GBM(d=2)":   lambda: GradientBoostingRegressor(max_depth=2, n_estimators=150, learning_rate=0.05, subsample=0.8, min_samples_leaf=4, random_state=SEED),
}
# 피처셋: Day 1 EDA 결과 반영 (QD_intercept/t80_min/C_avg/newstructure/knee 계열은 제외)
FEATURE_SETS = {
    "V1 var":                  ["dq_log_var"],
    "V2 var+min":              ["dq_log_var", "dq_log_abs_min"],
    "D4 dQ4":                  ["dq_log_var", "dq_log_abs_min", "dq_skew", "dq_kurt"],
    "D6 dQ4+QD2+QDmax":        ["dq_log_var", "dq_log_abs_min", "dq_skew", "dq_kurt", "QD_2", "QD_max_minus_2"],
    "S7 dQ4+chg+Tavg+IRmin":   ["dq_log_var", "dq_log_abs_min", "dq_skew", "dq_kurt", "chargetime_avg_2_6", "Tavg_mean_2_100", "IR_min_2_100"],
}


def load_data(data_dir: Path) -> pd.DataFrame:
    feat = pd.read_csv(data_dir / "features_cell_level.csv")
    meta = pd.read_csv(data_dir / "meta_clean.csv")[["cell_id", "policy", "new_structure"]]
    return feat.merge(meta, on="cell_id")


def valid_combos():
    for fs, cols in FEATURE_SETS.items():
        for mn in MODELS:
            if len(cols) == 1 and mn not in ("Linear", "GPR", "GBM(d=2)"):
                continue
            if mn == "PLS(2)" and len(cols) < 2:
                continue
            yield fs, cols, mn


def cv_score(d, cols, make, splits):
    X, y = d[cols].values, d.log_life.values
    s = [mape(y[te], make().fit(X[tr], y[tr]).predict(X[te])) for tr, te in splits]
    return float(np.mean(s)), float(np.std(s))


def compare_models(TR: pd.DataFrame, VA: pd.DataFrame) -> pd.DataFrame:
    """Train 안에서 정책단위 CV(선택 기준) / 랜덤 CV(Day 1 방식, 비교용) / Valid 를 모두 계산."""
    rows = []
    for fs, cols, mn in valid_combos():
        d = TR.dropna(subset=cols).reset_index(drop=True)
        g_m, g_s = cv_score(d, cols, MODELS[mn], repeated_group_folds(d.policy.values, 5, 10, 0))
        r_m, r_s = cv_score(d, cols, MODELS[mn], random_folds(len(d), 5, 10, 0))
        e_m, _ = cv_score(d, cols, MODELS[mn], extrapolation_splits(d))
        m = MODELS[mn]().fit(d[cols].values, d.log_life.values)
        v = VA.dropna(subset=cols)
        rows.append(dict(featset=fs, model=mn, n_feat=len(cols), CV_group=g_m, CV_group_sd=g_s,
                         CV_random=r_m, CV_random_sd=r_s, CV_extrap=e_m, score_stage2=(g_m + e_m) / 2, Valid=mape(v.log_life.values, m.predict(v[cols].values))))
    return pd.DataFrame(rows)


def fit_eval(fit_df, eval_df, cols, mn):
    f = fit_df.dropna(subset=cols); e = eval_df.dropna(subset=cols)
    m = MODELS[mn]().fit(f[cols].values, f.log_life.values)
    pred = 10 ** np.ravel(m.predict(e[cols].values))
    out = e[["cell_id", "batch", "policy", "new_structure", "cycle_life"]].copy()
    out["pred"] = pred
    out["ape"] = np.abs(pred - out.cycle_life) / out.cycle_life * 100
    return mape(e.log_life.values, np.log10(pred)), out


def holdout_robustness(b1, cols, mn, n=30):
    s = []
    for seed in range(n):
        try:
            tr, va, _ = group_holdout(b1, seed)
        except RuntimeError:
            continue
        s.append(fit_eval(b1.loc[tr], b1.loc[va], cols, mn)[0])
    return np.array(s)


STAGES = {"stage1": ("1차 (사전 확정)", "CV_group"),          # 정책단위 CV 최소
          "stage2": ("개선 후 (외삽 검증 반영)", "score_stage2")}   # (정책단위 CV + 외삽 CV)/2 최소
PLOT_TAG = {"stage1": "Stage 1 (pre-registered)", "stage2": "Stage 2 (extrapolation-aware)"}   # 그림 제목은 영문 (한글 폰트 없는 환경 대비)


def perf_table(train_cv, train_sd, valid, rob, n_valid, perf, fs, mn):
    t2 = perf[2]["test"]
    rows = [("Train (Batch 1 CV)", train_cv, f"정책단위 Repeated 5-Fold x10, SD={train_sd:.1f}"),
            ("Valid (Batch 1 Hold-out)", valid, f"정책단위 Hold-out {n_valid}셀; 30개 시드 평균 {rob.mean():.1f}+-{rob.std():.1f} (범위 {rob.min():.1f}~{rob.max():.1f})"),
            ("Test (Batch 2)", t2, f"Batch1 전체({perf[2]['n_fit']}셀) 재학습, n={perf[2]['n']}; Train 부분만 학습 시 {perf[2]['test_trainonly']:.1f}"),
            ("Gap (Train-Valid)", valid - train_cv, "(+) : 과적합 의심  [Valid - Train]"),
            ("Gap (Valid-Test)", t2 - valid, "(+) : 배치간 일반화 저하 의심  [Test - Valid]"),
            ("Gap (Target-Test)", t2 - TARGET_MAPE, f"Target : 원논문 {TARGET_MAPE}%  [Test - Target]")]
    if 3 in perf:
        t3 = perf[3]["test"]
        rows += [("Test (Batch 3)", t3, f"n={perf[3]['n']}"), ("Gap (Batch2-Batch3)", t3 - t2, "Test 성능 간 비교  [Batch3 - Batch2]"),
                 ("Gap (Target-Test) [Batch 3]", t3 - TARGET_MAPE, "Batch 3 기준, 원논문 성능 비교")]
    out = pd.DataFrame(rows, columns=["구분", "MAPE (%)", "비고"]); out["MAPE (%)"] = out["MAPE (%)"].round(1)
    return out


def plot_pred(P, T, b1, fs, mn, title_tag, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    lo, hi = P.cycle_life.min() * .85, max(P.cycle_life.max(), P.pred.max()) * 1.1
    ax[0].plot([lo, hi], [lo, hi], color="#8a8984", lw=1, ls="--")
    for lab, g, c, mk in [("Valid (B1 hold-out)", P[P.split == "Valid"], "#2a78d6", "o"),
                          ("Test B2 legacy", T[~T.new_structure], "#eb6834", "o"),
                          ("Test B2 newstructure", T[T.new_structure], "#eb6834", "D"),
                          ("Test B3 (all newstructure)", P[P.split == "Test(B3)"], "#2f9e6e", "s")]:
        ax[0].scatter(g.cycle_life, g.pred, s=34, c=c, marker=mk, edgecolor="white", lw=.8, label=lab, zorder=3)
    ax[0].axvline(b1.cycle_life.min(), color="#8a8984", lw=.8, ls=":"); ax[0].text(b1.cycle_life.min(), hi * .97, " Train min", fontsize=8, va="top", color="#5c5b56")
    ax[0].set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi), xlabel="True cycle life", ylabel="Predicted cycle life", title=f"{title_tag}: {fs} / {mn}"); ax[0].legend(fontsize=8)
    Ts = T.sort_values("cycle_life")
    ax[1].bar(range(len(Ts)), Ts.ape, color=np.where(Ts.new_structure, "#eb6834", "#86b6ef"), width=.75)
    ax[1].axhline(TARGET_MAPE, color="#2b2b29", ls="--", lw=1); ax[1].text(0, TARGET_MAPE + 1, " paper 9.1%", fontsize=8)
    ax[1].set(xlabel="Batch 2 cells (sorted by true life)", ylabel="APE (%)", title="Batch 2 error per cell (orange = newstructure)")
    for s_ in ("top", "right"): ax[0].spines[s_].set_visible(False); ax[1].spines[s_].set_visible(False)
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def run_stage(stage, df, b1, TR, VA, cv, eval_batches, out):
    label, key = STAGES[stage]
    best = cv.sort_values(key).iloc[0]; fs, mn = best.featset, best.model; cols = FEATURE_SETS[fs]
    print(f"\n=== {label} :: 선택 기준 {key} -> {fs} / {mn}  (CV_group={best.CV_group:.1f}, CV_extrap={best.CV_extrap:.1f}) ===")
    valid, valid_pred = fit_eval(TR, VA, cols, mn)
    rob = holdout_robustness(b1, cols, mn)
    perf, preds = {}, [valid_pred.assign(split="Valid")]
    for b in eval_batches:
        tb = df[df.batch == b].reset_index(drop=True)
        t_full, p = fit_eval(b1, tb, cols, mn); t_tr, _ = fit_eval(TR, tb, cols, mn)
        perf[b] = dict(test=t_full, test_trainonly=t_tr, n=len(p), n_fit=len(b1.dropna(subset=cols))); preds.append(p.assign(split=f"Test(B{b})"))
    perf_df = perf_table(best.CV_group, best.CV_group_sd, valid, rob, len(VA), perf, fs, mn)
    print(perf_df.to_string(index=False))
    P = pd.concat(preds, ignore_index=True); P["below_train_min"] = P.cycle_life < b1.cycle_life.min()
    T = P[P.split == "Test(B2)"]
    err = pd.concat([T.groupby("new_structure").ape.agg(["count", "mean", "median", "max"]).reset_index().rename(columns={"new_structure": "group"}).assign(by="new_structure"),
                     T.groupby("below_train_min").ape.agg(["count", "mean", "median", "max"]).reset_index().rename(columns={"below_train_min": "group"}).assign(by="below_train_min")])
    T = T.assign(signed_err_pct=(T.pred - T.cycle_life) / T.cycle_life * 100)
    print("Test(B2) 평균 부호오차(예측-실제)/실제 = %.1f%%" % T.signed_err_pct.mean()); print(err.round(1).to_string(index=False))
    d = out / stage; d.mkdir(parents=True, exist_ok=True)
    perf_df.to_csv(d / "model_performance.csv", index=False); P.round(2).to_csv(d / "predictions.csv", index=False)
    err.round(2).to_csv(d / "error_analysis.csv", index=False)
    plot_pred(P, T, b1, fs, mn, PLOT_TAG[stage], out / "figures" / f"pred_vs_true_{stage}.png")
    return dict(label=label, fs=fs, mn=mn, perf=perf_df, perf_raw=perf, valid=valid, train=best.CV_group, T=T, signed=T.signed_err_pct.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="results")
    ap.add_argument("--eval-batches", type=int, nargs="+", default=[2])
    a = ap.parse_args(); out = Path(a.out); (out / "figures").mkdir(parents=True, exist_ok=True)

    df = load_data(Path(a.data)); b1 = df[df.batch == 1].reset_index(drop=True)
    tr_i, va_i, chosen = group_holdout(b1, SEED)
    TR, VA = b1.loc[tr_i].reset_index(drop=True), b1.loc[va_i].reset_index(drop=True)
    print(f"Batch1 {len(b1)}셀 -> Train {len(TR)} / Valid {len(VA)} (정책 단위, Valid 정책: {chosen})")

    cv = compare_models(TR, VA); cv.sort_values("score_stage2").to_csv(out / "cv_comparison.csv", index=False)
    pd.set_option("display.width", 220)
    print(cv.sort_values("score_stage2").round(1).to_string())

    res = {s: run_stage(s, df, b1, TR, VA, cv, a.eval_batches, out) for s in STAGES}
    s1, s2 = res["stage1"]["perf"], res["stage2"]["perf"]
    comb = s1[["구분", "MAPE (%)"]].rename(columns={"MAPE (%)": "1차 MAPE (%)"}).merge(
        s2.rename(columns={"MAPE (%)": "개선 후 MAPE (%)", "비고": "비고 (개선 후)"}), on="구분")
    comb.to_csv(out / "model_performance.csv", index=False)
    print("\n[최종 성능표]\n", comb.to_string(index=False))
    bfs, bmn = BASELINE; brow = cv[(cv.featset == bfs) & (cv.model == bmn)].iloc[0]
    bt = {b: fit_eval(b1, df[df.batch == b], FEATURE_SETS[bfs], bmn)[0] for b in a.eval_batches}
    rows = [dict(model=f"{bfs} / {bmn} (논문 Variance baseline)", train_cv=brow.CV_group, valid=brow.Valid, **{f"test_b{b}": bt[b] for b in bt})]
    rows += [dict(model=f"{r['fs']} / {r['mn']} ({r['label']})", train_cv=r["train"], valid=r["valid"],
                  **{f"test_b{b}": r["perf_raw"][b]["test"] for b in a.eval_batches}) for r in res.values()]
    pd.DataFrame(rows).round(1).to_csv(out / "baseline_vs_final.csv", index=False)

if __name__ == "__main__":
    main()
