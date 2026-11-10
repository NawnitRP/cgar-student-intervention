"""Invariant tests for CGAR (run: python tests/test_cgar.py). Uses the HistGB fallback if xgboost is missing."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from sklearn.model_selection import StratifiedKFold
import cgar
from cgar import (MondrianConformal, RecourseEngine, HybridMLPBoost, ece_top_label,
                  selective_curve_maxprob, Plausibility, make_booster)

rng = np.random.default_rng(0)
ok = lambda name, cond, extra="": print(("PASS " if cond else "FAIL ") + name, extra) or cond
res = []

# 1 conformal: class-conditional coverage >= 1-alpha on exchangeable data, exact quantile rank
n = 4000
y = (rng.random(n) < 0.2).astype(int)               # minority class 1 here
p1 = np.clip(0.5 + (y - 0.5) * 0.5 + rng.normal(0, 0.25, n), 0.01, 0.99)
P = np.column_stack([1 - p1, p1])
for alpha in (0.05, 0.10, 0.20):
    covs = []
    for rep in range(40):
        idx = rng.permutation(n); cal, te = idx[:1500], idx[1500:]
        cp = MondrianConformal(alpha).fit(P[cal], y[cal])
        S = cp.predict_set(P[te]); cov = S[np.arange(len(te)), y[te]]
        covs.append([cov[y[te] == 0].mean(), cov[y[te] == 1].mean()])
    m = np.mean(covs, 0)
    res.append(ok(f"conformal coverage alpha={alpha}", (m >= 1 - alpha - 0.01).all(), m.round(3)))
# rank formula
cp = MondrianConformal(0.1).fit(np.array([[0.9, 0.1]] * 5 + [[0.1, 0.9]] * 5),
                                np.array([0] * 5 + [1] * 5))   # n=5: rank ceil(6*0.9)=6 > 5 -> inf
res.append(ok("conformal too few calibration points -> qhat=inf", np.isinf(cp.qhat_).all(), cp.qhat_))
sc = np.arange(1, 20) / 20.0                           # 19 class-0 scores 0.05..0.95
Pc = np.column_stack([1 - sc, sc]); cp = MondrianConformal(0.1).fit(Pc, np.zeros(19, int))
# class-0 score = 1 - p0 = sc ; rank ceil(20*0.9)=18 -> 0.90
res.append(ok("conformal quantile rank", abs(cp.qhat_[0] - 0.90) < 1e-9, cp.qhat_[0]))

# 2 gate thresholds identity used in Proposition 1
cp = MondrianConformal(0.1); cp.qhat_ = np.array([0.7, 0.3])
f = np.linspace(0, 1, 1001); Pf = np.column_stack([1 - f, f]); S = cp.predict_set(Pf)
conf0 = S[:, 0] & ~S[:, 1]; t = min(0.7, 1 - 0.3)
res.append(ok("confident at-risk  <=>  f <= t (t=min(q0,1-q1))", conf0.sum() > 0 and f[conf0].max() <= t + 1e-9, f[conf0].max()))

# 3 ECE: calibrated -> ~0 ; over-confident -> large
pp = rng.random(200000); yy = (rng.random(200000) < pp).astype(int)
res.append(ok("ECE calibrated ~0", ece_top_label(yy, pp) < 0.01, round(ece_top_label(yy, pp), 4)))
res.append(ok("ECE overconfident large", ece_top_label(yy, np.where(pp > .5, .99, .01)) > 0.2))

# 4 selective curve monotone sanity
yy2 = (rng.random(3000) < 0.5).astype(int); pp2 = np.clip(yy2 * 0.3 + 0.35 + rng.normal(0, .15, 3000), 0, 1)
cur = selective_curve_maxprob(yy2, pp2, [0.2, 0.5, 1.0])
res.append(ok("selective acc higher at low coverage", cur[0] >= cur[1] >= cur[2] - 0.02, cur.round(3)))

# 5 hybrid: OOF scores come from models that did not see the row; no-cross-fit is leakier
X = rng.normal(size=(600, 8)); logit = X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 1.0, 600); yb = (logit > 0).astype(int)
h = HybridMLPBoost(0, cross_fit=True).fit(X[:400], yb[:400]); hn = HybridMLPBoost(0, cross_fit=False).fit(X[:400], yb[:400])
from sklearn.metrics import roc_auc_score
# leakage mechanism (averaged over 4 data seeds): the naive in-sample score is inflated on the
# training rows, the cross-fitted (out-of-fold) score is honest (close to held-out AUC)
oo, ins, hos = [], [], []
for sd in range(4):
    r_ = np.random.default_rng(100 + sd)
    Xq = r_.normal(size=(600, 8)); yq = ((Xq[:, 0] + 0.5 * Xq[:, 1] + r_.normal(0, 1.0, 600)) > 0).astype(int)
    hq = HybridMLPBoost(0).fit(Xq[:400], yq[:400]); hnq = HybridMLPBoost(0, cross_fit=False).fit(Xq[:400], yq[:400])
    oo.append(roc_auc_score(yq[:400], hq.oof_)); ins.append(roc_auc_score(yq[:400], hnq.oof_))
    hos.append(roc_auc_score(yq[400:], hq.mlp_.predict_proba(hq.scaler_.transform(Xq[400:]))[:, 1]))
res.append(ok("naive in-sample score inflated; OOF score honest",
              np.mean(ins) > np.mean(oo) + 0.1 and abs(np.mean(oo) - np.mean(hos)) < 0.05,
              dict(oof=round(np.mean(oo), 3), insample=round(np.mean(ins), 3), heldout=round(np.mean(hos), 3))))
h = HybridMLPBoost(0).fit(X[:400], yb[:400])
pr = h.predict_proba(X[400:]); res.append(ok("predict_proba rows sum to 1", np.allclose(pr.sum(1), 1)))

# 6 recourse invariants
names = [f"f{i}" for i in range(8)]; Xt = X[:400]
f_pass = lambda A: h.predict_proba(A)[:, 1]
acts = {"f0": "up", "f1": "down", "f2": "both"}
eng = RecourseEngine(f_pass, Xt, names, acts, k_max=2, target=0.6)
bad = dict(imm=0, dirn=0, k=0, grid=0, cost=0, p=0, n=0)
for x0 in X[400:500]:
    if f_pass(x0[None])[0] >= 0.6: continue
    for r in (eng.recommend(x0), eng.recommend_beam(x0, 3)):
        bad["n"] += 1; x = r["x"]
        bad["imm"] += int(np.any(x[3:] != x0[3:]))
        bad["dirn"] += int(x[0] < x0[0] - 1e-12 or x[1] > x0[1] + 1e-12)
        bad["k"] += int(np.sum(x != x0) > 2)
        for j in (0, 1, 2):
            if x[j] != x0[j] and not np.any(np.isclose(eng.grid[j], x[j])): bad["grid"] += 1
        c = sum(abs(x[j] - x0[j]) / eng.scale[j] for j in (0, 1, 2))
        bad["cost"] += int(abs(c - r["cost"]) > 1e-6)
        bad["p"] += int(abs(f_pass(x[None])[0] - r["p"]) > 1e-6)
res.append(ok("recourse: immutable untouched / directions / k<=2 / on grid / cost / p consistent",
              all(v == 0 for k, v in bad.items() if k != "n") and bad["n"] > 20, bad))
# success flag matches target
r = eng.recommend(X[400]); res.append(ok("success flag == (p>=target)", r["success"] == (r["p"] >= 0.6)))
# already above target -> no change
xa = X[400:][np.argmax(f_pass(X[400:]))]; r = eng.recommend(xa)
res.append(ok("already >= target -> zero changes", len(r["changed"]) == 0 and r["success"]))
# unconstrained can change immutable ones
eu = RecourseEngine(f_pass, Xt, names, {n: "both" for n in names}, k_max=2, target=0.6)
chg = set()
for x0 in X[400:480]:
    if f_pass(x0[None])[0] < 0.6: chg |= set(eu.recommend(x0)["changed"])
res.append(ok("unconstrained touches non-actionable features", len(chg - set(acts)) > 0, sorted(chg)))
# max_reach bound >= any realised cost of a successful plan
okb = True
for x0 in X[400:480]:
    rr = eng.recommend(x0)
    if rr["changed"] and rr["cost"] > eng.max_reach(x0) + 1e-9: okb = False
res.append(ok("cost <= B_k(x)", okb))

# 7 plausibility: training points plausible, far points not
pl = Plausibility(Xt); res.append(ok("plausibility", pl.ok(Xt[:50]).all() and not pl.ok(Xt[:3] + 50).any()))

# 8 tuner: default always a candidate (tuned CV score >= default CV score)
yt = yb[:400]; params = cgar.tune_booster(Xt, yt, 0, n_iter=3)
res.append(ok("tune_booster returns dict", isinstance(params, dict), params))
print("\nALL PASSED" if all(res) else "\nSOME FAILED"); sys.exit(0 if all(res) else 1)
