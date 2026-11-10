"""
cgar.py - Confidence-Gated Actionable Recommendation (CGAR): core components.

Stage 1  HybridMLPBoost      cross-fitted MLP score stacked into a gradient-boosted tree model
Stage 2  MondrianConformal   class-conditional split conformal prediction -> confidence gate
Stage 3  RecourseEngine      actionable, sparse, bounded counterfactual recommendation
         Plausibility        nearest-neighbour density check for recommended profiles

Only numpy / scikit-learn are required. If `xgboost` is installed it is used as the
boosting component; otherwise scikit-learn's HistGradientBoostingClassifier is used
and `HAVE_XGB` is False (the experiment runner refuses to run in that case unless
you pass --allow-fallback, so a paper can never silently report the wrong learner).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:  # preferred boosting backend
    from xgboost import XGBClassifier

    HAVE_XGB = True
except Exception:  # pragma: no cover
    from sklearn.ensemble import HistGradientBoostingClassifier

    HAVE_XGB = False

BOOST_NAME = "XGBoost" if HAVE_XGB else "HistGB"


# --------------------------------------------------------------------------- #
# Stage 1: hybrid MLP -> boosting
# --------------------------------------------------------------------------- #
def make_booster(seed: int = 0, params: dict | None = None):
    p = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
             min_child_weight=1, reg_lambda=1.0)
    if params:
        p.update(params)
    if HAVE_XGB:
        return XGBClassifier(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"], learning_rate=p["learning_rate"],
            subsample=p["subsample"], colsample_bytree=p["colsample_bytree"],
            min_child_weight=p["min_child_weight"], reg_lambda=p["reg_lambda"],
            eval_metric="logloss", random_state=seed, n_jobs=1,
        )
    return HistGradientBoostingClassifier(
        max_iter=p["n_estimators"], max_depth=p["max_depth"], learning_rate=p["learning_rate"],
        l2_regularization=p["reg_lambda"], random_state=seed
    )


TUNE_SPACE = dict(n_estimators=[100, 200, 300, 500], max_depth=[2, 3, 4, 5, 6],
                  learning_rate=[0.02, 0.05, 0.1], subsample=[0.6, 0.8, 1.0],
                  colsample_bytree=[0.5, 0.8, 1.0], min_child_weight=[1, 3, 5],
                  reg_lambda=[0.5, 1.0, 5.0, 10.0])


def tune_booster(X, y, seed: int = 0, n_iter: int = 10, cv: int = 3) -> dict:
    """Random search on the TRAINING split only (stratified CV, AUC). The default configuration
    is always among the candidates, so tuning can only help on the validation criterion."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    cands = [{}] + [{k: v[rng.integers(len(v))] for k, v in TUNE_SPACE.items()}
                    for _ in range(n_iter)]
    skf = StratifiedKFold(cv, shuffle=True, random_state=seed)
    best, best_auc = {}, -1.0
    for c in cands:
        aucs = []
        for tr, va in skf.split(X, y):
            m = make_booster(seed, c).fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[va], m.predict_proba(X[va])[:, 1]))
        a = float(np.mean(aucs))
        if a > best_auc:
            best, best_auc = c, a
    return best


class TunedBooster:
    """Gradient-boosted trees with hyper-parameters chosen by CV on the training split."""

    def __init__(self, seed: int = 0, n_iter: int = 10):
        self.seed, self.n_iter = seed, n_iter

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, int)
        self.params_ = tune_booster(X, y, self.seed, self.n_iter)
        self.m_ = make_booster(self.seed, self.params_).fit(X, y)
        return self

    def predict_proba(self, X):
        return self.m_.predict_proba(np.asarray(X, float))


def make_mlp(seed: int = 0):
    # No early stopping: on small tabular datasets (a few hundred rows) the internal
    # validation split is too small and stopped training near initialisation. Fixed L2
    # regularisation instead; the same default is used for the MLP baseline and the hybrid.
    return MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-1,
        learning_rate_init=1e-3, max_iter=300, early_stopping=False, random_state=seed,
    )


class HybridMLPBoost:
    """MLP score (cross-fitted) + raw features -> gradient-boosted trees.

    Why a cross-fitted *score* and not the hidden-layer embedding: embeddings from
    different cross-fitting folds live in different (permuted) coordinate systems, so
    they cannot be stacked consistently; the predicted probability has the same meaning
    in every fold. Training rows receive out-of-fold MLP scores (no label leakage);
    test rows receive the score of an MLP trained on all training rows.
    """

    def __init__(self, seed: int = 0, n_folds: int = 5, cross_fit: bool = True,
                 tune: bool = False, n_iter: int = 10):
        self.tune, self.n_iter = tune, n_iter
        # cross_fit=False is the *ablation* "naive stacking": the booster is trained on the
        # in-sample MLP score of the very rows the MLP was fitted on (label leakage).
        self.seed, self.n_folds, self.cross_fit = seed, n_folds, cross_fit

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, int)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self.mlp_ = make_mlp(self.seed).fit(Xs, y)
        if self.cross_fit:
            oof = np.zeros(len(y))
            skf = StratifiedKFold(self.n_folds, shuffle=True, random_state=self.seed)
            for tr, va in skf.split(Xs, y):
                m = make_mlp(self.seed).fit(Xs[tr], y[tr])
                oof[va] = m.predict_proba(Xs[va])[:, 1]
        else:
            oof = self.mlp_.predict_proba(Xs)[:, 1]
        self.oof_ = oof     # kept for diagnostics / tests
        Z = np.column_stack([X, oof])
        params = tune_booster(Z, y, self.seed, self.n_iter) if self.tune else None
        self.params_ = params
        self.boost_ = make_booster(self.seed, params).fit(Z, y)
        return self

    def _stack(self, X):
        X = np.asarray(X, float)
        s = self.mlp_.predict_proba(self.scaler_.transform(X))[:, 1]
        return np.column_stack([X, s])

    def predict_proba(self, X):
        return self.boost_.predict_proba(self._stack(X))


# --------------------------------------------------------------------------- #
# Stage 2: class-conditional (Mondrian) split conformal prediction
# --------------------------------------------------------------------------- #
class MondrianConformal:
    """Split conformal with score 1 - p(true class); one threshold per true class.

    Class-conditional calibration gives coverage >= 1-alpha *for each class* (including
    the minority at-risk class) under exchangeability of calibration and test data.
    """

    def __init__(self, alpha: float = 0.10, mondrian: bool = True):
        self.alpha, self.mondrian = alpha, mondrian

    def fit(self, proba_cal, y_cal):
        proba_cal, y_cal = np.asarray(proba_cal), np.asarray(y_cal, int)
        K = proba_cal.shape[1]
        scores = 1.0 - proba_cal[np.arange(len(y_cal)), y_cal]
        self.qhat_ = np.full(K, np.inf)
        for k in range(K):
            s = np.sort(scores[y_cal == k] if self.mondrian else scores)
            n = len(s)
            r = int(np.ceil((n + 1) * (1 - self.alpha)))
            self.qhat_[k] = s[r - 1] if (n > 0 and r <= n) else np.inf
        return self

    def predict_set(self, proba):
        """Boolean matrix (n, K): True where class k is in the prediction set."""
        return (1.0 - np.asarray(proba)) <= self.qhat_[None, :]


def selective_curve_maxprob(y, p1, grid):
    """Accuracy among the `c` most confident predictions, for coverage fractions in grid."""
    conf = np.maximum(p1, 1 - p1)
    correct = ((p1 >= 0.5).astype(int) == y).astype(float)
    order = np.argsort(-conf, kind="stable")
    cs = np.cumsum(correct[order])
    out = []
    for c in grid:
        m = max(1, int(round(c * len(y))))
        out.append(cs[m - 1] / m)
    return np.array(out)


# --------------------------------------------------------------------------- #
# Stage 3: actionable counterfactual recommendation
# --------------------------------------------------------------------------- #
class RecourseEngine:
    """Greedy, sparse, bounded counterfactual search over *actionable* features only.

    actions : dict feature_name -> "up" | "down" | "both"   (feasible direction w.r.t. x0)
    Moves are restricted to values on a quantile grid of the training data (bounded and
    in-range), at most `k_max` distinct features are changed, and the search stops as soon
    as P(pass | x') >= target. Cost is the IQR-normalised L1 distance.
    """

    def __init__(self, predict_pass, X_train, feature_names, actions,
                 k_max=3, target=0.6, n_grid=9, max_steps=6):
        self.f, self.names = predict_pass, list(feature_names)
        self.k_max, self.target, self.max_steps = k_max, target, max_steps
        self.idx = [self.names.index(n) for n in actions]
        self.dir = {self.names.index(n): d for n, d in actions.items()}
        X = np.asarray(X_train, float)
        self.grid, self.scale = {}, {}
        for j in self.idx:
            col = X[:, j]
            u = np.unique(col)
            if len(u) > n_grid:
                u = np.unique(np.quantile(col, np.linspace(0.05, 0.95, n_grid)))
            self.grid[j] = u
            q25, q75 = np.percentile(col, [25, 75])
            s = q75 - q25
            if s <= 0:
                s = col.std()
            self.scale[j] = s if s > 0 else 1.0
        self.max_ratio = 0.0   # empirical Lipschitz-type constant: max gain in f per unit cost

    def max_reach(self, x0):
        """Largest feasible normalised change available in the best k_max actionable features
        (used by the feasibility bound of Proposition 1)."""
        x0 = np.asarray(x0, float)
        r = []
        for j in self.idx:
            d, g = self.dir[j], self.grid[j]
            ok = g[g > x0[j]] if d == "up" else g[g < x0[j]] if d == "down" else g[g != x0[j]]
            r.append(np.max(np.abs(ok - x0[j])) / self.scale[j] if len(ok) else 0.0)
        return float(np.sum(sorted(r, reverse=True)[: self.k_max]))

    def _moves(self, x, changed):
        rows = []
        for j in self.idx:
            if j not in changed and len(changed) >= self.k_max:
                continue
            d = self.dir[j]
            for v in self.grid[j]:
                if d == "up" and not v > x[j]:
                    continue
                if d == "down" and not v < x[j]:
                    continue
                if d == "both" and v == x[j]:
                    continue
                rows.append((j, v))
        return rows

    def recommend_beam(self, x0, width=3):
        """Beam search (width `width`) over the same bounded, sparse, direction-constrained move
        set. Keeps the `width` highest-probability partial plans per step; among plans that reach
        the target it returns the cheapest."""
        x0 = np.asarray(x0, float)
        p0 = float(self.f(x0[None])[0])
        beam = [(x0.copy(), frozenset(), p0, 0.0)]
        done = None
        for _ in range(self.max_steps):
            cand = []
            for x, ch, p, cost in beam:
                rows = self._moves(x, ch)
                if not rows:
                    continue
                C = np.repeat(x[None], len(rows), axis=0)
                for r, (j, v) in enumerate(rows):
                    C[r, j] = v
                pc = self.f(C)
                inc = np.array([abs(v - x[j]) / self.scale[j] for j, v in rows])
                self.max_ratio = max(self.max_ratio, float(np.max((pc - p) / (inc + 1e-9))))
                for r, (j, v) in enumerate(rows):
                    if pc[r] > p + 1e-6:
                        cand.append((C[r], ch | {j}, float(pc[r]), cost + float(inc[r])))
            if not cand:
                break
            hit = [c for c in cand if c[2] >= self.target]
            if hit:
                done = min(hit, key=lambda c: c[3])
                break
            cand.sort(key=lambda c: (-c[2], c[3]))
            seen, beam = set(), []
            for c in cand:
                key = tuple(c[0])
                if key not in seen:
                    seen.add(key)
                    beam.append(c)
                if len(beam) == width:
                    break
        x, ch, p, cost = done if done else max(beam, key=lambda c: c[2])
        diff = {self.names[j]: (float(x0[j]), float(x[j])) for j in ch if x[j] != x0[j]}
        return {"x": x, "p": p, "changed": diff, "cost": cost, "success": p >= self.target}

    def recommend(self, x0):
        x0 = np.asarray(x0, float)
        x, changed = x0.copy(), set()
        p = float(self.f(x[None])[0])
        steps = 0
        while p < self.target and steps < self.max_steps:
            rows = []
            for j in self.idx:
                if j not in changed and len(changed) >= self.k_max:
                    continue
                d = self.dir[j]
                for v in self.grid[j]:
                    if d == "up" and not v > x[j]:
                        continue
                    if d == "down" and not v < x[j]:
                        continue
                    if d == "both" and v == x[j]:
                        continue
                    rows.append((j, v))
            if not rows:
                break
            C = np.repeat(x[None], len(rows), axis=0)
            for r, (j, v) in enumerate(rows):
                C[r, j] = v
            pc = self.f(C)
            inc = np.array([abs(v - x[j]) / self.scale[j] for j, v in rows])
            self.max_ratio = max(self.max_ratio, float(np.max((pc - p) / (inc + 1e-9))))
            reach = pc >= self.target
            if reach.any():                      # cheapest move that reaches the target
                r = int(np.argmin(np.where(reach, inc, np.inf)))
            else:                                # otherwise best gain per unit cost
                gain = pc - p
                if gain.max() <= 1e-6:
                    break
                r = int(np.argmax(np.where(gain > 1e-6, gain / (inc + 1e-9), -np.inf)))
            j, v = rows[r]
            x[j] = v
            changed.add(j)
            p = float(pc[r])
            steps += 1
        diff = {self.names[j]: (float(x0[j]), float(x[j])) for j in changed if x[j] != x0[j]}
        cost = float(sum(abs(x[j] - x0[j]) / self.scale[j] for j in changed))
        return {"x": x, "p": p, "changed": diff, "cost": cost, "success": p >= self.target}


class Plausibility:
    """A recommended profile is plausible if its nearest training neighbour is no farther
    than the q-th percentile of leave-one-out nearest-neighbour distances in training."""

    def __init__(self, X_train, q=95):
        self.sc = StandardScaler().fit(np.asarray(X_train, float))
        Z = self.sc.transform(np.asarray(X_train, float))
        self.nn = NearestNeighbors(n_neighbors=2).fit(Z)
        d, _ = self.nn.kneighbors(Z)
        self.thr = float(np.percentile(d[:, 1], q))

    def ok(self, X):
        Z = self.sc.transform(np.asarray(X, float))
        d, _ = self.nn.kneighbors(Z, n_neighbors=1)
        return d[:, 0] <= self.thr


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def ece_top_label(y, p1, bins=10):
    y = np.asarray(y)
    conf = np.maximum(p1, 1 - p1)
    correct = ((p1 >= 0.5).astype(int) == y).astype(float)
    edges = np.linspace(0.5, 1.0, bins + 1)
    e = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b) if a > 0.5 else (conf >= a) & (conf <= b)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)
