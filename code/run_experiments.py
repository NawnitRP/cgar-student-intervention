#!/usr/bin/env python3
"""
run_experiments.py - reproduces every number, table and figure of the paper.

Example
-------
    python run_experiments.py --uci-dir data/uci --oulad-dir data/oulad --out ../paper/results

All LaTeX tables / figures are written into --out; the manuscript `\\input`s them, so no
number in the paper is typed by hand. A run configuration + library versions are written
to <out>/summary.json.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time

import matplotlib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import cgar
from cgar import (HybridMLPBoost, MondrianConformal, Plausibility, RecourseEngine, TunedBooster,
                  ece_top_label, make_booster, make_mlp, selective_curve_maxprob)
from datasets import Dataset, drop_overlap, load_oulad, load_uci_pair, load_xapi

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ALL_KEYS = ["LR", "RF", "MLP", "Boost", "BoostT", "HybridNaive", "Hybrid", "HybridT"]
MODEL_LABEL = {"LR": "Logistic Reg.", "RF": "Random Forest", "MLP": "MLP",
               "Boost": cgar.BOOST_NAME, "BoostT": f"{cgar.BOOST_NAME} (tuned)",
               "HybridNaive": f"MLP+{cgar.BOOST_NAME}, no cross-fit",
               "Hybrid": f"MLP+{cgar.BOOST_NAME} (ours)",
               "HybridT": f"MLP+{cgar.BOOST_NAME} (ours, tuned)"}
MODEL_KEYS = list(ALL_KEYS)     # trimmed in main() to the models actually run
PRIMARY = "Hybrid"              # predictor used by the gate and the recommender
# attributes used for the subgroup report (column name in the numeric feature table)
SENSITIVE = {"UCI-Math": ["sex_F", "address_R"], "UCI-Port": ["sex_F", "address_R"],
             "OULAD": ["gender_F", "disability_Y"], "xAPI": ["gender_F"]}


# ----------------------------------------------------------------------------- #
# helpers
# ----------------------------------------------------------------------------- #
def split3(y, groups, seed, f_train=0.6, f_cal=0.2):
    idx = np.arange(len(y))
    if groups is None:
        tr, rest = train_test_split(idx, train_size=f_train, stratify=y, random_state=seed)
        cal, te = train_test_split(rest, train_size=f_cal / (1 - f_train),
                                   stratify=y[rest], random_state=seed + 1)
    else:
        tr, rest = next(GroupShuffleSplit(1, train_size=f_train, random_state=seed)
                        .split(idx, y, groups))
        c, t = next(GroupShuffleSplit(1, train_size=f_cal / (1 - f_train),
                                      random_state=seed + 1).split(rest, y[rest], groups[rest]))
        cal, te = rest[c], rest[t]
    return tr, cal, te


def build_models(seed, tune=False):
    m = {
        "LR": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "RF": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1),
        "MLP": make_pipeline(StandardScaler(), make_mlp(seed)),
        "Boost": make_booster(seed),
        "HybridNaive": HybridMLPBoost(seed, cross_fit=False),   # ablation: leaky stacking
        "Hybrid": HybridMLPBoost(seed),
    }
    if tune:    # hyper-parameters chosen by 3-fold CV on the training split only
        m["BoostT"] = TunedBooster(seed)
        m["HybridT"] = HybridMLPBoost(seed, tune=True)
    return m


def clf_metrics(y, p1):
    pred = (p1 >= 0.5).astype(int)
    return dict(acc=accuracy_score(y, pred), f1=f1_score(y, pred, average="macro"),
                auc=roc_auc_score(y, p1), brier=float(np.mean((p1 - y) ** 2)),
                ece=ece_top_label(y, p1))


def conformal_stats(P_cal, y_cal, P_te, y_te, alpha):
    cp = MondrianConformal(alpha).fit(P_cal, y_cal)
    S = cp.predict_set(P_te)
    size = S.sum(1)
    covered = S[np.arange(len(y_te)), y_te]
    single = size == 1
    lab = S.argmax(1)
    st = dict(alpha=alpha, coverage=float(covered.mean()),
              cov_atrisk=float(covered[y_te == 0].mean()) if (y_te == 0).any() else np.nan,
              cov_pass=float(covered[y_te == 1].mean()) if (y_te == 1).any() else np.nan,
              answered=float(single.mean()),
              sel_acc=float((lab[single] == y_te[single]).mean()) if single.any() else np.nan,
              acc_all=float((P_te.argmax(1) == y_te).mean()))
    if single.any():   # max-probability threshold answering the same number of students
        st["thr_acc"] = float(selective_curve_maxprob(y_te, P_te[:, 1], [st["answered"]])[0])
    else:
        st["thr_acc"] = np.nan
    return st, S


def subset(ds: Dataset, mask) -> Dataset:
    return Dataset(ds.name, ds.X.loc[mask].reset_index(drop=True), ds.y[mask], ds.actions,
                   None if ds.groups is None else ds.groups[mask],
                   None if ds.period is None else ds.period[mask],
                   None if ds.keys is None else ds.keys[mask])


# ----------------------------------------------------------------------------- #
# experiments
# ----------------------------------------------------------------------------- #
def run_in_distribution(ds: Dataset, seed: int, args, out: dict):
    X, y, names = ds.X.to_numpy(float), ds.y, list(ds.X.columns)
    tr, cal, te = split3(y, ds.groups, seed)
    models, probs = build_models(seed, args.tune), {}
    for k, m in models.items():
        m.fit(X[tr], y[tr])
        probs[k] = m.predict_proba(X[te])[:, 1]
        out["pred"].append(dict(dataset=ds.name, seed=seed, model=k, **clf_metrics(y[te], probs[k])))
        # the same conformal gate on top of every predictor (does the predictor matter for the gate?)
        st_k, _ = conformal_stats(m.predict_proba(X[cal]), y[cal], m.predict_proba(X[te]),
                                  y[te], args.alpha)
        out["gate_models"].append(dict(dataset=ds.name, seed=seed, model=k, **st_k))

    hyb = models[PRIMARY]
    P_cal, P_te = hyb.predict_proba(X[cal]), hyb.predict_proba(X[te])

    # conformal operating point + alpha sweep + max-prob selective baseline
    st, S = conformal_stats(P_cal, y[cal], P_te, y[te], args.alpha)
    out["conf"].append(dict(dataset=ds.name, seed=seed, **st))
    for a in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30):
        s_a, _ = conformal_stats(P_cal, y[cal], P_te, y[te], a)
        out["curve"].append(dict(dataset=ds.name, seed=seed, kind="conformal", alpha=a,
                                 coverage=s_a["answered"], acc=s_a["sel_acc"]))
    grid = np.linspace(0.3, 1.0, 15)
    for c, a in zip(grid, selective_curve_maxprob(y[te], P_te[:, 1], grid)):
        out["curve"].append(dict(dataset=ds.name, seed=seed, kind="maxprob", alpha=np.nan,
                                 coverage=float(c), acc=float(a)))

    # ---- subgroup report of the gate (coverage / answered / at-risk flag recall) ------------- #
    size, argmax = S.sum(1), P_te.argmax(1)
    covered = S[np.arange(len(te)), y[te]]
    conf_risk = (size == 1) & S[:, 0]
    for attr in SENSITIVE.get(ds.name, []):
        if attr not in ds.X.columns:
            continue
        v = ds.X[attr].to_numpy()[te]
        for val in (0, 1):
            g = v == val
            if g.sum() < 5:
                continue
            risk = g & (y[te] == 0)
            out["fair"].append(dict(
                dataset=ds.name, seed=seed, attr=attr, value=val, n=int(g.sum()),
                coverage=float(covered[g].mean()),
                cov_atrisk=float(covered[risk].mean()) if risk.any() else np.nan,
                answered=float((size[g] == 1).mean()),
                flag_recall=float(conf_risk[risk].mean()) if risk.any() else np.nan))

    # ---- recommendation -------------------------------------------------------------------- #
    groups = {"confident": conf_risk, "uncertain": (argmax == 0) & (size != 1)}
    # Students whose predicted pass probability is already >= the target need no plan (the class-
    # conditional gate can mark a student 'at-risk' although f >= 0.5); they are excluded from the
    # recommendation evaluation and their share is reported (frac_trivial).
    p_all = hyb.predict_proba(X[te])[:, 1]
    trivial = p_all >= args.target
    frac_triv = {g: (float(trivial[m].mean()) if m.any() else np.nan) for g, m in groups.items()}
    groups = {g: m & ~trivial for g, m in groups.items()}
    plaus = Plausibility(X[tr])
    verifier = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1).fit(X[tr], y[tr])
    f = lambda A: hyb.predict_proba(A)[:, 1]
    engines = {
        "actionable": RecourseEngine(f, X[tr], names, ds.actions, args.k_max, args.target),
        "actionable-beam": RecourseEngine(f, X[tr], names, ds.actions, args.k_max, args.target),
        "unconstrained": RecourseEngine(f, X[tr], names, {n: "both" for n in names},
                                        args.k_max, args.target),
    }
    rng = np.random.default_rng(seed)
    sens_cols = [c for c in SENSITIVE.get(ds.name, []) if c in ds.X.columns]
    for gname, mask in groups.items():
        ids = np.where(mask)[0]
        if len(ids) > args.max_recs:
            ids = rng.choice(ids, args.max_recs, replace=False)
        for vname, eng in engines.items():
            row = dict(dataset=ds.name, seed=seed, variant=vname, group=gname, n=len(ids),
                       frac_trivial=frac_triv[gname])
            if len(ids) == 0:
                out["rec"].append(row)
                continue
            Xi = X[te][ids]
            res = [eng.recommend_beam(x, args.beam) if vname == "actionable-beam" else eng.recommend(x)
                   for x in Xi]
            Xn = np.array([r["x"] for r in res])
            succ = np.array([r["success"] for r in res])
            p0 = f(Xi)
            vp0, vp1 = verifier.predict_proba(Xi)[:, 1], verifier.predict_proba(Xn)[:, 1]
            agree = vp0 < 0.5
            row.update(
                valid_model=float(succ.mean()),
                valid_verif=float((vp1[agree] >= 0.5).mean()) if agree.any() else np.nan,
                n_agree=int(agree.sum()),
                n_feat=float(np.mean([len(r["changed"]) for r, s in zip(res, succ) if s])) if succ.any() else np.nan,
                cost=float(np.mean([r["cost"] for r, s in zip(res, succ) if s])) if succ.any() else np.nan,
                plaus=float(plaus.ok(Xn).mean()),
                immut=float(np.mean([any(c not in ds.actions for c in r["changed"]) for r in res])),
                prec_atrisk=float((y[te][ids] == 0).mean()),
                p0_mean=float(p0.mean()),
                p0_succ=float(p0[succ].mean()) if succ.any() else np.nan,
                p0_fail=float(p0[~succ].mean()) if (~succ).any() else np.nan,
            )
            if vname == "actionable" and succ.sum() >= 5:
                from scipy.stats import spearmanr
                gap = args.target - p0[succ]
                cst = np.array([r["cost"] for r, s_ in zip(res, succ) if s_])
                row["rho_gap_cost"] = float(spearmanr(gap, cst).correlation) if np.std(cst) > 0 else np.nan
            if vname != "unconstrained":
                # Proposition 1: a plan cannot exist when f(x) < tau - L * B_k(x)  (L = empirical constant)
                L = eng.max_ratio
                B = np.array([eng.max_reach(x) for x in Xi])
                row.update(lhat=float(L), cert_infeasible=float((p0 < args.target - L * B).mean()),
                           cert_correct=float((~succ[p0 < args.target - L * B]).mean())
                           if (p0 < args.target - L * B).any() else np.nan)
            out["rec"].append(row)
            if vname == "actionable":
                for i, r in enumerate(res):
                    out["rec_student"].append(dict(
                        dataset=ds.name, seed=seed, group=gname, success=bool(r["success"]),
                        **{c: float(ds.X[c].to_numpy()[te][ids[i]]) for c in sens_cols}))
                if seed == 0 and gname == "confident" and args.export_plans:
                    for i, r in enumerate(res[: args.export_plans]):
                        plan = "; ".join(f"{k}: {a:g} -> {b:g}" for k, (a, b) in r["changed"].items())
                        out["plans"].append(dict(dataset=ds.name, student=int(te[ids[i]]),
                                                 p_pass_before=round(float(p0[i]), 3),
                                                 p_pass_after=round(float(r["p"]), 3),
                                                 plan=plan or "(no feasible plan; referred)",
                                                 rating_1to5_feasible="", rating_1to5_helpful="",
                                                 comment=""))


def run_shift(src: Dataset, tgt: Dataset, label: str, seed: int, args, out: dict):
    Xs, ys = src.X.to_numpy(float), src.y
    Xt, yt = tgt.X.to_numpy(float), tgt.y
    tr, cal, te = split3(ys, src.groups, seed)
    hyb = HybridMLPBoost(seed, tune=args.tune).fit(Xs[tr], ys[tr])
    P_cal = hyb.predict_proba(Xs[cal])
    for where, Xe, ye in (("in-distribution", Xs[te], ys[te]), ("shifted", Xt, yt)):
        P = hyb.predict_proba(Xe)
        st, _ = conformal_stats(P_cal, ys[cal], P, ye, args.alpha)
        out["shift"].append(dict(setting=label, where=where, seed=seed,
                                 acc=st["acc_all"], auc=roc_auc_score(ye, P[:, 1]), **st))


def run_cutoff(args, out: dict):
    """Early-warning curve on OULAD: the primary predictor and the conformal gate at several
    prediction days (information available up to that day only)."""
    for cut in args.cutoffs:
        ds = load_oulad(args.oulad_dir, cut, args.oulad_sample)
        X, y = ds.X.to_numpy(float), ds.y
        for seed in range(args.cutoff_seeds):
            tr, cal, te = split3(y, ds.groups, seed)
            m = build_models(seed, args.tune)[PRIMARY].fit(X[tr], y[tr])
            P_te = m.predict_proba(X[te])
            st, _ = conformal_stats(m.predict_proba(X[cal]), y[cal], P_te, y[te], args.alpha)
            out["cutoff"].append(dict(cutoff=cut, seed=seed, **clf_metrics(y[te], P_te[:, 1]),
                                      coverage=st["coverage"], answered=st["answered"],
                                      sel_acc=st["sel_acc"]))
        print(f"[cutoff {cut}] done", flush=True)


# ----------------------------------------------------------------------------- #
# aggregation / LaTeX
# ----------------------------------------------------------------------------- #
def ms(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0


def fmt(m, s, d=3, bold=False, pm=True):
    if np.isnan(m):
        return "--"
    t = f"{m:.{d}f}$\\pm${s:.{d}f}" if pm else f"{m:.{d}f}"
    return f"\\textbf{{{t}}}" if bold else t


def tex_table(path, header, rows, colspec):
    L = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        L.append("\\midrule" if r == "MIDRULE" else " & ".join(r) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


def holm(p):
    p = np.asarray(p, float)
    order, adj, run = np.argsort(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def write_tables(df: dict, outdir: str, args):
    P = df["pred"]
    rows = []
    for dsn in P.dataset.unique():
        sub = P[P.dataset == dsn]
        best = {}
        for m in ("acc", "f1", "auc", "brier", "ece"):
            means = sub.groupby("model")[m].mean()
            best[m] = means.idxmin() if m in ("brier", "ece") else means.idxmax()
        if rows:
            rows.append("MIDRULE")
        for k in MODEL_KEYS:
            s = sub[sub.model == k]
            cells = [dsn if k == MODEL_KEYS[0] else "", MODEL_LABEL[k]]
            for m in ("acc", "f1", "auc", "brier", "ece"):
                cells.append(fmt(*ms(s[m]), bold=(best[m] == k)))
            g = df["gate_models"]     # the same conformal gate on top of every predictor
            g = g[(g.dataset == dsn) & (g.model == k)]
            cells += [fmt(*ms(g.answered), pm=False, d=2), fmt(*ms(g.sel_acc), pm=False, d=3)]
            rows.append(cells)
    tex_table(f"{outdir}/tab_prediction.tex",
              ["Dataset", "Model", "Acc.", "Macro-F1", "AUC", "Brier$\\downarrow$", "ECE$\\downarrow$",
               "Answ.", "Acc.$_{\\mathrm{ans}}$"],
              rows, "llccccccc")

    # significance of Hybrid vs each baseline (paired over seeds), Holm-adjusted
    rows = []
    for dsn in P.dataset.unique():
        sub = P[P.dataset == dsn]
        pv = {"auc": [], "f1": []}
        diffs = {"auc": [], "f1": []}
        effs = {}
        bases = [k for k in MODEL_KEYS if k not in (PRIMARY, "LR", "Hybrid" if args.tune else "") and k in set(sub.model)]
        for b in bases:
            for m in ("auc", "f1"):
                a = sub[sub.model == PRIMARY].sort_values("seed")[m].to_numpy()
                c = sub[sub.model == b].sort_values("seed")[m].to_numpy()
                d = a - c
                diffs[m].append(d.mean())
                pv[m].append(1.0 if np.allclose(d, 0) or len(d) < 6 else wilcoxon(a, c).pvalue)
                dz = d.mean() / d.std(ddof=1) if len(d) > 1 and d.std(ddof=1) > 0 else 0.0
                effs.setdefault(m, []).append(dz)
        adj = {m: holm(pv[m]) for m in pv}
        if rows:
            rows.append("MIDRULE")
        for i, b in enumerate(bases):
            rows.append([dsn if i == 0 else "", MODEL_LABEL[b],
                         f"{diffs['auc'][i]:+.3f}", f"{effs['auc'][i]:+.2f}", f"{adj['auc'][i]:.3f}",
                         f"{diffs['f1'][i]:+.3f}", f"{adj['f1'][i]:.3f}"])
    tex_table(f"{outdir}/tab_significance.tex",
              ["Dataset", "Baseline", "$\\Delta$AUC", "$d_z$", "$p_{\\mathrm{Holm}}$", "$\\Delta$F1", "$p_{\\mathrm{Holm}}$"],
              rows, "llccccc")

    # the same conformal gate on top of each predictor
    G = df["gate_models"]
    rows = []
    for dsn in G.dataset.unique():
        if rows:
            rows.append("MIDRULE")
        gate_keys = [k for k in MODEL_KEYS if k not in ("LR", "RF")]
        for k in gate_keys:
            s = G[(G.dataset == dsn) & (G.model == k)]
            rows.append([dsn if k == gate_keys[0] else "", MODEL_LABEL[k], fmt(*ms(s.coverage)),
                         fmt(*ms(s.cov_atrisk)), fmt(*ms(s.answered)), fmt(*ms(s.sel_acc))])
    tex_table(f"{outdir}/tab_gate_models.tex",
              ["Dataset", "Predictor", "Coverage", "Cov. at-risk", "Answered", "Acc. (answered)"],
              rows, "llcccc")

    C = df["conf"]
    rows = []
    for dsn in C.dataset.unique():
        s = C[C.dataset == dsn]
        d = (s.sel_acc - s.thr_acc).to_numpy()
        rows.append([dsn, fmt(*ms(s.coverage)), fmt(*ms(s.cov_atrisk)), fmt(*ms(s.cov_pass)),
                     fmt(*ms(s.answered)), fmt(*ms(s.sel_acc)), fmt(*ms(s.thr_acc)),
                     f"{np.nanmean(d):+.3f}"])
    tex_table(f"{outdir}/tab_conformal.tex",
              ["Dataset", "Cov.", "Cov. at-risk", "Cov. pass", "Answered", "Acc. gate", "Acc. thr.", "Gate$-$thr."],
              rows, "lccccccc")

    R = df["rec"]
    rows = []
    for dsn in R.dataset.unique():
        if rows:
            rows.append("MIDRULE")
        first = True
        for vname in ("actionable", "actionable-beam", "unconstrained"):
            for gname in ("confident", "uncertain"):
                s = R[(R.dataset == dsn) & (R.variant == vname) & (R.group == gname)]
                if s.empty or (vname == "unconstrained" and gname == "uncertain"):
                    continue    # (kept in raw_rec.csv; omitted from the table to save space)
                rows.append([dsn if first else "", vname, gname, f"{int(s.n.sum())}",
                             fmt(*ms(s.get("valid_model", [])), pm=False, d=2),
                             fmt(*ms(s.get("valid_verif", [])), pm=False, d=2),
                             fmt(*ms(s.get("n_feat", [])), pm=False, d=2),
                             fmt(*ms(s.get("cost", [])), pm=False, d=2),
                             fmt(*ms(s.get("plaus", [])), pm=False, d=2),
                             fmt(*ms(s.get("immut", [])), pm=False, d=2),
                             fmt(*ms(s.get("prec_atrisk", [])), pm=False, d=2)])
                first = False
    tex_table(f"{outdir}/tab_recourse.tex",
              ["Dataset", "Variant", "Group", "$n$", "Valid$_M$", "Valid$_V$", "\\#Feat.", "Cost",
               "Plaus.", "Immut.", "Prec."], rows, "lllrcccccccc"[:11])

    # confidence vs feasibility: initial P(pass) and plan success, certificate of Proposition 1
    rows = []
    for dsn in R.dataset.unique():
        if rows:
            rows.append("MIDRULE")
        first = True
        for gname in ("confident", "uncertain"):
            s = R[(R.dataset == dsn) & (R.variant == "actionable") & (R.group == gname)]
            if s.empty or s.n.sum() == 0:
                continue
            rows.append([dsn if first else "", gname, fmt(*ms(s.p0_mean), pm=False, d=2),
                         fmt(*ms(s.p0_succ), pm=False, d=2), fmt(*ms(s.p0_fail), pm=False, d=2),
                         fmt(*ms(s.valid_model), pm=False, d=2),
                         fmt(*ms(s.frac_trivial), pm=False, d=2),
                         fmt(*ms(s.cert_infeasible), pm=False, d=2),
                         fmt(*ms(s.rho_gap_cost), pm=False, d=2)])
            first = False
    tex_table(f"{outdir}/tab_tension.tex",
              ["Dataset", "Group", "$\\bar f(x)$", "$\\bar f$ (plan)", "$\\bar f$ (no plan)", "Valid$_M$",
               "Triv.", "Cert.", "$\\rho_{\\mathrm{gap,cost}}$"], rows, "llcccccccc"[:9])

    # subgroup report
    F, RS = df["fair"], df["rec_student"]
    rows = []
    if len(F):
        for dsn in F.dataset.unique():
            for attr in F[F.dataset == dsn].attr.unique():
                if rows:
                    rows.append("MIDRULE")
                for val in (0, 1):
                    s = F[(F.dataset == dsn) & (F.attr == attr) & (F.value == val)]
                    if s.empty:
                        continue
                    first_row = not any(isinstance(r, list) and r[0] == dsn for r in rows)
                    pr = RS[(RS.dataset == dsn) & (RS.group == "confident") & (RS[attr] == val)] \
                        if attr in RS.columns else RS.iloc[0:0]
                    rows.append([dsn if first_row else "", attr.replace("_", r"\_") + f"={val}", f"{int(s.n.mean())}",
                                 fmt(*ms(s.coverage), pm=False, d=2), fmt(*ms(s.cov_atrisk), pm=False, d=2),
                                 fmt(*ms(s.answered), pm=False, d=2), fmt(*ms(s.flag_recall), pm=False, d=2),
                                 f"{pr.success.mean():.2f}" if len(pr) else "--"])
        tex_table(f"{outdir}/tab_fairness.tex",
                  ["Dataset", "Group", "$n$", "Cov.", "Cov. at-risk", "Answered", "Flag recall", "Plan rate"],
                  rows, "llrccccc")

    if len(df["cutoff"]):
        Cu = df["cutoff"]
        rows = [[f"{int(c)}", fmt(*ms(g.auc)), fmt(*ms(g.acc)), fmt(*ms(g.ece)),
                 fmt(*ms(g.coverage)), fmt(*ms(g.answered)), fmt(*ms(g.sel_acc))]
                for c, g in Cu.groupby("cutoff")]
        tex_table(f"{outdir}/tab_cutoff.tex",
                  ["Cutoff day", "AUC", "Acc.", "ECE", "Coverage", "Answered", "Acc. (answered)"],
                  rows, "lcccccc")

    if len(df["shift"]):
        Sh = df["shift"]
        rows = []
        for lab in Sh.setting.unique():
            if rows:
                rows.append("MIDRULE")
            for w in ("in-distribution", "shifted"):
                s = Sh[(Sh["setting"] == lab) & (Sh["where"] == w)]  # NB: Sh.where is a DataFrame method
                rows.append([lab if w == "in-distribution" else "", w, fmt(*ms(s.acc)), fmt(*ms(s.auc)),
                             fmt(*ms(s.coverage)), fmt(*ms(s.cov_atrisk)), fmt(*ms(s.answered)),
                             fmt(*ms(s.sel_acc))])
        tex_table(f"{outdir}/tab_shift.tex",
                  ["Setting", "Test data", "Acc.", "AUC", "Coverage", "Cov. at-risk", "Answered", "Acc. (answered)"],
                  rows, "llcccccc")


def write_figures(df: dict, outdir: str):
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "font.family": "serif"})
    Cv = df["curve"]
    dsets = list(Cv.dataset.unique())
    fig, axes = plt.subplots(1, len(dsets), figsize=(7.1, 2.2), squeeze=False)
    for ax, dsn in zip(axes[0], dsets):
        s = Cv[Cv.dataset == dsn]
        b = s[s.kind == "maxprob"].groupby("coverage").acc.mean()
        ax.plot(b.index, b.values, ls="--", lw=1.0, color="0.45", label="Max-prob. threshold")
        c = s[s.kind == "conformal"].groupby("alpha")[["coverage", "acc"]].mean().sort_index()
        ax.plot(c.coverage, c.acc, ls="-", lw=1.2, marker="o", ms=3.5, color="black",
                label="Conformal gate")
        for a, r in c.iterrows():
            if round(a, 2) in (0.05, 0.10, 0.20):      # label a few operating points only
                ax.annotate(f"$\\alpha$={a:.2f}", (r.coverage, r.acc), textcoords="offset points",
                            xytext=(4, 4), fontsize=6, color="0.25")
        ax.set_title(dsn, fontsize=8)
        ax.set_xlabel("Fraction answered")
        ax.grid(True, lw=0.3, color="0.85")
    axes[0][0].set_ylabel("Accuracy on answered")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(f"{outdir}/fig_selective.pdf")
    plt.close(fig)

    R = df["rec"].dropna(subset=["valid_verif"])
    if len(R):
        dsets = list(R.dataset.unique())
        fig, ax = plt.subplots(figsize=(3.4, 2.4))
        combos = [("actionable", "confident", "0.1", ""), ("actionable", "uncertain", "0.55", ""),
                  ("unconstrained", "confident", "0.1", "///"), ("unconstrained", "uncertain", "0.8", "///")]
        w = 0.2
        for i, (v, g, col, h) in enumerate(combos):
            m, e = [], []
            for dsn in dsets:
                s = R[(R.dataset == dsn) & (R.variant == v) & (R.group == g)].valid_verif
                mm, ss = ms(s)
                m.append(0 if np.isnan(mm) else mm)
                e.append(0 if np.isnan(ss) else ss)
            ax.bar(np.arange(len(dsets)) + (i - 1.5) * w, m, w, yerr=e, color=col, hatch=h,
                   edgecolor="black", lw=0.5, error_kw={"lw": 0.6}, label=f"{v}, {g}")
        ax.set_xticks(range(len(dsets)))
        ax.set_xticklabels(dsets)
        ax.set_ylabel("Verifier-confirmed validity")
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=False, fontsize=6, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.12))
        fig.tight_layout()
        fig.savefig(f"{outdir}/fig_recourse.pdf", bbox_inches="tight")
        plt.close(fig)


# ----------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uci-dir")
    ap.add_argument("--oulad-dir")
    ap.add_argument("--out", default="../paper/results")
    ap.add_argument("--n-seeds", type=int, default=20, help="repeated splits (UCI, xAPI)")
    ap.add_argument("--oulad-seeds", type=int, default=10, help="repeated splits for OULAD (slow)")
    ap.add_argument("--tune", action="store_true",
                    help="also run CV-tuned XGBoost and a tuned hybrid; the tuned hybrid becomes the "
                         "predictor used by the gate and the recommender")
    ap.add_argument("--beam", type=int, default=3, help="beam width of the beam-search recommender")
    ap.add_argument("--xapi-csv", default=None, help="path to xAPI-Edu-Data.csv (optional 3rd dataset)")
    ap.add_argument("--cutoffs", default="", help="comma-separated OULAD cutoff days, e.g. 30,60,90")
    ap.add_argument("--cutoff-seeds", type=int, default=5)
    ap.add_argument("--export-plans", type=int, default=0,
                    help="export N example plans per dataset to advisor_plans.csv (for an advisor rating)")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--target", type=float, default=0.60, help="recourse target P(pass)")
    ap.add_argument("--k-max", type=int, default=3, help="max features changed per recommendation")
    ap.add_argument("--max-recs", type=int, default=60, help="recommendations per group per seed")
    ap.add_argument("--cutoff-day", type=int, default=90)
    ap.add_argument("--oulad-sample", type=int, default=None)
    ap.add_argument("--allow-fallback", action="store_true",
                    help="allow HistGradientBoosting when xgboost is missing (smoke tests only)")
    args = ap.parse_args()

    if not cgar.HAVE_XGB and not args.allow_fallback:
        sys.exit("xgboost is not installed (pip install xgboost). Use --allow-fallback only for "
                 "smoke tests; results must not be reported as XGBoost results.")
    if not (args.uci_dir or args.oulad_dir):
        sys.exit("give --uci-dir and/or --oulad-dir")
    import os
    os.makedirs(args.out, exist_ok=True)
    global PRIMARY, MODEL_KEYS
    PRIMARY = "HybridT" if args.tune else "Hybrid"
    MODEL_KEYS = [k for k in ALL_KEYS if args.tune or k not in ("BoostT", "HybridT")]
    if args.tune:   # the tuned hybrid is the proposed predictor; the untuned ones are references
        MODEL_LABEL["Boost"] = f"{cgar.BOOST_NAME} (default)"
        MODEL_LABEL["Hybrid"] = f"MLP+{cgar.BOOST_NAME} (default)"
    args.cutoffs = [int(c) for c in args.cutoffs.split(",") if c.strip()]

    data, shifts = {}, []
    if args.uci_dir:
        pair = load_uci_pair(args.uci_dir)
        data.update(pair)
        # Most Math students also appear in the Portuguese file (370+ of 395 match on the
        # dataset authors' merge keys). Removing them from the Math file would leave ~25 students,
        # so the overlap is removed from the Portuguese file in BOTH directions: the two sides of
        # every shift experiment then share no student, and Math stays whole.
        por_unique = drop_overlap(pair["UCI-Port"], pair["UCI-Math"])
        shifts.append(("UCI-Math$\\to$UCI-Port", pair["UCI-Math"], por_unique))
        shifts.append(("UCI-Port$\\to$UCI-Math", por_unique, pair["UCI-Math"]))
    if args.xapi_csv:
        data["xAPI"] = load_xapi(args.xapi_csv)
    if args.oulad_dir:
        ou = load_oulad(args.oulad_dir, args.cutoff_day, args.oulad_sample)
        data["OULAD"] = ou
        a = np.array([p.startswith("2013") for p in ou.period])
        b = np.array([p.startswith("2014") for p in ou.period])
        if a.any() and b.any():
            shifts.append(("OULAD 2013$\\to$2014", subset(ou, a), subset(ou, b)))

    out = {k: [] for k in ("pred", "gate_models", "conf", "curve", "rec", "shift", "fair",
                           "rec_student", "plans", "cutoff")}
    t0 = time.time()
    n_for = lambda name: args.oulad_seeds if name == "OULAD" else args.n_seeds
    for seed in range(max(n_for(d.name) for d in data.values())):
        for ds in data.values():
            if seed < n_for(ds.name):
                run_in_distribution(ds, seed, args, out)
        for label, s, t in shifts:
            if seed < n_for("OULAD" if label.startswith("OULAD") else ""):
                run_shift(s, t, label, seed, args, out)
        print(f"[seed {seed + 1}] {time.time() - t0:.0f}s", flush=True)
    if args.cutoffs and args.oulad_dir:
        run_cutoff(args, out)

    df = {k: pd.DataFrame(v) for k, v in out.items()}
    for k, v in df.items():
        if k == "plans":
            if len(v):
                v.to_csv(f"{args.out}/advisor_plans.csv", index=False)
        else:
            v.to_csv(f"{args.out}/raw_{k}.csv", index=False)
    write_tables(df, args.out, args)
    tex_table(f"{args.out}/tab_datasets.tex",
              ["Dataset", "Records", "Features", "Pass rate", "Actionable"],
              [[n, f"{len(d.y)}", f"{d.X.shape[1]}", f"{d.y.mean():.2f}", f"{len(d.actions)}"]
               for n, d in data.items()], "lrrrr")
    write_figures(df, args.out)

    summary = dict(args=vars(args), boosting_backend=cgar.BOOST_NAME, primary_predictor=PRIMARY,
                   dataset_sizes={n: dict(n=len(d.y), pass_rate=float(d.y.mean()), n_features=d.X.shape[1])
                                  for n, d in data.items()},
                   shift_target_sizes={l: len(t.y) for l, _, t in shifts},
                   versions=dict(python=platform.python_version(), numpy=np.__version__,
                                 pandas=pd.__version__, sklearn=sklearn.__version__),
                   runtime_s=round(time.time() - t0, 1))
    with open(f"{args.out}/summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("done ->", args.out)


if __name__ == "__main__":
    main()
