"""
datasets.py - loaders for the three public benchmarks used in the paper.

  * UCI Student Performance (Cortez & Silva, 2008): student-mat.csv, student-por.csv
  * Open University Learning Analytics Dataset, OULAD (Kuzilek et al., 2017)

Every loader returns a `Dataset` with a purely numeric feature table, a binary label
(1 = pass, 0 = at-risk) and the *actionability specification* (which features a student
or advisor can change, and in which direction). Everything else is treated as immutable
(demographics, past performance, enrolment facts).

Data are NOT bundled. Download them yourself (see README.md).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame                     # all columns float
    y: np.ndarray                       # 1 = pass, 0 = at-risk
    actions: dict                       # column -> "up" | "down" | "both"
    groups: np.ndarray | None = None    # student id (for group-aware splitting)
    period: np.ndarray | None = None    # presentation label (for temporal shift)
    keys: np.ndarray | None = None      # UCI merge keys (for overlap removal)


# ----------------------------------------------------------------------------- #
# UCI student performance
# ----------------------------------------------------------------------------- #
UCI_BINARY = ["schoolsup", "famsup", "paid", "activities", "nursery", "higher",
              "internet", "romantic"]
UCI_CATEGORICAL = ["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
                   "reason", "guardian"]
# attributes the dataset authors use to identify students present in both files
UCI_MERGE_KEYS = ["school", "sex", "age", "address", "famsize", "Pstatus", "Medu",
                  "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet"]
# actionable behaviours / supports (feasible direction is an explicit modelling assumption)
UCI_ACTIONS = {"studytime": "up", "absences": "down", "goout": "down", "Dalc": "down",
               "Walc": "down", "paid": "up", "schoolsup": "up", "famsup": "up",
               "activities": "both"}


def load_uci_pair(uci_dir: str, pass_mark: int = 10, use_g1: bool = True) -> dict:
    """Returns {"UCI-Math": Dataset, "UCI-Port": Dataset} with a shared column layout.

    G3 (final grade) defines the label (pass if G3 >= pass_mark). G2 and G3 are removed;
    G1 (first-period grade) is kept as an immutable 'past performance' feature when
    `use_g1` is True (early-warning setting).
    """
    mat = pd.read_csv(os.path.join(uci_dir, "student-mat.csv"), sep=";")
    por = pd.read_csv(os.path.join(uci_dir, "student-por.csv"), sep=";")
    t = pd.concat([mat.assign(_src="mat"), por.assign(_src="por")], ignore_index=True)

    keys = t[UCI_MERGE_KEYS].astype(str).agg("|".join, axis=1).to_numpy()
    y_all = (t["G3"] >= pass_mark).astype(int).to_numpy()
    src = t["_src"].to_numpy()

    drop = ["G2", "G3", "_src"] + ([] if use_g1 else ["G1"])
    f = t.drop(columns=drop).copy()
    for c in UCI_BINARY:
        f[c] = (f[c].astype(str).str.strip().str.lower() == "yes").astype(float)
    f = pd.get_dummies(f, columns=UCI_CATEGORICAL, dtype=float).astype(float)

    out = {}
    for name, tag in (("UCI-Math", "mat"), ("UCI-Port", "por")):
        m = src == tag
        out[name] = Dataset(name, f.loc[m].reset_index(drop=True), y_all[m],
                            dict(UCI_ACTIONS), keys=keys[m])
    return out


def drop_overlap(target: Dataset, source: Dataset) -> Dataset:
    """Remove from `target` the students that also appear in `source` (same merge keys),
    so that cross-dataset transfer is not contaminated by identical students."""
    keep = ~np.isin(target.keys, source.keys)
    return Dataset(target.name, target.X.loc[keep].reset_index(drop=True), target.y[keep],
                   target.actions, None, None, target.keys[keep])


# ----------------------------------------------------------------------------- #
# OULAD
# ----------------------------------------------------------------------------- #
OULAD_CATEGORICAL = ["gender", "region", "highest_education", "imd_band", "age_band",
                     "disability", "code_module"]
OULAD_ACTIONS = {"clicks_total": "up", "active_days": "up", "clicks_last4w": "up",
                 "n_submitted": "up", "studied_credits": "down"}


def load_oulad(oulad_dir: str, cutoff_day: int = 90, sample: int | None = None,
               seed: int = 0) -> Dataset:
    """Early-warning view of OULAD at `cutoff_day` (days since course start).

    Only information available up to the cutoff is used. Students who unregistered on or
    before the cutoff are excluded (they are no longer enrolled when the prediction is
    issued). Label: Pass/Distinction = 1, Fail/Withdrawn = 0.
    """
    p = lambda fn: os.path.join(oulad_dir, fn)
    key = ["code_module", "code_presentation", "id_student"]

    info = pd.read_csv(p("studentInfo.csv"))
    reg = pd.read_csv(p("studentRegistration.csv"), usecols=key + ["date_unregistration"])
    info = info.merge(reg, on=key, how="left")
    gone = info["date_unregistration"].notna() & (info["date_unregistration"] <= cutoff_day)
    info = info.loc[~gone].copy()

    # studentVle.csv is ~450 MB (10M+ rows): read in chunks and keep only rows up to the cutoff
    # so memory stays small on a laptop.
    chunks = pd.read_csv(p("studentVle.csv"), usecols=key + ["date", "sum_click"],
                         chunksize=1_000_000)
    vle = pd.concat([c[c["date"] <= cutoff_day] for c in chunks], ignore_index=True)
    act = vle.groupby(key).agg(clicks_total=("sum_click", "sum"),
                               active_days=("date", "nunique")).reset_index()
    rec = (vle[vle["date"] > cutoff_day - 28].groupby(key)["sum_click"].sum()
           .rename("clicks_last4w").reset_index())

    sa = pd.read_csv(p("studentAssessment.csv"))
    asm = pd.read_csv(p("assessments.csv"), usecols=["code_module", "code_presentation",
                                                     "id_assessment"])
    sa = sa.merge(asm, on="id_assessment", how="inner")
    sa = sa[sa["date_submitted"] <= cutoff_day]
    asg = sa.groupby(key).agg(n_submitted=("id_assessment", "nunique"),
                              mean_score_early=("score", "mean")).reset_index()

    df = (info.merge(act, on=key, how="left").merge(rec, on=key, how="left")
              .merge(asg, on=key, how="left"))
    for c in ["clicks_total", "active_days", "clicks_last4w", "n_submitted"]:
        df[c] = df[c].fillna(0.0)
    df["has_early_score"] = df["mean_score_early"].notna().astype(float)
    df["mean_score_early"] = df["mean_score_early"].fillna(0.0)
    df["imd_band"] = df["imd_band"].fillna("missing")
    if sample is not None and sample < len(df):
        df = df.sample(sample, random_state=seed)
    df = df.reset_index(drop=True)

    y = df["final_result"].isin(["Pass", "Distinction"]).astype(int).to_numpy()
    groups = df["id_student"].to_numpy()
    period = df["code_presentation"].astype(str).to_numpy()
    num = ["num_of_prev_attempts", "studied_credits", "clicks_total", "active_days",
           "clicks_last4w", "n_submitted", "mean_score_early", "has_early_score"]
    X = pd.concat([df[num].astype(float),
                   pd.get_dummies(df[OULAD_CATEGORICAL], dtype=float)], axis=1)
    return Dataset("OULAD", X, y, dict(OULAD_ACTIONS), groups, period)


# ----------------------------------------------------------------------------- #
# xAPI-Edu-Data (Amrieh et al., 2016) - optional third dataset (CSV downloaded by the user)
# ----------------------------------------------------------------------------- #
XAPI_REQUIRED = ["gender", "NationalITy", "PlaceofBirth", "StageID", "GradeID", "SectionID",
                 "Topic", "Semester", "Relation", "raisedhands", "VisITedResources",
                 "AnnouncementsView", "Discussion", "ParentAnsweringSurvey",
                 "ParentschoolSatisfaction", "StudentAbsenceDays", "Class"]
XAPI_CATEGORICAL = ["gender", "NationalITy", "PlaceofBirth", "StageID", "GradeID", "SectionID",
                    "Topic", "Semester", "Relation", "ParentschoolSatisfaction"]
XAPI_ACTIONS = {"raisedhands": "up", "VisITedResources": "up", "AnnouncementsView": "up",
                "Discussion": "up", "absent_above7": "down"}


def load_xapi(csv_path: str) -> Dataset:
    """xAPI-Edu-Data: 480 students, behavioural LMS counts + demographics; Class in {L,M,H}.
    Label: M/H = 1 (pass), L = 0 (at-risk). Behaviour counts and absence are actionable; all
    demographic / school-structure fields are immutable."""
    df = pd.read_csv(csv_path)
    missing = [c for c in XAPI_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"xAPI file is missing columns {missing}; expected the original "
                         "xAPI-Edu-Data.csv (480 rows).")
    y = df["Class"].isin(["M", "H"]).astype(int).to_numpy()
    num = df[["raisedhands", "VisITedResources", "AnnouncementsView", "Discussion"]].astype(float)
    num["absent_above7"] = (df["StudentAbsenceDays"].astype(str).str.contains("Above")).astype(float)
    num["parent_survey"] = (df["ParentAnsweringSurvey"].astype(str) == "Yes").astype(float)
    X = pd.concat([num, pd.get_dummies(df[XAPI_CATEGORICAL], dtype=float)], axis=1).astype(float)
    return Dataset("xAPI", X, y, dict(XAPI_ACTIONS))
