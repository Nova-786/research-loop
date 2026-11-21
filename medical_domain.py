"""Medical research domain: build a diagnostic classifier on a public dataset.

Dataset: scikit-learn's Breast Cancer Wisconsin (Diagnostic), 569 samples, 30 features.
Research prototype only -- NOT a clinical tool.

Integrity guards (these are the medical-specific part of the harness):
  * 30% stratified TEST set is locked away; the proposer never sees it.
  * Scaling and feature selection live INSIDE the CV pipeline -> no preprocessing leakage.
  * The score is repeated stratified CV AUROC on the dev set, minus a small penalty for
    fold-to-fold instability (a model that wins by luck on one fold is not trusted).
  * Final report includes a bootstrap confidence interval, because 171 test samples is small.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from research_loop import Domain

warnings.filterwarnings("ignore")


class MedicalDomain(Domain):
    name = "medical_diagnostic_classifier"
    space = {
        "model": ("cat", ["logreg", "svm", "rf", "gb"]),
        "k_features": ("int", 3, 30),
        "C": ("log", 1e-3, 1e2),                 # logreg / svm
        "n_estimators": ("int", 20, 250),        # rf / gb
        "max_depth": ("int", 1, 8),              # rf / gb
        "learning_rate": ("log", 0.01, 0.5),     # gb
        "class_weight_balanced": ("cat", [0, 1]),
    }

    def __init__(self, seed: int = 0):
        X, y = load_breast_cancer(return_X_y=True)
        self.X_dev, self.X_test, self.y_dev, self.y_test = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=seed)
        self.seed = seed

    def baseline(self):
        return {"model": "logreg", "k_features": 30, "C": 1.0, "n_estimators": 100, "max_depth": 3,
                "learning_rate": 0.1, "class_weight_balanced": 0}

    def _pipeline(self, c) -> Pipeline:
        cw = "balanced" if c["class_weight_balanced"] else None
        m = c["model"]
        if m == "logreg":
            clf = LogisticRegression(C=c["C"], class_weight=cw, max_iter=2000)
        elif m == "svm":
            clf = SVC(C=c["C"], class_weight=cw, probability=True, random_state=self.seed)
        elif m == "rf":
            clf = RandomForestClassifier(n_estimators=c["n_estimators"], max_depth=c["max_depth"],
                                         class_weight=cw, random_state=self.seed, n_jobs=1)
        else:
            clf = GradientBoostingClassifier(n_estimators=min(c["n_estimators"], 150), max_depth=min(c["max_depth"], 4),
                                             learning_rate=c["learning_rate"], random_state=self.seed)
        return Pipeline([("scale", StandardScaler()),
                         ("select", SelectKBest(f_classif, k=int(c["k_features"]))),
                         ("clf", clf)])

    def evaluate(self, config, split):
        pipe = self._pipeline(config)
        if split == "val":
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=self.seed)
            aucs = cross_val_score(pipe, self.X_dev, self.y_dev, cv=cv, scoring="roc_auc")
            return {"score": float(aucs.mean() - 0.5 * aucs.std()), "auroc": float(aucs.mean()),
                    "auroc_std": float(aucs.std())}
        pipe.fit(self.X_dev, self.y_dev)
        p = pipe.predict_proba(self.X_test)[:, 1]
        auc = roc_auc_score(self.y_test, p)
        rng = np.random.default_rng(0)
        boots = []
        for _ in range(500):
            i = rng.integers(0, len(p), len(p))
            if len(set(self.y_test[i])) == 2:
                boots.append(roc_auc_score(self.y_test[i], p[i]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return {"score": float(auc), "auroc": float(auc), "ci95_low": float(lo), "ci95_high": float(hi),
                "brier": float(brier_score_loss(self.y_test, p))}
