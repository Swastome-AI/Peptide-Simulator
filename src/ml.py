"""Step 03 - compare machine-learning models for BBB-peptide prediction."""
import difflib
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.under_sampling import RandomUnderSampler
    HAS_IMB = True
except Exception:  # pragma: no cover
    HAS_IMB = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False

METRICS = ["AUROC", "AUPRC", "Accuracy", "Balanced_Acc", "F1", "MCC",
           "Precision", "Recall", "Specificity"]

# name -> list of (param, kind, min, max, default)  - used by the UI AND the factory
HYPERPARAMS = {
    "Logistic Regression": [("C", "float", 0.01, 100.0, 1.0)],
    "Random Forest": [("n_estimators", "int", 50, 1000, 300), ("max_depth", "int", 0, 50, 0)],
    "Extra Trees": [("n_estimators", "int", 50, 1000, 300), ("max_depth", "int", 0, 50, 0)],
    "Gradient Boosting": [("n_estimators", "int", 50, 600, 200),
                          ("learning_rate", "float", 0.01, 1.0, 0.1),
                          ("max_depth", "int", 1, 8, 3)],
    "HistGradientBoosting": [("max_iter", "int", 50, 600, 200),
                             ("learning_rate", "float", 0.01, 1.0, 0.1)],
    "AdaBoost": [("n_estimators", "int", 50, 600, 200), ("learning_rate", "float", 0.01, 2.0, 0.5)],
    "SVM (RBF)": [("C", "float", 0.01, 100.0, 1.0)],
    "k-Nearest Neighbors": [("n_neighbors", "int", 1, 51, 7)],
    "Gaussian Naive Bayes": [],
    "MLP (Neural Net)": [("hidden", "int", 8, 256, 64), ("alpha", "float", 0.0, 1.0, 0.001)],
    "Decision Tree": [("max_depth", "int", 1, 30, 5)],
}
if HAS_XGB:
    HYPERPARAMS["XGBoost"] = [("n_estimators", "int", 50, 800, 300),
                              ("learning_rate", "float", 0.01, 1.0, 0.1),
                              ("max_depth", "int", 1, 10, 4)]
MODEL_NAMES = list(HYPERPARAMS)
DEFAULT_MODELS = ["Logistic Regression", "Random Forest", "Extra Trees",
                  "Gradient Boosting", "SVM (RBF)", "k-Nearest Neighbors"]


def default_hp(name):
    return {p: d for p, _, _, _, d in HYPERPARAMS[name]}


def make_classifier(name, hp, seed, balanced, pos_weight):
    hp = {**default_hp(name), **(hp or {})}
    cw = "balanced" if balanced else None
    if name == "Logistic Regression":
        return LogisticRegression(C=hp["C"], max_iter=3000, class_weight=cw)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=hp["n_estimators"], max_depth=hp["max_depth"] or None,
                                      class_weight=cw, random_state=seed, n_jobs=-1)
    if name == "Extra Trees":
        return ExtraTreesClassifier(n_estimators=hp["n_estimators"], max_depth=hp["max_depth"] or None,
                                    class_weight=cw, random_state=seed, n_jobs=-1)
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(n_estimators=hp["n_estimators"], learning_rate=hp["learning_rate"],
                                          max_depth=hp["max_depth"], random_state=seed)
    if name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(max_iter=hp["max_iter"], learning_rate=hp["learning_rate"],
                                              class_weight=cw, random_state=seed)
    if name == "AdaBoost":
        return AdaBoostClassifier(n_estimators=hp["n_estimators"], learning_rate=hp["learning_rate"],
                                  random_state=seed)
    if name == "SVM (RBF)":
        return SVC(C=hp["C"], kernel="rbf", gamma="scale", probability=True,
                   class_weight=cw, random_state=seed)
    if name == "k-Nearest Neighbors":
        return KNeighborsClassifier(n_neighbors=hp["n_neighbors"])
    if name == "Gaussian Naive Bayes":
        return GaussianNB()
    if name == "MLP (Neural Net)":
        return MLPClassifier(hidden_layer_sizes=(hp["hidden"],), alpha=hp["alpha"],
                             max_iter=800, random_state=seed)
    if name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=hp["max_depth"], class_weight=cw, random_state=seed)
    if name == "XGBoost" and HAS_XGB:
        return XGBClassifier(n_estimators=hp["n_estimators"], learning_rate=hp["learning_rate"],
                             max_depth=hp["max_depth"], scale_pos_weight=pos_weight if balanced else 1.0,
                             eval_metric="logloss", random_state=seed, n_jobs=-1)
    raise ValueError(f"Unknown model {name}")


def build_pipeline(name, hp, cfg, seed, n_features, min_class_train, pos_weight=1.0):
    steps = []
    if cfg["scaler"] == "Standard":
        steps.append(("scale", StandardScaler()))
    elif cfg["scaler"] == "MinMax":
        steps.append(("scale", MinMaxScaler()))
    if cfg["selector"] != "None":
        score = f_classif if cfg["selector"] == "ANOVA F-test" else mutual_info_classif
        steps.append(("select", SelectKBest(score, k=min(cfg["k_features"], n_features))))
    sampler = cfg["sampler"]
    if sampler != "None":
        if not HAS_IMB:
            raise RuntimeError("imbalanced-learn is not installed (pip install imbalanced-learn).")
        k = max(1, min(5, min_class_train - 1))
        if sampler == "Random undersampling":
            steps.append(("sample", RandomUnderSampler(random_state=seed)))
        elif sampler == "SMOTE":
            steps.append(("sample", SMOTE(k_neighbors=k, random_state=seed)))
        elif sampler == "Borderline-SMOTE":
            steps.append(("sample", BorderlineSMOTE(k_neighbors=k, m_neighbors=max(k, 5),
                                                    random_state=seed)))
    steps.append(("clf", make_classifier(name, hp, seed, cfg["class_weight"], pos_weight)))
    return (ImbPipeline if sampler != "None" else Pipeline)(steps)


# ------------------------------------------------------------- clustering ----
def cluster_sequences(seqs, identity=0.8):
    """Greedy, CD-HIT-style clustering. A peptide joins the first representative
    it matches at >= `identity` (difflib similarity ratio); else it starts a new cluster.
    Returns an int array of cluster ids aligned with `seqs`."""
    seqs = list(seqs)
    order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]))
    reps, rep_cid = [], []
    cid = np.zeros(len(seqs), dtype=int)
    for i in order:
        s, assigned = seqs[i], None
        for r, c in zip(reps, rep_cid):
            if min(len(s), len(r)) / max(len(s), len(r)) < identity:
                continue
            if difflib.SequenceMatcher(None, s, r, autojunk=False).ratio() >= identity:
                assigned = c
                break
        if assigned is None:
            assigned = len(reps)
            reps.append(s)
            rep_cid.append(assigned)
        cid[i] = assigned
    return cid


# --------------------------------------------------------------- metrics -----
def compute_metrics(y, proba, thr=0.5):
    y = np.asarray(y)
    pred = (np.asarray(proba) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return {
        "AUROC": roc_auc_score(y, proba) if both else np.nan,
        "AUPRC": average_precision_score(y, proba) if both else np.nan,
        "Accuracy": accuracy_score(y, pred),
        "Balanced_Acc": balanced_accuracy_score(y, pred),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) else np.nan,
    }


# ------------------------------------------------------------ experiment -----
def run_experiment(df, cfg, model_names, model_hp, on_progress=None):
    """df needs columns Sequence, Label. cfg keys:
       feature_sets, test_size, split ('Random stratified' | 'Cluster-aware'),
       identity, cv_folds, scaler, selector, k_features, sampler, class_weight,
       threshold, seed."""
    from .features import featurize
    seed = cfg["seed"]
    seqs = df["Sequence"].tolist()
    y = df["Label"].to_numpy().astype(int)
    Xdf = featurize(seqs, cfg["feature_sets"])
    X, names = Xdf.to_numpy(float), list(Xdf.columns)

    groups = cluster_sequences(seqs, cfg["identity"]) if cfg["split"] == "Cluster-aware" else None

    idx = np.arange(len(y))
    if cfg["test_size"] > 0:
        if groups is None:
            tr, te = train_test_split(idx, test_size=cfg["test_size"], stratify=y, random_state=seed)
        else:
            sgk = StratifiedGroupKFold(n_splits=max(2, round(1 / cfg["test_size"])),
                                       shuffle=True, random_state=seed)
            tr, te = next(sgk.split(X, y, groups))
    else:
        tr, te = idx, np.array([], dtype=int)
    Xtr, ytr = X[tr], y[tr]
    gtr = groups[tr] if groups is not None else None

    min_class = int(np.bincount(ytr).min())
    folds = max(2, min(cfg["cv_folds"], min_class))
    if groups is None:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = list(splitter.split(Xtr, ytr))
    else:
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = list(splitter.split(Xtr, ytr, gtr))
    pos_weight = float((ytr == 0).sum() / max(1, (ytr == 1).sum()))
    min_fold_class = max(2, int(min_class * (1 - 1 / folds)))

    cv_rows, test_rows, oof, test_proba, fitted = [], [], {}, {}, {}
    for n_done, name in enumerate(model_names):
        t0 = time.time()
        if on_progress:
            on_progress(n_done / len(model_names), f"Training {name} ...")
        hp = model_hp.get(name, {})
        fold_metrics, oof_p = [], np.full(len(ytr), np.nan)
        for trn, val in splits:
            pipe = build_pipeline(name, hp, cfg, seed, Xtr.shape[1], min_fold_class, pos_weight)
            pipe.fit(Xtr[trn], ytr[trn])
            p = pipe.predict_proba(Xtr[val])[:, 1]
            oof_p[val] = p
            fold_metrics.append(compute_metrics(ytr[val], p, cfg["threshold"]))
        fm = pd.DataFrame(fold_metrics)
        row = {"Model": name}
        for m in METRICS:
            row[f"{m}_mean"], row[f"{m}_std"] = fm[m].mean(), fm[m].std(ddof=0)
        row["Seconds"] = round(time.time() - t0, 1)
        cv_rows.append(row)
        oof[name] = (ytr, oof_p)

        final = build_pipeline(name, hp, cfg, seed, Xtr.shape[1], min(min_class, 6), pos_weight)
        final.fit(Xtr, ytr)
        fitted[name] = final
        if len(te):
            tp = final.predict_proba(X[te])[:, 1]
            test_proba[name] = tp
            test_rows.append({"Model": name, **compute_metrics(y[te], tp, cfg["threshold"])})
    if on_progress:
        on_progress(1.0, "Done")
    return {
        "cfg": cfg, "names": names, "X": X, "y": y, "seqs": seqs,
        "train_idx": tr, "test_idx": te, "groups": groups, "cv_folds": folds,
        "cv_table": pd.DataFrame(cv_rows), "test_table": pd.DataFrame(test_rows),
        "oof": oof, "test_proba": test_proba, "fitted": fitted,
        "n_clusters": int(len(np.unique(groups))) if groups is not None else None,
    }


def refit_on_all(result, name, model_hp):
    """Fit one model on 100 % of the data (use for the final screening model)."""
    cfg, y, X = result["cfg"], result["y"], result["X"]
    pos_weight = float((y == 0).sum() / max(1, (y == 1).sum()))
    pipe = build_pipeline(name, model_hp.get(name, {}), cfg, cfg["seed"], X.shape[1],
                          min(int(np.bincount(y).min()), 6), pos_weight)
    pipe.fit(X, y)
    return pipe


def importance_table(pipe, X, y, names, n_repeats=10, seed=0):
    r = permutation_importance(pipe, X, y, scoring="roc_auc", n_repeats=n_repeats,
                               random_state=seed, n_jobs=1)
    return (pd.DataFrame({"Feature": names, "Importance": r.importances_mean,
                          "Std": r.importances_std})
            .sort_values("Importance", ascending=False).reset_index(drop=True))
