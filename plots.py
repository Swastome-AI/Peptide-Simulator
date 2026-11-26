"""Matplotlib plots - white background, black text, colour-blind-safe palette."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from .features import AA

BLUE, ORANGE = "#0072B2", "#D55E00"
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9",
           "#F0E442", "#8C564B", "#7F7F7F", "#17BECF", "#BCBD22", "#393B79"]


def set_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "text.color": "black", "axes.labelcolor": "black", "axes.edgecolor": "black",
        "xtick.color": "black", "ytick.color": "black", "axes.titlecolor": "black",
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.8,
        "axes.axisbelow": True, "legend.frameon": False,
    })


set_style()


def _fig(w=6.5, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax


# ------------------------------------------------------------- data plots ----
def label_counts(bbb, non):
    fig, ax = _fig(4.5, 3.6)
    bars = ax.bar(["Crosses BBB", "Non-BBB"], [len(bbb), len(non)], color=[BLUE, ORANGE],
                  edgecolor="black", linewidth=0.8)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{int(b.get_height())}",
                ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Peptides")
    ax.set_title("Class balance")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def source_counts(bbb, non):
    both = pd.concat([bbb.assign(Class="Crosses BBB"), non.assign(Class="Non-BBB")])
    both = both.assign(Source=both["Source"].str.split(";")).explode("Source")
    tab = both.groupby(["Source", "Class"]).size().unstack(fill_value=0)
    fig, ax = _fig(6.5, 3.8)
    left = np.zeros(len(tab))
    for cls, col in [("Crosses BBB", BLUE), ("Non-BBB", ORANGE)]:
        if cls in tab:
            ax.barh(tab.index, tab[cls], left=left, color=col, label=cls,
                    edgecolor="black", linewidth=0.6)
            left += tab[cls].to_numpy()
    ax.set_xlabel("Peptides (a peptide in 2 sources counts in both)")
    ax.set_title("Peptides per source")
    ax.legend()
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def length_hist(bbb, non):
    fig, ax = _fig()
    hi = int(max(bbb["Length"].max(), non["Length"].max()))
    bins = np.arange(0, hi + 3, 2)
    ax.hist(bbb["Length"], bins=bins, alpha=0.75, color=BLUE, label="Crosses BBB", edgecolor="black", linewidth=0.5)
    ax.hist(non["Length"], bins=bins, alpha=0.65, color=ORANGE, label="Non-BBB", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Peptide length (aa)")
    ax.set_ylabel("Peptides")
    ax.set_title("Length distribution")
    ax.legend()
    fig.tight_layout()
    return fig


def aac_compare(bbb, non):
    def comp(df):
        tot = sum(len(s) for s in df["Sequence"])
        return np.array([sum(s.count(a) for s in df["Sequence"]) / tot * 100 for a in AA])
    pb, nb = comp(bbb), comp(non)
    x = np.arange(20)
    fig, ax = _fig(9, 4)
    ax.bar(x - 0.2, pb, 0.4, color=BLUE, label="Crosses BBB", edgecolor="black", linewidth=0.5)
    ax.bar(x + 0.2, nb, 0.4, color=ORANGE, label="Non-BBB", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(list(AA))
    ax.set_ylabel("Composition (%)")
    ax.set_title("Amino-acid composition")
    ax.legend()
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def feature_box(Xdf, y, features):
    fig, axes = plt.subplots(1, len(features), figsize=(3.4 * len(features), 4), squeeze=False)
    for ax, f in zip(axes[0], features):
        data = [Xdf.loc[y == 1, f], Xdf.loc[y == 0, f]]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                        medianprops=dict(color="black", linewidth=1.5), tick_labels=["BBB", "Non-BBB"])
        for patch, c in zip(bp["boxes"], [BLUE, ORANGE]):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
        ax.set_title(f.replace("PHY_", ""), fontsize=11)
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------- ML plots ----
def model_comparison(cv_table, metric, test_table=None):
    t = cv_table.sort_values(f"{metric}_mean", ascending=True)
    fig, ax = _fig(7, max(3.2, 0.55 * len(t) + 1.2))
    y = np.arange(len(t))
    ax.barh(y - 0.2 if test_table is not None and len(test_table) else y, t[f"{metric}_mean"], height=0.38,
            xerr=t[f"{metric}_std"], color=BLUE, edgecolor="black", linewidth=0.6,
            label="Cross-validation (mean ± SD)", capsize=3)
    if test_table is not None and len(test_table):
        tv = test_table.set_index("Model").loc[t["Model"], metric].to_numpy()
        ax.barh(y + 0.2, tv, height=0.38, color=ORANGE, edgecolor="black", linewidth=0.6,
                label="Hold-out test")
    ax.set_yticks(y)
    ax.set_yticklabels(t["Model"])
    ax.set_xlabel(metric)
    ax.set_xlim(0, 1.05 if metric not in ("MCC",) else 1.0)
    ax.set_title(f"Model comparison - {metric}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def metric_heatmap(cv_table):
    cols = [f"{m}_mean" for m in ["AUROC", "AUPRC", "Accuracy", "Balanced_Acc", "F1", "MCC",
                                  "Precision", "Recall", "Specificity"]]
    data = cv_table.set_index("Model")[cols]
    data.columns = [c.replace("_mean", "") for c in cols]
    fig, ax = _fig(8, max(3, 0.5 * len(data) + 1.3))
    im = ax.imshow(data.to_numpy(float), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(data.columns, rotation=35, ha="right")
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(data.index)
    ax.grid(False)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data.iat[i, j]
            ax.text(j, i, "-" if np.isnan(v) else f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.6 else "black", fontsize=9)
    ax.set_title("Cross-validation metrics")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    return fig


def curves(pairs, kind="roc", title=None):
    """pairs: {model: (y_true, y_score)} ; kind 'roc' or 'pr'."""
    fig, ax = _fig(6, 5)
    for i, (name, (y, p)) in enumerate(pairs.items()):
        ok = ~np.isnan(p)
        if kind == "roc":
            xs, ys, _ = roc_curve(y[ok], p[ok])
        else:
            ys, xs, _ = precision_recall_curve(y[ok], p[ok])
        ax.plot(xs, ys, color=PALETTE[i % len(PALETTE)], linewidth=2, label=name)
    if kind == "roc":
        ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
    else:
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
    ax.set_title(title or ("ROC curves" if kind == "roc" else "Precision-recall curves"))
    ax.legend(fontsize=9, loc="lower right" if kind == "roc" else "lower left")
    fig.tight_layout()
    return fig


def confusion(y, proba, thr, title):
    pred = (np.asarray(proba) >= thr).astype(int)
    cm = np.zeros((2, 2), int)
    for a, b in zip(y, pred):
        cm[a, b] += 1
    fig, ax = _fig(4.2, 3.8)
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-BBB", "BBB"])
    ax.set_yticklabels(["Non-BBB", "BBB"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=15, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def importance_bar(imp, top=20, title="Permutation importance (drop in AUROC)"):
    t = imp.head(top).iloc[::-1]
    fig, ax = _fig(7, max(3.2, 0.3 * len(t) + 1))
    ax.barh(t["Feature"], t["Importance"], xerr=t["Std"], color=BLUE, edgecolor="black",
            linewidth=0.5, capsize=2)
    ax.set_xlabel("Mean decrease in AUROC when shuffled")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def score_bars(tbl, thr, top=30):
    t = tbl.head(top).iloc[::-1]
    fig, ax = _fig(7, max(3, 0.32 * len(t) + 1))
    cols = [BLUE if p >= thr else ORANGE for p in t["BBB_probability"]]
    ax.barh(t["Sequence"], t["BBB_probability"], color=cols, edgecolor="black", linewidth=0.5)
    ax.axvline(thr, color="black", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Predicted probability of crossing the BBB")
    ax.set_title("Screening result")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig
