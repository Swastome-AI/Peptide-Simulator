"""Peptide Simulator - Module 2 (BBB penetration) pipeline.

Tab 1  Data builder      any source -> general format -> bbb_peptides.csv / non_bbb_peptides.csv
Tab 2  Model comparison  features, imbalance handling, cluster-aware splits, 11 ML models
Tab 3  Explain           permutation importance, feature distributions, misclassified peptides
Tab 4  Predict & screen  score new peptides with the chosen model
"""
import io
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import data_builder as db
from src import features as ft
from src import ml, plots

st.set_page_config(page_title="Peptide Simulator", page_icon="🧬", layout="wide")
ss = st.session_state
ss.setdefault("sources", {})      # name -> general-format DataFrame
ss.setdefault("built", None)      # output of db.build_datasets
ss.setdefault("exp", None)        # output of ml.run_experiment
ss.setdefault("final", None)      # dict with the model used in tab 4

LABEL_MODES = {
    "All peptides CROSS the BBB (positives)": "all_positive",
    "NONE cross the BBB (negatives)": "all_negative",
    "Read the label from a TEXT column": "values",
    "Read the label from a NUMBER column (threshold)": "numeric",
}


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


# =============================================================== sidebar =====
with st.sidebar:
    st.title("🧬 Peptide Simulator")
    st.caption("Module 2 · Blood-brain-barrier penetration")
    st.markdown("**Pipeline status**")
    st.markdown(f"{'✅' if ss.built else '⬜'} 1 · Data built")
    st.markdown(f"{'✅' if ss.exp else '⬜'} 2 · Models compared")
    st.markdown(f"{'✅' if ss.final else '⬜'} 3-4 · Final model ready")
    st.divider()
    if st.button("🔄 Reset everything", key="reset_all"):
        for k in ["sources", "built", "exp", "final"]:
            ss[k] = {} if k == "sources" else None
        st.rerun()
    st.divider()
    st.caption("Coming next: Module 1 (gut stability, SGF/SIF) and Module 3 "
               "(glymphatic / CSF clearance) as extra tabs.")

st.title("Peptide Simulator")
tab1, tab2, tab3, tab4 = st.tabs(["1 · Data builder", "2 · Model comparison",
                                  "3 · Explain", "4 · Predict & screen"])


# ============================================================ TAB 1 ==========
def file_mapper(f):
    """Let the user map ANY uploaded table onto the general format."""
    try:
        df = db.read_table(f.name, f.getvalue())
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read {f.name}: {e}")
        return
    with st.expander(f"📄 {f.name} · {len(df)} rows · {len(df.columns)} columns"):
        st.dataframe(df.head(5), use_container_width=True)
        cols = list(df.columns)
        g = db.guess_column(cols, db.SEQ_GUESSES)
        seq_col = st.selectbox("Sequence column", cols, index=cols.index(g) if g in cols else 0,
                               key=f"seq_{f.name}")
        mode_label = st.radio("How are the labels defined?", list(LABEL_MODES), key=f"mode_{f.name}")
        mode = LABEL_MODES[mode_label]
        label_col, pos_v, neg_v, thr, contains = None, "", "", 0.5, False
        if mode in ("values", "numeric"):
            lg = db.guess_column(cols, db.LABEL_GUESSES)
            label_col = st.selectbox("Label column", cols, index=cols.index(lg) if lg in cols else 0,
                                     key=f"lab_{f.name}")
            st.caption("Values in that column: " + ", ".join(
                f"{k} ({v})" for k, v in df[label_col].astype(str).value_counts().head(8).items()))
        if mode == "values":
            c1, c2 = st.columns(2)
            pos_v = c1.text_input("Values meaning 'crosses BBB' (comma separated)", "yes, 1, true",
                                  key=f"pos_{f.name}")
            neg_v = c2.text_input("Values meaning 'does not cross' (blank = everything else)",
                                  "no, 0, false", key=f"neg_{f.name}")
            contains = st.checkbox("Match as substring (e.g. 'crosses' matches 'Crosses BBB')",
                                   key=f"cont_{f.name}")
        if mode == "numeric":
            thr = st.number_input("Value >= threshold means 'crosses BBB'", value=0.5,
                                  key=f"thr_{f.name}")
        source = st.text_input("Source name", os.path.splitext(f.name)[0], key=f"src_{f.name}")
        if st.button("➕ Add to pipeline", key=f"add_{f.name}"):
            try:
                std, dropped = db.standardize(df, seq_col, source, mode, label_col, pos_v, neg_v,
                                              thr, contains)
                ss.sources[source] = std
                st.success(f"Added {len(std)} peptides ({int(std.Label.sum())} BBB / "
                           f"{int((std.Label == 0).sum())} non-BBB)"
                           + (f"; {dropped} rows had no usable label and were skipped." if dropped else ""))
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not convert this file: {e}")


with tab1:
    st.subheader("Step 02 · Build the BBB and non-BBB peptide files")
    st.caption("Every source is first converted to one **general format** - "
               "`Sequence | Label (1 = crosses BBB, 0 = does not) | Source` - then cleaned, "
               "merged and balanced.")

    with st.expander("⚙️ Cleaning & balancing parameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        min_len = c1.slider("Minimum length (aa)", 2, 30, 5, key="min_len")
        max_len = c1.slider("Maximum length (aa)", 10, 100, 50, key="max_len")
        lowercase_mode = c2.radio("Lowercase letters (D-amino acids in some databases)",
                                  ["reject", "uppercase"], key="lowercase_mode",
                                  help="'reject' keeps only pure L-peptides.")
        drop_conflicts = c2.checkbox("Remove sequences labelled both BBB and non-BBB", True,
                                     key="drop_conflicts")
        neg_ratio = c3.selectbox("Negatives per positive", [1, 2, 5, 10, "all"], index=0,
                                 key="neg_ratio", help="1 = balanced; 10 = Augur-style imbalance")
        length_match = c3.checkbox("Match negative lengths to positive lengths", True, key="length_match")
        fill_short = c3.checkbox("Fill missing length matches with any length", False, key="fill_short")
        seed1 = c3.number_input("Random seed", value=42, step=1, key="seed1")

    st.markdown("#### 1 · Add data")
    a, b, c = st.columns(3)
    with a:
        st.markdown("**B3Pdb** (BBB-crossing peptides)")
        if st.button("⬇️ Download from B3Pdb", key="btn_b3pdb"):
            msg = st.empty()
            try:
                with st.spinner("Scraping B3Pdb (about a minute) ..."):
                    ss.sources["B3Pdb"] = db.scrape_b3pdb(on_progress=msg.caption)
                st.success(f"B3Pdb: {len(ss.sources['B3Pdb'])} rows added")
            except Exception as e:  # noqa: BLE001
                st.error(str(e))
    with b:
        st.markdown("**UniProt** (presumed non-BBB peptides)")
        with st.expander("Query settings"):
            excl = st.text_area("Exclude terms (one per line)", "\n".join(db.DEFAULT_EXCLUDE),
                                key="excl_terms", height=150)
            max_rows = st.number_input("Max sequences to download", 100, 20000, 5000, step=500,
                                       key="uni_max")
            terms = [t.strip() for t in excl.splitlines() if t.strip()]
            query = st.text_area("UniProt query (editable)",
                                 db.build_uniprot_query(min_len, max_len, terms), key="uni_query")
        if st.button("⬇️ Download UniProt negatives", key="btn_uniprot"):
            msg = st.empty()
            try:
                with st.spinner("Downloading from UniProt ..."):
                    ss.sources["UniProt"] = db.fetch_uniprot_negatives(query, int(max_rows), msg.caption)
                st.success(f"UniProt: {len(ss.sources['UniProt'])} rows added")
            except Exception as e:  # noqa: BLE001
                st.error(str(e))
    with c:
        st.markdown("**Demo** (synthetic, for testing)")
        if st.button("🧪 Load synthetic demo data", key="btn_demo"):
            ss.sources["DEMO_synthetic"] = db.make_demo_data(150, 600)
            st.warning("Synthetic peptides - only to test the app, never for results.")

    st.markdown("**Upload your own files** - BBPpredict, Brainpeps, PepBenchmark, a spreadsheet, "
                "FASTA ... anything with a sequence column.")
    template = pd.DataFrame({"Sequence": ["YGGFM", "KLFMALVAFLRFLT"], "Label": [1, 0],
                             "Source": ["my_lab", "my_lab"]})
    st.download_button("📥 General-format template (CSV)", csv_bytes(template),
                       "general_format_template.csv", key="dl_template")
    uploads = st.file_uploader("Drop files here", accept_multiple_files=True, key="uploads",
                               type=["csv", "tsv", "txt", "xlsx", "xls", "fasta", "fa", "faa", "fas"])
    for f in uploads or []:
        file_mapper(f)

    if ss.sources:
        st.markdown("**Sources in the pipeline**")
        summary = pd.DataFrame([{"Source": k, "Rows": len(v), "BBB (1)": int(v.Label.sum()),
                                 "Non-BBB (0)": int((v.Label == 0).sum())}
                                for k, v in ss.sources.items()])
        st.dataframe(summary, hide_index=True, use_container_width=True)
        rm = st.multiselect("Remove sources", list(ss.sources), key="rm_sources")
        if rm and st.button("🗑️ Remove selected", key="btn_remove"):
            for k in rm:
                ss.sources.pop(k, None)
            st.rerun()

    st.markdown("#### 2 · Build")
    if st.button("🚀 Build BBB / non-BBB files", type="primary", key="btn_build"):
        try:
            ss.built = db.build_datasets(
                list(ss.sources.values()), min_len, max_len, lowercase_mode, drop_conflicts,
                None if neg_ratio == "all" else int(neg_ratio), length_match, fill_short, int(seed1))
            ss.exp, ss.final = None, None      # old models no longer match the data
        except Exception as e:  # noqa: BLE001
            st.error(str(e))

    if ss.built:
        r = ss.built
        bbb, non = r["bbb_df"], r["non_df"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("BBB peptides", len(bbb))
        m2.metric("Non-BBB peptides", len(non))
        m3.metric("Negative pool", r["negative_pool"])
        m4.metric("Label conflicts", r["merge_info"]["label_conflicts"])
        if r["unmatched_negatives"]:
            st.warning(f"{r['unmatched_negatives']} negatives could not be length-matched "
                       "(not enough of that length in the pool). Tick 'Fill missing length "
                       "matches' or add more negatives.")
        d1, d2, d3 = st.columns(3)
        d1.download_button("⬇️ bbb_peptides.csv", csv_bytes(bbb), "bbb_peptides.csv", key="dl_bbb")
        d2.download_button("⬇️ non_bbb_peptides.csv", csv_bytes(non), "non_bbb_peptides.csv", key="dl_non")
        d3.download_button("⬇️ combined.csv", csv_bytes(pd.concat([bbb, non])), "combined_peptides.csv",
                           key="dl_comb")
        with st.expander("🧹 Cleaning report (why rows were removed)"):
            st.dataframe(r["clean_report"], hide_index=True)
            st.caption(f"Duplicates merged: {r['merge_info']['duplicates_merged']}")
        p1, p2, p3, p4 = st.tabs(["Balance & sources", "Length", "Composition", "Physicochemical"])
        with p1:
            q1, q2 = st.columns(2)
            with q1:
                show(plots.label_counts(bbb, non))
            with q2:
                show(plots.source_counts(bbb, non))
        with p2:
            show(plots.length_hist(bbb, non))
        with p3:
            show(plots.aac_compare(bbb, non))
        with p4:
            both = pd.concat([bbb, non], ignore_index=True)
            phy = ft.featurize(both["Sequence"], ["PHY"])
            pick = st.multiselect("Descriptors", list(phy.columns),
                                  default=["PHY_pI", "PHY_Charge_pH7.4", "PHY_GRAVY", "PHY_Length"],
                                  key="phy_pick")
            if pick:
                show(plots.feature_box(phy, both["Target_Label"].to_numpy(), pick))


# ============================================================ TAB 2 ==========
def get_ml_data():
    """Data for tab 2: output of tab 1, or two uploaded step-02 files."""
    options = ["Use the files built in Tab 1", "Upload bbb_peptides.csv and non_bbb_peptides.csv"]
    src = st.radio("Data for machine learning", options, key="ml_src",
                   index=0 if ss.built else 1, horizontal=True)
    if src == options[0]:
        if not ss.built:
            st.info("Nothing built yet - build the files in Tab 1 or upload them here.")
            return None
        pos, neg = ss.built["bbb_df"], ss.built["non_df"]
    else:
        u1, u2 = st.columns(2)
        fp = u1.file_uploader("bbb_peptides.csv (positives)", key="up_pos", type=["csv", "tsv", "txt", "xlsx"])
        fn = u2.file_uploader("non_bbb_peptides.csv (negatives)", key="up_neg", type=["csv", "tsv", "txt", "xlsx"])
        if not (fp and fn):
            return None
        pos, neg = db.read_table(fp.name, fp.getvalue()), db.read_table(fn.name, fn.getvalue())
        for d, n in ((pos, "positive"), (neg, "negative")):
            if "Sequence" not in d.columns:
                st.error(f"The {n} file needs a 'Sequence' column.")
                return None
    df = pd.concat([pos.assign(Label=1), neg.assign(Label=0)], ignore_index=True)
    if "Source" not in df:
        df["Source"] = "uploaded"
    df, _ = db.clean_sequences(df[["Sequence", "Label", "Source"]], 2, 500, "uppercase")
    return df.drop_duplicates("Sequence").reset_index(drop=True)


with tab2:
    st.subheader("Step 03 · Which machine-learning model works best?")
    data = get_ml_data()
    if data is not None:
        n1, n0 = int(data.Label.sum()), int((data.Label == 0).sum())
        st.caption(f"{len(data)} peptides · {n1} BBB · {n0} non-BBB")

        with st.expander("⚙️ Parameters", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Features**")
                fsets = st.multiselect("Feature sets", ft.FEATURE_SETS, default=["AAC", "GAAC", "PHY"],
                                       key="fsets",
                                       help="AAC: 20 · GAAC: 5 groups · DPC: 400 dipeptides · PHY: 12 descriptors")
                scaler = st.selectbox("Scaling", ["Standard", "MinMax", "None"], key="scaler")
                selector = st.selectbox("Feature selection", ["None", "ANOVA F-test", "Mutual information"],
                                        key="selector")
                k_feat = st.slider("Keep top-k features", 5, 200, 30, key="k_feat",
                                   disabled=selector == "None")
            with c2:
                st.markdown("**Validation**")
                split = st.radio("Train/test split", ["Random stratified", "Cluster-aware"], key="split",
                                 help="Cluster-aware keeps similar peptides together, so the test set "
                                      "is really unseen (prevents data leakage).")
                identity = st.slider("Cluster identity cut-off", 0.4, 1.0, 0.8, 0.05, key="identity",
                                     disabled=split != "Cluster-aware")
                test_size = st.slider("Hold-out test fraction", 0.0, 0.5, 0.2, 0.05, key="test_size")
                folds = st.slider("Cross-validation folds", 3, 10, 5, key="folds")
            with c3:
                st.markdown("**Imbalance & decision**")
                samplers = ["None", "Random undersampling", "SMOTE", "Borderline-SMOTE"]
                sampler = st.selectbox("Resampling (training folds only)", samplers, key="sampler")
                if sampler != "None" and not ml.HAS_IMB:
                    st.error("imbalanced-learn is not installed.")
                cw = st.checkbox("Class weights 'balanced'", True, key="cw")
                thr = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05, key="thr")
                seed2 = st.number_input("Random seed", value=42, step=1, key="seed2")

        models = st.multiselect("Models to compare", ml.MODEL_NAMES, default=ml.DEFAULT_MODELS, key="models")
        hp = {}
        with st.expander("🔧 Model hyper-parameters"):
            for name in models:
                specs = ml.HYPERPARAMS[name]
                if not specs:
                    continue
                st.markdown(f"**{name}**")
                cols = st.columns(len(specs))
                hp[name] = {}
                for col, (p, kind, lo, hi, dflt) in zip(cols, specs):
                    hp[name][p] = col.slider(p, lo, hi, dflt, key=f"hp_{name}_{p}")

        if st.button("▶️ Run model comparison", type="primary", key="btn_run"):
            if not fsets or not models:
                st.error("Pick at least one feature set and one model.")
            else:
                cfg = dict(feature_sets=fsets, test_size=test_size, split=split, identity=identity,
                           cv_folds=folds, scaler=scaler, selector=selector, k_features=k_feat,
                           sampler=sampler, class_weight=cw, threshold=thr, seed=int(seed2))
                bar = st.progress(0.0)
                try:
                    with st.spinner("Training models ..."):
                        ss.exp = ml.run_experiment(data, cfg, models, hp,
                                                   on_progress=lambda f, m: bar.progress(min(f, 1.0), text=m))
                    ss.exp["hp"] = hp
                    ss.final = None
                except Exception as e:  # noqa: BLE001
                    st.error(f"Training failed: {e}")
                bar.empty()

    if ss.exp:
        E = ss.exp
        cfg, cv, te = E["cfg"], E["cv_table"], E["test_table"]
        st.markdown("### Results")
        info = f"{len(E['train_idx'])} training · {len(E['test_idx'])} hold-out test · {E['cv_folds']}-fold CV"
        if E["n_clusters"]:
            info += f" · {E['n_clusters']} sequence clusters"
        st.caption(info)
        metric = st.selectbox("Rank models by", ml.METRICS, index=0, key="rank_metric")
        order = cv.sort_values(f"{metric}_mean", ascending=False)["Model"].tolist()
        st.success(f"🏆 Best by cross-validated {metric}: **{order[0]}** "
                   f"({cv.set_index('Model').loc[order[0], metric + '_mean']:.3f})")

        table = cv.set_index("Model").loc[order]
        show_tbl = pd.DataFrame({
            f"{m} (CV)": table[f"{m}_mean"].round(3).astype(str) + " ± " + table[f"{m}_std"].round(3).astype(str)
            for m in ["AUROC", "AUPRC", "F1", "MCC"]}, index=table.index)
        show_tbl["Seconds"] = table["Seconds"]
        if len(te):
            t2 = te.set_index("Model").loc[order]
            show_tbl["AUROC (test)"] = t2["AUROC"].round(3)
            show_tbl["MCC (test)"] = t2["MCC"].round(3)
        st.dataframe(show_tbl.reset_index(), hide_index=True, use_container_width=True)
        st.download_button("⬇️ Full results (CSV)", csv_bytes(cv.merge(te, on="Model", how="left",
                           suffixes=("_cv", "_test")) if len(te) else cv), "model_comparison.csv",
                           key="dl_results")

        g1, g2 = st.columns(2)
        with g1:
            show(plots.model_comparison(cv, metric, te if len(te) else None))
        with g2:
            show(plots.metric_heatmap(cv))
        h1, h2 = st.columns(2)
        with h1:
            show(plots.curves(E["oof"], "roc", "ROC - cross-validation (out-of-fold)"))
        with h2:
            show(plots.curves(E["oof"], "pr", "Precision-recall - cross-validation"))
        if len(te):
            ytest = E["y"][E["test_idx"]]
            k1, k2 = st.columns(2)
            with k1:
                show(plots.curves({m: (ytest, p) for m, p in E["test_proba"].items()}, "roc",
                                  "ROC - hold-out test"))
            with k2:
                cm_model = st.selectbox("Confusion matrix for", order, key="cm_model")
                show(plots.confusion(ytest, E["test_proba"][cm_model], cfg["threshold"],
                                     f"{cm_model} (test, threshold {cfg['threshold']})"))
        st.caption("Pick the model with the cross-validation score, then report the hold-out test "
                   "score once. Choosing by test score would make the test set optimistic.")

        st.markdown("### Choose the model for Tabs 3-4")
        f1c, f2c = st.columns([2, 2])
        chosen = f1c.selectbox("Final model", order, key="final_model")
        refit = f2c.checkbox("Re-train on ALL data (recommended for screening)", True, key="refit_all")
        if st.button("✅ Use this model", key="btn_final"):
            with st.spinner("Preparing model ..."):
                pipe = ml.refit_on_all(E, chosen, E.get("hp", {})) if refit else E["fitted"][chosen]
            ss.final = {"name": chosen, "pipe": pipe, "cfg": cfg, "names": E["names"],
                        "train_seqs": set(E["seqs"]), "min_len": min(map(len, E["seqs"])),
                        "max_len": max(map(len, E["seqs"])), "refit": refit}
            st.success(f"{chosen} is ready - go to Tab 3 or Tab 4.")
        if ss.final:
            buf = io.BytesIO()
            joblib.dump({k: ss.final[k] for k in ("name", "pipe", "cfg", "names")}, buf)
            st.download_button("⬇️ Download model (.joblib)", buf.getvalue(), "bbb_model.joblib",
                               key="dl_model")


# ============================================================ TAB 3 ==========
with tab3:
    st.subheader("Explain the model")
    if not ss.exp:
        st.info("Run the model comparison in Tab 2 first.")
    else:
        E = ss.exp
        names_m = list(E["fitted"])
        model = st.selectbox("Model", names_m, key="exp_model")
        has_test = len(E["test_idx"]) > 0
        if has_test:
            idx, y_eval, p_eval = E["test_idx"], E["y"][E["test_idx"]], E["test_proba"][model]
            where = "hold-out test set"
        else:
            y_eval, p_eval = E["oof"][model]
            idx, where = E["train_idx"], "cross-validation (out-of-fold)"
        st.caption(f"Evaluated on the {where}.")

        e1, e2 = st.tabs(["Which features matter?", "Where does the model fail?"])
        with e1:
            r1, r2 = st.columns(2)
            reps = r1.slider("Shuffle repeats", 3, 30, 10, key="perm_reps")
            topn = r2.slider("Features to show", 5, 40, 20, key="perm_top")
            if st.button("Compute permutation importance", key="btn_perm"):
                with st.spinner("Shuffling features ..."):
                    ss["imp"] = ml.importance_table(E["fitted"][model], E["X"][idx], E["y"][idx],
                                                    E["names"], reps, E["cfg"]["seed"])
                    ss["imp_model"] = model
            if ss.get("imp") is not None:
                st.caption(f"Importance for: {ss['imp_model']}")
                show(plots.importance_bar(ss["imp"], topn))
                st.dataframe(ss["imp"].round(4), hide_index=True, use_container_width=True)
            st.markdown("**Feature distributions**")
            feats = st.multiselect("Features", E["names"], default=E["names"][:1], key="dist_feats")
            if feats:
                Xdf = pd.DataFrame(E["X"], columns=E["names"])
                show(plots.feature_box(Xdf, E["y"], feats[:5]))
        with e2:
            tbl = pd.DataFrame({"Sequence": [E["seqs"][i] for i in idx], "Actual": y_eval,
                                "BBB_probability": p_eval.round(3)})
            tbl["Predicted"] = (tbl["BBB_probability"] >= E["cfg"]["threshold"]).astype(int)
            wrong = tbl[tbl["Actual"] != tbl["Predicted"]].copy()
            wrong["Confidence"] = (wrong["BBB_probability"] - E["cfg"]["threshold"]).abs().round(3)
            st.write(f"**{len(wrong)}** misclassified of {len(tbl)} ({len(wrong) / max(1, len(tbl)):.0%}). "
                     "Most confident mistakes first - check these peptides in the literature.")
            st.dataframe(wrong.sort_values("Confidence", ascending=False), hide_index=True,
                         use_container_width=True)


# ============================================================ TAB 4 ==========
with tab4:
    st.subheader("Predict & screen new peptides")
    if not ss.final:
        st.info("Choose a final model at the bottom of Tab 2 first.")
    else:
        F = ss.final
        st.caption(f"Model: **{F['name']}** · features: {', '.join(F['cfg']['feature_sets'])} · "
                   f"trained on {'all data' if F['refit'] else 'the training split'}")
        text = st.text_area("Peptide sequences (one per line, or FASTA)", "", height=140, key="pred_text",
                            placeholder="e.g. your microbiota peptides")
        up = st.file_uploader("...or upload a file (csv / fasta / txt)", key="pred_file",
                              type=["csv", "tsv", "txt", "fasta", "fa", "xlsx"])
        thr4 = st.slider("Decision threshold", 0.1, 0.9, float(F["cfg"]["threshold"]), 0.05, key="thr4")
        if st.button("🔍 Predict", type="primary", key="btn_predict"):
            raw = []
            if text.strip():
                if text.lstrip().startswith(">"):
                    raw += db.read_fasta_text(text)["Sequence"].tolist()
                else:
                    raw += [t.strip() for t in text.splitlines() if t.strip()]
            if up is not None:
                d = db.read_table(up.name, up.getvalue())
                col = db.guess_column(d.columns, db.SEQ_GUESSES) or d.columns[0]
                raw += d[col].astype(str).tolist()
            ok, bad = [], []
            for s in raw:
                clean, why = db.parse_peptide(s, "uppercase")
                (ok if clean else bad).append(clean if clean else (s, why))
            ss["pred_ok"], ss["pred_bad"] = ok, bad
        if ss.get("pred_ok"):
            seqs = list(dict.fromkeys(ss["pred_ok"]))
            X = ft.featurize(seqs, F["cfg"]["feature_sets"]).to_numpy(float)
            proba = F["pipe"].predict_proba(X)[:, 1]
            out = pd.DataFrame({"Sequence": seqs, "Length": [len(s) for s in seqs],
                                "BBB_probability": proba.round(3)})
            out["Prediction"] = ["Crosses BBB" if p >= thr4 else "Non-BBB" for p in proba]
            out["In_training_set"] = [s in F["train_seqs"] for s in seqs]
            out["Length_in_training_range"] = out["Length"].between(F["min_len"], F["max_len"])
            out = out.sort_values("BBB_probability", ascending=False).reset_index(drop=True)
            st.dataframe(out, hide_index=True, use_container_width=True)
            show(plots.score_bars(out, thr4))
            st.download_button("⬇️ Predictions (CSV)", csv_bytes(out), "bbb_predictions.csv", key="dl_pred")
            if out["In_training_set"].any():
                st.warning("Some peptides were in the training data - their scores are optimistic.")
            if not out["Length_in_training_range"].all():
                st.warning(f"Some peptides are outside the training length range "
                           f"({F['min_len']}-{F['max_len']} aa); treat those scores with caution.")
            st.caption("A score is a model estimate from sequence only. It is not experimental "
                       "proof of BBB penetration.")
        if ss.get("pred_bad"):
            with st.expander(f"{len(ss['pred_bad'])} sequences could not be used"):
                st.dataframe(pd.DataFrame(ss["pred_bad"], columns=["Input", "Reason"]), hide_index=True)
