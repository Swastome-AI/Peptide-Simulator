"""Step 02 - build the BBB / non-BBB peptide files from ANY source.

Everything is converted to ONE general format first:

    Sequence | Label (1 = crosses BBB, 0 = does not) | Source

so data from B3Pdb, BBPpredict, Brainpeps, PepBenchmark, UniProt or your own
spreadsheet can be mixed freely.
"""
import io
import re
import time
from io import StringIO

import numpy as np
import pandas as pd
import requests

from .features import AA, isoelectric_point

VALID_AA = set(AA)
STD_COLS = ["Sequence", "Label", "Source"]

DEFAULT_EXCLUDE = [
    "blood brain barrier", "brain", "brainpeps", "b3pdb", "permeation",
    "permeability", "venom", "toxin", "transmembrane", "transport",
    "transfer", "membrane", "neuro", "hemolysis",
]
SEQ_GUESSES = ["sequence", "seq", "peptide", "peptide sequence",
               "peptide sequence (1-letter)", "cleaned_sequence", "aa_sequence"]
LABEL_GUESSES = ["label", "class", "y", "target", "target_label", "bbb",
                 "bbb_cross", "permeability", "response"]


# ------------------------------------------------------------------ reading --
def read_fasta_text(text):
    rows, header, cur = [], None, []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(">"):
            if header is not None:
                rows.append((header, "".join(cur)))
            header, cur = line[1:], []
        elif line:
            cur.append(line)
    if header is not None:
        rows.append((header, "".join(cur)))
    return pd.DataFrame(rows, columns=["Header", "Sequence"])


def read_table(name, data):
    """Read an uploaded file (csv / tsv / txt / xlsx / fasta) into a DataFrame."""
    lower = name.lower()
    head = data[:300].decode("utf-8", "ignore").lstrip()
    if lower.endswith((".fasta", ".fa", ".faa", ".fas")) or head.startswith(">"):
        return read_fasta_text(data.decode("utf-8", "ignore"))
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data))
    try:
        return pd.read_csv(io.BytesIO(data), sep=None, engine="python")
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(data), sep=None, engine="python", encoding="latin-1")


def guess_column(columns, guesses):
    lower = {str(c).lower().strip(): c for c in columns}
    for g in guesses:
        if g in lower:
            return lower[g]
    return None


# ------------------------------------------------------- to general format ---
def standardize(df, seq_col, source, label_mode="all_positive", label_col=None,
                pos_values="", neg_values="", threshold=None, contains=False):
    """Convert any table to the general format. Returns (std_df, n_unlabelled_dropped).

    label_mode
      all_positive   every row is a BBB-crossing peptide
      all_negative   every row is a non-BBB peptide
      values         read the label column; text in `pos_values` -> 1,
                     text in `neg_values` -> 0 (comma separated, case-insensitive).
                     If `neg_values` is blank, everything that is not positive -> 0.
      numeric        label column is a number; value >= threshold -> 1 else 0
    """
    out = pd.DataFrame({"Sequence": df[seq_col].astype(str)})
    dropped = 0
    if label_mode == "all_positive":
        out["Label"] = 1
    elif label_mode == "all_negative":
        out["Label"] = 0
    elif label_mode == "numeric":
        vals = pd.to_numeric(df[label_col], errors="coerce")
        out["Label"] = np.where(vals.isna(), np.nan, (vals >= float(threshold)).astype(float))
    elif label_mode == "values":
        col = df[label_col].astype(str).str.strip().str.lower()
        pos = [v.strip().lower() for v in pos_values.split(",") if v.strip()]
        neg = [v.strip().lower() for v in neg_values.split(",") if v.strip()]
        match = (lambda vs: col.apply(lambda x: any(v in x for v in vs))) if contains \
            else (lambda vs: col.isin(vs))
        is_pos = match(pos) if pos else pd.Series(False, index=col.index)
        if neg:
            is_neg = match(neg)
            lab = pd.Series(np.nan, index=col.index)
            lab[is_neg] = 0.0
            lab[is_pos] = 1.0
        else:
            lab = pd.Series(0.0, index=col.index)
            lab[is_pos] = 1.0
        out["Label"] = lab
    else:
        raise ValueError(f"Unknown label_mode: {label_mode}")
    dropped = int(out["Label"].isna().sum())
    out = out.dropna(subset=["Label"]).copy()
    out["Label"] = out["Label"].astype(int)
    out["Source"] = source
    return out[STD_COLS].reset_index(drop=True), dropped


# ----------------------------------------------------------------- cleaning --
def parse_peptide(raw, lowercase_mode="reject"):
    """Return (clean_sequence, None) or (None, reason)."""
    seq = str(raw).strip().strip('"')
    if seq.upper() in ("", "NA", "NAN", "NONE"):
        return None, "empty"
    core = re.sub(r"^(Ac|H)-", "", seq, flags=re.I)
    core = re.sub(r"-(NH\d*|OH|amide)$", "", core, flags=re.I)
    if re.search(r"[-()\[\]]", core):
        return None, "modified / cyclic notation"
    if any(c.islower() for c in core):
        if lowercase_mode == "reject":
            return None, "lowercase (D-amino acid)"
        core = core.upper()
    core = core.upper()
    if not core or not set(core) <= VALID_AA:
        return None, "non-standard letters (X, B, Z, U ...)"
    return core, None


def clean_sequences(df, min_len=5, max_len=50, lowercase_mode="reject"):
    """Return (clean_df, report_df). Report counts every reason a row was dropped."""
    reasons, seqs = [], []
    for raw in df["Sequence"]:
        s, why = parse_peptide(raw, lowercase_mode)
        seqs.append(s)
        reasons.append(why)
    d = df.copy()
    d["Sequence"] = seqs
    d["_why"] = reasons
    n_in = len(d)
    kept = d[d["_why"].isna()].copy()
    short = (kept["Sequence"].str.len() < min_len) | (kept["Sequence"].str.len() > max_len)
    reasons_all = d["_why"].dropna().tolist() + [f"length outside {min_len}-{max_len}"] * int(short.sum())
    kept = kept[~short].drop(columns="_why")
    rep = pd.Series(reasons_all, dtype=str).value_counts().rename_axis("Reason").reset_index(name="Rows")
    rep.loc[len(rep)] = ["KEPT", len(kept)]
    rep.loc[len(rep)] = ["TOTAL IN", n_in]
    return kept.reset_index(drop=True), rep


def merge_sources(df, drop_conflicts=True):
    """Collapse duplicates (joining Source names) and remove label conflicts."""
    n0 = len(df)
    merged = (df.groupby(["Sequence", "Label"])["Source"]
              .agg(lambda s: ";".join(sorted(set(s)))).reset_index())
    n_dupes = n0 - len(merged)
    pos = set(merged.loc[merged.Label == 1, "Sequence"])
    neg = set(merged.loc[merged.Label == 0, "Sequence"])
    both = pos & neg
    if drop_conflicts and both:
        merged = merged[~merged["Sequence"].isin(both)]
    return merged.reset_index(drop=True), {"duplicates_merged": n_dupes,
                                          "label_conflicts": len(both),
                                          "conflicts_removed": drop_conflicts}


# --------------------------------------------------------------- downloads ---
def build_uniprot_query(min_len, max_len, exclude_terms):
    q = f"reviewed:true AND length:[{min_len} TO {max_len}]"
    return q + "".join(f' NOT "{t}"' for t in exclude_terms)


def fetch_uniprot_negatives(query, max_rows=5000, on_progress=None):
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {"query": query, "format": "tsv", "fields": "accession,length,sequence",
              "size": 500}
    frames, total, page = [], 0, 0
    while url:
        for attempt in range(1, 6):
            try:
                r = requests.get(url, params=params, timeout=60)
                if r.status_code == 400:
                    raise RuntimeError("UniProt rejected the query:\n" + r.text[:500])
                r.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                if on_progress:
                    on_progress(f"UniProt attempt {attempt}/5 failed ({type(e).__name__}); retrying")
                time.sleep(4 * attempt)
        else:
            raise RuntimeError("UniProt download failed after 5 retries - try again.")
        chunk = pd.read_csv(StringIO(r.text), sep="\t")
        frames.append(chunk)
        total += len(chunk)
        page += 1
        if on_progress:
            on_progress(f"UniProt page {page}: {total} rows so far")
        if total >= max_rows:
            break
        url = r.links.get("next", {}).get("url")
        params = None
    df = pd.concat(frames, ignore_index=True).head(max_rows)
    return pd.DataFrame({"Sequence": df["Sequence"], "Label": 0, "Source": "UniProt"})


def scrape_b3pdb(on_progress=None):
    """Download all B3Pdb categories (BBB-crossing peptides). Returns general format."""
    url = "https://webs.iiitd.edu.in/raghava/b3pdb/browse_sub1.php"
    frames = []
    for category in range(1, 7):
        page = 1
        while True:
            df = None
            for attempt in range(1, 6):
                try:
                    r = requests.get(url, params={"token": f"ppl{category}", "col": 7,
                                                  "page": page}, timeout=30)
                    r.raise_for_status()
                    try:
                        df = pd.read_html(StringIO(r.text))[0]
                    except ValueError:
                        df = None
                    break
                except requests.exceptions.RequestException as e:
                    if on_progress:
                        on_progress(f"B3Pdb ppl{category} p{page} attempt {attempt}/5 failed "
                                    f"({type(e).__name__})")
                    time.sleep(4 * attempt)
            else:
                raise RuntimeError(f"B3Pdb: could not fetch category {category}, page {page}.")
            if df is None or len(df) == 0:
                break
            frames.append(df[["B3pdbID", "PEPTIDE SEQUENCE (1-letter)"]])
            if on_progress:
                on_progress(f"B3Pdb category {category}, page {page}")
            page += 1
    raw = pd.concat(frames, ignore_index=True)
    return pd.DataFrame({"Sequence": raw["PEPTIDE SEQUENCE (1-letter)"], "Label": 1,
                         "Source": "B3Pdb"})


def make_demo_data(n_pos=150, n_neg=300, seed=0):
    """SYNTHETIC peptides, only for testing the app. Positives are enriched in R/K/Y/F/W."""
    rng = np.random.default_rng(seed)
    aa = np.array(list(AA))
    base = np.ones(20)
    pos_w = base.copy()
    for a, w in {"R": 3.0, "K": 2.0, "Y": 2.5, "F": 1.8, "W": 1.8, "G": 1.5}.items():
        pos_w[AA.index(a)] = w
    pos_w /= pos_w.sum()
    base /= base.sum()
    def gen(n, w, lo, hi):
        return ["".join(rng.choice(aa, size=rng.integers(lo, hi + 1), p=w)) for _ in range(n)]
    return pd.concat([
        pd.DataFrame({"Sequence": gen(n_pos, pos_w, 6, 20), "Label": 1, "Source": "DEMO_synthetic"}),
        pd.DataFrame({"Sequence": gen(n_neg, base, 6, 30), "Label": 0, "Source": "DEMO_synthetic"}),
    ], ignore_index=True)


# ------------------------------------------------------------------ output ---
def length_matched_sample(pos, neg, ratio, fill_shortfall, seed):
    if ratio is None:
        return neg, 0
    rng = np.random.default_rng(seed)
    picks, short = [], 0
    for L, n_pos in pos["Sequence"].str.len().value_counts().items():
        pool = neg[neg["Sequence"].str.len() == L]
        want = int(n_pos * ratio)
        take = min(want, len(pool))
        short += want - take
        if take:
            picks.append(pool.sample(take, random_state=int(rng.integers(1e9))))
    chosen = pd.concat(picks) if picks else neg.iloc[0:0]
    if short and fill_shortfall:
        rest = neg.drop(chosen.index)
        extra = rest.sample(min(short, len(rest)), random_state=seed)
        chosen = pd.concat([chosen, extra])
        short -= len(extra)
    return chosen, short


def finish(df, prefix):
    df = df.sort_values("Sequence").reset_index(drop=True)
    out = pd.DataFrame({
        "Peptide_ID": [f"{prefix}_{i:04d}" for i in range(1, len(df) + 1)],
        "Sequence": df["Sequence"],
        "Target_Result": df["Label"].map({1: "Crosses_BBB", 0: "Non_BBB"}),
        "Target_Label": df["Label"],
        "Length": df["Sequence"].str.len(),
        "pI": df["Sequence"].apply(isoelectric_point),
        "Source": df["Source"],
    })
    return out


def build_datasets(std_frames, min_len=5, max_len=50, lowercase_mode="reject",
                   drop_conflicts=True, neg_ratio=1, length_match=True,
                   fill_shortfall=False, seed=42):
    """Full step-02 pipeline. Returns dict with bbb_df, non_df, reports."""
    if not std_frames:
        raise ValueError("Add at least one data source first.")
    allp = pd.concat(std_frames, ignore_index=True)
    cleaned, clean_report = clean_sequences(allp, min_len, max_len, lowercase_mode)
    merged, merge_info = merge_sources(cleaned, drop_conflicts)
    pos = merged[merged.Label == 1]
    neg = merged[merged.Label == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError(f"Need both classes. Found {len(pos)} positives and "
                         f"{len(neg)} negatives after cleaning.")
    n_pool = len(neg)
    if length_match:
        neg, short = length_matched_sample(pos, neg, neg_ratio, fill_shortfall, seed)
    else:
        short = 0
        if neg_ratio is not None and len(neg) > len(pos) * neg_ratio:
            neg = neg.sample(int(len(pos) * neg_ratio), random_state=seed)
    return {
        "bbb_df": finish(pos, "BBB"),
        "non_df": finish(neg, "NON"),
        "clean_report": clean_report,
        "merge_info": merge_info,
        "negative_pool": n_pool,
        "unmatched_negatives": short,
    }
