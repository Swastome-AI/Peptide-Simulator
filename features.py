"""Peptide feature engineering (no Biopython needed, so the app stays light).

Feature sets
  AAC   amino-acid composition (20)
  GAAC  grouped AAC: aliphatic / aromatic / positive / negative / uncharged (5)
  DPC   dipeptide composition (400)
  PHY   physicochemical descriptors (length, MW, pI, charge, GRAVY, ...)
"""
import numpy as np
import pandas as pd

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}

GAAC_GROUPS = {
    "aliphatic": "GAVLMI",
    "aromatic": "FYW",
    "positive": "KRH",
    "negative": "DE",
    "uncharged": "STCPNQ",
}

# Kyte-Doolittle hydropathy
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
      "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
      "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}

# Average residue masses (Da)
MASS = {"A": 71.0788, "R": 156.1875, "N": 114.1038, "D": 115.0886, "C": 103.1388,
        "E": 129.1155, "Q": 128.1307, "G": 57.0519, "H": 137.1411, "I": 113.1594,
        "L": 113.1594, "K": 128.1741, "M": 131.1926, "F": 147.1766, "P": 97.1167,
        "S": 87.0782, "T": 101.1051, "W": 186.2132, "Y": 163.1760, "V": 99.1326}
WATER = 18.0153

# pKa values (same set Biopython uses)
POS_PK = {"Nterm": 9.0, "K": 10.0, "R": 12.0, "H": 5.98}
NEG_PK = {"Cterm": 2.0, "D": 4.05, "E": 4.45, "C": 9.0, "Y": 10.0}
PK_C = {"D": 4.55, "E": 4.75}
PK_N = {"A": 7.59, "M": 7.0, "S": 6.93, "P": 8.36, "T": 6.82, "V": 7.44, "E": 7.7}

FEATURE_SETS = ["AAC", "GAAC", "DPC", "PHY"]
PHY_NAMES = ["Length", "MW", "pI", "Charge_pH7.4", "GRAVY", "Aromaticity",
             "Aliphatic_index", "Frac_positive", "Frac_negative", "Frac_polar",
             "Frac_Gly_Pro", "Frac_Cys"]


def net_charge(seq, ph=7.4):
    pk_n = PK_N.get(seq[0], POS_PK["Nterm"])
    pk_c = PK_C.get(seq[-1], NEG_PK["Cterm"])
    ch = 1 / (1 + 10 ** (ph - pk_n)) - 1 / (1 + 10 ** (pk_c - ph))
    for aa in "KRH":
        ch += seq.count(aa) / (1 + 10 ** (ph - POS_PK[aa]))
    for aa in "DECY":
        ch -= seq.count(aa) / (1 + 10 ** (NEG_PK[aa] - ph))
    return ch


def isoelectric_point(seq):
    lo, hi = 0.0, 14.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if net_charge(seq, mid) > 0:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 3)


def molecular_weight(seq):
    return sum(MASS[a] for a in seq) + WATER


def physchem(seq):
    n = len(seq)
    frac = lambda letters: sum(seq.count(a) for a in letters) / n
    aliph = 100 * (frac("A") + 2.9 * frac("V") + 3.9 * (frac("I") + frac("L")))
    return {
        "Length": n,
        "MW": molecular_weight(seq),
        "pI": isoelectric_point(seq),
        "Charge_pH7.4": net_charge(seq, 7.4),
        "GRAVY": sum(KD[a] for a in seq) / n,
        "Aromaticity": frac("FYW"),
        "Aliphatic_index": aliph,
        "Frac_positive": frac("KR"),
        "Frac_negative": frac("DE"),
        "Frac_polar": frac("STNQ"),
        "Frac_Gly_Pro": frac("GP"),
        "Frac_Cys": frac("C"),
    }


def featurize(seqs, feature_sets=("AAC", "PHY")):
    """Return a DataFrame (one row per sequence) with all requested features."""
    seqs = list(seqs)
    blocks = []
    if "AAC" in feature_sets:
        m = np.zeros((len(seqs), 20))
        for i, s in enumerate(seqs):
            for a in s:
                m[i, AA_IDX[a]] += 1
            m[i] /= len(s)
        blocks.append(pd.DataFrame(m, columns=[f"AAC_{a}" for a in AA]))
    if "GAAC" in feature_sets:
        m = np.array([[sum(s.count(a) for a in g) / len(s) for g in GAAC_GROUPS.values()]
                      for s in seqs])
        blocks.append(pd.DataFrame(m, columns=[f"GAAC_{k}" for k in GAAC_GROUPS]))
    if "DPC" in feature_sets:
        m = np.zeros((len(seqs), 400))
        for i, s in enumerate(seqs):
            if len(s) > 1:
                for a, b in zip(s, s[1:]):
                    m[i, AA_IDX[a] * 20 + AA_IDX[b]] += 1
                m[i] /= (len(s) - 1)
        blocks.append(pd.DataFrame(m, columns=[f"DPC_{a}{b}" for a in AA for b in AA]))
    if "PHY" in feature_sets:
        rows = [physchem(s) for s in seqs]
        blocks.append(pd.DataFrame(rows)[PHY_NAMES].add_prefix("PHY_"))
    if not blocks:
        raise ValueError("Select at least one feature set.")
    return pd.concat(blocks, axis=1)
