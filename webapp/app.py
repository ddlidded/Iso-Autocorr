import io
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from flask import Flask, Response, flash, redirect, render_template, request, send_file, url_for
from scipy.optimize import nnls


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev")


TR_SYMBOLS = ["C", "N", "D", "O"]
TR_NAMES = ["13C", "15N", "2D", "18O"]
# Natural isotope abundances used by the MATLAB code (Autocorr.m)
ABUNDANCE = [0.0107, 0.00364, 0.00001, 0.00187]


@dataclass(frozen=True)
class MetaResult:
    id: int
    name: str
    formula: str
    corr_abs: np.ndarray  # (rows, samples)
    corr_pct: np.ndarray  # (rows, samples)
    corr_tic: np.ndarray  # (samples,)


def label_autodetect(labels: Sequence[str]) -> List[int]:
    """Return 1-based tracer indices (C=1,N=2,D=3,O=4) detected in the isotopeLabel strings."""
    blob = "".join(labels)
    found = []
    if "C13" in blob:
        found.append(1)
    if "N15" in blob:
        found.append(2)
    if "D2" in blob:
        found.append(3)
    if "O18" in blob:
        found.append(4)
    return found


def _binom_coeff(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def dbinom(a: int, b: int, r: float) -> float:
    return _binom_coeff(a, b) * ((1.0 - r) ** (a - b)) * (r**b)


def isocorr_A(distin: np.ndarray, n: int, ab_A: float, im_A: float) -> Tuple[np.ndarray, np.ndarray]:
    """Python port of isocorr_A.m.

    distin: (n+1, samples)
    returns (distout_abs, distout_pct)
    """
    if distin.ndim == 1:
        distin = distin.reshape(-1, 1)
    if distin.shape[0] != n + 1:
        raise ValueError(f"distin rows ({distin.shape[0]}) != n+1 ({n+1})")

    r = ab_A
    r1 = im_A

    M = np.zeros((n + 1, n + 1), dtype=float)
    for i in range(1, n + 2):
        for j in range(1, i + 1):
            a = (n + 1) - j
            b = i - j
            M[i - 1, j - 1] = _binom_coeff(a, b) * ((1.0 - r) ** (a - b)) * (r**b)

    Mp = np.zeros((n + 1, n + 1), dtype=float)
    for i in range(1, n + 2):
        for j in range(i, n + 2):
            a = j - 1
            b = j - i
            Mp[i - 1, j - 1] = _binom_coeff(a, b) * ((1.0 - r1) ** (a - b)) * (r1**b)

    N = M @ Mp
    distout = np.zeros_like(distin, dtype=float)
    for col in range(distin.shape[1]):
        x, _ = nnls(N, distin[:, col])
        distout[:, col] = x

    denom = distout.sum(axis=0, keepdims=True) + 1e-10
    distout_pct = distout / denom
    return distout, distout_pct


def isocorr_AB(
    distin: np.ndarray,
    n: int,
    m: int,
    ab_A: float,
    ab_B: float,
    im_A: float,
    im_B: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Python port of isocorr_AB.m.

    distin: ((n+1)*(m+1), samples)
    returns (distout_abs, distout_pct)
    """
    if distin.ndim == 1:
        distin = distin.reshape(-1, 1)

    dim = (n + 1) * (m + 1)
    if distin.shape[0] != dim:
        raise ValueError(f"distin rows ({distin.shape[0]}) != (n+1)*(m+1) ({dim})")

    M = np.zeros((dim, dim), dtype=float)
    for i in range(1, dim + 1):
        A_num_row = (i - 1) // (m + 1)
        B_num_row = (i - 1) % (m + 1)
        for j in range(1, i + 1):
            A_num_col = (j - 1) // (m + 1)
            B_num_col = (j - 1) % (m + 1)

            a1 = n - A_num_col
            b1 = A_num_row - A_num_col
            a2 = m - B_num_col
            b2 = B_num_row - B_num_col
            if b1 >= 0 and b2 >= 0:
                M[i - 1, j - 1] = dbinom(a1, b1, ab_A) * dbinom(a2, b2, ab_B)

    Mp = np.zeros((dim, dim), dtype=float)
    for i in range(1, dim + 1):
        A_num_row = (i - 1) // (m + 1)
        B_num_row = (i - 1) % (m + 1)
        for j in range(i, dim + 1):
            A_num_col = (j - 1) // (m + 1)
            B_num_col = (j - 1) % (m + 1)

            a1 = A_num_col
            b1 = A_num_col - A_num_row
            a2 = B_num_col
            b2 = B_num_col - B_num_row
            if b1 >= 0 and b2 >= 0:
                Mp[i - 1, j - 1] = dbinom(a1, b1, im_A) * dbinom(a2, b2, im_B)

    N = M @ Mp
    distout = np.zeros_like(distin, dtype=float)
    for col in range(distin.shape[1]):
        x, _ = nnls(N, distin[:, col])
        distout[:, col] = x

    denom = distout.sum(axis=0, keepdims=True) + 1e-10
    distout_pct = distout / denom
    return distout, distout_pct


def str2ab(label: str, A: str, B: str) -> Tuple[int, int]:
    """Python port of str2AB.m (no printing; raises ValueError on parse issues)."""
    parts = label.split("-")
    if len(parts) == 1:
        return 0, 0
    if len(parts) == 3:
        if label and label[0] == A:
            return int(parts[-1]), 0
        if label and label[0] == B:
            return 0, int(parts[-1])
        raise ValueError(f"Unexpected tracer in isotopeLabel '{label}' (expected {A}/{B})")
    if len(parts) == 4:
        return int(parts[-2]), int(parts[-1])
    raise ValueError(f"Cannot parse isotopeLabel '{label}'")


def ab_getcounts(labels: Sequence[str], A: str, B: str) -> np.ndarray:
    counts = np.zeros((len(labels), 2), dtype=int)
    for i, s in enumerate(labels):
        a, b = str2ab(s, A, B)
        counts[i, 0] = a
        counts[i, 1] = b
    return counts


def ab_getfulldt(dt: np.ndarray, counts: np.ndarray, A_num: int, B_num: int) -> Tuple[np.ndarray, List[int]]:
    """Python port of AB_getfulldt.m.

    dt: (rows_present, samples)
    counts: (rows_present, 2)
    returns (fulldt, reduced_idx)
    - fulldt: ((A_num+1)*(B_num+1), samples)
    - reduced_idx: positions (0-based) of present rows in full table, ordered as in input
    """
    # MATLAB equivalent:
    #   v1=repmat(0:B_num,1,A_num+1);
    #   v2=reshape(repmat(0:A_num,B_num+1,1),1,(B_num+1)*(A_num+1));
    #   cn=[v1;v2]'; cn=cn(:,[2,1]);
    #
    # Note: MATLAB reshape is column-major; the v2 pattern is therefore:
    #   [0,0,..., 1,1,..., 2,2,...] where each A repeats (B_num+1) times.
    v1 = np.tile(np.arange(0, B_num + 1, dtype=int), A_num + 1)
    v2 = np.repeat(np.arange(0, A_num + 1, dtype=int), B_num + 1)
    cn = np.stack([v1, v2], axis=1)[:, [1, 0]]  # swap columns like MATLAB cn(:,[2,1])

    fulldt_rows: List[np.ndarray] = []
    idx_r: List[int] = []
    idx_j: List[int] = []

    for j in range(cn.shape[0]):
        matches = np.where((counts == cn[j]).all(axis=1))[0]
        if matches.size == 0:
            fulldt_rows.append(np.zeros((dt.shape[1],), dtype=float))
        else:
            tp = int(matches[0])
            idx_r.append(tp)
            idx_j.append(j)
            fulldt_rows.append(dt[tp].astype(float))

    fulldt = np.vstack(fulldt_rows)
    order = np.argsort(np.array(idx_r, dtype=int))
    reduced_idx = [idx_j[k] for k in order.tolist()]
    return fulldt, reduced_idx


def formula_counts(formula: str) -> List[int]:
    """Return [C,N,H,O,S,P] counts; supports parentheses and simple formulas.

    This is a lightweight replacement for the MATLAB formula2mass() usage in Autocorr.m,
    where only element counts are needed.
    """

    # Drop charge annotations like "+" / "-" and surrounding quotes
    f = formula.strip().strip('"')
    f = f.replace("+", "").replace("-", "")

    tokens: List[Tuple[str, int]] = []

    def parse_int(i: int) -> Tuple[int, int]:
        if i >= len(f) or not f[i].isdigit():
            return 1, i
        j = i
        while j < len(f) and f[j].isdigit():
            j += 1
        return int(f[i:j]), j

    def parse_group(i: int) -> Tuple[dict, int]:
        counts: dict = {}
        while i < len(f):
            ch = f[i]
            if ch == "(":
                inner, i = parse_group(i + 1)
                if i >= len(f) or f[i] != ")":
                    raise ValueError(f"Unmatched '(' in formula '{formula}'")
                mult, i = parse_int(i + 1)
                for k, v in inner.items():
                    counts[k] = counts.get(k, 0) + v * mult
                continue
            if ch == ")":
                return counts, i
            if ch.isupper():
                sym = ch
                i += 1
                if i < len(f) and f[i].islower():
                    sym += f[i]
                    i += 1
                mult, i = parse_int(i)
                counts[sym] = counts.get(sym, 0) + mult
                continue
            if ch in " []{},;":
                i += 1
                continue
            raise ValueError(f"Unexpected character '{ch}' in formula '{formula}'")
        return counts, i

    parsed, end_i = parse_group(0)
    if end_i != len(f):
        # parse_group stops early only if it hits ')'
        raise ValueError(f"Unexpected trailing input in formula '{formula}'")

    symbols = ["C", "N", "H", "O", "S", "P"]
    return [int(parsed.get(s, 0)) for s in symbols]


def run_autocorr(df: pd.DataFrame, impurity: Sequence[float]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Replicates Autocorr.m for a single input table."""

    required_cols = {"medMz", "isotopeLabel", "parent", "compound", "formula", "medRt", "ppmDiff"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Clean up common CSV formatting issues: trim whitespace on all string-like columns
    # (MATLAB readtable()/writetable tends to drop trailing spaces).
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            df[col] = df[col].astype("string").str.strip()

    # Cut empty rows: T=T(1:length(find([T.medMz]>0)),:)
    if "medMz" in df.columns:
        positive = df["medMz"].fillna(0).to_numpy(dtype=float) > 0
        if positive.any():
            last_pos = int(np.where(positive)[0][-1])
            df = df.iloc[: last_pos + 1].copy()
        else:
            df = df.iloc[0:0].copy()

    parent_idx = list(df.columns).index("parent")
    start_col = parent_idx + 1
    sample_cols = list(df.columns)[start_col:]

    # Group heads: rows where isotopeLabel == 'C12 PARENT'
    grp_heads = df.index[df["isotopeLabel"].astype(str) == "C12 PARENT"].to_list()
    if not grp_heads:
        raise ValueError("Could not find any 'C12 PARENT' rows in isotopeLabel")
    grp_heads.append(int(df.index[-1]) + 1)  # sentinel like MATLAB

    meta_results: List[MetaResult] = []

    for i in range(len(grp_heads) - 1):
        start = grp_heads[i]
        end_exclusive = grp_heads[i + 1]
        sub = df.loc[start : end_exclusive - 1].copy()
        # Match MATLAB Autocorr.m behavior: overwrite metaGroupId to sequential group index.
        if "metaGroupId" in df.columns:
            df.loc[start : end_exclusive - 1, "metaGroupId"] = i + 1
        else:
            # If missing, add it (kept before sample columns doesn't matter for output).
            df.loc[start : end_exclusive - 1, "metaGroupId"] = i + 1

        labels = sub["isotopeLabel"].astype(str).tolist()
        dt = sub[sample_cols].to_numpy(dtype=float)

        try:
            tp = formula_counts(str(sub["formula"].iloc[0]))
        except Exception as e:
            raise ValueError(f"Formula parse failed at group starting row {start}: {e}")

        out = label_autodetect(labels)

        if len(out) == 0:
            corr_abs = dt
            denom = corr_abs.sum(axis=0, keepdims=True) + 1e-10
            corr_pct = corr_abs / denom

        elif len(out) == 1:
            trA = out[0]
            A_num = int(tp[trA - 1])
            ab_A = ABUNDANCE[trA - 1]
            im_A = float(impurity[trA - 1])
            counts = ab_getcounts(labels, TR_SYMBOLS[trA - 1], TR_SYMBOLS[trA - 1])
            fulldt, reduced_idx = ab_getfulldt(dt, counts, A_num, 0)
            corr_full_abs, corr_full_pct = isocorr_A(fulldt, A_num, ab_A, im_A)
            corr_abs = corr_full_abs[reduced_idx, :]
            corr_pct = corr_full_pct[reduced_idx, :]

        elif len(out) == 2:
            trA, trB = out[0], out[1]
            A_num = int(tp[trA - 1])
            B_num = int(tp[trB - 1])
            ab_A = ABUNDANCE[trA - 1]
            ab_B = ABUNDANCE[trB - 1]
            im_A = float(impurity[trA - 1])
            im_B = float(impurity[trB - 1])
            counts = ab_getcounts(labels, TR_SYMBOLS[trA - 1], TR_SYMBOLS[trB - 1])
            fulldt, reduced_idx = ab_getfulldt(dt, counts, A_num, B_num)
            corr_full_abs, corr_full_pct = isocorr_AB(fulldt, A_num, B_num, ab_A, ab_B, im_A, im_B)
            corr_abs = corr_full_abs[reduced_idx, :]
            corr_pct = corr_full_pct[reduced_idx, :]

        else:
            raise ValueError(f"More than 2 labeled terms detected in group starting row {start}")

        name = str(sub["compound"].iloc[0])
        formula = str(sub["formula"].iloc[0])
        corr_tic = corr_abs.sum(axis=0)

        meta_results.append(
            MetaResult(
                id=i + 1,
                name=name,
                formula=formula,
                corr_abs=corr_abs,
                corr_pct=corr_pct,
                corr_tic=corr_tic,
            )
        )

    cat_abs = np.vstack([m.corr_abs for m in meta_results]) if meta_results else np.zeros((0, len(sample_cols)))
    cat_pct = np.vstack([m.corr_pct for m in meta_results]) if meta_results else np.zeros((0, len(sample_cols)))

    df_corr_abs = df.copy()
    df_corr_pct = df.copy()
    # Ensure sample columns can hold floats (avoids dtype warnings and future errors).
    df_corr_abs = df_corr_abs.astype({c: float for c in sample_cols})
    df_corr_pct = df_corr_pct.astype({c: float for c in sample_cols})
    df_corr_abs.loc[:, sample_cols] = cat_abs.astype(float)
    df_corr_pct.loc[:, sample_cols] = cat_pct.astype(float)

    total_rows = []
    for m in meta_results:
        row = {"ID": m.id, "Name": m.name, "formula": m.formula}
        for col_name, v in zip(sample_cols, m.corr_tic.tolist()):
            row[col_name] = v
        total_rows.append(row)
    df_total = pd.DataFrame(total_rows, columns=["ID", "Name", "formula"] + sample_cols)

    return df, df_corr_pct, df_corr_abs, df_total


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.post("/autocorr")
def autocorr_route():
    if "file" not in request.files:
        flash("No file uploaded")
        return redirect(url_for("index"))

    f = request.files["file"]
    if not f.filename:
        flash("No file selected")
        return redirect(url_for("index"))

    try:
        impurity = [
            float(request.form.get("impurity_c", "0.01")),
            float(request.form.get("impurity_n", "0.01")),
            float(request.form.get("impurity_d", "0.01")),
            float(request.form.get("impurity_o", "0.01")),
        ]
    except ValueError:
        flash("Impurity values must be numeric")
        return redirect(url_for("index"))

    try:
        df = pd.read_csv(f)
        original, cor_pct, cor_abs, total = run_autocorr(df, impurity)

        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            original.to_excel(writer, index=False, sheet_name="original")
            cor_pct.to_excel(writer, index=False, sheet_name="cor_pct")
            cor_abs.to_excel(writer, index=False, sheet_name="cor_abs")
            total.to_excel(writer, index=False, sheet_name="total")
        out.seek(0)

        base = os.path.splitext(os.path.basename(f.filename))[0]
        download_name = f"{base}_cor.xlsx"
        return send_file(
            out,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            as_attachment=True,
            download_name=download_name,
        )

    except Exception as e:
        flash(f"Failed to process file: {e}")
        return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=True)
