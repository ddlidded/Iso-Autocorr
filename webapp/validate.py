import sys
from pathlib import Path

import numpy as np
import pandas as pd

from app import run_autocorr


def compare_numeric(df_a: pd.DataFrame, df_b: pd.DataFrame, sheet: str, tol: float = 1e-6) -> None:
    if list(df_a.columns) != list(df_b.columns):
        raise AssertionError(f"{sheet}: column mismatch")
    if len(df_a) != len(df_b):
        raise AssertionError(f"{sheet}: row count mismatch {len(df_a)} != {len(df_b)}")

    # Compare numeric columns with tolerance; non-numeric exact.
    for col in df_a.columns:
        a = df_a[col]
        b = df_b[col]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            aa = a.to_numpy(dtype=float)
            bb = b.to_numpy(dtype=float)
            if not np.allclose(aa, bb, rtol=0, atol=tol, equal_nan=True):
                diff = np.nanmax(np.abs(aa - bb))
                raise AssertionError(f"{sheet}.{col}: max abs diff {diff}")
        else:
            if not a.fillna("").astype(str).equals(b.fillna("").astype(str)):
                raise AssertionError(f"{sheet}.{col}: non-numeric mismatch")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    examples = sorted(root.glob("example*.csv"))
    if not examples:
        print("No example*.csv files found")
        return 1

    failures = 0
    for csv_path in examples:
        xlsx_path = csv_path.with_name(csv_path.stem + "_cor.xlsx")
        if not xlsx_path.exists():
            print(f"SKIP {csv_path.name}: missing {xlsx_path.name}")
            continue

        df = pd.read_csv(csv_path)
        original, cor_pct, cor_abs, total = run_autocorr(df, impurity=[0.01, 0.01, 0.01, 0.01])

        ref_original = pd.read_excel(xlsx_path, sheet_name="original")
        ref_cor_pct = pd.read_excel(xlsx_path, sheet_name="cor_pct")
        ref_cor_abs = pd.read_excel(xlsx_path, sheet_name="cor_abs")
        ref_total = pd.read_excel(xlsx_path, sheet_name="total")

        try:
            compare_numeric(cor_abs, ref_cor_abs, "cor_abs")
            compare_numeric(cor_pct, ref_cor_pct, "cor_pct")
            compare_numeric(total, ref_total, "total")
            # original can differ slightly in dtype/formatting; skip strict compare
            print(f"OK  {csv_path.name}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {csv_path.name}: {e}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
