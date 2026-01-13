# scripts/patch_matches_csv.py
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd


CANDIDATES = [
    Path("matches.csv"),
    Path("data/matches.csv"),
    Path("data/processed/matches.csv"),
]


def _find_matches_csv() -> Path:
    env = os.getenv("MATCHES_CSV")
    if env:
        p = Path(env)
        if p.exists():
            return p
        raise FileNotFoundError(f"MATCHES_CSV points to missing file: {p}")

    for p in CANDIDATES:
        if p.exists():
            return p

    raise FileNotFoundError(
        "Could not find matches.csv. Set MATCHES_CSV env var or place it in one of: "
        + ", ".join(str(p) for p in CANDIDATES)
    )


def _parse_base_date(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()

    # Detect YYYYMMDD (8 digits) vs already formatted dates
    is_yyyymmdd = s.str.fullmatch(r"\d{8}", na=False)

    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if is_yyyymmdd.any():
        out.loc[is_yyyymmdd] = pd.to_datetime(s.loc[is_yyyymmdd], format="%Y%m%d", errors="coerce")
    if (~is_yyyymmdd).any():
        out.loc[~is_yyyymmdd] = pd.to_datetime(s.loc[~is_yyyymmdd], errors="coerce")

    return out


def main() -> None:
    path = _find_matches_csv()
    df = pd.read_csv(path)

    # Base column preference: tourney_date -> date
    base_col = "tourney_date" if "tourney_date" in df.columns else ("date" if "date" in df.columns else None)
    if base_col is None:
        raise ValueError("matches.csv must contain either 'tourney_date' or 'date' column")

    base_dt = _parse_base_date(df[base_col])

    # Create/patch match_date
    if "match_date" not in df.columns:
        # Insert after base_col if possible
        insert_at = list(df.columns).index(base_col) + 1
        df.insert(insert_at, "match_date", "")
    else:
        df["match_date"] = df["match_date"].fillna("").astype(str)

    # Fill only missing/blank values to avoid overwriting if you later compute offsets by round
    missing = df["match_date"].astype(str).str.strip().eq("") | df["match_date"].isna()
    df.loc[missing, "match_date"] = base_dt.dt.strftime("%Y-%m-%d")

    # Optional: if you want to guarantee no NaN strings
    df["match_date"] = df["match_date"].fillna("")

    df.to_csv(path, index=False)
    print(f"Patched {path}: match_date filled for {int(missing.sum())} rows using '{base_col}'")


if __name__ == "__main__":
    main()
