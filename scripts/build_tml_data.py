#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import os
import re
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# ============================================================
# GitHub private-safe fetch (fallback), + prefer local cache
# ============================================================

def github_get(url: str, timeout: int = 60) -> requests.Response:
    """
    GET robust per GitHub:
    - Si és URL github.com/.../blob/... (HTML), la converteix a Contents API
      i demana el raw (funciona amb repos privats amb token).
    - Si ja és raw.githubusercontent.com, prova igualment amb token si n'hi ha.
    """
    token = (
        os.getenv("GITHUB_TOKEN")
        or os.getenv("GH_TOKEN")
        or os.getenv("GH_PAT")
        or ""
    )

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # https://github.com/{owner}/{repo}/blob/{ref}/{path}
    m = re.match(r"^https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.*)$", url)
    if m:
        owner, repo, ref, path = m.group(1), m.group(2), m.group(3), m.group(4)
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={ref}"

        headers_api = dict(headers)
        headers_api["Accept"] = "application/vnd.github.raw"

        r = requests.get(api_url, headers=headers_api, timeout=timeout)
        r.raise_for_status()
        return r

    r = requests.get(url, headers=headers if headers else None, timeout=timeout)
    r.raise_for_status()
    return r


def fetch_csv(url: str, timeout: int = 60, local_path: Path | None = None) -> pd.DataFrame:
    """
    1) Si local_path existeix -> llegeix local (zero network, ideal per Actions i dev)
    2) Si no -> baixa via github_get(url) (suporta repos privats amb token)
    """
    if local_path is not None and local_path.exists():
        return pd.read_csv(local_path)

    r = github_get(url, timeout=timeout)
    return pd.read_csv(io.StringIO(r.text))


# ============================================================
# Build helpers
# ============================================================

def year_range(a: int, b: int):
    return list(range(a, b + 1))


def yyyymmdd_to_date(x):
    try:
        x = str(int(x)).strip()
        return f"{x[:4]}-{x[4:6]}-{x[6:8]}"
    except Exception:
        return None


def norm_surface(s):
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    if "hard" in s:
        return "hard"
    if "clay" in s:
        return "clay"
    if "grass" in s:
        return "grass"
    if "carpet" in s:
        return "indoor-hard"
    return ""


def map_level(lvl):
    m = {"G": "GS", "A": "ATP", "D": "Davis", "F": "Finals"}
    return m.get(str(lvl).strip(), str(lvl).strip())


def build_from_tml(df: pd.DataFrame, seed: int = 2026, do_flip: bool = True):
    cols_req = [
        "tourney_id","tourney_name","surface","tourney_level","tourney_date","match_num",
        "winner_id","winner_ioc","loser_id","loser_ioc","best_of","round","minutes",
        "w_ace","w_df","w_svpt","w_1stIn","w_1stWon","w_2ndWon","w_SvGms","w_bpSaved","w_bpFaced",
        "l_ace","l_df","l_svpt","l_1stIn","l_1stWon","l_2ndWon","l_SvGms","l_bpSaved","l_bpFaced"
    ]
    for c in cols_req:
        if c not in df.columns:
            df[c] = np.nan

    df["date"] = df["tourney_date"].apply(yyyymmdd_to_date)
    df["surface_norm"] = df["surface"].apply(norm_surface)
    df["level_norm"] = df["tourney_level"].apply(map_level)
    df["indoor_flag"] = (df["surface_norm"] == "indoor-hard").astype(int)
    df["best_of_5"] = (pd.to_numeric(df["best_of"], errors="coerce") == 5).astype(int)
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce")

    df["match_id"] = df.apply(
        lambda r: f"{r.get('tourney_id','')}_{r.get('match_num','')}_{r.get('date','')}",
        axis=1
    )

    rng = np.random.default_rng(seed)
    flip = rng.integers(0, 2, size=len(df)).astype(bool) if do_flip else np.zeros(len(df), dtype=bool)

    player_a_id = df["winner_id"].astype(str).where(~flip, df["loser_id"].astype(str))
    player_b_id = df["loser_id"].astype(str).where(~flip, df["winner_id"].astype(str))
    player_a_country = df["winner_ioc"].astype(str).where(~flip, df["loser_ioc"].astype(str))
    player_b_country = df["loser_ioc"].astype(str).where(~flip, df["winner_ioc"].astype(str))

    matches = pd.DataFrame({
        "match_id": df["match_id"],
        "date": df["date"],
        "tournament": df["tourney_name"].astype(str),
        "city": df["tourney_name"].astype(str),
        "country": "",
        "level": df["level_norm"],
        "round": df["round"].astype(str),
        "best_of_5": df["best_of_5"],
        "surface": df["surface_norm"],
        "indoor": df["indoor_flag"],
        "player_a_id": player_a_id,
        "player_b_id": player_b_id,
        "winner_id": df["winner_id"].astype(str),
        "duration_minutes": df["minutes"],
        "player_a_country": player_a_country,
        "player_b_country": player_b_country,
    })
    matches = matches.dropna(subset=["date"])
    matches = matches[matches["surface"].isin(["hard", "clay", "grass", "indoor-hard"])]

    w_bp_conv = (pd.to_numeric(df["l_bpFaced"], errors="coerce") - pd.to_numeric(df["l_bpSaved"], errors="coerce")).clip(lower=0)
    l_bp_conv = (pd.to_numeric(df["w_bpFaced"], errors="coerce") - pd.to_numeric(df["w_bpSaved"], errors="coerce")).clip(lower=0)

    pts_w = pd.DataFrame({
        "match_id": df["match_id"],
        "player_id": df["winner_id"].astype(str),
        "aces": pd.to_numeric(df["w_ace"], errors="coerce"),
        "double_faults": pd.to_numeric(df["w_df"], errors="coerce"),
        "first_sv_in": pd.to_numeric(df["w_1stIn"], errors="coerce"),
        "first_sv_pts_won": pd.to_numeric(df["w_1stWon"], errors="coerce"),
        "second_sv_pts_won": pd.to_numeric(df["w_2ndWon"], errors="coerce"),
        "bp_faced": pd.to_numeric(df["w_bpFaced"], errors="coerce"),
        "bp_saved": pd.to_numeric(df["w_bpSaved"], errors="coerce"),
        "bp_opp": pd.to_numeric(df["l_bpFaced"], errors="coerce"),
        "bp_conv": w_bp_conv,
        "service_games": pd.to_numeric(df["w_SvGms"], errors="coerce"),
        "return_games": pd.to_numeric(df["l_SvGms"], errors="coerce"),
        "hold_games_won": (pd.to_numeric(df["w_SvGms"], errors="coerce") - l_bp_conv).clip(lower=0),
        "break_games_won": w_bp_conv,
        "tb_played": np.nan,
        "tb_won": np.nan
    })

    pts_l = pd.DataFrame({
        "match_id": df["match_id"],
        "player_id": df["loser_id"].astype(str),
        "aces": pd.to_numeric(df["l_ace"], errors="coerce"),
        "double_faults": pd.to_numeric(df["l_df"], errors="coerce"),
        "first_sv_in": pd.to_numeric(df["l_1stIn"], errors="coerce"),
        "first_sv_pts_won": pd.to_numeric(df["l_1stWon"], errors="coerce"),
        "second_sv_pts_won": pd.to_numeric(df["l_2ndWon"], errors="coerce"),
        "bp_faced": pd.to_numeric(df["l_bpFaced"], errors="coerce"),
        "bp_saved": pd.to_numeric(df["l_bpSaved"], errors="coerce"),
        "bp_opp": pd.to_numeric(df["w_bpFaced"], errors="coerce"),
        "bp_conv": l_bp_conv,
        "service_games": pd.to_numeric(df["l_SvGms"], errors="coerce"),
        "return_games": pd.to_numeric(df["w_SvGms"], errors="coerce"),
        "hold_games_won": (pd.to_numeric(df["l_SvGms"], errors="coerce") - w_bp_conv).clip(lower=0),
        "break_games_won": l_bp_conv,
        "tb_played": np.nan,
        "tb_won": np.nan
    })

    points = pd.concat([pts_w, pts_l], ignore_index=True)
    return matches, points


# ============================================================
# Main
# ============================================================

def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Aquesta és la carpeta local del repo (preferida)
    local_cache_dir = data_dir / "tml_cache"
    local_cache_dir.mkdir(parents=True, exist_ok=True)

    start_year = 2015
    remote_end_year = 2026

    # Opció A (recomanada): apunta a URL "blob" (el helper la converteix a API raw)
    # IMPORTANT: posa el ref correcte (main o el que sigui)
    OWNER = "adriaparcerisas"
    REPO  = "Tennis-Bets"
    REF   = "main"
    RAW_BASE = f"https://github.com/{OWNER}/{REPO}/blob/{REF}/data/tml_cache"

    all_df = []

    # 1) Years
    for y in range(start_year, remote_end_year + 1):
        url = f"{RAW_BASE}/{y}.csv"
        lp = local_cache_dir / f"{y}.csv"
        try:
            df = fetch_csv(url, timeout=60, local_path=lp)
            df["__year"] = y
            all_df.append(df)
            print(f"[OK] {y}: {len(df)} rows ({'local' if lp.exists() else 'remote'})")
        except Exception as e:
            print(f"[WARN] {y}: {e}")

    # 2) Ongoing
    try:
        url_ongo = f"{RAW_BASE}/ongoing_tourneys.csv"
        lp_ongo  = local_cache_dir / "ongoing_tourneys.csv"
        df_ongo = fetch_csv(url_ongo, timeout=60, local_path=lp_ongo)
        df_ongo["__year"] = remote_end_year
        all_df.append(df_ongo)
        print(f"[OK] ongoing_tourneys: {len(df_ongo)} rows ({'local' if lp_ongo.exists() else 'remote'})")
    except Exception as e:
        print(f"[WARN] ongoing_tourneys: {e}")

    # 3) Local 2026 override / add (tal com ho tens)
    local_2026_path = data_dir / "2026_nou.csv"
    if local_2026_path.exists():
        df_2026 = pd.read_csv(local_2026_path)
        df_2026["__year"] = 2026
        all_df.append(df_2026)
        print(f"[OK] local 2026_nou.csv: {len(df_2026)} rows")
    else:
        print("[WARN] data/2026_nou.csv not found. 2026 will NOT be included.")

    if not all_df:
        raise RuntimeError("No data loaded (remote or local).")

    raw_all = pd.concat(all_df, ignore_index=True)

    # Mantinc exactament el teu comportament actual:
    matches, points = build_from_tml(raw_all, seed=2026, do_flip=True)

    matches_path = data_dir / "matches.csv"
    points_path  = data_dir / "points_sets_games.csv"

    matches.to_csv(matches_path, index=False)
    points.to_csv(points_path, index=False)

    print(f"[SAVED] {matches_path} rows={len(matches)}")
    print(f"[SAVED] {points_path} rows={len(points)}")


if __name__ == "__main__":
    main()
