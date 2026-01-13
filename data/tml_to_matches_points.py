#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data/tml_to_matches_points.py

Ingest TML-Database (GitHub/raw endpoints) -> build data/matches.csv and data/points_sets_games.csv

Usage:
  python tml_to_matches_points.py --data_dir data --years 2018-2026 --include_ongoing 1 --random_flip 1 --use_cache 1

Notes:
- Si el repo és privat, raw.githubusercontent.com donarà 404.
  Aquest script fa fallback automàtic a GitHub Contents API si hi ha token:
    env: GITHUB_TOKEN / GH_TOKEN / GH_PAT
"""

import argparse
import io
import sys
import os
import re
import datetime as dt
import pandas as pd
import numpy as np
import requests

#RAW_BASE = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master"
RAW_BASE = "https://raw.githubusercontent.com/adriaparcerisas/Tennis-Bets/main/data/tml_cache"


# -------------------------
# GitHub private-safe fetch
# -------------------------

def _get_github_token() -> str:
    return (
        os.getenv("GITHUB_TOKEN")
        or os.getenv("GH_TOKEN")
        or os.getenv("GH_PAT")
        or ""
    )

def _github_raw_via_api(owner: str, repo: str, ref: str, path: str, timeout: int = 60) -> str:
    """
    Baixa fitxer via GitHub Contents API amb Accept: application/vnd.github.raw.
    Funciona per repos privats si hi ha token.
    """
    token = _get_github_token()
    headers = {"Accept": "application/vnd.github.raw"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={ref}"
    r = requests.get(api_url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text

def _fetch_text(url: str, timeout: int = 60) -> str:
    """
    - Si url és raw.githubusercontent.com/... -> intenta primer Contents API (si privat)
    - Si falla -> prova raw directe.
    """
    m = re.match(r"^https?://raw\.githubusercontent\.com/([^/]+)/([^/]+)/([^/]+)/(.*)$", url)
    if m:
        owner, repo, ref, path = m.group(1), m.group(2), m.group(3), m.group(4)
        try:
            return _github_raw_via_api(owner, repo, ref, path, timeout=timeout)
        except Exception:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.text

    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text

def fetch_csv(url: str, timeout: int = 60) -> pd.DataFrame:
    text = _fetch_text(url, timeout=timeout)
    return pd.read_csv(io.StringIO(text))


# -------------------------
# CLI helpers
# -------------------------

def year_range(s: str):
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(s)]


# -------------------------
# Build helpers (same logic)
# -------------------------

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


# -------------------------
# Main
# -------------------------

def main(args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--years", type=str, default="2015-2026", help="e.g. 2015-2026 or single year 2026")
    ap.add_argument("--include_ongoing", type=int, default=1, help="1 to include ongoing_tourneys.csv")
    ap.add_argument("--random_flip", type=int, default=1, help="1 to randomize orientation A/B")
    ap.add_argument("--seed", type=int, default=2026)

    # Robustesa
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--use_cache", type=int, default=1, help="1 to cache remote CSVs to data/tml_cache")
    cfg = ap.parse_args(args)

    years = year_range(cfg.years)
    all_df = []

    data_dir = cfg.data_dir
    os.makedirs(data_dir, exist_ok=True)

    # local cache dir dins data_dir
    cache_dir = os.path.join(data_dir, "tml_cache")
    if int(cfg.use_cache) == 1:
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(name: str) -> str:
        return os.path.join(cache_dir, name)

    # Info token (no l’imprimim)
    has_token = bool(_get_github_token())
    print(f"[github] token_present={has_token} (needed only if repo is private and cache miss)")

    # Years
    for y in years:
        url = f"{RAW_BASE}/{y}.csv"
        try:
            # cache first
            if int(cfg.use_cache) == 1 and os.path.exists(_cache_path(f"{y}.csv")):
                df = pd.read_csv(_cache_path(f"{y}.csv"))
                print(f"[CACHE] {y}: {len(df)} rows")
            else:
                df = fetch_csv(url, timeout=int(cfg.timeout))
                if int(cfg.use_cache) == 1:
                    df.to_csv(_cache_path(f"{y}.csv"), index=False)
                print(f"[OK] {y}: {len(df)} rows")

            df["__year"] = y
            all_df.append(df)

        except Exception as e:
            print(f"[WARN] {y}: {type(e).__name__}: {e}")

    # ongoing
    if int(cfg.include_ongoing) == 1:
        url = f"{RAW_BASE}/ongoing_tourneys.csv"
        try:
            if int(cfg.use_cache) == 1 and os.path.exists(_cache_path("ongoing_tourneys.csv")):
                df_ongo = pd.read_csv(_cache_path("ongoing_tourneys.csv"))
                print(f"[CACHE] ongoing_tourneys: {len(df_ongo)} rows")
            else:
                df_ongo = fetch_csv(url, timeout=int(cfg.timeout))
                if int(cfg.use_cache) == 1:
                    df_ongo.to_csv(_cache_path("ongoing_tourneys.csv"), index=False)
                print(f"[OK] ongoing_tourneys: {len(df_ongo)} rows")

            df_ongo["__year"] = int(dt.date.today().year)
            all_df.append(df_ongo)

        except Exception as e:
            print(f"[WARN] ongoing_tourneys: {type(e).__name__}: {e}")

    if not all_df:
        print("No data fetched. Exiting.")
        sys.exit(1)

    raw_all = pd.concat(all_df, ignore_index=True)

    matches, points = build_from_tml(raw_all, seed=cfg.seed, do_flip=bool(cfg.random_flip))

    # Upsert matches
    match_path = os.path.join(cfg.data_dir, "matches.csv")
    if os.path.exists(match_path):
        old = pd.read_csv(match_path, dtype=str)
        merged = pd.concat([old, matches.astype(str)]).drop_duplicates(subset=["match_id"], keep="last")
    else:
        merged = matches.astype(str)
    merged.to_csv(match_path, index=False)
    print(f"[SAVED] {match_path} rows={len(merged)}")

    # Upsert points
    pts_path = os.path.join(cfg.data_dir, "points_sets_games.csv")
    if os.path.exists(pts_path):
        oldp = pd.read_csv(pts_path, dtype=str)
        mergedp = pd.concat([oldp, points.astype(str)]).drop_duplicates(subset=["match_id","player_id"], keep="last")
    else:
        mergedp = points.astype(str)
    mergedp.to_csv(pts_path, index=False)
    print(f"[SAVED] {pts_path} rows={len(mergedp)}")


if __name__ == "__main__":
    main()
