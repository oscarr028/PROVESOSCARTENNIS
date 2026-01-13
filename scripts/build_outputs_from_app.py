#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/build_outputs_from_app.py

Replica offline del pipeline de la Tab "Refresh & Train" de l'app:
1) Descarrega TML (anys + ongoing opcional)
2) Construeix:
   - data/matches.csv            (build_matches_from_tml)
   - data/points_sets_games.csv  (build_points_from_tml)
3) Aplica cutoffs d'entrenament (tournament/match/both) com a l'app
4) Feature building (tennis_model_pipeline_v2):
   - outputs/features_player_pre.csv
   - outputs/dataset_match_level_raw.csv
   - outputs/dataset_match_level.csv  (amb meta features v2: NOMÉS home flags)
   - outputs/model_columns.txt        (EXACTAMENT les columnes entrenades)
5) Entrena model + calibració isotònica:
   - outputs/model_logistic.pkl o outputs/model_lightgbm.pkl
   - outputs/scaler.pkl (si Logistic)
   - outputs/calibrator_isotonic.pkl + outputs/isotonic.pkl
   - outputs/preds_test.csv
   - outputs/train_metrics.json
6) Prediccions upcoming (si n'hi ha):
   - outputs/predictions_upcoming.csv
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

# Optional LightGBM
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False


# =========================
# Paths / Repo detection
# =========================

def get_repo_dir() -> Path:
    # repo/
    #   scripts/build_outputs_from_app.py
    #   tennis_model_pipeline_v2.py
    #   data/
    return Path(__file__).resolve().parents[1]

REPO_DIR = get_repo_dir()
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

DATA_DIR  = REPO_DIR / "data"
OUT_DIR   = DATA_DIR / "outputs"
CACHE_DIR = DATA_DIR / "tml_cache"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# IMPORTANT:
# - raw.githubusercontent.com dóna 404 si el repo és privat
# - amb els helpers de sota farem fallback automàtic a GitHub API (amb token) si cal
RAW_BASE = "https://raw.githubusercontent.com/adriaparcerisas/Tennis-Bets/main/data/tml_cache"


# =========================
# Pipeline import
# =========================

try:
    from tennis_model_pipeline_v2 import (
        compute_pre_match_features_v2,
        make_match_features,
    )
except Exception as e:
    raise RuntimeError(
        "No puc importar tennis_model_pipeline_v2. "
        "Assegura't que tennis_model_pipeline_v2.py és al root del repo o al PYTHONPATH.\n"
        f"Detall: {type(e).__name__}: {e}"
    )


# =========================
# GitHub fetch (private-safe)
# =========================

def _get_github_token() -> str:
    return (
        os.getenv("GITHUB_TOKEN")
        or os.getenv("GH_TOKEN")
        or os.getenv("GH_PAT")
        or ""
    )

def _github_raw_via_api(owner: str, repo: str, ref: str, path: str, timeout: int = 30) -> str:
    """
    Baixa un fitxer via GitHub Contents API amb Accept: application/vnd.github.raw.
    Funciona per repos privats si hi ha token, i per repos públics (subjecte a rate limit).
    """
    token = _get_github_token()
    headers = {"Accept": "application/vnd.github.raw"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={ref}"
    r = requests.get(api_url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text

def _fetch_text(url: str, timeout: int = 30) -> str:
    """
    - Si URL és raw.githubusercontent.com/... -> intenta GitHub API (més robust en repos privats)
    - Si no, fa requests.get normal.
    """
    m = re.match(r"^https?://raw\.githubusercontent\.com/([^/]+)/([^/]+)/([^/]+)/(.*)$", url)
    if m:
        owner, repo, ref, path = m.group(1), m.group(2), m.group(3), m.group(4)
        # Prova via API (suporta privat amb token)
        try:
            return _github_raw_via_api(owner, repo, ref, path, timeout=timeout)
        except Exception:
            # Fallback a raw directe (funciona si el repo és públic)
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.text

    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


# =========================
# Helpers
# =========================

def _norm_surface(s: str) -> str:
    s = str(s or "").strip().lower()
    if "carpet" in s: return "indoor-hard"
    if "indoor" in s: return "indoor-hard"
    if "hard" in s:   return "hard"
    if "clay" in s:   return "clay"
    if "grass" in s:  return "grass"
    return "hard"


def _parse_yyyymmdd_series(s: pd.Series) -> pd.Series:
    ss = s.astype(str).str.strip()
    ss = ss.str.replace(r"\.0$", "", regex=True)
    digits = ss.str.replace(r"\D", "", regex=True)
    is8 = digits.str.len().eq(8)

    dt8 = pd.to_datetime(digits.where(is8), format="%Y%m%d", errors="coerce")
    dtany = pd.to_datetime(ss.where(~is8), errors="coerce")
    out = dt8.fillna(dtany)

    try:
        out = out.dt.tz_localize(None)
    except Exception:
        pass
    return out


def fetch_year_csv(year: int, timeout: int = 30, use_cache: bool = True) -> pd.DataFrame:
    """
    Local-first:
      - si data/tml_cache/{year}.csv existeix i use_cache=True -> llegeix local
      - si no -> baixa (private-safe) i opcionalment guarda a cache
    """
    cache_path = CACHE_DIR / f"{year}.csv"
    if use_cache and cache_path.exists():
        df = pd.read_csv(cache_path, dtype={"tourney_date": "string"})
        df["__src_year"] = year
        return df

    url = f"{RAW_BASE}/{year}.csv"
    text = _fetch_text(url, timeout=timeout)
    df = pd.read_csv(io.StringIO(text), dtype={"tourney_date": "string"})
    df["__src_year"] = year

    if use_cache:
        df.to_csv(cache_path, index=False)
    return df


def fetch_ongoing_csv(timeout: int = 30, use_cache: bool = True) -> pd.DataFrame:
    """
    Local-first:
      - si data/tml_cache/ongoing_tourneys.csv existeix i use_cache=True -> llegeix local
      - si no -> baixa (private-safe) i opcionalment guarda a cache
    """
    cache_path = CACHE_DIR / "ongoing_tourneys.csv"
    url = f"{RAW_BASE}/ongoing_tourneys.csv"

    if use_cache and cache_path.exists():
        df = pd.read_csv(cache_path, dtype={"tourney_date": "string"})
        df["__src_year"] = int(pd.Timestamp.today().year)
        return df

    text = _fetch_text(url, timeout=timeout)
    df = pd.read_csv(io.StringIO(text), dtype={"tourney_date": "string"})
    df["__src_year"] = int(pd.Timestamp.today().year)

    if use_cache:
        df.to_csv(cache_path, index=False)
    return df


def build_matches_from_tml(df_all: pd.DataFrame) -> pd.DataFrame:
    df = df_all.copy()
    needed = ["tourney_id","tourney_name","surface","tourney_level","tourney_date",
              "match_num","winner_id","loser_id","best_of","round"]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan

    df["tourney_date"] = _parse_yyyymmdd_series(df["tourney_date"])
    df["surface_norm"] = df["surface"].apply(_norm_surface)
    df["best_of_5"] = (pd.to_numeric(df["best_of"], errors="coerce") == 5).astype(int)

    # PSEUDO MATCH DATE (1 dia per ronda dins de cada torneig)
    ROUND_STAGE = {
        "Q1": 0, "Q2": 1, "Q3": 2,
        "RR": 3,
        "R128": 4, "R64": 5, "R32": 6, "R16": 7,
        "QF": 8, "SF": 9,
        "BR": 10,
        "F": 11,
        "1R": 5, "2R": 6, "3R": 7, "4R": 8,
        "R1": 5, "R2": 6, "R3": 7, "R4": 8,
    }

    r = df["round"].astype(str).str.strip().str.upper()
    stage = r.map(ROUND_STAGE)
    min_stage = stage.groupby(df["tourney_id"].astype(str)).transform("min")
    offset_days = (stage - min_stage).fillna(0).clip(lower=0).astype(int)
    df["match_date"] = df["tourney_date"] + pd.to_timedelta(offset_days, unit="D")

    def sort_pair(wid, lid):
        try:
            a, b = sorted([str(wid), str(lid)])
        except Exception:
            a, b = str(wid), str(lid)
        return a, b

    a_ids, b_ids = [], []
    for wid, lid in zip(df["winner_id"], df["loser_id"]):
        a, b = sort_pair(wid, lid)
        a_ids.append(a); b_ids.append(b)

    df["player_a_id"] = a_ids
    df["player_b_id"] = b_ids

    df["match_id"] = (
        df["tourney_id"].astype(str) + "_" +
        df["tourney_date"].dt.strftime("%Y%m%d").fillna("00000000") + "_" +
        df["match_num"].astype(str)
    )

    matches = pd.DataFrame({
        "match_id": df["match_id"].astype(str),
        "date": df["tourney_date"].dt.strftime("%Y-%m-%d"),
        "match_date": df["match_date"].dt.strftime("%Y-%m-%d"),
        "tournament": df["tourney_name"].astype(str),
        "city": df["tourney_name"].astype(str),
        "country": np.nan,
        "level": df["tourney_level"].astype(str),
        "round": df["round"].astype(str),
        "best_of_5": df["best_of_5"].astype(int),
        "surface": df["surface_norm"].astype(str),
        "indoor": (df["surface_norm"] == "indoor-hard").astype(int),
        "player_a_id": df["player_a_id"].astype(str),
        "player_b_id": df["player_b_id"].astype(str),
        # IMPORTANT: NO forcem a str aquí; mantenim NaNs si n'hi ha
        "winner_id": df["winner_id"],
        "duration_minutes": pd.to_numeric(df.get("minutes", np.nan), errors="coerce"),
    })
    return matches


def build_points_from_tml(df_all: pd.DataFrame) -> pd.DataFrame:
    df = df_all.copy()
    df["tourney_date"] = _parse_yyyymmdd_series(df["tourney_date"])
    df["match_id"] = (
        df["tourney_id"].astype(str) + "_" +
        df["tourney_date"].dt.strftime("%Y%m%d").fillna("00000000") + "_" +
        df["match_num"].astype(str)
    )

    rows = []
    for _, r in df.iterrows():
        mid = str(r["match_id"])
        rows.append(dict(
            match_id=mid, player_id=str(r.get("winner_id","")),
            service_games=pd.to_numeric(r.get("w_SvGms", np.nan), errors="coerce"),
            return_games=pd.to_numeric(r.get("l_SvGms", np.nan), errors="coerce"),
            hold_games_won=np.nan, break_games_won=np.nan,
            aces=pd.to_numeric(r.get("w_ace", np.nan), errors="coerce"),
            double_faults=pd.to_numeric(r.get("w_df", np.nan), errors="coerce"),
            first_sv_in=pd.to_numeric(r.get("w_1stIn", np.nan), errors="coerce"),
            first_sv_pts_won=pd.to_numeric(r.get("w_1stWon", np.nan), errors="coerce"),
            second_sv_pts_won=pd.to_numeric(r.get("w_2ndWon", np.nan), errors="coerce"),
            second_sv_attempts=np.nan,
        ))
        rows.append(dict(
            match_id=mid, player_id=str(r.get("loser_id","")),
            service_games=pd.to_numeric(r.get("l_SvGms", np.nan), errors="coerce"),
            return_games=pd.to_numeric(r.get("w_SvGms", np.nan), errors="coerce"),
            hold_games_won=np.nan, break_games_won=np.nan,
            aces=pd.to_numeric(r.get("l_ace", np.nan), errors="coerce"),
            double_faults=pd.to_numeric(r.get("l_df", np.nan), errors="coerce"),
            first_sv_in=pd.to_numeric(r.get("l_1stIn", np.nan), errors="coerce"),
            first_sv_pts_won=pd.to_numeric(r.get("l_1stWon", np.nan), errors="coerce"),
            second_sv_pts_won=pd.to_numeric(r.get("l_2ndWon", np.nan), errors="coerce"),
            second_sv_attempts=np.nan,
        ))
    return pd.DataFrame(rows)


def _parse_match_dates(s: pd.Series) -> pd.Series:
    ss = s.astype(str).str.strip()
    ss = ss.str.replace(r"\.0$", "", regex=True)
    ss_norm = ss.str.replace("/", "-", regex=False)

    digits = ss_norm.str.replace(r"\D", "", regex=True)
    mask_8 = digits.str.len().eq(8)

    dt_8   = pd.to_datetime(digits.where(mask_8), format="%Y%m%d", errors="coerce")
    dt_any = pd.to_datetime(ss_norm.where(~mask_8), errors="coerce")
    return dt_8.fillna(dt_any)


# =========================
# Domestic enrich (meta locals a data/meta/*)
# =========================

META_DIR = DATA_DIR / "meta"

def _safe_csv(path: Path, **kw) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kw)
    except Exception:
        return pd.DataFrame()

def load_players_meta_domestic() -> pd.DataFrame:
    fp = META_DIR / "players_meta.csv"
    df = _safe_csv(fp)
    if "player_id" in df.columns:
        df["player_id"] = df["player_id"].astype(str)
    if "handedness" in df.columns:
        df["handedness"] = df["handedness"].astype(str).str.upper().str[0].where(lambda s: s.isin(["L","R"]), "")
    if "nationality_iso" in df.columns:
        df["nationality_iso"] = df["nationality_iso"].astype(str).str.upper().str[:2]
    return df[["player_id","nationality_iso","handedness"]].drop_duplicates() if len(df) else df

def load_tournaments_meta_domestic() -> pd.DataFrame:
    fp = META_DIR / "tournaments_meta.csv"
    df = _safe_csv(fp)
    if "surface_norm" in df.columns:
        df["surface_norm"] = df["surface_norm"].astype(str).str.lower()
    if "country_iso" in df.columns:
        df["country_iso"] = df["country_iso"].astype(str).str.upper().str[:2]
    if "tournament_key" in df.columns:
        df["tournament_key"] = df["tournament_key"].astype(str).str.strip().str.lower()
    return df[["tournament_key","tournament_display","country_iso","surface_norm"]].drop_duplicates() if len(df) else df

def _norm_tournament_key(x: str) -> str:
    s = str(x).strip().lower()
    s = s.replace("(", " ").replace(")", " ").replace("/", " ").replace("-", " ").replace("  "," ")
    s = s.replace(" masters 1000", " masters").replace(" atp ", " ").replace(" wta ", " ")
    s = "_".join(s.split())
    s = s.replace("paris", "paris_masters")
    s = s.replace("basel", "basel_open")
    s = s.replace("roland_garros", "french_open")
    return f"atp_{s}" if "wta" not in s and not s.startswith("atp_") else s

def enrich_matches_domestic(matches_df: pd.DataFrame) -> pd.DataFrame:
    df = matches_df.copy()
    for c in ["player_a_id","player_b_id","tournament","surface"]:
        if c not in df.columns:
            df[c] = "" if c != "surface" else "hard"
    df["player_a_id"] = df["player_a_id"].astype(str)
    df["player_b_id"] = df["player_b_id"].astype(str)

    P = load_players_meta_domestic()
    T = load_tournaments_meta_domestic()

    # players
    if len(P):
        P = P.rename(columns={"player_id":"_pid"})
        df = df.merge(P.rename(columns={"_pid":"player_a_id"}), on="player_a_id", how="left")
        df = df.rename(columns={"nationality_iso":"A_nation","handedness":"A_is_left"})
        df["A_is_left"] = df["A_is_left"].fillna("").astype(str).str.upper().str[0]
        df["A_is_left"] = df["A_is_left"].where(df["A_is_left"].isin(["L","R"]), "")

        df = df.merge(P.rename(columns={"_pid":"player_b_id"}), on="player_b_id", how="left")
        df = df.rename(columns={"nationality_iso":"B_nation","handedness":"B_is_left"})
        df["B_is_left"] = df["B_is_left"].fillna("").astype(str).str.upper().str[0]
        df["B_is_left"] = df["B_is_left"].where(df["B_is_left"].isin(["L","R"]), "")
    else:
        for c in ["A_nation","B_nation","A_is_left","B_is_left"]:
            if c not in df.columns:
                df[c] = ""

    # tournaments
    df["tournament_key"] = df.get("tournament","").apply(_norm_tournament_key)
    if len(T):
        df = df.merge(T, on="tournament_key", how="left")
        df["tournament_country_iso"] = df["country_iso"].fillna("")
        df["surface"] = np.where(
            df["surface"].notna() & (df["surface"] != ""),
            df["surface"],
            df["surface_norm"].fillna(df.get("surface","hard"))
        )
    else:
        df["tournament_country_iso"] = ""

    df["A_nation"] = df["A_nation"].fillna("").astype(str).str.upper().str[:2]
    df["B_nation"] = df["B_nation"].fillna("").astype(str).str.upper().str[:2]
    df["tournament_country_iso"] = df["tournament_country_iso"].fillna("").astype(str).str.upper().str[:2]

    df["is_home_A"] = (df["A_nation"] != "") & (df["A_nation"] == df["tournament_country_iso"])
    df["is_home_B"] = (df["B_nation"] != "") & (df["B_nation"] == df["tournament_country_iso"])
    return df


# =========================
# Meta v2 (data/players_meta.csv i data/tournaments_meta.csv)
# NOMÉS per calcular "home"
# =========================

def _coerce_country_id(x):
    if pd.isna(x): return "??"
    v = str(x).strip()
    if len(v) == 2: return v.upper()
    MAP = {
        "spain":"ES","espanya":"ES","argentina":"AR","italy":"IT","italia":"IT",
        "united states":"US","usa":"US","france":"FR","germany":"DE","australia":"AU",
        "england":"GB","united kingdom":"GB","canada":"CA","romania":"RO","serbia":"RS",
        "sweden":"SE","china":"CN","austria":"AT","kazakhstan":"KZ","qatar":"QA",
        "monaco":"MC","netherlands":"NL","denmark":"DK"
    }
    return MAP.get(v.lower(), "??")

def load_players_meta_v2(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["player_id","country_id"])
    p = pd.read_csv(path)
    p.columns = [c.strip().lower() for c in p.columns]

    id_col = next((c for c in ["player_id","id","playerid","atp_id"] if c in p.columns), None)
    if not id_col:
        return pd.DataFrame(columns=["player_id","country_id"])
    p = p.rename(columns={id_col: "player_id"})

    c_col = next((c for c in ["country_id","country_code","countryid","country","nationality"] if c in p.columns), None)
    if c_col:
        p["country_id"] = p[c_col].apply(_coerce_country_id)
    else:
        p["country_id"] = "??"

    out = p[["player_id","country_id"]].drop_duplicates("player_id")
    out["player_id"] = out["player_id"].astype(str)
    return out

def load_tournaments_meta_v2(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["tournament","country_id"])
    t = pd.read_csv(path)
    t.columns = [c.strip().lower() for c in t.columns]

    name_col = next((c for c in ["tournament","name","tournament_name","tourney","tournament_title"] if c in t.columns), None)
    if not name_col:
        return pd.DataFrame(columns=["tournament","country_id"])
    t = t.rename(columns={name_col: "tournament"})

    c_col = next((c for c in ["country_id","country_code","country","countryid"] if c in t.columns), None)
    if c_col:
        t["country_id"] = t[c_col].apply(_coerce_country_id)
    else:
        t["country_id"] = "N/A"

    out = t[["tournament","country_id"]].drop_duplicates("tournament")
    out["tournament"] = out["tournament"].astype(str)
    return out

def add_meta_features_v2(df: pd.DataFrame,
                         matches_df: pd.DataFrame,
                         pmeta: pd.DataFrame,
                         tmeta: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Afegeix NOMÉS features numèriques necessàries per calcular "home":
      - A_is_home, B_is_home, home_advantage
      - best_of_5 (per robustesa; sovint també ja ve del pipeline)
    """
    df = df.copy()

    # Porta clau des de matches_df
    mcols = [c for c in ["match_id","player_a_id","player_b_id","tournament","best_of_5"] if c in matches_df.columns]
    if "match_id" not in df.columns:
        raise ValueError("add_meta_features_v2: falta 'match_id' a dataset_raw")
    if mcols:
        df = df.merge(matches_df[mcols].drop_duplicates("match_id"), on="match_id", how="left")

    for c, default in [("player_a_id",""), ("player_b_id",""), ("tournament",""), ("best_of_5",0)]:
        if c not in df.columns:
            df[c] = default

    # Players → country
    if not pmeta.empty:
        pm = pmeta.copy()
        pm.columns = [c.strip().lower() for c in pm.columns]
        if "player_id" not in pm.columns:
            cand = [c for c in pm.columns if c in ("id","atp_id")]
            if cand:
                pm = pm.rename(columns={cand[0]:"player_id"})
        pm["player_id"] = pm["player_id"].astype(str)

        pA = pm.rename(columns={"player_id":"player_a_id","country_id":"A_country_id"})[["player_a_id","A_country_id"]].drop_duplicates("player_a_id")
        pB = pm.rename(columns={"player_id":"player_b_id","country_id":"B_country_id"})[["player_b_id","B_country_id"]].drop_duplicates("player_b_id")

        df["player_a_id"] = df["player_a_id"].astype(str)
        df["player_b_id"] = df["player_b_id"].astype(str)
        df = df.merge(pA, on="player_a_id", how="left")
        df = df.merge(pB, on="player_b_id", how="left")
    else:
        df["A_country_id"] = "??"
        df["B_country_id"] = "??"

    # Tournament → country
    if not tmeta.empty:
        tm = tmeta.copy()
        tm.columns = [c.strip().lower() for c in tm.columns]
        if "tournament" not in tm.columns:
            for c in ["name","tournament_name","tourney","tournament_title"]:
                if c in tm.columns:
                    tm = tm.rename(columns={c:"tournament"})
                    break
        keep = ["tournament"]
        if "country_id" in tm.columns:
            keep.append("country_id")
        tm = tm[keep].drop_duplicates("tournament").rename(columns={"country_id":"tourney_country_id"})

        df["tournament"] = df["tournament"].astype(str)
        tm["tournament"] = tm["tournament"].astype(str)
        df = df.merge(tm, on="tournament", how="left")

    if "tourney_country_id" not in df.columns:
        df["tourney_country_id"] = "N/A"

    # Derivades numèriques (neutral -> no home)
    def _neutral(series: pd.Series) -> pd.Series:
        return series.astype(str).str.upper().isin(["", "N/A","NA","??","NEUTRAL","NONE","NULL","NAN"])

    A_c = df["A_country_id"].astype(str).str.upper().fillna("??")
    B_c = df["B_country_id"].astype(str).str.upper().fillna("??")
    T_c = df["tourney_country_id"].astype(str).str.upper().fillna("N/A")

    neu = _neutral(T_c)
    A_home = (~neu) & (A_c == T_c)
    B_home = (~neu) & (B_c == T_c)

    df["A_is_home"] = A_home.astype(int)
    df["B_is_home"] = B_home.astype(int)
    df["home_advantage"] = df["A_is_home"] - df["B_is_home"]

    df["best_of_5"] = pd.to_numeric(df.get("best_of_5", 0), errors="coerce").fillna(0).astype(int)

    added_cols = ["A_is_home","B_is_home","home_advantage","best_of_5"]
    for c in added_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf,-np.inf], np.nan).fillna(0)

    return df, added_cols


# =========================
# Training
# =========================

def train_models(dataset: pd.DataFrame,
                 model_cols_candidate: Optional[List[str]] = None,
                 use_lgb: bool = False) -> Tuple[object, Optional[StandardScaler], IsotonicRegression, dict, tuple, List[str]]:
    df = dataset.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    n = len(df)
    if n < 50:
        raise ValueError(f"Dataset massa petit per entrenar (n={n}).")

    cut80 = int(0.80 * n)
    cut90 = int(0.90 * n)
    train = df.iloc[:cut80].copy()
    valid = df.iloc[cut80:cut90].copy()
    test  = df.iloc[cut90:].copy()

    forbidden_exact = {
        "odds_home","odds_away",
        "y_home_win","winner_id","match_id","date","commence_time",
        "best_side","best_edge","stake_% (half kelly)",
        "edge_home","edge_away","kelly_home","kelly_away",
        "fair_odds_home","fair_odds_away",
        "unit_return","source","pred_time_utc","model_name"
    }

    def _is_bad_feature(c: str) -> bool:
        cl = c.lower()
        return (
            (c in forbidden_exact) or
            ("odds" in cl) or
            ("edge" in cl) or
            ("kelly" in cl) or
            ("fair_odds" in cl) or
            ("unit_return" in cl) or
            ("best_side" in cl) or
            ("best_edge" in cl) or
            ("decision" in cl) or
            ("hint" in cl)
        )

    # Candidates -> final used
    if model_cols_candidate:
        base = [c for c in model_cols_candidate if c in df.columns]
    else:
        base = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()

    model_cols_used = [c for c in base if (c != "y_home_win") and (not _is_bad_feature(c))]
    if not model_cols_used:
        raise ValueError("Sense features vàlides després del filtre anti-leak.")

    # Persistim EXACTAMENT les columnes usades
    (OUT_DIR / "model_columns.txt").write_text("\n".join(model_cols_used), encoding="utf-8")

    y_tr = pd.to_numeric(train["y_home_win"], errors="coerce").fillna(0).astype(int).values
    y_va = pd.to_numeric(valid["y_home_win"], errors="coerce").fillna(0).astype(int).values

    def _to_X(dff: pd.DataFrame, cols: List[str]) -> np.ndarray:
        Xdf = dff.reindex(columns=cols, fill_value=0.0)
        Xdf = Xdf.apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan).fillna(0.0)
        return Xdf.values

    X_tr = _to_X(train, model_cols_used)
    X_va = _to_X(valid, model_cols_used)

    # Train
    if use_lgb and HAS_LGB and len(train):
        dtr = lgb.Dataset(X_tr, label=y_tr)
        dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
        params = dict(
            objective="binary",
            metric="binary_logloss",
            learning_rate=0.05,
            num_leaves=63,
            feature_fraction=0.9,
            bagging_fraction=0.9,
            bagging_freq=1,
            min_data_in_leaf=50,
            seed=2026,
            verbose=-1,
            force_row_wise=True
        )
        booster = lgb.train(
            params, dtr, num_boost_round=5000,
            valid_sets=[dtr, dva],
            valid_names=["train","valid"],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)]
        )
        model = booster
        scaler = None
        p_va = booster.predict(X_va, num_iteration=booster.best_iteration)
    else:
        scaler = StandardScaler()
        X_trs = scaler.fit_transform(X_tr)
        X_vas = scaler.transform(X_va)
        lr = LogisticRegression(C=1.0, max_iter=300)
        lr.fit(X_trs, y_tr)
        model = lr
        p_va = lr.predict_proba(X_vas)[:, 1]

    # Calibratge isotònic (sobre valid)
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_va, y_va)

    # Avaluació test
    X_te = _to_X(test, model_cols_used)
    if use_lgb and HAS_LGB and hasattr(model, "predict") and not hasattr(model, "predict_proba"):
        p_te = model.predict(X_te, num_iteration=getattr(model, "best_iteration", None))
    else:
        X_tes = scaler.transform(X_te) if scaler is not None else X_te
        p_te = model.predict_proba(X_tes)[:, 1]

    y_te = pd.to_numeric(test["y_home_win"], errors="coerce").fillna(0).astype(int).values

    def safe_auc(y, p):
        try:
            return roc_auc_score(y, p)
        except Exception:
            return np.nan

    metrics = dict(
        valid_logloss=float(log_loss(y_va, np.clip(p_va, 1e-6, 1 - 1e-6))) if len(valid) else np.nan,
        valid_auc=float(safe_auc(y_va, p_va)) if len(valid) else np.nan,
        valid_brier=float(brier_score_loss(y_va, p_va)) if len(valid) else np.nan,
        test_logloss=float(log_loss(y_te, np.clip(p_te, 1e-6, 1 - 1e-6))) if len(test) and len(np.unique(y_te)) > 1 else np.nan,
        test_auc=float(safe_auc(y_te, p_te)) if len(test) and len(np.unique(y_te)) > 1 else np.nan,
        test_brier=float(brier_score_loss(y_te, p_te)) if len(test) else np.nan,
        n_train=int(len(train)), n_valid=int(len(valid)), n_test=int(len(test)),
        n_features=int(len(model_cols_used)),
    )

    # Save models
    if use_lgb and HAS_LGB and not hasattr(model, "predict_proba"):
        joblib.dump(model, OUT_DIR / "model_lightgbm.pkl")
    else:
        joblib.dump(model, OUT_DIR / "model_logistic.pkl")
        joblib.dump(scaler, OUT_DIR / "scaler.pkl")

    joblib.dump(iso, OUT_DIR / "calibrator_isotonic.pkl")
    joblib.dump(iso, OUT_DIR / "isotonic.pkl")  # compat

    # Save preds test
    cols_for_preds = [c for c in ["match_id","date","player_a_id","player_b_id"] if c in test.columns]
    preds_test = test[cols_for_preds].copy() if cols_for_preds else test[["match_id","date"]].copy()
    preds_test["p_home_win"] = np.clip(p_te, 1e-6, 1 - 1e-6)
    preds_test["y_home_win"] = y_te
    preds_test.to_csv(OUT_DIR / "preds_test.csv", index=False)

    return model, scaler, iso, metrics, (train, valid, test), model_cols_used


def predict_upcoming(model, scaler, iso, dataset: pd.DataFrame, model_cols_used: List[str], matches: pd.DataFrame) -> pd.DataFrame:
    today = pd.Timestamp.today().normalize()

    # "unknown winner" robust: NaN o strings tipus "nan"/""/"none"
    w = matches.get("winner_id", pd.Series(index=matches.index, dtype="object"))
    w_str = w.astype(str).str.strip().str.lower()
    unknown_winner = w.isna() | w_str.isin(["", "nan", "none", "null"])

    mdate = pd.to_datetime(matches.get("date", pd.Series(index=matches.index, dtype="object")), errors="coerce")
    future_or_today = mdate >= today

    cand_ids = matches[unknown_winner | future_or_today]["match_id"].astype(str)
    ds = dataset[dataset["match_id"].astype(str).isin(set(cand_ids))].copy()
    if not len(ds):
        return pd.DataFrame()

    Xdf = ds.reindex(columns=model_cols_used, fill_value=0.0)
    Xdf = Xdf.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = Xdf.values

    if hasattr(model, "predict_proba"):
        if scaler is not None:
            X = scaler.transform(X)
        p = model.predict_proba(X)[:, 1]
    else:
        try:
            p = model.predict(X, num_iteration=getattr(model, "best_iteration", None))
        except TypeError:
            p = model.predict(X)

    if iso is not None:
        p = iso.transform(p)

    out = ds[["match_id","date"]].copy()
    out["p_home_win"] = np.clip(p, 1e-6, 1 - 1e-6)
    return out


# =========================
# Eval helpers
# =========================

def _prep_eval_predictions(dataset_df, matches_df, model, scaler, iso, model_cols_for_eval):
    Xdf = dataset_df.reindex(columns=model_cols_for_eval, fill_value=0.0)
    Xdf = Xdf.apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan).fillna(0.0)
    X = Xdf.values
    if scaler is not None and hasattr(scaler, "transform"):
        X = scaler.transform(X)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if hasattr(model, "predict_proba"):
        p_raw = model.predict_proba(X)[:, 1]
    else:
        try:
            p_raw = model.predict(X, num_iteration=getattr(model, "best_iteration", None))
        except TypeError:
            p_raw = model.predict(X)

    if iso is not None:
        p_raw = iso.transform(p_raw)

    p_clip = np.clip(p_raw, 1e-6, 1 - 1e-6)

    eval_df = dataset_df[["match_id","date","y_home_win"]].copy()
    eval_df["p_home_win"] = p_clip

    merge_cols = [c for c in ["match_id","odds_home","odds_away","best_of_5"] if c in matches_df.columns]
    if merge_cols:
        eval_df = eval_df.merge(matches_df[merge_cols], on="match_id", how="left")
    return eval_df

def _hi_conf_accuracy(eval_df, p_hi=0.60, p_lo=0.40):
    df = eval_df.copy()
    y = pd.to_numeric(df["y_home_win"], errors="coerce")
    bin_ok = y.isin([0,1])
    pick_home = df["p_home_win"] >= p_hi
    pick_away = df["p_home_win"] <= p_lo
    pick_mask = (pick_home | pick_away) & bin_ok
    df_picks = df[pick_mask].copy()
    if not len(df_picks):
        return {"hc_n_picks": 0, "hc_hit_rate": np.nan, "hc_cov_pct": 0.0}
    picks_home_mask = pick_home[pick_mask]
    y_sub = y[pick_mask]
    bet_hit = np.where(picks_home_mask, (y_sub == 1).astype(int), (y_sub == 0).astype(int))
    return {
        "hc_n_picks": int(len(df_picks)),
        "hc_hit_rate": float(np.mean(bet_hit)) if len(bet_hit) else np.nan,
        "hc_cov_pct": 100.0 * len(df_picks) / len(df) if len(df) else 0.0,
    }

def _edge(prob, odd):
    if isinstance(odd,(int,float,np.floating)) and np.isfinite(odd) and odd > 0:
        return prob - (1.0/odd)
    return np.nan

def _backtest_simple(eval_df):
    rows = []
    if not {"odds_home","odds_away"}.issubset(eval_df.columns):
        return {"bt_n_bets": 0, "bt_hit_rate": np.nan, "bt_avg_edge": np.nan, "bt_coverage_pct": 0.0}

    for _, rr in eval_df.iterrows():
        if not pd.notna(rr.get("y_home_win")):
            continue
        p_h = float(rr["p_home_win"]); p_a = 1.0 - p_h
        oh = pd.to_numeric(rr.get("odds_home"), errors="coerce")
        oa = pd.to_numeric(rr.get("odds_away"), errors="coerce")
        if not (np.isfinite(oh) and oh > 1 and np.isfinite(oa) and oa > 1):
            continue

        eh = _edge(p_h, oh)
        ea = _edge(p_a, oa)
        if not (np.isfinite(eh) or np.isfinite(ea)):
            continue

        if np.nanmax([eh, ea]) <= 0:
            continue

        side_home = (eh >= ea)
        y = int(rr["y_home_win"])
        hit = 1 if ((side_home and y == 1) or ((not side_home) and y == 0)) else 0
        rows.append({"hit": hit, "edge": float(np.nanmax([eh, ea]))})

    if not rows:
        return {"bt_n_bets": 0, "bt_hit_rate": np.nan, "bt_avg_edge": np.nan, "bt_coverage_pct": 0.0}

    bt = pd.DataFrame(rows)
    return {
        "bt_n_bets": int(len(bt)),
        "bt_hit_rate": float(bt["hit"].mean()),
        "bt_avg_edge": float(bt["edge"].mean()),
        "bt_coverage_pct": 100.0 * len(bt) / len(eval_df) if len(eval_df) else 0.0,
    }


# =========================
# Main
# =========================

def parse_args():
    p = argparse.ArgumentParser(description="Build outputs (offline) igual que la Tab 'Refresh & Train' de l'app.")
    p.add_argument("--year-start", type=int, default=2015)
    p.add_argument("--year-end", type=int, default=int(pd.Timestamp.today().year))
    p.add_argument("--include-ongoing", action="store_true", default=True)
    p.add_argument("--no-include-ongoing", dest="include_ongoing", action="store_false")

    p.add_argument("--use-lgb", action="store_true", default=False)
    p.add_argument("--timeout", type=int, default=30)
    p.add_argument("--no-cache", action="store_true", default=False)

    # Cutoffs (com UI)
    p.add_argument("--train-cutoff-enabled", action="store_true", default=False)
    p.add_argument("--train-cutoff-date", type=str, default="2026-01-01")
    p.add_argument("--match-cutoff-enabled", action="store_true", default=True)
    p.add_argument("--match-cutoff-date", type=str, default="2026-01-01")
    p.add_argument("--cutoff-mode", type=str, default="Match-level",
                   choices=["Tournament-level", "Match-level", "Both (stricter)"])

    return p.parse_args()


def main():
    args = parse_args()

    year_start = int(args.year_start)
    year_end = int(args.year_end)
    include_ongoing = bool(args.include_ongoing)
    use_lgb = bool(args.use_lgb)

    use_cache = not bool(args.no_cache)
    timeout = int(args.timeout)

    train_cutoff_enabled = bool(args.train_cutoff_enabled)
    train_cutoff_date = pd.to_datetime(args.train_cutoff_date).date()

    match_cutoff_enabled = bool(args.match_cutoff_enabled)
    match_cutoff_date = pd.to_datetime(args.match_cutoff_date).date()

    cutoff_mode = str(args.cutoff_mode)

    print(f"[config] years={year_start}..{year_end} include_ongoing={include_ongoing} use_lgb={use_lgb} HAS_LGB={HAS_LGB}")
    print(f"[config] cache={use_cache} timeout={timeout}s")
    print(f"[config] train_cutoff_enabled={train_cutoff_enabled} train_cutoff_date={train_cutoff_date}")
    print(f"[config] match_cutoff_enabled={match_cutoff_enabled} match_cutoff_date={match_cutoff_date}")
    print(f"[config] cutoff_mode={cutoff_mode}")
    print(f"[paths] REPO_DIR={REPO_DIR}")
    print(f"[paths] DATA_DIR={DATA_DIR}")
    print(f"[paths] OUT_DIR={OUT_DIR}")

    # Info token (sense imprimir-lo)
    has_token = bool(_get_github_token())
    print(f"[github] token_present={has_token} (needed only if cache misses and repo is private)")

    # 1) Fetch TML
    frames = []
    for y in range(year_start, year_end + 1):
        try:
            dfy = fetch_year_csv(y, timeout=timeout, use_cache=use_cache)
            frames.append(dfy)
            print(f"[fetch] {y}: rows={len(dfy):,}")
        except Exception as e:
            print(f"[fetch][warn] {y}: {type(e).__name__}: {e}")

    if include_ongoing:
        try:
            dfo = fetch_ongoing_csv(timeout=timeout, use_cache=use_cache)
            frames.append(dfo)
            print(f"[fetch] ongoing_tourneys: rows={len(dfo):,}")
        except Exception as e:
            print(f"[fetch][warn] ongoing_tourneys: {type(e).__name__}: {e}")

    if not frames:
        raise RuntimeError("No s'ha pogut descarregar cap CSV TML.")

    df_all = pd.concat(frames, ignore_index=True)
    print(f"[fetch] df_all rows={len(df_all):,}")

    # 2) Build matches & points (guardem COMPLET)
    matches = build_matches_from_tml(df_all)
    points  = build_points_from_tml(df_all)

    matches_path = DATA_DIR / "matches.csv"
    points_path  = DATA_DIR / "points_sets_games.csv"

    matches.to_csv(matches_path, index=False)
    points.to_csv(points_path, index=False)
    print(f"[save] {matches_path} rows={len(matches):,}")
    print(f"[save] {points_path} rows={len(points):,}")

    # 2b) Apply training filters EN MEMÒRIA
    matches_train = matches.copy()
    points_train  = points.copy()

    if "date" in matches_train.columns:
        matches_train["date"] = _parse_match_dates(matches_train["date"])
    if "match_date" in matches_train.columns:
        matches_train["match_date"] = _parse_match_dates(matches_train["match_date"])

    if "tourney_id" not in matches_train.columns:
        matches_train["tourney_id"] = matches_train["match_id"].astype(str).str.split("_").str[0]

    def apply_tournament_cutoff(df: pd.DataFrame, cutoff_ts: pd.Timestamp) -> pd.DataFrame:
        if "date" not in df.columns or df["date"].isna().all():
            print("[cutoff][warn] Tournament-level cutoff: 'date' missing/empty -> skip.")
            return df
        if "tourney_id" not in df.columns or df["tourney_id"].isna().all():
            print("[cutoff][warn] Tournament-level cutoff: 'tourney_id' missing/empty -> skip.")
            return df
        tournament_start = df.groupby("tourney_id")["date"].transform("min")
        return df[tournament_start < cutoff_ts].copy()

    def apply_match_cutoff(df: pd.DataFrame, cutoff_ts: pd.Timestamp) -> pd.DataFrame:
        if "match_date" not in df.columns or df["match_date"].isna().all():
            print("[cutoff][warn] Match-level cutoff: 'match_date' missing/empty -> skip.")
            return df
        return df[df["match_date"] < cutoff_ts].copy()

    before_n = len(matches_train)

    if cutoff_mode == "Tournament-level":
        if train_cutoff_enabled:
            matches_train = apply_tournament_cutoff(matches_train, pd.Timestamp(train_cutoff_date))
    elif cutoff_mode == "Match-level":
        if match_cutoff_enabled:
            matches_train = apply_match_cutoff(matches_train, pd.Timestamp(match_cutoff_date))
    else:  # Both (stricter)
        if train_cutoff_enabled:
            matches_train = apply_tournament_cutoff(matches_train, pd.Timestamp(train_cutoff_date))
        if match_cutoff_enabled:
            matches_train = apply_match_cutoff(matches_train, pd.Timestamp(match_cutoff_date))

    if "match_id" in matches_train.columns and "match_id" in points_train.columns:
        allowed_ids = set(matches_train["match_id"].astype(str))
        points_train["match_id"] = points_train["match_id"].astype(str)
        points_train = points_train[points_train["match_id"].isin(allowed_ids)].copy()

    removed = before_n - len(matches_train)
    print(f"[cutoff] removed={removed:,} remaining matches_train={len(matches_train):,} points_train={len(points_train):,}")

    # 3) Feature building (pipeline v2 + meta-home only)
    matches_enr = enrich_matches_domestic(matches_train)

    feats = compute_pre_match_features_v2(matches_enr, points_train)
    dataset_raw, model_cols_base = make_match_features(feats, matches_enr)

    pmeta = load_players_meta_v2(DATA_DIR / "players_meta.csv")
    tmeta = load_tournaments_meta_v2(DATA_DIR / "tournaments_meta.csv")

    dataset_final, added_cols = add_meta_features_v2(dataset_raw, matches_enr, pmeta, tmeta)

    # Candidate cols (abans del filtre final al train)
    model_cols_candidate = list(model_cols_base) + list(added_cols or [])
    _leak_words = ("odds", "edge", "kelly", "stake", "best_side", "decision", "hint", "fair_odds")
    model_cols_candidate = [c for c in model_cols_candidate if all(w not in str(c).lower() for w in _leak_words)]

    # Save dataset/files
    feats_path     = OUT_DIR / "features_player_pre.csv"
    ds_raw_path    = OUT_DIR / "dataset_match_level_raw.csv"
    ds_final_path  = OUT_DIR / "dataset_match_level.csv"

    feats.to_csv(feats_path, index=False)
    dataset_raw.to_csv(ds_raw_path, index=False)
    dataset_final.to_csv(ds_final_path, index=False)

    print(f"[save] {feats_path} rows={len(feats):,}")
    print(f"[save] {ds_raw_path} rows={len(dataset_raw):,}")
    print(f"[save] {ds_final_path} rows={len(dataset_final):,}")

    # 4) Train + metrics
    model, scaler, iso, metrics, splits, model_cols_used = train_models(
        dataset_final,
        model_cols_candidate=model_cols_candidate,
        use_lgb=use_lgb
    )
    print(f"[train] model_columns.txt written with n_cols={len(model_cols_used)}")

    eval_df = _prep_eval_predictions(dataset_final, matches_enr, model, scaler, iso, model_cols_used)
    hc_stats = _hi_conf_accuracy(eval_df, p_hi=0.60, p_lo=0.40)
    bt_stats = _backtest_simple(eval_df)

    metrics_to_save = dict(metrics)
    metrics_to_save["timestamp"]  = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    metrics_to_save["model_type"] = model.__class__.__name__
    metrics_to_save.update(hc_stats)
    metrics_to_save.update(bt_stats)

    with open(OUT_DIR / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_to_save, f, indent=2)

    print("[train] saved train_metrics.json")
    print("[train] metrics:", json.dumps(metrics_to_save, indent=2))

    # 5) Predict upcoming
    preds = predict_upcoming(model, scaler, iso, dataset_final, model_cols_used, matches_train)
    if len(preds):
        preds_path = OUT_DIR / "predictions_upcoming.csv"
        preds.to_csv(preds_path, index=False)
        print(f"[predict] saved {preds_path} rows={len(preds):,}")
    else:
        print("[predict] no upcoming/unknown-winner matches detected.")

    print("[done] outputs built successfully.")


if __name__ == "__main__":
    main()
