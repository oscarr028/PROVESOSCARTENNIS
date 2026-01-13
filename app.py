#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# streamlit_app.py — v2 with Odds API diagnostics, multi-region search, and events fallback

import os, io, sys, time, math, json, datetime as dt
from typing import List, Tuple, Optional, Dict
import pandas as pd
import numpy as np
import requests
import streamlit as st
from pathlib import Path
import shutil

# Optional libs
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss


# Try to import our pipeline v2 (expected in the same folder or PYTHONPATH)
PIPE_OK = False
try:
    from tennis_model_pipeline_v2 import (
        compute_pre_match_features_v2,
        make_match_features,
        enrich_matches_domestic,
    )
    PIPE_OK = True
except Exception as e:
    PIPE_OK = False


#RAW_BASE = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master"
RAW_BASE = "https://raw.githubusercontent.com/adriaparcerisas/Tennis-Bets/main/data/tml_cache"


import __main__

def get_app_dir() -> Path:
    main_file = getattr(__main__, "__file__", None)
    if main_file:
        return Path(main_file).resolve().parent
    return Path.cwd().resolve()

APP_DIR  = get_app_dir()
DATA_DIR = APP_DIR / "data"
OUT_DIR  = DATA_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = OUT_DIR / "predictions_log.csv"

st.set_page_config(page_title="Tennis Predictions (TML + Elo v2)", layout="wide")

# ─── MIGRACIÓ D’ARXIUS ANTICS (outputs/ → data/outputs/) ──────────
def load_artifacts(OUT_DIR: str):
    import os, joblib
    model = scaler = iso = None
    model_cols = None

    # model (lgbm o logistic)
    for name in ["model_lightgbm.pkl", "model_logistic.pkl"]:
        p = os.path.join(OUT_DIR, name)
        if os.path.exists(p):
            model = joblib.load(p)
            break

    sp = os.path.join(OUT_DIR, "scaler.pkl")
    if os.path.exists(sp):
        scaler = joblib.load(sp)

    ip = os.path.join(OUT_DIR, "calibrator_isotonic.pkl")
    if os.path.exists(ip):
        iso = joblib.load(ip)

    cp = os.path.join(OUT_DIR, "model_columns.txt")
    if os.path.exists(cp):
        with open(cp, "r") as f:
            model_cols = [line.strip() for line in f if line.strip()]

    return model, scaler, iso, model_cols

def migrate_legacy_outputs():
    legacy_dirs = [APP_DIR/"outputs"]  # possibles rutes antigues
    moved = []
    for d in legacy_dirs:
        if d.exists() and d.is_dir():
            for p in d.glob("*.csv"):
                target = OUT_DIR/p.name
                if not target.exists():
                    try:
                        shutil.move(str(p), str(target)); moved.append(p.name)
                    except Exception:
                        pass
    # si el directori antic queda buit, el pots eliminar
    try:
        if (APP_DIR/"outputs").exists() and not any((APP_DIR/"outputs").iterdir()):
            (APP_DIR/"outputs").rmdir()
    except Exception:
        pass
    return moved

_just_moved = migrate_legacy_outputs()
if _just_moved:
    st.info(f"📦 Migrats a data/outputs/: {', '.join(_just_moved)}")

# ─── NAME CONFIG ─── 
def _norm_surface(s: str) -> str:
    s = str(s or "").strip().lower()
    if "carpet" in s: return "indoor-hard"
    if "indoor" in s: return "indoor-hard"
    if "hard" in s: return "hard"
    if "clay" in s: return "clay"
    if "grass" in s: return "grass"
    return "hard"

import re, unicodedata

def _norm_name(s: str) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
    s = s.lower().strip().replace(".", " ")
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s

def _name_tokens(s: str):
    s = _norm_name(s)
    return [t for t in s.split() if t]

def _surname_like(query: str, full_name: str) -> bool:
    """
    True si TOTS els tokens de 'query' (p.ex. 'alcaraz' o 'davidovich fokina')
    apareixen entre els tokens del 'full_name' de TML.
    """
    q = set(_name_tokens(query))
    f = set(_name_tokens(full_name))
    return len(q) > 0 and q.issubset(f)

def _parse_yyyymmdd_series(s: pd.Series) -> pd.Series:
    """
    Converteix valors tipus YYYYMMDD (int/str) a datetime.
    Si no és YYYYMMDD, prova parse genèric.
    """
    ss = s.astype(str).str.strip()
    ss = ss.str.replace(r"\.0$", "", regex=True)  # 20251020.0 -> 20251020

    digits = ss.str.replace(r"\D", "", regex=True)
    is8 = digits.str.len().eq(8)

    dt8 = pd.to_datetime(digits.where(is8), format="%Y%m%d", errors="coerce")
    dtany = pd.to_datetime(ss.where(~is8), errors="coerce")

    out = dt8.fillna(dtany)

    # Si ve timezone-aware, ho deixem naive
    try:
        out = out.dt.tz_localize(None)
    except Exception:
        pass

    return out


# =========================
# GitHub private-safe fetch (for RAW_BASE)
# =========================

def _get_github_token() -> str:
    # Streamlit Cloud: st.secrets["GITHUB_TOKEN"] (si el poses)
    # Local: export GITHUB_TOKEN=... / GH_TOKEN=... / GH_PAT=...
    tok = ""
    try:
        tok = st.secrets.get("GITHUB_TOKEN", "") or st.secrets.get("GH_TOKEN", "") or st.secrets.get("GH_PAT", "")
    except Exception:
        tok = ""
    return tok or os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN") or os.getenv("GH_PAT") or ""

def _github_raw_via_api(owner: str, repo: str, ref: str, path: str, timeout: int = 30) -> str:
    """
    Llegeix un fitxer del repo (inclòs privat) via GitHub Contents API.
    Retorna text CSV.
    """
    token = _get_github_token()
    headers = {"Accept": "application/vnd.github.raw"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={ref}"
    r = requests.get(api_url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text

def _fetch_text_private_safe(url: str, timeout: int = 30) -> str:
    """
    1) Si és raw.githubusercontent.com, prova Contents API (necessita token si repo privat)
    2) Si falla, prova raw directe
    3) Si no és raw, prova requests.get normal
    """
    m = re.match(r"^https?://raw\.githubusercontent\.com/([^/]+)/([^/]+)/([^/]+)/(.*)$", url)
    if m:
        owner, repo, ref, path = m.group(1), m.group(2), m.group(3), m.group(4)
        # Primer: API (funciona en privat amb token)
        try:
            return _github_raw_via_api(owner, repo, ref, path, timeout=timeout)
        except Exception:
            # Segon: raw directe (funciona en públic o si tens accés per algun altre mecanisme)
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.text

    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


# =========================
# Fetchers used by the app
# =========================

def fetch_year_csv(year: int) -> pd.DataFrame:
    """
    - Intenta baixar del RAW_BASE (privat-safe via API)
    - Si falla, prova fallback local: data/tml_cache/<year>.csv (si existeix)
    """
    url = f"{RAW_BASE}/{year}.csv"

    # fallback local (no canvia res si no existeix)
    local_path = DATA_DIR / "tml_cache" / f"{year}.csv"

    try:
        text = _fetch_text_private_safe(url, timeout=30)
        df = pd.read_csv(io.StringIO(text), dtype={"tourney_date": "string"})
        df["__src_year"] = year
        return df
    except Exception as e:
        # fallback local
        if local_path.exists():
            df = pd.read_csv(local_path, dtype={"tourney_date": "string"})
            df["__src_year"] = year
            return df

        # error explícit i actionable
        tok_present = bool(_get_github_token())
        raise RuntimeError(
            f"fetch_year_csv({year}) failed.\n"
            f"- Tried remote: {url}\n"
            f"- Local fallback missing: {local_path}\n"
            f"- Token present: {tok_present}\n"
            f"Root error: {type(e).__name__}: {e}\n\n"
            f"If the repo is private, set GITHUB_TOKEN (or GH_TOKEN/GH_PAT) in Streamlit secrets or env vars."
        )

def fetch_ongoing_csv() -> pd.DataFrame:
    """
    Mateixa lògica que fetch_year_csv, per ongoing_tourneys.csv.
    """
    url = f"{RAW_BASE}/ongoing_tourneys.csv"
    local_path = DATA_DIR / "tml_cache" / "ongoing_tourneys.csv"

    try:
        text = _fetch_text_private_safe(url, timeout=30)
        df = pd.read_csv(io.StringIO(text), dtype={"tourney_date": "string"})
        df["__src_year"] = int(pd.Timestamp.today().year)
        return df
    except Exception as e:
        if local_path.exists():
            df = pd.read_csv(local_path, dtype={"tourney_date": "string"})
            df["__src_year"] = int(pd.Timestamp.today().year)
            return df

        tok_present = bool(_get_github_token())
        raise RuntimeError(
            "fetch_ongoing_csv() failed.\n"
            f"- Tried remote: {url}\n"
            f"- Local fallback missing: {local_path}\n"
            f"- Token present: {tok_present}\n"
            f"Root error: {type(e).__name__}: {e}\n\n"
            f"If the repo is private, set GITHUB_TOKEN (or GH_TOKEN/GH_PAT) in Streamlit secrets or env vars."
        )


def build_name_id_map(df_all: pd.DataFrame) -> Dict[str, str]:
    name_to_id = {}
    cols = ["winner_name","winner_id","loser_name","loser_id"]
    for c in cols:
        if c not in df_all.columns:
            df_all[c] = np.nan
    for _, r in df_all[["winner_name","winner_id"]].dropna().iterrows():
        name_to_id[_norm_name(r["winner_name"])] = str(r["winner_id"])
    for _, r in df_all[["loser_name","loser_id"]].dropna().iterrows():
        name_to_id[_norm_name(r["loser_name"])] = str(r["loser_id"])
    return name_to_id

def build_matches_from_tml(df_all: pd.DataFrame) -> pd.DataFrame:
    df = df_all.copy()
    needed = ["tourney_id","tourney_name","surface","tourney_level","tourney_date",
              "match_num","winner_id","loser_id","best_of","round"]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan

    df["tourney_date"] = _parse_yyyymmdd_series(df["tourney_date"])
    df["surface_norm"] = df["surface"].apply(_norm_surface)
    df["best_of_5"] = (pd.to_numeric(df["best_of"], errors="coerce")==5).astype(int)

    # --- PSEUDO MATCH DATE (1 dia per ronda dins de cada torneig) ---
    ROUND_STAGE = {
        # Qualifying (si apareix)
        "Q1": 0, "Q2": 1, "Q3": 2,
    
        # Round robin: tot el mateix dia
        "RR": 3,
    
        # Main draw
        "R128": 4, "R64": 5, "R32": 6, "R16": 7,
        "QF": 8, "SF": 9,
    
        # Tercer i quart lloc (olímpics, etc.)
        "BR": 10,
    
        "F": 11,
    
        # Aliases (per si algun CSV ve diferent)
        "1R": 5, "2R": 6, "3R": 7, "4R": 8,
        "R1": 5, "R2": 6, "R3": 7, "R4": 8,
    }
    
    r = df["round"].astype(str).str.strip().str.upper()
    stage = r.map(ROUND_STAGE)
    
    # Per torneig, la primera ronda existent serà offset 0; la següent offset 1, etc.
    min_stage = stage.groupby(df["tourney_id"].astype(str)).transform("min")
    
    # Si alguna ronda no està al mapping -> la deixem al mateix dia (offset 0)
    offset_days = (stage - min_stage).fillna(0)
    offset_days = offset_days.clip(lower=0).astype(int)
    
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
    df["winner_id"] = df["winner_id"].astype(str)

    df["match_id"] = (
        df["tourney_id"].astype(str) + "_" +
        df["tourney_date"].dt.strftime("%Y%m%d").fillna("00000000") + "_" +
        df["match_num"].astype(str)
    )

    matches = pd.DataFrame({
        "match_id": df["match_id"].astype(str),
        "date": df["tourney_date"].dt.strftime("%Y-%m-%d"),          # data “oficial” del CSV
        "match_date": df["match_date"].dt.strftime("%Y-%m-%d"),      # data pseudo per rondes (per filtrar)
        "tournament": df["tourney_name"].astype(str),
        "city": df["tourney_name"].astype(str),
        "country": np.nan,
        "level": df["tourney_level"].astype(str),
        "round": df["round"].astype(str),
        "best_of_5": df["best_of_5"].astype(int),
        "surface": df["surface_norm"].astype(str),
        "indoor": (df["surface_norm"]=="indoor-hard").astype(int),
        "player_a_id": df["player_a_id"].astype(str),
        "player_b_id": df["player_b_id"].astype(str),
        "winner_id": df["winner_id"].astype(str),
        "duration_minutes": pd.to_numeric(df.get("minutes", np.nan), errors="coerce")
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

def train_models(dataset: pd.DataFrame, model_cols: list | None = None, use_lgb: bool = False):
    """
    Entrena model (Logistic o LightGBM) amb split temporal 80/10/10,
    aplica calibratge isotònic, i evita filtracions (odds/edge/kelly/etc) a les features.
    Desa: model, scaler (si n'hi ha), calibrador i model_columns.txt a OUT_DIR.

    Retorna: model, scaler|None, iso, metrics, (train, valid, test)
    """
    import os, joblib
    import numpy as np

    df = dataset.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    n = len(df)
    if n < 50:
        raise ValueError(f"Dataset massa petit per entrenar (n={n}).")

    # --- splits temporals
    cut80 = int(0.80 * n); cut90 = int(0.90 * n)
    train = df.iloc[:cut80].copy()
    valid = df.iloc[cut80:cut90].copy()
    test  = df.iloc[cut90:].copy()

    # --- filtre anti-leak: treu odds/edge/kelly/fair_odds/targets/ids/temps
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
            cl.startswith("edge") or
            cl.startswith("kelly") or
            ("fair_odds" in cl) or
            ("unit_return" in cl) or
            ("best_side" in cl) or
            ("best_edge" in cl)
        )

    if model_cols:
        base = [c for c in model_cols if c in df.columns]
    else:
        # agafa numèriques/booleans com a punt de partida
        base = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()

    model_cols_used = [c for c in base if not _is_bad_feature(c)]
    if "y_home_win" in model_cols_used:
        model_cols_used.remove("y_home_win")

    if not model_cols_used:
        raise ValueError("Sense features vàlides després del filtre anti-leak.")

    # desa la llista de columnes
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "model_columns.txt"), "w") as f:
        f.write("\n".join(model_cols_used))
    leak = [c for c in model_cols if ("odds" in c.lower() or c.lower().startswith(("edge","kelly")) or "fair_odds" in c.lower())]
    if leak:
        st.error(f"Model columns tenen leak d'odds/features prohibides: {leak}. Reentrena amb filtre anti-leak.")


    # --- entrenament
    y_tr = train["y_home_win"].values
    y_va = valid["y_home_win"].values

    X_tr = train[model_cols_used].fillna(0.0).values
    X_va = valid[model_cols_used].fillna(0.0).values

    if use_lgb and HAS_LGB and len(train):
        dtr = lgb.Dataset(X_tr, label=y_tr)
        dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
        params = dict(
            objective='binary', metric='binary_logloss',
            learning_rate=0.05, num_leaves=63,
            feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
            min_data_in_leaf=50, seed=2026, verbose=-1, force_row_wise=True
        )
        booster = lgb.train(
            params, dtr, num_boost_round=5000,
            valid_sets=[dtr, dva], valid_names=['train','valid'],
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

    # --- calibratge isotònic sobre VALID
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_va, y_va)

    # --- avaluació en TEST
    X_te = test[model_cols_used].fillna(0.0).values
    if use_lgb and HAS_LGB and isinstance(model, lgb.basic.Booster):
        p_te = model.predict(X_te, num_iteration=model.best_iteration)
    else:
        X_tes = scaler.transform(X_te)
        p_te = model.predict_proba(X_tes)[:, 1]
    y_te = test["y_home_win"].values

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
        n_features=int(len(model_cols_used))
    )

    # --- desa models i calibrador
    if use_lgb and HAS_LGB and isinstance(model, lgb.basic.Booster):
        joblib.dump(model, os.path.join(OUT_DIR, "model_lightgbm.pkl"))
        scaler_path = None
    else:
        joblib.dump(model, os.path.join(OUT_DIR, "model_logistic.pkl"))
        joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.pkl"))
        scaler_path = os.path.join(OUT_DIR, "scaler.pkl")
    joblib.dump(iso, os.path.join(OUT_DIR, "calibrator_isotonic.pkl"))

    # --- info útil (en consola)
    if model_cols:
        dropped = [c for c in model_cols if c not in model_cols_used]
        if dropped:
            print(f"[train_models] Dropped {len(dropped)} features per anti-leak: {dropped[:12]}{' ...' if len(dropped)>12 else ''}")
    print(f"[train_models] Features usades: {len(model_cols_used)} | Guardades a model_columns.txt")

        # --- desa preds del TEST per a "Offline backtest" ---
    try:
        preds_test = test[["match_id","date","player_a_id","player_b_id"]].copy()
    except Exception:
        # si no tens aquestes columnes exactes, ajusta al que tinguis
        preds_test = test[["match_id","date"]].copy()
    preds_test["p_home_win"] = np.clip(p_te, 1e-6, 1-1e-6)
    preds_test["y_home_win"] = y_te
    os.makedirs(OUT_DIR, exist_ok=True)
    #preds_test["date"] = pd.to_datetime(preds_test["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    preds_test.to_csv(os.path.join(OUT_DIR, "preds_test.csv"), index=False)

    # >>> GUARDAR MÈTRIQUES D'ENTRENAMENT <<<
    import json, time
    metrics_to_save = metrics.copy()
    metrics_to_save["timestamp"] = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    metrics_to_save["model_type"] = model.__class__.__name__
    with open(os.path.join(OUT_DIR, "train_metrics.json"), "w") as f:
        json.dump(metrics_to_save, f, indent=2)


    return model, (None if (use_lgb and HAS_LGB and isinstance(model, lgb.basic.Booster)) else scaler), iso, metrics, (train, valid, test)


def predict_upcoming(model, scaler, iso, dataset: pd.DataFrame, model_cols: List[str], matches: pd.DataFrame) -> pd.DataFrame:
    today = pd.Timestamp.today().normalize()

    m = matches.copy()

    # --- 1) pending/undecided robust (NaN o string buit o "nan") ---
    if "winner_id" in m.columns:
        w = m["winner_id"]
        w_str = w.astype(str).str.strip().str.lower()
        undecided = w.isna() | (w_str.eq("")) | (w_str.eq("nan"))
    else:
        undecided = pd.Series(True, index=m.index)

    # --- 2) data de referència: match_date (prioritat) i fallback a date ---
    md = pd.to_datetime(m["match_date"], errors="coerce") if "match_date" in m.columns else pd.Series(pd.NaT, index=m.index)
    d  = pd.to_datetime(m["date"], errors="coerce")       if "date" in m.columns       else pd.Series(pd.NaT, index=m.index)
    ref_date = md.fillna(d)

    # upcoming si: no decidit o ref_date >= today
    future = ref_date.notna() & (ref_date >= today)
    cand_ids = m[undecided | future]["match_id"].astype(str)

    ds = dataset[dataset["match_id"].astype(str).isin(set(cand_ids))].copy()
    if not len(ds):
        return pd.DataFrame()

    # --- 3) predict ---
    if hasattr(model, "predict_proba"):
        if scaler is not None:
            X = scaler.transform(ds[model_cols].fillna(0.0))
        else:
            X = ds[model_cols].fillna(0.0).values
        p = model.predict_proba(X)[:, 1]
    else:
        X = ds[model_cols].fillna(0.0).values
        try:
            p = model.predict(X, num_iteration=getattr(model, "best_iteration", None))
        except TypeError:
            p = model.predict(X)

    if iso is not None:
        p = iso.transform(p)

    out = ds[["match_id","date"]].copy()
    out["p_home_win"] = np.clip(p, 1e-6, 1 - 1e-6)

    # --- 4) retorna també match_date i, si existeix, reemplaça out["date"] per match_date ---
    if "match_date" in m.columns:
        m_dates = m[["match_id","match_date"]].copy()
        m_dates["match_date"] = pd.to_datetime(m_dates["match_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        out = out.merge(m_dates, on="match_id", how="left")

        # Prioritza match_date com a 'date' de sortida (perquè tot el pipeline/lògica usa 'date')
        out["date"] = np.where(out["match_date"].notna() & (out["match_date"] != ""), out["match_date"], out["date"])

    return out


def _get_home_away(e):
    # Prefer home_team/away_team if present; fallback to teams[]
    home = e.get("home_team"); away = e.get("away_team")
    if home and away:
        return str(home), str(away)
    teams = e.get("teams", [])
    if isinstance(teams, list) and len(teams) == 2:
        a, b = sorted(map(str, teams))
        return a, b
    return None, None

def fixtures_from_oddsapi_markets(
    api_key: str,
    sports_keys: list[str] | None = None,
    regions: list[str] | None = None,
    markets: list[str] | tuple[str, ...] = ("h2h",),
    odds_mode: str = "best",
    per_key_regions: dict[str, list[str]] | None = None,
):
    import pandas as pd

    # Accés als índexs de noms
    MAP_FULL   = st.session_state.get("MAP_FULL", {})
    MAP_ALIAS  = st.session_state.get("MAP_ALIAS", {})
    INDEX_LAST = st.session_state.get("INDEX_LAST", {})

    def resolve_name_for_feed(s):
        pid, mode, conf = resolve_player_name(s, MAP_FULL, MAP_ALIAS, INDEX_LAST)
        if not pid:
            return (f"UNMATCHED:{s}", mode, conf)
        return (pid, mode, conf)

    # helper sports_keys
    def _get_tennis_keys():
        code, data, _ = get_json("/sports", {"apiKey": api_key})
        if code != 200 or not isinstance(data, list):
            return []
        return [s.get("key") for s in data if "tennis" in (s.get("key",""))]

    if sports_keys is None:
        sports_keys = _get_tennis_keys()
    if not sports_keys:
        return pd.DataFrame(columns=[
            "match_id","sport_key","region","date","tournament",
            "player_a_id","player_b_id","player_a_name","player_b_name",
            "odds_home","odds_away","n_books","resolve_a_mode","resolve_b_mode","resolve_a_conf","resolve_b_conf",
            "level","round","best_of_5","surface","indoor"
        ])

    if regions is None or not len(regions):
        regions = ["eu"]
    if not any(m.lower()=="h2h" for m in markets):
        markets = tuple(list(markets)+["h2h"])

    rows = []
    for sk in sports_keys:
        reg_list = per_key_regions.get(sk, regions) if per_key_regions else regions
        for reg in reg_list:
            code, data, _ = get_json(
                f"/sports/{sk}/odds",
                {"regions": reg, "markets": ",".join(markets), "oddsFormat": "decimal", "apiKey": api_key}
            )
            if code != 200 or not isinstance(data, list) or not data:
                continue

            for e in data:
                a_name = (e.get("home_team") or "").strip()
                b_name = (e.get("away_team") or "").strip()

                # resolució a IDs
                a_id, a_mode, a_conf = resolve_name_for_feed(a_name)
                b_id, b_mode, b_conf = resolve_name_for_feed(b_name)

                # recopilem preus h2h
                home_prices, away_prices = [], []
                for bk in (e.get("bookmakers") or []):
                    for mkt in (bk.get("markets") or []):
                        if mkt.get("key") != "h2h": 
                            continue
                        outs = mkt.get("outcomes") or []
                        # lookup robust pel nom "normalitzat"
                        norm_map = {}
                        for o in outs:
                            nm = _norm_name(o.get("name",""))
                            pr = o.get("price")
                            try: pr = float(pr)
                            except: pr = None
                            if nm and pr: norm_map[nm] = pr
                        an = _norm_name(a_name); bn = _norm_name(b_name)
                        if an in norm_map and bn in norm_map:
                            home_prices.append(norm_map[an]); away_prices.append(norm_map[bn])
                            continue
                        # fallback literal
                        lit = { (o.get("name") or "").strip(): o.get("price") for o in outs if o.get("price") }
                        if a_name in lit and b_name in lit:
                            try:
                                home_prices.append(float(lit[a_name])); away_prices.append(float(lit[b_name]))
                            except Exception:
                                pass

                if not home_prices or not away_prices:
                    continue

                if str(odds_mode).lower()=="average":
                    oh = sum(home_prices)/len(home_prices)
                    oa = sum(away_prices)/len(away_prices)
                else:
                    oh = max(home_prices); oa = max(away_prices)

                rows.append(dict(
                    match_id=f"oddsapi_{e.get('id')}",
                    sport_key=sk,
                    region=reg,
                    date=str(e.get("commence_time",""))[:10],
                    tournament=e.get("sport_title","OddsAPI Tennis"),
                    player_a_id=a_id, player_b_id=b_id,
                    player_a_name=a_name, player_b_name=b_name,
                    resolve_a_mode=a_mode, resolve_b_mode=b_mode,
                    resolve_a_conf=round(a_conf, 2), resolve_b_conf=round(b_conf, 2),
                    odds_home=oh, odds_away=oa, n_books=len(home_prices),
                    level="A", round="", best_of_5=0, surface="hard", indoor=0,
                ))

    return pd.DataFrame(rows)

def estimate_odds_credits(markets, try_all_regions, n_keys):
    n_regions = 4 if try_all_regions else 1
    n_markets = len(markets) if isinstance(markets, (list, tuple)) else 1
    return int(n_regions * n_markets * max(0, n_keys))

def _norm(s: str) -> str:
    s = str(s or "").lower()
    try: s = s.encode("ascii","ignore").decode()
    except: pass
    for ch in "-_":
        s = s.replace(ch, " ")
    return " ".join(s.split())

# sinònims perquè TML i l’API usen noms diferents
_TOURNEY_SYNONYMS = {
    "australian open": ["australian", "melbourne", "aus open"],
    "french open": ["roland garros", "paris (gs)", "french open", "paris gs"],
    "wimbledon": ["wimbledon", "london"],
    "us open": ["us open", "new york"],
    "indian wells": ["indian wells"],
    "miami open": ["miami"],
    "madrid open": ["madrid"],
    "italian open": ["rome", "roma", "italian open"],
    "monte carlo masters": ["monte carlo"],
    "canadian open": ["canadian", "toronto", "montreal"],
    "cincinnati open": ["cincinnati"],
    "shanghai masters": ["shanghai"],
    "paris masters": ["paris masters", "bercy"],
    "dubai": ["dubai"],
    "qatar open": ["doha", "qatar"],
    "china open": ["beijing", "china open"],
    "wuhan open": ["wuhan"],
}

def tml_active_names(today_only: bool = True) -> list[str]:
    """Extreu noms de torneig actius de TML ongoing_tourneys.csv considerant finestra setmanal."""
    try:
        df = fetch_ongoing_csv()
    except Exception:
        return []
    if "tourney_date" in df.columns:
        df["tourney_date"] = _parse_yyyymmdd_series(df["tourney_date"])
        if today_only:
            today = pd.Timestamp.today().normalize()
            # finestra setmana (quali + finals)
            mask = (df["tourney_date"] <= today) & (today <= df["tourney_date"] + pd.Timedelta(days=9))
            df = df[mask]
    names = sorted(set(df.get("tourney_name", pd.Series(dtype=str)).dropna().map(_norm)))
    return names


def map_tml_to_sport_keys(api_key: str, tml_names: list[str]) -> list[str]:
    """Retorna sport_keys de tennis que matchen amb els noms TML (amb sinònims)."""
    if not tml_names:
        return []
    sports, _ = oddsapi_sports(api_key)
    tennis = [s for s in sports if "tennis" in s.get("key","")]
    keys = []
    # Prepara llistes de paraules clau
    wanted = set(tml_names)
    for base, syns in _TOURNEY_SYNONYMS.items():
        if any(k in " ".join(tml_names) for k in syns+[base]):
            wanted.add(base)
            wanted.update(syns)
    wanted = {_norm(w) for w in wanted}
    # Fes match contra sport_title i key
    for s in tennis:
        title = _norm(s.get("sport_title",""))
        key   = _norm(s.get("key",""))
        if any(w in title or w in key for w in wanted):
            keys.append(s["key"])
    return sorted(set(keys))

def suggest_region_for_tourney_name(name: str) -> str:
    """Heurística molt simple per triar regió de llibres."""
    n = _norm(name)
    if any(k in n for k in ["australian", "brisbane", "adelaide", "sydney", "melbourne"]):
        return "au"
    if any(k in n for k in ["us open", "new york", "indian wells", "miami", "cincinnati"]):
        return "us"
    if any(k in n for k in ["toronto", "montreal", "canadian"]):
        return "us"
    if any(k in n for k in ["wimbledon", "london"]):
        return "uk"
    if any(k in n for k in ["qatar", "doha", "dubai"]):
        return "uk"
    if any(k in n for k in ["shanghai", "beijing", "wuhan"]):
        return "uk"  # UK sol tenir bons llibres per Àsia; canvia-ho si vols
    # per defecte, europa continental
    return "eu"

def per_key_region_from_tml(keys: list[str]) -> dict[str,str]:
    """Assigna una regió a cada sport_key segons el nom."""
    m = {}
    for k in keys:
        # usa el nom “bonic” de l’API per triar regió
        title = k.replace("tennis_atp_","").replace("tennis_wta_","").replace("_"," ")
        m[k] = suggest_region_for_tourney_name(title)
    return m

# ========= PREDICTION LOGGING & RESULT BACKFILL =========
#from pathlib import Path
#APP_DIR = Path(__file__).resolve().parent
#DATA_DIR = APP_DIR / "data"
#OUT_DIR = DATA_DIR / "outputs"
#OUT_DIR.mkdir(parents=True, exist_ok=True)
#LOG_PATH = OUT_DIR / "predictions_log.csv"

def _now_utc_iso():
    return pd.Timestamp.utcnow().isoformat(timespec="seconds") + "Z"

def _to_str(x):
    return "" if pd.isna(x) else str(x)

def log_predictions(out_df: pd.DataFrame, source: str, model_name: str = "unknown", skip_no_bet: bool=True):
    """
    Desa prediccions al log. Si skip_no_bet=True, NO desa files sense aposta
    (best_side == 'no bet' o stake == 0 o best_edge <= 0).
    """
    if out_df is None or not len(out_df):
        return False

    df = out_df.copy()

    # normalitza columnes clau
    for c in ["best_side","stake_% (half kelly)","best_edge"]:
        if c not in df.columns: df[c] = np.nan

    if skip_no_bet:
        bs = df["best_side"].astype(str).str.lower()
        stake = pd.to_numeric(df["stake_% (half kelly)"], errors="coerce").fillna(0)/100.0
        bedge = pd.to_numeric(df["best_edge"], errors="coerce")
        mask = (
            bs.str.startswith("home") | bs.str.startswith("away")
        ) & (stake > 0) & (bedge > 0)
        df = df[mask].copy()
        if not len(df):
            return False  # no hi ha res a desar

    # camps mínims
    for c in ["match_id","date","player_a_id","player_b_id","player_a_name","player_b_name",
              "p_home_win","p_away_win","odds_home","odds_away",
              "edge_home","edge_away","best_side","best_edge","stake_% (half kelly)"]:
        if c not in df.columns:
            df[c] = np.nan if c not in {"match_id","date","player_a_id","player_b_id","player_a_name","player_b_name"} else ""

    df["pred_time_utc"] = pd.Timestamp.utcnow().isoformat(timespec="seconds") + "Z"
    df["source"] = source
    df["model_name"] = model_name

    # Normalitza 'date' a 'YYYY-MM-DD' (evita Nones)
    if "date" in df.columns:
        d = pd.to_datetime(df["date"], errors="coerce")
        df["date"] = np.where(
            d.notna(),
            d.dt.strftime("%Y-%m-%d"),
            df["date"].astype(str).fillna("")
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if LOG_PATH.exists():
        base = pd.read_csv(LOG_PATH, dtype=str)
        pd.concat([base, df], ignore_index=True).to_csv(LOG_PATH, index=False)
    else:
        df.to_csv(LOG_PATH, index=False)

    # --- mensual (opcional; robust) ---
    try:
        m = pd.Timestamp.utcnow().strftime("%Y-%m")  # p.ex. '2025-10'
        dfm = pd.read_csv(LOG_PATH, dtype=str)
    
        if "date" in dfm.columns:
            # assegura string i filtra per prefix del mes actual
            dates = dfm["date"].astype(str).fillna("")
            month_mask = dates.str.startswith(m)  # 'YYYY-MM'
            df_month = dfm[month_mask].copy()
        else:
            df_month = dfm.iloc[0:0].copy()
    
        (OUT_DIR / f"predictions_log_{m}.csv").write_text(
            df_month.to_csv(index=False),
            encoding="utf-8"
        )
    except Exception as e:
        # opcional: ignora o mostra un avís suau
        # st.warning(f"No s'ha pogut escriure el log mensual: {e}")
        pass

    return True



# --- Backfill de resultats sense dependència de fetch_tml ---
def _safe_str(x):
    s = "" if pd.isna(x) else str(x)
    return s.strip()

def _pair_key(a, b):  # per si no el tens
    a = "" if a is None else str(a)
    b = "" if b is None else str(b)
    return "|".join(sorted([a.strip(), b.strip()]))

def _pair_key_name(a, b):
    return _pair_key(_norm_name(a), _norm_name(b))

def _load_tml_recent(days_back: int = 60) -> pd.DataFrame:
    import os, pandas as pd
    dfs = []

    # 1) memòria
    try:
        if "df_all" in st.session_state:
            dfm = st.session_state["df_all"].copy()
            need = {"tourney_date","winner_id","loser_id","winner_name","loser_name"}
            if need.issubset(dfm.columns):
                dfs.append(dfm)
    except Exception:
        pass

    # 2) locals
    today = pd.Timestamp.today().normalize()
    start = today - pd.Timedelta(days=days_back+21)
    years = sorted({start.year, today.year})
    for y in years:
        for p in [
            os.path.join(DATA_DIR, f"{y}.csv"),
            os.path.join(DATA_DIR, f"tml_{y}.csv"),
            os.path.join(DATA_DIR, "tml_cache", f"{y}.csv"),
        ]:
            if os.path.exists(p):
                try: dfs.append(pd.read_csv(p))
                except Exception: pass

    # 3) fallback GitHub raw
    if not dfs:
        base = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master"
        for y in years:
            try:
                dfs.append(pd.read_csv(f"{base}/{y}.csv"))
            except Exception:
                pass

    if not dfs:
        return pd.DataFrame(columns=[
            "tourney_date","winner_id","loser_id","winner_name","loser_name",
            "pair_key","pair_key_name","tourney_date8"
        ])

    tml = pd.concat(dfs, ignore_index=True)
    tml["tourney_date"] = _parse_yyyymmdd_series(tml.get("tourney_date", pd.Series(dtype=str)))
    tml = tml.dropna(subset=["tourney_date"])

    for c in ["winner_id","loser_id","winner_name","loser_name"]:
        if c not in tml.columns: tml[c] = ""
        tml[c] = tml[c].astype(str)

    tml["pair_key"]      = tml.apply(lambda r: _pair_key(r["winner_id"], r["loser_id"]), axis=1)
    tml["pair_key_name"] = tml.apply(lambda r: _pair_key_name(r["winner_name"], r["loser_name"]), axis=1)
    tml["tourney_date8"] = pd.to_numeric(tml["tourney_date"].dt.strftime("%Y%m%d"), errors="coerce")
    return tml

def debug_backfill_pending(sample_n=30, days_back=90, date_tol_days=15):
    import pandas as pd
    if not os.path.exists(LOG_PATH):
        st.warning("No predictions_log.csv.")
        return
    log = pd.read_csv(LOG_PATH)
    tml = _load_tml_recent(days_back)
    if tml.empty:
        st.error("TML buit. Revisa Refresh & Train o connexió a GitHub.")
        return
    if "y_home_win" in log.columns:
        pend = log[~log["y_home_win"].isin([0,1])].copy()
    else:
        pend = log.copy()
    if not len(pend):
        st.info("No hi ha files pendents.")
        return
    pend["_date"] = pd.to_datetime(pend["date"], errors="coerce")
    tol = pd.Timedelta(days=date_tol_days)
    rows = []
    for _, r in pend.head(sample_n).iterrows():
        d0 = r["_date"]
        cand = tml.copy()
        if pd.notna(d0):
            mask = (cand["tourney_date"] >= d0 - tol) & (cand["tourney_date"] <= d0 + tol)
            cand = cand[mask] if mask.any() else cand
        # compteigs
        hit_id   = len(cand[cand["pair_key"] == _pair_key(str(r.get("player_a_id","")), str(r.get("player_b_id","")))])
        pk_name  = _pair_key_name(str(r.get("player_a_name","")), str(r.get("player_b_name","")))
        hit_name = len(cand[cand["pair_key_name"] == pk_name])
        rows.append(dict(
            date=r.get("date",""),
            a=r.get("player_a_name",""),
            b=r.get("player_b_name",""),
            a_id=r.get("player_a_id",""),
            b_id=r.get("player_b_id",""),
            hits_by_id=hit_id,
            hits_by_name=hit_name,
            date_ok=pd.notna(d0),
        ))
    st.dataframe(pd.DataFrame(rows))


def backfill_results_from_tml(days_back: int = 60, date_tol_days: int = 15):
    """
    Emplena al log: winner_id, y_home_win, decided_time_utc, unit_return, i
    (nou) decided_src_date amb la data EXACTA del partit TML que hem fet servir.
    Només backfilleja si |tourney_date - date| <= date_tol_days.
    """
    if not LOG_PATH.exists():
        return False, "No predictions_log.csv yet."

    log = pd.read_csv(LOG_PATH)
    if not len(log):
        return True, "Log is empty."

    # indexos pendents (sense resultat)
    if "y_home_win" in log.columns:
        pending_idx = log.index[log["y_home_win"].isna() | (log["y_home_win"]=="")]
    else:
        pending_idx = log.index

    # normalitza claus i dates
    for c in ["player_a_id","player_b_id","match_id","date"]:
        if c not in log.columns: log[c] = ""
    log["player_a_id"] = log["player_a_id"].astype(str)
    log["player_b_id"] = log["player_b_id"].astype(str)
    log["pair_key"]    = log.apply(lambda r: "|".join(sorted([str(r["player_a_id"]).strip(),
                                                              str(r["player_b_id"]).strip()])), axis=1)
    log["_date"]       = pd.to_datetime(log.get("date", pd.Series(dtype=str)), errors="coerce")

    # carrega TML recent
    tml = _load_tml_recent(days_back)   # la teva funció existent
    if tml.empty:
        return False, "No puc llegir TML de cap font (memòria/locals)."

    grp = tml.groupby("pair_key")
    tol = pd.Timedelta(days=date_tol_days)

    # assegura columnes de sortida
    for c in ["winner_id","y_home_win","decided_time_utc","unit_return","decided_src_date"]:
        if c not in log.columns:
            log[c] = np.nan

    filled = 0
    today = pd.Timestamp.today().normalize()

    for idx in pending_idx:
        row = log.loc[idx]
        pk  = row["pair_key"]
        if pk not in grp.groups:
            continue

        cand = grp.get_group(pk).copy()
        d0   = row["_date"]  # data del log (predicció)
        if pd.notna(d0):
            mask = (cand["tourney_date"] >= d0 - tol) & (cand["tourney_date"] <= d0 + tol)
            cand = cand[mask]
        # si no hi ha cap partit dins tolerància → NO backfill
        if not len(cand):
            continue

        # escull el més proper a d0 (si d0 és NaT, agafa el darrer passat)
        if pd.notna(d0):
            cand = cand.assign(_diff=(cand["tourney_date"] - d0).abs()).sort_values("_diff")
        else:
            cand = cand.sort_values("tourney_date", ascending=False)

        w    = str(cand.iloc[0]["winner_id"])
        m_dt = pd.to_datetime(cand.iloc[0]["tourney_date"], errors="coerce")
        if pd.isna(m_dt):   # seguretat
            continue
        # no omplim resultats futurs per error
        if m_dt > today:
            continue

        a = str(row["player_a_id"]).strip()
        b = str(row["player_b_id"]).strip()
        if w == a:
            y = 1
        elif w == b:
            y = 0
        else:
            # si tens matching per nom, podries afegir-lo aquí (opcional)
            continue

        log.at[idx, "winner_id"]        = w
        log.at[idx, "y_home_win"]       = y
        log.at[idx, "decided_time_utc"] = pd.Timestamp.utcnow().isoformat(timespec="seconds") + "Z"
        log.at[idx, "decided_src_date"] = m_dt.strftime("%Y-%m-%d")

        # retorn si seguíem la recomanació (unitats amb stake_%):
        bs = str(row.get("best_side","")).lower()
        oh = pd.to_numeric(row.get("odds_home"), errors="coerce")
        oa = pd.to_numeric(row.get("odds_away"), errors="coerce")
        stake_pct = pd.to_numeric(row.get("stake_% (half kelly)"), errors="coerce")
        stake = 0.0 if not np.isfinite(stake_pct) else float(stake_pct)/100.0

        ret = 0.0
        if bs.startswith("home") and np.isfinite(oh) and oh > 1:
            ret = (oh - 1.0) if y == 1 else -1.0
        elif bs.startswith("away") and np.isfinite(oa) and oa > 1:
            ret = (oa - 1.0) if y == 0 else -1.0
        else:
            ret = 0.0

        log.at[idx, "unit_return"] = stake * ret
        filled += 1

    log.to_csv(LOG_PATH, index=False)
    return True, f"Backfilled {filled} rows. Log stored at {LOG_PATH.name}"



NAME_MAP_PATH = os.path.join(OUT_DIR, "name_to_id.csv")

def _safe_str(x):
    s = "" if pd.isna(x) else str(x)
    return s.strip()

def build_name_id_map_from_tml(df_all: pd.DataFrame) -> dict:
    """
    De TML (winner_name/id, loser_name/id), crea un diccionari norm_name -> ATP player_id (str)
    triant l'ID més freqüent per cada nom normalitzat.
    """
    parts = []
    for nm, pid in [("winner_name","winner_id"), ("loser_name","loser_id")]:
        if nm in df_all.columns and pid in df_all.columns:
            tmp = df_all[[nm, pid]].dropna()
            tmp["norm"] = tmp[nm].map(_norm_name)
            tmp[pid] = tmp[pid].astype(str)
            parts.append(tmp[["norm", pid]])
    if not parts:
        return {}

    cat = pd.concat(parts, ignore_index=True)
    # ID més freqüent per nom normalitzat
    top = cat.groupby("norm")[pid].agg(lambda s: s.value_counts().index[0])
    # desa CSV per reutilitzar
    out = top.rename("player_id").to_frame()
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(NAME_MAP_PATH)
    return top.to_dict()

def load_name_id_map() -> dict:
    if os.path.exists(NAME_MAP_PATH):
        ser = pd.read_csv(NAME_MAP_PATH)
        # format: columns ['norm','player_id'] si guardem amb index=False; o ['norm','player_id'] si amb index reset
        if "norm" in ser.columns and "player_id" in ser.columns:
            return dict(zip(ser["norm"], ser["player_id"].astype(str)))
        # si guardat amb index com a primera columna:
        ser.columns = [c.lower() for c in ser.columns]
        if ser.shape[1] >= 2:
            return dict(zip(ser.iloc[:,0].astype(str), ser.iloc[:,1].astype(str)))
    return {}

import os, re, unicodedata
from collections import Counter, defaultdict

def _split_tokens(s: str):
    s = _norm_name(s); toks = s.split()
    return toks

def build_name_indices_from_tml(df_all):
    """
    Construeix:
      - MAP_FULL:    "carlos alcaraz" → "104745"
      - MAP_ALIAS:   "alcaraz" (multi cognom) / "c alcaraz" → "104745" (si és unívoc)
      - INDEX_LAST:  "alcaraz" → { "104745": freq, ... } (per cognom curt, pot ser ambigu)
    """
    if df_all is None or not len(df_all):
        return {}, {}, {}

    cols_ok = set(df_all.columns)
    if not {"winner_name","winner_id","loser_name","loser_id"} <= cols_ok:
        return {}, {}, {}

    full_counts   = defaultdict(Counter)  # norm_full -> Counter(pid)
    alias_counts  = defaultdict(Counter)  # alias (last_multi / initial_last) -> Counter(pid)
    last_counts   = defaultdict(Counter)  # last_token -> Counter(pid)

    def _feed(name, pid):
        if not isinstance(name, str): return
        pid = str(pid)
        toks = _split_tokens(name)
        if not toks: return
        norm_full = " ".join(toks)
        # last_multi: tot menys el primer token (gestiona "davidovich fokina")
        last_multi = " ".join(toks[1:]) if len(toks) >= 2 else ""
        last_token = toks[-1]
        initial_last = (toks[0][0] + " " + last_token) if toks and last_token else ""

        full_counts[norm_full][pid] += 1
        if last_multi:
            alias_counts[last_multi][pid] += 1
        if initial_last.strip():
            alias_counts[initial_last.strip()][pid] += 1
        if last_token:
            last_counts[last_token][pid] += 1

    for nm, pid in [("winner_name","winner_id"), ("loser_name","loser_id")]:
        sub = df_all[[nm, pid]].dropna()
        for name, p in zip(sub[nm].values, sub[pid].astype(str).values):
            _feed(name, p)

    def _pick(counter: Counter):
        # tria el més freqüent
        return counter.most_common(1)[0][0] if counter else None

    MAP_FULL   = {k: _pick(v) for k, v in full_counts.items()}
    MAP_ALIAS  = {k: _pick(v) for k, v in alias_counts.items()}
    INDEX_LAST = {k: dict(v) for k, v in last_counts.items()}
    return MAP_FULL, MAP_ALIAS, INDEX_LAST

def resolve_player_name(q: str, MAP_FULL, MAP_ALIAS, INDEX_LAST, min_conf_last: float = 0.55):
    """
    Resol 'q' cap a un player_id amb heurístiques:
      1) match exacte per nom complet normalitzat
      2) alias: "cognom(s)" (sense 1r nom) o "inicial cognom"
      3) cognom curt: si unívoc → OK; si múltiple → tria el més freqüent
         (si la confiança < min_conf_last, marca 'LOW_CONF')
    Retorna: (player_id or None, mode, confidence)
    """
    toks = _split_tokens(q)
    if not toks:
        return (None, "EMPTY", 0.0)

    # si la cadena comença amb inicial, la descartem per provar alias més net
    if len(toks) >= 2 and len(toks[0]) == 1:
        cand_input = " ".join(toks[1:])
    else:
        cand_input = " ".join(toks)

    # 1) FULL
    if cand_input in MAP_FULL:
        return (MAP_FULL[cand_input], "FULL", 1.0)

    # 2) ALIAS
    if cand_input in MAP_ALIAS:
        return (MAP_ALIAS[cand_input], "ALIAS", 0.9)

    # 3) LAST (cognom curt)
    last = toks[-1]
    bucket = INDEX_LAST.get(last, {})
    if not bucket:
        return (None, "UNMATCHED", 0.0)

    # tria el més freqüent; calcula confiança relativa
    c = Counter(bucket)
    pid, cnt = c.most_common(1)[0]
    conf = cnt / max(1, sum(c.values()))
    mode = "LAST_UNIQUE" if len(c) == 1 else ("LAST" if conf >= min_conf_last else "LAST_LOWCONF")
    return (pid, mode, conf)

def clean_log_df(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np, pandas as pd
    out = df.copy()

    # assegura columnes
    must_str = ["match_id","player_a_id","player_b_id","player_a_name","player_b_name","best_side","source","model_name"]
    for c in must_str:
        if c not in out.columns: out[c] = ""
        out[c] = out[c].astype(str)

    must_num = ["y_home_win","odds_home","odds_away","stake_% (half kelly)","unit_return","unit_return_calc","best_edge"]
    for c in must_num:
        if c not in out.columns: out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # dates
    if "date" not in out.columns: out["date"] = ""
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    # decided
    out["decided"] = out["y_home_win"].isin([0,1])

    # stake en unitats
    out["stake"] = (pd.to_numeric(out.get("stake_% (half kelly)"), errors="coerce").fillna(0.0) / 100.0)

    # (re)calcula unit_return_calc per coherència
    def realized_unit(row):
        import numpy as np
        if not bool(row["decided"]): return np.nan
        bs = str(row.get("best_side","")).lower()
        y  = row.get("y_home_win")
        st = row.get("stake", 0.0)
        oh = row.get("odds_home"); oa = row.get("odds_away")
        if st <= 0 or (bs == "" or bs == "no bet"): return 0.0
        if bs.startswith("home"):
            return st * ((oh - 1.0) if (pd.notna(oh) and oh>1 and y==1) else (-1.0 if pd.notna(oh) and oh>1 else 0.0))
        if bs.startswith("away"):
            return st * ((oa - 1.0) if (pd.notna(oa) and oa>1 and y==0) else (-1.0 if pd.notna(oa) and oa>1 else 0.0))
        return 0.0

    out["unit_return_calc"] = out.apply(realized_unit, axis=1)

    return out

def add_outcome_columns(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np, pandas as pd
    out = df.copy()
    # assegura camps bàsics
    for c in ["p_home_win","best_side","y_home_win"]:
        if c not in out.columns: out[c] = np.nan if c!="best_side" else ""
    if "decided" not in out.columns:
        out["decided"] = out["y_home_win"].isin([0,1])
    if "stake" not in out.columns:
        out["stake"] = pd.to_numeric(out.get("stake_% (half kelly)"), errors="coerce").fillna(0)/100.0

    # costat predit pel model (sense odds): home si p>=0.5
    out["pred_side"] = np.where(pd.to_numeric(out["p_home_win"], errors="coerce")>=0.5, "home", "away")

    # costat apostat (si stake>0), extret de best_side
    bs = out["best_side"].astype(str).str.lower()
    out["bet_side"] = np.where(out["stake"]>0,
                               np.where(bs.str.startswith("home"), "home",
                               np.where(bs.str.startswith("away"), "away", "")),
                               "")

    # encert de la predicció (sempre que hi hagi resultat)
    y = pd.to_numeric(out["y_home_win"], errors="coerce")
    pred_ok = np.where(~out["decided"], np.nan,
                       np.where(out["pred_side"]=="home", (y==1).astype(float),
                       np.where(out["pred_side"]=="away", (y==0).astype(float), np.nan)))
    out["pred_correct"] = pred_ok

    # encert de la recomanació (només si s’ha apostat)
    bet_ok = np.where(~out["decided"] | (out["bet_side"]==""), np.nan,
                      np.where(out["bet_side"]=="home", (y==1).astype(float),
                      np.where(out["bet_side"]=="away", (y==0).astype(float), np.nan)))
    out["bet_correct"] = bet_ok

    # emoji: si hi ha aposta usem bet_correct; sinó pred_correct; si pendent ⌛
    def _emoji(row):
        if not bool(row["decided"]):
            return "⌛"
        v = row["bet_correct"]
        if pd.notna(v):
            return "✅" if v==1 else "❌"
        v = row["pred_correct"]
        if pd.notna(v):
            return "✅" if v==1 else "❌"
        return "⌛"
    out["result_emoji"] = out.apply(_emoji, axis=1)
    return out

def decide_with_filters_PREV(p_h, oh, p_a, oa, kh, ka, edge_h, edge_a,
                        min_edge=0.02, min_prob_margin=0.025,
                        fav_cutoff=2.0, min_kelly_fav=0.05, min_kelly_dog=0.05,
                        stake_cap_pct=5.0):
    """
    Devuelve: (best_side, best_edge, stake_half_pct, reason)
    - best_side: "home(A)" | "away(B)" | "no bet"
    - best_edge: float o np.nan
    - stake_half_pct: porcentaje recomendado (½ Kelly) ya capado [0..stake_cap_pct]
    - reason: texto breve explicando por qué no pasa filtros (si aplica)
    """
    import numpy as np
    # No odds -> no bet pero mostrar prob/fair odds fuera
    valid_oh = isinstance(oh, (int,float)) and np.isfinite(oh) and oh > 1.0
    valid_oa = isinstance(oa, (int,float)) and np.isfinite(oa) and oa > 1.0
    if not (valid_oh or valid_oa):
        return "no bet", np.nan, 0.0, "no_odds"

    # Edge positivo en alguna banda
    e_h = edge_h if isinstance(edge_h,(int,float)) else np.nan
    e_a = edge_a if isinstance(edge_a,(int,float)) else np.nan
    if (not np.isfinite(e_h)) and (not np.isfinite(e_a)):
        return "no bet", np.nan, 0.0, "no_edge_values"

    # Elegimos por mayor edge (>0). Si ambos ≤ 0 → no bet.
    cand_side = None; cand_edge = None; cand_p = None; cand_k = None; cand_odds = None
    if np.nanmax([e_h, e_a]) <= 0:
        return "no bet", float(np.nanmax([e_h, e_a])), 0.0, "edge≤0"

    if np.isfinite(e_h) and (e_h >= (e_a if np.isfinite(e_a) else -1e9)):
        cand_side, cand_edge, cand_p, cand_k, cand_odds = "home(A)", e_h, p_h, kh, oh
    else:
        cand_side, cand_edge, cand_p, cand_k, cand_odds = "away(B)", e_a, p_a, ka, oa

    # 1) margen mínimo |p-0.5|
    if abs(cand_p - 0.5) < min_prob_margin:
        return "no bet", cand_edge, 0.0, "low_prob_margin"

    # 2) edge mínimo
    if cand_edge < min_edge:
        return "no bet", cand_edge, 0.0, "edge_below_min"

    # 3) Kelly mínimo (depende de favorito vs dog)
    is_fav = cand_odds < fav_cutoff
    k_req = min_kelly_fav if is_fav else min_kelly_dog
    if cand_k < k_req:
        return "no bet", cand_edge, 0.0, f"kelly<{k_req:.2f} ({'fav' if is_fav else 'dog'})"

    # stake ½ Kelly en %, capado
    stake_half = max(min(cand_k * 50.0, stake_cap_pct), 0.0)
    return cand_side, cand_edge, stake_half, ""

def decide_with_filters(
    p_h, oh, p_a, oa, kh, ka, eh, ea,
    *,
    # llindars
    fav_p_min=0.60,          # si p >= això → el considerem favorit
    dog_p_max=0.35,          # si p <= això → el considerem underdog
    edge_min=0.025,          # mínim edge pur (p - 1/odds)
    margin_min=0.025,        # mínim marge vs implied prob (p - 1/odds) permes per apostar
    kelly_min=0.02,          # Kelly mínima per considerar-ho “val real”
    prob_gap_min=0.05,       # si |p-0.5| < això → és coinflip
    # caps de stake (½-Kelly, en % banca)
    cap_fav=5.0, cap_dog=2.5, cap_mid=1.5, cap_global=8.0,
    # multiplicadors de risc
    risk_fav=1.00, risk_mid=0.75, risk_dog=0.55,
    risk_book_dog=0.60       # penal extra si apostes al “no favorit” per quotes (odd ≥ 2.0)
):
    """
    Retorna: (best_side, best_edge, stake_half, reason)

    - best_side: "home(A)" | "away(B)" | "no bet"
    - best_edge: edge del costat triat (o NaN)
    - stake_half: % banca recomanada (ja amb 1/2 Kelly, risk_mult i caps)
    - reason   : etiqueta breu del bucket/raó
    """
    import numpy as np

    def _finite(x):
        return isinstance(x, (int, float, np.floating)) and np.isfinite(x)

    # 0) si no hi ha dues quotes vàlides, no podem avaluar
    if not (_finite(oh) and oh > 1 and _finite(oa) and oa > 1):
        return "no bet", np.nan, 0.0, "no_odds"

    # 1) triem quin costat candidate mirem (el de més edge vàlid)
    home_ok = _finite(eh)
    away_ok = _finite(ea)

    if home_ok and away_ok:
        if eh >= ea:
            side = "home(A)"; edge = eh; p = p_h; k = kh; o = oh
        else:
            side = "away(B)"; edge = ea; p = p_a; k = ka; o = oa
    elif home_ok:
        side = "home(A)"; edge = eh; p = p_h; k = kh; o = oh
    elif away_ok:
        side = "away(B)"; edge = ea; p = p_a; k = ka; o = oa
    else:
        return "no bet", np.nan, 0.0, "no_edge"

    # 2) filtres durs edge / kelly
    if edge < edge_min:
        return "no bet", edge, 0.0, "edge_below_min"

    if (not _finite(k)) or (k < kelly_min):
        return "no bet", edge, 0.0, "kelly_below_min"

    # 3) marge vs implied prob de la quota del costat triat
    implied = 1.0 / o if _finite(o) and o > 0 else np.nan
    margin  = p - implied if _finite(implied) else np.nan

    # coinflip safety:
    # si és gairebé 50/50 (|p-0.5| petit) i a sobre el marge no és prou gran,
    # no entrem
    if abs(p - 0.5) < prob_gap_min:
        if (not _finite(margin)) or (margin < max(edge_min, margin_min)):
            return "no bet", edge, 0.0, "low_prob_margin"

    # marge mínim absolut per protegir soroll
    if (not _finite(margin)) or (margin < margin_min):
        return "no bet", edge, 0.0, "margin_below_min"

    # 4) bucket de risc i caps
    if p >= fav_p_min:
        bucket = "fav"
        cap = cap_fav
        risk_mult = risk_fav
    elif p <= dog_p_max:
        bucket = "dog"
        cap = cap_dog
        risk_mult = risk_dog
    else:
        bucket = "mid"
        cap = cap_mid
        risk_mult = risk_mid

    # penal extra si segons la quota vas amb l’underdog del book (odd ≥ 2.0)
    is_book_dog = _finite(o) and (o >= 2.0)
    if is_book_dog:
        risk_mult *= risk_book_dog
        bucket += "_bookdog"

    # stake recomanat (% banca) = 50 * Kelly * multiplicador de risc
    stake_pct = 50.0 * k * risk_mult
    stake_pct = float(max(0.0, min(stake_pct, cap, cap_global)))

    return side, edge, stake_pct, bucket



def alt_market_hint(
    p_h, oh, p_a, oa, edge_h, edge_a, best_of_5: bool,
    margin_min: float, edge_min: float, alt_edge_soft: float = 0.0,
    fav_p_min: float = 0.60, dog_p_max: float = 0.35, c_edge_soft: float = 0.02
) -> str:
    """
    Retorna suggeriments només quan NO hi ha ML (max_edge < edge_min) i hi ha una mica de senyal.
    Escenaris:
      A) Dog amb opcions (p_dog ∈ (dog_p_max, ~0.47) i partit igualat) → +jocs / over sets
      B) Model igualat però llibre decanta l’altre → over jocs / petit handicap al infravalorat
      C) *Favorit clar per model* (p_fav ≥ fav_p_min) però sense valor ML,
         i *edge del favorit* ∈ [c_edge_soft, edge_min) → -jocs o 2–0/3–0
    """
    import numpy as np
    def _ok(x): return isinstance(x, (int, float)) and np.isfinite(x)

    eh = edge_h if _ok(edge_h) else float("-inf")
    ea = edge_a if _ok(edge_a) else float("-inf")
    max_edge = max(eh, ea)

    # Gating general: només si no dona per ML
    if not (max_edge < edge_min):
        return ""

    hints = []

    # ----------------------
    # Escenari A: dog amb opcions
    # ----------------------
    if _ok(oh) and _ok(oa):
        dog_is_home = oh > oa
        p_dog = float(p_h) if dog_is_home else float(p_a)
        if (p_dog > dog_p_max) and (p_dog < 0.47) and (abs(p_dog - 0.5) < max(margin_min, 0.07)):
            hints.append("Underdog con opciones: mira handicap +juegos (+3.5/+4.5) o over sets.")

    # ----------------------
    # Escenari B: igualat per model però llibre decanta l'altre
    # ----------------------
    choose_home = eh >= ea
    p_cand = float(p_h) if choose_home else float(p_a)
    book_fav_home = (_ok(oh) and _ok(oa) and oh < oa)
    book_fav_away = (_ok(oh) and _ok(oa) and oa < oh)
    if abs(p_cand - 0.5) < margin_min:
        if (choose_home and book_fav_away) or ((not choose_home) and book_fav_home):
            if not best_of_5:
                hints.append("Partido igualado por modelo pero libro decanta: prueba over juegos (22.5) o pequeño handicap al infravalorado.")
            else:
                hints.append("Partido igualado por modelo pero libro decanta: prueba over juegos (37.5–39.5) o pequeño handicap al infravalorado.")

    # ----------------------
    # Escenari C (revisat): favorit del model amb edge suficient per SUGERIR
    # ----------------------
    p_fav  = max(p_h, p_a)
    fav_is_home = (p_h >= p_a)
    fav_edge = eh if fav_is_home else ea

    if (p_fav >= fav_p_min) and _ok(fav_edge) and (fav_edge >= c_edge_soft) and (fav_edge < edge_min):
        if not best_of_5:
            if p_fav >= 0.75:
                hints.append("Favorito claro: considera ganar 2–0 (BO3) o -juegos (-3.5).")
            else:
                hints.append("Favorito sin valor ML: considera -juegos (-2.5/-3.5).")
        else:
            if p_fav >= 0.80:
                hints.append("Favorito claro (BO5): considera 3–0 o -juegos.")
            else:
                hints.append("Favorito sin valor ML (BO5): considera -juegos.")

    # Si no s'activa res, no suggerim
    return " | ".join(dict.fromkeys(hints))





# -------------------------------
# UI
# -------------------------------
st.title("🎾 PROVES OSCAR TENNIS)")

with st.sidebar:
    
    # ============================================================
    # TOGGLE (SIDEBAR) — EXCLOURE TORNEJOS PER TEST (cutoff inclusiu)
    # ============================================================
    st.header("Training data cutoff (testing)")
    
    TRAIN_CUTOFF_ENABLED = st.checkbox(
        "Exclude tournaments from cutoff date (inclusive)",
        value=False
    )
    TRAIN_CUTOFF_DATE = st.date_input(
        "Cutoff date (inclusive)",
        value=pd.to_datetime("2026-01-01").date()
    )
    
    st.divider()
    
    MATCH_CUTOFF_ENABLED = st.checkbox(
        "Exclude matches by match_date (inclusive)",
        value=False
    )
    MATCH_CUTOFF_DATE = st.date_input(
        "Match cutoff (inclusive)",
        value=pd.to_datetime("2026-01-01").date(),
        key="match_cutoff_date"
    )
    
    CUTOFF_MODE = st.radio(
        "Cutoff mode",
        options=["Tournament-level", "Match-level", "Both (stricter)"],
        index=1
    )

    
    st.header("Config")
    y1, y2 = st.columns(2)
    YEAR_START = y1.number_input("From year", value=2015, min_value=1968, max_value=2026, step=1)
    YEAR_END   = y2.number_input("To year", value=2026, min_value=1968, max_value=2026, step=1)
    INCLUDE_ONGOING = st.checkbox("Include ongoing_tourneys", value=True, help="Adds latest ongoing matches into 2026")
    USE_LGB = st.checkbox("Use LightGBM if available", value=False)
    
    st.subheader("Bet filters")
    #MIN_EDGE = st.number_input("Min edge (prob - 1/odds)", 0.0, 0.2, 0.02, 0.005)
    #MIN_PROB_MARGIN = st.number_input("Min |p - 0.5|", 0.0, 0.5, 0.05, 0.01)
    #FAV_CUTOFF = st.number_input("Límite favorito (odds < x)", 1.5, 3.0, 2.0, 0.05)
    #MIN_KELLY_FAV = st.number_input("Min Kelly favorito", 0.0, 1.0, 0.08, 0.01)
    #MIN_KELLY_DOG = st.number_input("Min Kelly underdog", 0.0, 1.0, 0.15, 0.01)
    #STAKE_CAP_PCT = st.number_input("Cap stake (% bankroll)", 0.0, 20.0, 3.0, 0.5)
    
    #EDGE_MIN = st.number_input("Min edge (prob - 1/odds) para ML", 0.0, 0.2, 0.05, 0.005)
    #MARGIN_MIN = st.number_input("Min |p - 0.5|", 0.0, 0.5, 0.05, 0.01)
    #KELLY_MIN = st.number_input("Kelly mínimo (ML)", 0.0, 0.2, 0.02, 0.005)
    #CAP_FAV = st.number_input("Cap stake favorito (%)", 0.0, 10.0, 5.0, 0.5)
    #CAP_DOG = st.number_input("Cap stake underdog (%)", 0.0, 10.0, 3.0, 0.5)
    #CAP_MID = st.number_input("Cap stake zona media (%)", 0.0, 10.0, 1.5, 0.5)
    #CAP_GLOBAL = st.number_input("Cap stake global (%)", 0.0, 20.0, 10.0, 0.5)


    # mínim valor que exigim a l'edge pur (p - 1/odds) perquè considerem apostar
    EDGE_MIN = st.number_input(
        "Min edge ML (p - 1/odds)",
        min_value=0.0, max_value=0.2, value=0.05, step=0.005
    )
    
    MKT_MARGIN_MIN = st.number_input(
        "Min model-vs-odds margin (p - implied)",
        min_value=0.0, max_value=0.1, value=0.05, step=0.005
    )
    
    MARGIN_MIN = st.number_input(
        "Min |p - 0.5| (coinflip veto)",
        min_value=0.0, max_value=0.5, value=0.05, step=0.01
    )
    
    KELLY_MIN = st.number_input(
        "Kelly mínimo (ML)",
        min_value=0.0, max_value=0.2, value=0.02, step=0.005
    )
    
    PROB_GAP_MIN = 0.05  # si vols també fer-ho slider, cap problema
    
    CAP_FAV = st.number_input(
        "Cap stake favorito (%)",
        min_value=0.0, max_value=3.0, value=1.0, step=0.5
    )
    CAP_DOG = st.number_input(
        "Cap stake underdog (%)",
        min_value=0.0, max_value=1.0, value=1.0, step=0.5
    )
    CAP_MID = st.number_input(
        "Cap stake zona media (%)",
        min_value=0.0, max_value=2.0, value=1.0, step=0.5
    )
    CAP_GLOBAL = st.number_input(
        "Cap stake global (%)",
        min_value=0.0, max_value=3.0, value=1.0, step=0.5
    )
    
    # buckets i risc (si no els tenies ja definits enlloc, afegeix-los aquí també)
    FAV_P_MIN = 0.60
    DOG_P_MAX = 0.35
    RISK_FAV = 1.00
    RISK_MID = 0.75
    RISK_DOG = 0.55
    RISK_BOOK_DOG = 0.60
    # Mínim value% requerit per permetre picks d'underdog (model p<0.5)
    MIN_DOG_VALUE_PCT = float(os.getenv("MIN_DOG_VALUE_PCT", 10.0))
    
    ALT_EDGE_SOFT = 0.0
    C_EDGE_SOFT = 0.05



    # nuevo: umbral "blando" para SUGERIR (no apostar ML)
    ALT_EDGE_SOFT = st.number_input("Min edge para sugerir (no ML)", -0.05, 0.1, 0.0, 0.005)
    C_EDGE_SOFT   = st.number_input("Min edge (favorito) para sugerir", 0.0, 0.1, 0.05, 0.005)  # ← 0.02


def _coerce_country_id(x):
    if pd.isna(x):
        return "??"
    v = str(x).strip()
    if len(v) == 2:   # ja és codi
        return v.upper()
    MAP = {
        "spain":"ES","espanya":"ES","argentina":"AR","italy":"IT","italia":"IT",
        "united states":"US","usa":"US","france":"FR","germany":"DE",
        "australia":"AU","england":"GB","united kingdom":"GB","canada":"CA",
        "romania":"RO","serbia":"RS","sweden":"SE","china":"CN","austria":"AT",
        "kazakhstan":"KZ","qatar":"QA","monaco":"MC","netherlands":"NL","denmark":"DK"
    }
    return MAP.get(v.lower(), "??")

def _norm_bool_lefty(x):
    v = str(x).strip().lower()
    return 1 if v in {"1","true","t","yes","y","left","l","left-handed","lefty"} else 0



# ============ META LOADING ============

META_DIR = os.path.join(DATA_DIR, "meta")

def _safe_csv(path, **kw):
    try:
        return pd.read_csv(path, **kw)
    except Exception:
        return pd.DataFrame()

def load_players_meta():
    fp = os.path.join(META_DIR, "players_meta.csv")
    df = _safe_csv(fp)
    # normalitza
    if "player_id" in df.columns:
        df["player_id"] = df["player_id"].astype(str)
    if "handedness" in df.columns:
        df["handedness"] = df["handedness"].astype(str).str.upper().str[0].where(lambda s: s.isin(["L","R"]), "")
    if "nationality_iso" in df.columns:
        df["nationality_iso"] = df["nationality_iso"].astype(str).str.upper().str[:2]
    return df[["player_id","nationality_iso","handedness"]].drop_duplicates() if len(df) else df

def load_tournaments_meta():
    fp = os.path.join(META_DIR, "tournaments_meta.csv")
    df = _safe_csv(fp)
    if "surface_norm" in df.columns:
        df["surface_norm"] = df["surface_norm"].astype(str).str.lower()
    if "country_iso" in df.columns:
        df["country_iso"] = df["country_iso"].astype(str).str.upper().str[:2]
    if "tournament_key" in df.columns:
        df["tournament_key"] = df["tournament_key"].astype(str).str.strip().str.lower()
    return df[["tournament_key","tournament_display","country_iso","surface_norm"]].drop_duplicates() if len(df) else df

def _norm_tournament_key(x: str) -> str:
    """Intenta treure una clau estable del camp 'tournament' dels teus fixtures/matches."""
    s = str(x).strip().lower()
    s = s.replace("(", " ").replace(")", " ").replace("/", " ").replace("-", " ").replace("  "," ")
    s = s.replace(" masters 1000", " masters").replace(" atp ", " ").replace(" wta ", " ")
    s = "_".join(s.split())
    # uns quants alias ràpids
    s = s.replace("paris", "paris_masters")
    s = s.replace("basel", "basel_open")
    s = s.replace("roland_garros", "french_open")
    return f"atp_{s}" if "wta" not in s and not s.startswith("atp_") else s

# ============ ENRICH ============

def enrich_matches_domestic(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Afegeix:
      - A_nation, B_nation
      - A_is_left, B_is_left
      - tournament_country_iso, is_home_A, is_home_B
    No trenca res si falten metadades (deixa buit).
    """
    df = matches_df.copy()
    # columns mínimes
    for c in ["player_a_id","player_b_id","tournament","surface"]:
        if c not in df.columns:
            df[c] = "" if c != "surface" else "hard"
    df["player_a_id"] = df["player_a_id"].astype(str)
    df["player_b_id"] = df["player_b_id"].astype(str)

    P = load_players_meta()
    T = load_tournaments_meta()

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
            if c not in df.columns: df[c] = ""

    # tournaments
    df["tournament_key"] = df.get("tournament","").apply(_norm_tournament_key)
    if len(T):
        df = df.merge(T, on="tournament_key", how="left")
        df["tournament_country_iso"] = df["country_iso"].fillna("")
        # surface normalitzada si ve de meta
        df["surface"] = np.where(df["surface"].notna() & (df["surface"]!=""),
                                 df["surface"], df["surface_norm"].fillna(df.get("surface","hard")))
    else:
        df["tournament_country_iso"] = ""

    # home flags
    df["A_nation"] = df["A_nation"].fillna("").astype(str).str.upper().str[:2]
    df["B_nation"] = df["B_nation"].fillna("").astype(str).str.upper().str[:2]
    df["tournament_country_iso"] = df["tournament_country_iso"].fillna("").astype(str).str.upper().str[:2]

    A = df["A_nation"].map(_cc)
    B = df["B_nation"].map(_cc)
    T = df["tournament_country_iso"].map(_cc)
    
    df["is_home_A"] = (A != "") & (A == T)
    df["is_home_B"] = (B != "") & (B == T)


    return df

# ============ STAKE DISCRETITZACIÓ (sobre % half-Kelly) ============

def discretize_stake_units(stake_pct: float) -> float:
    """
    stake_pct està en percentatge (p. ex. 1.2 = 1.2%).
    Map:
      < 1.00   → 0.5u
      1.00–1.50 → 1.0u
      1.51–4.00 → 2.0u
      > 4.00   → 3.0u
    """
    try:
        s = float(stake_pct)
    except Exception:
        return 0.0
    if not np.isfinite(s) or s <= 0: return 0.0
    if s < 1.00: return 0.5
    if s <= 1.50: return 1.0
    if s <= 4.00: return 2.0
    return 3.0

# ============ POST-FILTER (noves normes) ============

def apply_post_decision_overrides(row: pd.Series,
                                  odds_min: float = 1.30,
                                  edge_min: float = None,   # passa EDGE_MIN si vols
                                  extra_edge_away_dog: float = 0.02,
                                  strong_disagree_cap: tuple = (0.60, 0.40)  # (p_mod>=, p_book<=)
                                  ):
    """
    Rep una fila amb:
      - best_side, best_edge, stake_% (half kelly), odds_home/away, p_home_win, p_away_win
      - is_home_A, is_home_B (si disponibles)
    Aplica:
      1) odds massa curtes → no bet
      2) desacord fort model vs book → cap d’unitats
      3) dog fora de casa → exigeix edge extra o cap units
    Retorna: (best_side, stake_pct, reason_append, stake_units)
    """
    bs  = str(row.get("best_side","")).lower()
    be  = pd.to_numeric(row.get("best_edge"), errors="coerce")
    oh  = pd.to_numeric(row.get("odds_home"), errors="coerce")
    oa  = pd.to_numeric(row.get("odds_away"), errors="coerce")
    ph  = pd.to_numeric(row.get("p_home_win"), errors="coerce")
    pa  = pd.to_numeric(row.get("p_away_win"), errors="coerce")
    stake_pct = pd.to_numeric(row.get("stake_% (half kelly)"), errors="coerce")
    reason = str(row.get("decision_reason",""))

    # Si no hi ha pick, surt tal qual
    if bs == "" or bs == "no bet":
        return ("no bet", 0.0, reason, 0.0)

    # 1) Odds massa curtes
    chosen_odds = np.nan
    p_mod = np.nan
    is_home_flag = None
    if bs.startswith("home"):
        chosen_odds = oh
        p_mod = ph
        is_home_flag = bool(row.get("is_home_A", False))
    elif bs.startswith("away"):
        chosen_odds = oa
        p_mod = pa
        is_home_flag = bool(row.get("is_home_B", False))

    if isinstance(chosen_odds,(int,float,np.floating)) and np.isfinite(chosen_odds):
        if chosen_odds < odds_min:
            return ("no bet", 0.0, reason + "; odds_too_short", 0.0)

    # 2) Desacord fort model vs book
    p_book = np.nan
    if isinstance(chosen_odds,(int,float,np.floating)) and chosen_odds>1.0:
        p_book = 1.0/float(chosen_odds)

    units = discretize_stake_units(stake_pct)
    if np.isfinite(p_mod) and np.isfinite(p_book):
        if (p_mod >= strong_disagree_cap[0]) and (p_book <= strong_disagree_cap[1]):
            # cap 1.0u si p_book ∈ (0.40,0.50]; cap 0.5u si < 0.40
            cap_u = 1.0 if p_book >= 0.40 else 0.5
            if units > cap_u:
                units = cap_u
                reason += "; strong_disagreement_cap"

    # 3) Dog fora de casa → exigeix edge extra
    if edge_min is None:
        edge_min = 0.0
    e_chosen = be
    if str(bs).startswith("home"):
        # dog si p_mod < 0.5
        is_dog = np.isfinite(p_mod) and (p_mod < 0.50)
    else:
        is_dog = np.isfinite(p_mod) and (p_mod < 0.50)

    if is_dog and (is_home_flag is False):
        # si edge insuficient, no bet; si suficient, com a mínim cap a 1u
        need = edge_min + extra_edge_away_dog
        if not (isinstance(e_chosen,(int,float,np.floating)) and np.isfinite(e_chosen) and e_chosen >= need):
            return ("no bet", 0.0, reason + "; away_dog_edge_short", 0.0)
        if units > 1.0:
            units = 1.0
            reason += "; away_dog_cap1u"

    return (row.get("best_side",""), float(stake_pct if np.isfinite(stake_pct) else 0.0), reason, float(units))

# ---- META LOADERS v2 (robustos i sense col·lisions) ----
import pandas as pd, numpy as np, os

def _coerce_country_id(x):
    if pd.isna(x): return "??"
    v = str(x).strip()
    if len(v) == 2:
        return v.upper()
    MAP = {
        "spain":"ES","espanya":"ES","argentina":"AR","italy":"IT","italia":"IT",
        "united states":"US","usa":"US","france":"FR","germany":"DE","australia":"AU",
        "england":"GB","united kingdom":"GB","canada":"CA","romania":"RO","serbia":"RS",
        "sweden":"SE","china":"CN","austria":"AT","kazakhstan":"KZ","qatar":"QA",
        "monaco":"MC","netherlands":"NL","denmark":"DK"
    }
    return MAP.get(v.lower(), "??")

def _norm_bool_lefty(x):
    v = str(x).strip().lower()
    return 1 if v in {"1","true","t","yes","y","left","l","left-handed","lefty"} else 0

def load_players_meta_v2(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["player_id","country_id","is_lefty"])
    p = pd.read_csv(path)
    p.columns = [c.strip().lower() for c in p.columns]

    id_col = next((c for c in ["player_id","id","playerid","atp_id"] if c in p.columns), None)
    if not id_col:
        return pd.DataFrame(columns=["player_id","country_id","is_lefty"])
    p = p.rename(columns={id_col: "player_id"})

    c_col = next((c for c in ["country_id","country_code","countryid","country","nationality"] if c in p.columns), None)
    if c_col: p["country_id"] = p[c_col].apply(_coerce_country_id)
    else:     p["country_id"] = "??"

    h_col = next((c for c in ["is_lefty","lefty","handedness","hand","plays","dominant_hand"] if c in p.columns), None)
    if h_col: p["is_lefty"] = p[h_col].apply(_norm_bool_lefty)
    else:     p["is_lefty"] = 0

    out = p[["player_id","country_id","is_lefty"]].drop_duplicates("player_id")
    out["player_id"] = out["player_id"].astype(str)
    return out

def load_tournaments_meta_v2(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["tournament","country_id","level"])
    t = pd.read_csv(path)
    t.columns = [c.strip().lower() for c in t.columns]

    name_col = next((c for c in ["tournament","name","tournament_name","tourney","tournament_title"] if c in t.columns), None)
    if not name_col:
        return pd.DataFrame(columns=["tournament","country_id","level"])
    t = t.rename(columns={name_col: "tournament"})

    c_col = next((c for c in ["country_id","country_code","country","countryid"] if c in t.columns), None)
    if c_col: t["country_id"] = t[c_col].apply(_coerce_country_id)
    else:     t["country_id"] = "N/A"

    lvl_col = next((c for c in ["level","tier","category"] if c in t.columns), None)
    t["level"] = t[lvl_col].astype(str) if lvl_col else "ATP"

    out = t[["tournament","country_id","level"]].drop_duplicates("tournament")
    out["tournament"] = out["tournament"].astype(str)
    return out

# ---- META FEATURES v2 (robustes i numèriques) ----
import pandas as pd
import numpy as np

HOME_NEUTRAL_CODES = {"", "N/A", "NA", "??", "NEUTRAL", "NONE", "NULL"}

def _cc(x: object) -> str:
    """Country code net: retorna codi ISO2 en majúscules o '' si no és usable."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip().upper()
    if s in HOME_NEUTRAL_CODES:
        return ""
    # si ve "ES " o "espanya" no ho arreglem aquí; assumim que meta ja està neta ISO2
    # però almenys controlem longitud
    if len(s) != 2:
        return ""
    return s

def compute_home_advantage(A_country: object, B_country: object, T_country: object, is_neutral: object = 0):
    """
    Retorna (A_is_home, B_is_home, home_advantage) amb la norma:
      - només hi ha senyal si EXACTAMENT un jugador està al seu país
      - si tots dos home o tots dos away -> neutral (0)
      - si torneig neutral/unknown -> neutral (0)
    """
    try:
        neutral = int(pd.to_numeric(is_neutral, errors="coerce") or 0)
    except Exception:
        neutral = 0

    Tc = _cc(T_country)
    if neutral == 1 or Tc == "":
        return 0, 0, 0

    Ac = _cc(A_country)
    Bc = _cc(B_country)

    A_home = 1 if (Ac != "" and Ac == Tc) else 0
    B_home = 1 if (Bc != "" and Bc == Tc) else 0

    # diferència ja fa neutral si (1,1) o (0,0)
    return A_home, B_home, (A_home - B_home)


def add_meta_features_v2(df: pd.DataFrame,
                         matches_df: pd.DataFrame,
                         pmeta: pd.DataFrame,
                         tmeta: pd.DataFrame):
    """
    - Assegura ID/tournament des de matches_df
    - Afegeix meta de jugadors (country_id, is_lefty) i tournaments (country_id, level)
    - Deriva només features NUMÈRIQUES per entrenar
    Retorna: (df_amb_features, added_cols_list)
    """
    df = df.copy()

    # 0) Porta camps clau des de matches (si hi són)
    mcols = [c for c in ["match_id", "player_a_id", "player_b_id", "tournament", "surface", "best_of_5"]
             if c in matches_df.columns]
    if "match_id" not in df.columns:
        raise ValueError("add_meta_features_v2: falta 'match_id' a dataset_raw")
    if mcols:
        df = df.merge(matches_df[mcols].drop_duplicates("match_id"), on="match_id", how="left")

    # garanteix existència
    for c, default in [("player_a_id", ""), ("player_b_id", ""), ("tournament", ""), ("best_of_5", 0)]:
        if c not in df.columns:
            df[c] = default

    # 1) Players meta → A/B
    if not pmeta.empty:
        pmeta = pmeta.copy()
        pmeta.columns = [c.strip().lower() for c in pmeta.columns]

        if "player_id" not in pmeta.columns:
            cand = [c for c in pmeta.columns if c in ("id", "atp_id")]
            if cand:
                pmeta = pmeta.rename(columns={cand[0]: "player_id"})

        pmeta["player_id"] = pmeta["player_id"].astype(str)

        pA = pmeta.rename(columns={
            "player_id": "player_a_id",
            "country_id": "A_country_id",
            "is_lefty": "A_is_lefty"
        })[["player_a_id", "A_country_id", "A_is_lefty"]].drop_duplicates("player_a_id")

        pB = pmeta.rename(columns={
            "player_id": "player_b_id",
            "country_id": "B_country_id",
            "is_lefty": "B_is_lefty"
        })[["player_b_id", "B_country_id", "B_is_lefty"]].drop_duplicates("player_b_id")

        df["player_a_id"] = df["player_a_id"].astype(str)
        df["player_b_id"] = df["player_b_id"].astype(str)

        df = df.merge(pA, on="player_a_id", how="left")
        df = df.merge(pB, on="player_b_id", how="left")
    else:
        for c in ["A_country_id", "B_country_id"]:
            df[c] = "??"
        for c in ["A_is_lefty", "B_is_lefty"]:
            df[c] = 0

    # 2) Tournament meta
    if not tmeta.empty:
        tm = tmeta.copy()
        tm.columns = [c.strip().lower() for c in tm.columns]
        name_col = "tournament" if "tournament" in tm.columns else None

        if not name_col:
            for c in ["name", "tournament_name", "tourney", "tournament_title"]:
                if c in tm.columns:
                    tm = tm.rename(columns={c: "tournament"})
                    name_col = "tournament"
                    break

        if name_col:
            keep = ["tournament"]
            if "country_id" in tm.columns:
                keep.append("country_id")
            if "level" in tm.columns:
                keep.append("level")

            tm = tm[keep].drop_duplicates("tournament")
            tm = tm.rename(columns={
                "country_id": "tourney_country_id",
                "level": "tourney_level"
            })

            df["tournament"] = df["tournament"].astype(str)
            tm["tournament"] = tm["tournament"].astype(str)
            df = df.merge(tm, on="tournament", how="left")

    # defaults torneo
    if "tourney_country_id" not in df.columns:
        df["tourney_country_id"] = "N/A"
    if "tourney_level" not in df.columns:
        df["tourney_level"] = "ATP"

    # 3) Derivades NUMÈRIQUES
    #   - home flags robustos (neutral/unknown -> 0, i només senyal si exactament un és home)

    HOME_NEUTRAL_CODES = {"", "N/A", "NA", "??", "NEUTRAL", "NONE", "NULL"}

    def _clean_country_series(s: pd.Series) -> pd.Series:
        # Normalitza a ISO2 en majúscules o '' si no és usable
        x = s.astype(str).str.strip().str.upper()
        x = x.where(~x.isin(HOME_NEUTRAL_CODES), "")
        x = x.where(x.str.len() == 2, "")  # evita "USA", "ESP", etc.
        return x

    A_c = _clean_country_series(df.get("A_country_id", pd.Series([""] * len(df), index=df.index)))
    B_c = _clean_country_series(df.get("B_country_id", pd.Series([""] * len(df), index=df.index)))
    T_c = _clean_country_series(df.get("tourney_country_id", pd.Series([""] * len(df), index=df.index)))

    # Si el torneig és neutral/unknown (T_c == ''), tot queda neutral
    A_home = (T_c != "") & (A_c != "") & (A_c == T_c)
    B_home = (T_c != "") & (B_c != "") & (B_c == T_c)

    df["A_is_home"] = A_home.astype(int)
    df["B_is_home"] = B_home.astype(int)

    # Diferència: +1 si només A és home, -1 si només B, 0 en la resta (inclòs "tots dos home")
    df["home_advantage"] = df["A_is_home"] - df["B_is_home"]

    #   - lefty flags (ja numèrics)
    df["A_is_lefty"] = pd.to_numeric(df.get("A_is_lefty", 0), errors="coerce").fillna(0).astype(int)
    df["B_is_lefty"] = pd.to_numeric(df.get("B_is_lefty", 0), errors="coerce").fillna(0).astype(int)

    #   - bo5 flag
    df["best_of_5"] = pd.to_numeric(df.get("best_of_5", 0), errors="coerce").fillna(0).astype(int)

    #   - codi de nivell (nota: factorize és estable dins el mateix df, però no garanteix mapping estable cross-run)
    df["tourney_level_code"] = pd.factorize(df["tourney_level"].fillna("ATP").astype(str))[0].astype(int)

    # Features numèriques que afegirem al model (PAS 2: treiem A_is_home i B_is_home)
    added_cols = [
        "A_is_lefty", "B_is_lefty",
        "home_advantage",
        "best_of_5", "tourney_level_code"
    ]

    # Safety: substitueix NaNs només per a les features del model
    for c in added_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    return df, added_cols





#tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Refresh & Train", "Preview Data", "Predict (upload fixtures)", "Fetch fixtures (Odds API)", "Diagnostics", "History & Results", "About/Help"])
tab1, tab2, tab3 = st.tabs(["Refresh & Train", "Predict (upload fixtures)", "History & Results"])

with tab1:
    import os, json, joblib, hashlib
    import numpy as np
    import pandas as pd
    import streamlit as st

    # ============================================================
    # META HELPERS (alineats amb Tab2)
    # ============================================================
    def _meta_paths():
        pmeta_path = os.path.join(str(DATA_DIR), "players_meta.csv")
        tmeta_path = os.path.join(str(DATA_DIR), "tournaments_meta.csv")
        return pmeta_path, tmeta_path

    def load_meta_tables():
        """
        Carrega players_meta.csv i tournaments_meta.csv.
        Normalitza columnes i crea camps mínims:
          - pmeta: player_id(str), country_id(str), is_lefty(int)
          - tmeta: tournament(str), country_id(str), level(str), is_neutral(int)
        """
        pmeta_path, tmeta_path = _meta_paths()
        pmeta = pd.read_csv(pmeta_path) if os.path.exists(pmeta_path) else pd.DataFrame()
        tmeta = pd.read_csv(tmeta_path) if os.path.exists(tmeta_path) else pd.DataFrame()

        # ---- Players meta ----
        if not pmeta.empty:
            pm = pmeta.copy()
            pm.columns = [c.strip().lower() for c in pm.columns]

            # id
            if "player_id" not in pm.columns:
                cand = [c for c in pm.columns if c in ("id", "atp_id")]
                if cand:
                    pm = pm.rename(columns={cand[0]: "player_id"})
            if "player_id" in pm.columns:
                pm["player_id"] = pm["player_id"].astype(str)
            else:
                pm["player_id"] = ""

            # country
            if "country_id" not in pm.columns:
                pm["country_id"] = "??"
            pm["country_id"] = pm["country_id"].astype(str)

            # lefty
            if "is_lefty" not in pm.columns:
                if "handedness" in pm.columns:
                    pm["is_lefty"] = pm["handedness"].astype(str).str.upper().str.startswith("L").astype(int)
                else:
                    pm["is_lefty"] = 0
            pm["is_lefty"] = pd.to_numeric(pm["is_lefty"], errors="coerce").fillna(0).astype(int)

            # (compat) camps extra si existeixen, però no afecten el "home"
            for c in ["serve_score", "return_score"]:
                if c not in pm.columns:
                    pm[c] = np.nan

            pmeta = pm

        # ---- Tournaments meta ----
        if not tmeta.empty:
            tm = tmeta.copy()
            tm.columns = [c.strip().lower() for c in tm.columns]

            # tournament name column
            if "tournament" not in tm.columns:
                for cand in ["name", "tournament_name", "tourney", "tournament_title"]:
                    if cand in tm.columns:
                        tm = tm.rename(columns={cand: "tournament"})
                        break
            if "tournament" in tm.columns:
                tm["tournament"] = tm["tournament"].astype(str)
            else:
                tm["tournament"] = ""

            # country + level
            if "country_id" not in tm.columns:
                tm["country_id"] = "N/A"
            tm["country_id"] = tm["country_id"].astype(str)

            if "level" not in tm.columns:
                tm["level"] = "ATP"
            tm["level"] = tm["level"].astype(str)

            # neutral flag (només per anul·lar home; NO és feature del model)
            if "is_neutral" not in tm.columns:
                if "neutral" in tm.columns:
                    tm["is_neutral"] = (
                        (tm["neutral"].astype(str).str.lower().isin(["1", "true", "yes", "neutral"])) |
                        (tm["country_id"].astype(str).str.upper().isin(["N/A", "NA", "??", "NEUTRAL", ""]))
                    ).astype(int)
                else:
                    tm["is_neutral"] = (tm["country_id"].astype(str).str.upper().isin(["N/A", "NA", "??", "NEUTRAL", ""])).astype(int)
            tm["is_neutral"] = pd.to_numeric(tm["is_neutral"], errors="coerce").fillna(0).astype(int)

            tmeta = tm

        return pmeta, tmeta

    def add_meta_features_v2(df: pd.DataFrame,
                             matches_df: pd.DataFrame,
                             pmeta: pd.DataFrame,
                             tmeta: pd.DataFrame):
        """
        VERSIÓ ALINEADA AMB TAB2:
        - "home" només via home_advantage = A_home - B_home
        - A_home/B_home només si:
            * torneig NO neutral
            * torneig té country_id usable
            * country_id jugador == country_id torneig (normalitzat)
        - NO afegeix home_A/home_B/is_neutral com a features del model
        Retorna: (df_amb_features, added_cols_list)
        """
        df = df.copy()

        # 0) Porta camps clau des de matches (si hi són)
        mcols = [c for c in ["match_id", "player_a_id", "player_b_id", "tournament", "surface", "best_of_5"]
                 if c in matches_df.columns]
        if "match_id" not in df.columns:
            raise ValueError("add_meta_features_v2: falta 'match_id' a dataset_raw")
        if mcols:
            df = df.merge(matches_df[mcols].drop_duplicates("match_id"), on="match_id", how="left")

        # garanteix existència
        for c, default in [("player_a_id", ""), ("player_b_id", ""), ("tournament", ""), ("best_of_5", 0)]:
            if c not in df.columns:
                df[c] = default

        # 1) Players meta → A/B (country_id + is_lefty)
        if not pmeta.empty:
            pm = pmeta.copy()
            pm.columns = [c.strip().lower() for c in pm.columns]

            if "player_id" not in pm.columns:
                cand = [c for c in pm.columns if c in ("id", "atp_id")]
                if cand:
                    pm = pm.rename(columns={cand[0]: "player_id"})

            pm["player_id"] = pm["player_id"].astype(str) if "player_id" in pm.columns else ""
            if "country_id" not in pm.columns:
                pm["country_id"] = "??"
            if "is_lefty" not in pm.columns:
                if "handedness" in pm.columns:
                    pm["is_lefty"] = pm["handedness"].astype(str).str.upper().str.startswith("L").astype(int)
                else:
                    pm["is_lefty"] = 0

            pm["country_id"] = pm["country_id"].astype(str)
            pm["is_lefty"] = pd.to_numeric(pm["is_lefty"], errors="coerce").fillna(0).astype(int)

            pA = pm.rename(columns={
                "player_id": "player_a_id",
                "country_id": "A_country_id",
                "is_lefty": "A_is_lefty"
            })[["player_a_id", "A_country_id", "A_is_lefty"]].drop_duplicates("player_a_id")

            pB = pm.rename(columns={
                "player_id": "player_b_id",
                "country_id": "B_country_id",
                "is_lefty": "B_is_lefty"
            })[["player_b_id", "B_country_id", "B_is_lefty"]].drop_duplicates("player_b_id")

            df["player_a_id"] = df["player_a_id"].astype(str)
            df["player_b_id"] = df["player_b_id"].astype(str)

            df = df.merge(pA, on="player_a_id", how="left")
            df = df.merge(pB, on="player_b_id", how="left")
        else:
            df["A_country_id"] = "??"
            df["B_country_id"] = "??"
            df["A_is_lefty"] = 0
            df["B_is_lefty"] = 0

        # 2) Tournament meta (country_id + level + is_neutral intern)
        if not tmeta.empty:
            tm = tmeta.copy()
            tm.columns = [c.strip().lower() for c in tm.columns]

            if "tournament" not in tm.columns:
                for cand in ["name", "tournament_name", "tourney", "tournament_title"]:
                    if cand in tm.columns:
                        tm = tm.rename(columns={cand: "tournament"})
                        break

            if "tournament" in tm.columns:
                keep = ["tournament"]
                if "country_id" in tm.columns:
                    keep.append("country_id")
                if "level" in tm.columns:
                    keep.append("level")
                if "is_neutral" in tm.columns:
                    keep.append("is_neutral")

                tm = tm[keep].drop_duplicates("tournament").rename(columns={
                    "country_id": "tourney_country_id",
                    "level": "tourney_level"
                })

                if "is_neutral" not in tm.columns:
                    tm["is_neutral"] = 0

                df["tournament"] = df["tournament"].astype(str)
                tm["tournament"] = tm["tournament"].astype(str)
                df = df.merge(tm, on="tournament", how="left")

        # defaults torneig
        if "tourney_country_id" not in df.columns:
            df["tourney_country_id"] = "N/A"
        if "tourney_level" not in df.columns:
            df["tourney_level"] = "ATP"
        if "is_neutral" not in df.columns:
            df["is_neutral"] = 0

        # 3) Derivades NUMÈRIQUES (HOME: només home_advantage)
        HOME_NEUTRAL_CODES = {"", "N/A", "NA", "??", "NEUTRAL", "NONE", "NULL"}

        def _clean_code(s: pd.Series) -> pd.Series:
            x = s.astype(str).str.strip().str.upper()
            x = x.where(~x.isin(HOME_NEUTRAL_CODES), "")
            return x

        A_c = _clean_code(df["A_country_id"].fillna("??"))
        B_c = _clean_code(df["B_country_id"].fillna("??"))
        T_c = _clean_code(df["tourney_country_id"].fillna("N/A"))

        # si el torneig és neutral, anul·la qualsevol "home"
        neu_flag = pd.to_numeric(df.get("is_neutral", 0), errors="coerce").fillna(0).astype(int)
        T_c = T_c.where(neu_flag.eq(0), "")

        A_home = (T_c != "") & (A_c != "") & (A_c == T_c)
        B_home = (T_c != "") & (B_c != "") & (B_c == T_c)

        df["home_advantage"] = (A_home.astype(int) - B_home.astype(int)).astype(int)

        # Lefty flags numèrics
        df["A_is_lefty"] = pd.to_numeric(df.get("A_is_lefty", 0), errors="coerce").fillna(0).astype(int)
        df["B_is_lefty"] = pd.to_numeric(df.get("B_is_lefty", 0), errors="coerce").fillna(0).astype(int)

        # Bo5 flag numèric
        df["best_of_5"] = pd.to_numeric(df.get("best_of_5", 0), errors="coerce").fillna(0).astype(int)

        # Level code (mateixa lògica que Tab2, per consistència)
        df["tourney_level_code"] = pd.factorize(df["tourney_level"].fillna("ATP").astype(str))[0].astype(int)

        added_cols = [
            "A_is_lefty", "B_is_lefty",
            "home_advantage",
            "best_of_5", "tourney_level_code"
        ]

        for c in added_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).replace([np.inf, -np.inf], 0.0)

        return df, added_cols

    # ============================================================
    # UTILITATS ENTRENAMENT
    # ============================================================
    def _save_model_bundle(model, scaler, iso, out_dir, use_lgb=True):
        os.makedirs(out_dir, exist_ok=True)
        if scaler is not None:
            joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
        if iso is not None:
            joblib.dump(iso, os.path.join(out_dir, "isotonic.pkl"))
        model_name = "model_lightgbm.pkl" if use_lgb else "model_logistic.pkl"
        joblib.dump(model, os.path.join(out_dir, model_name))

    def _dataset_fingerprint(df: pd.DataFrame) -> dict:
        """
        Crea un fingerprint lleuger per detectar canvis.
        Fem servir: n_rows, n_unique_matches, max(date), i un hash curt d'una part estable.
        """
        try:
            n_rows = int(len(df))
            n_uniq = int(df["match_id"].nunique()) if "match_id" in df.columns else n_rows
            max_date = (
                pd.to_datetime(df["date"], errors="coerce").max().strftime("%Y-%m-%d")
                if "date" in df.columns else ""
            )
            # hash sobre (match_id + y_home_win) com a resum
            key_cols = []
            for c in ["match_id", "y_home_win"]:
                if c in df.columns:
                    key_cols.append(c)
            if key_cols:
                tmp = df[key_cols].astype(str).agg("|".join, axis=1).str.encode("utf-8")
                hh = hashlib.md5(b"".join(tmp[: min(len(tmp), 100000)])).hexdigest()
            else:
                hh = "na"
            return {"n_rows": n_rows, "n_unique": n_uniq, "max_date": max_date, "hash": hh}
        except Exception:
            return {"n_rows": -1, "n_unique": -1, "max_date": "", "hash": "err"}

    def _has_changed(new_fp: dict, path_json: str) -> bool:
        if not os.path.exists(path_json):
            return True
        try:
            old = json.load(open(path_json, "r"))
        except Exception:
            return True
        return any(new_fp.get(k) != old.get(k) for k in ["n_rows", "n_unique", "max_date", "hash"])

    # ============================================================
    # 1) FETCH TML
    # ============================================================
    st.subheader("1) Fetch TML data")
    with st.status("Downloading TML CSVs…", expanded=False) as status:
        frames = []
        for y in range(int(YEAR_START), int(YEAR_END) + 1):
            try:
                frames.append(fetch_year_csv(int(y)))
            except Exception as e:
                st.warning(f"{y}: {e}")
        if INCLUDE_ONGOING:
            try:
                frames.append(fetch_ongoing_csv())
            except Exception as e:
                st.warning(f"ongoing_tourneys: {e}")

        if not frames:
            st.error("No CSVs downloaded.")
            status.update(state="error")
            st.stop()

        df_all = pd.concat(frames, ignore_index=True)
        status.update(
            label=f"Downloaded {len(frames)} file(s), rows={len(df_all):,}",
            state="complete"
        )

        st.session_state["df_all"] = df_all
        if "df_all" in st.session_state:
            MAP_FULL, MAP_ALIAS, INDEX_LAST = build_name_indices_from_tml(st.session_state["df_all"])
            st.session_state["MAP_FULL"] = MAP_FULL
            st.session_state["MAP_ALIAS"] = MAP_ALIAS
            st.session_state["INDEX_LAST"] = INDEX_LAST
            st.success(f"Nom-index: FULL={len(MAP_FULL)}, ALIAS={len(MAP_ALIAS)}, LAST={len(INDEX_LAST)}")
        else:
            st.warning("Carrega TML (Refresh) per construir índex de noms.")

    # ============================================================
    # 2) BUILD matches & points (GUARDEM COMPLETS, i després filtrem per training)
    # ============================================================
    st.subheader("2) Build matches & points")
    with st.status("Building matches.csv / points_sets_games.csv…", expanded=False) as status:
        matches = build_matches_from_tml(df_all)
        points = build_points_from_tml(df_all)

        from pathlib import Path
        import time

        DATA_DIR.mkdir(parents=True, exist_ok=True)

        matches_path = DATA_DIR / "matches.csv"
        points_path = DATA_DIR / "points_sets_games.csv"

        # Guardem fitxers complets (no filtrats)
        matches.to_csv(matches_path, index=False)
        points.to_csv(points_path, index=False)

        st.write("Saved matches to:", str(matches_path.resolve()), "mtime:", time.ctime(matches_path.stat().st_mtime))
        st.write("Saved points  to:", str(points_path.resolve()), "mtime:", time.ctime(points_path.stat().st_mtime))

        status.update(
            label=f"Saved matches.csv ({len(matches):,}) & points_sets_games.csv ({len(points):,})",
            state="complete"
        )

    import re

    def _find_date_col(df: pd.DataFrame):
        for c in ["date", "match_date", "tourney_date", "tournament_date", "start_date", "tourney_start_date"]:
            if c in df.columns:
                return c
        # fallback: primera columna que contingui "date"
        for c in df.columns:
            if "date" in c.lower():
                return c
        return None

    def _parse_match_dates(s: pd.Series) -> pd.Series:
        ss = s.astype(str).str.strip()
        ss = ss.str.replace(r"\.0$", "", regex=True)      # 20251020.0 -> 20251020
        ss_norm = ss.str.replace("/", "-", regex=False)   # / -> -

        digits = ss_norm.str.replace(r"\D", "", regex=True)
        mask_8 = digits.str.len().eq(8)

        dt_8 = pd.to_datetime(digits.where(mask_8), format="%Y%m%d", errors="coerce")
        dt_any = pd.to_datetime(ss_norm.where(~mask_8), errors="coerce")

        return dt_8.fillna(dt_any)

    # ============================================================
    # 2b) APLICAR FILTRE TEMPORAL PER ENTRENAMENT (EN MEMÒRIA)
    # ============================================================
    matches_train = matches.copy()
    points_train = points.copy()

    # Parse robust dates (si existeixen)
    if "date" in matches_train.columns:
        matches_train["date"] = _parse_match_dates(matches_train["date"])

    if "match_date" in matches_train.columns:
        matches_train["match_date"] = _parse_match_dates(matches_train["match_date"])

    # Assegura tourney_id si vols fer filtre de torneig sencer
    if "tourney_id" not in matches_train.columns:
        if "match_id" in matches_train.columns:
            matches_train["tourney_id"] = matches_train["match_id"].astype(str).str.split("_").str[0]
        else:
            matches_train["tourney_id"] = np.nan

    # --- Cutoffs independents ---
    cutoff_tour_ts = pd.Timestamp(TRAIN_CUTOFF_DATE) if TRAIN_CUTOFF_ENABLED else None
    cutoff_match_ts = pd.Timestamp(MATCH_CUTOFF_DATE) if MATCH_CUTOFF_ENABLED else None

    # ---------- 1) Tournament-level filter (torneig sencer) ----------
    def apply_tournament_cutoff(df: pd.DataFrame, cutoff_ts: pd.Timestamp) -> pd.DataFrame:
        if cutoff_ts is None:
            return df

        if "date" not in df.columns or df["date"].isna().all():
            st.warning("Tournament-level cutoff selected, but 'date' is missing/empty. Skipping tournament-level cutoff.")
            return df

        if "tourney_id" not in df.columns or df["tourney_id"].isna().all():
            st.warning("Tournament-level cutoff selected, but 'tourney_id' is missing/empty. Skipping tournament-level cutoff.")
            return df

        tournament_start = df.groupby("tourney_id")["date"].transform("min")
        # cutoff inclusiu => ens quedem amb < cutoff_ts
        return df[tournament_start < cutoff_ts].copy()

    # ---------- 2) Match-level filter (per pseudo-calendari) ----------
    def apply_match_cutoff(df: pd.DataFrame, cutoff_ts: pd.Timestamp) -> pd.DataFrame:
        if cutoff_ts is None:
            return df

        if "match_date" not in df.columns or df["match_date"].isna().all():
            st.warning("Match-level cutoff selected, but 'match_date' is missing/empty. Skipping match-level cutoff.")
            return df

        # cutoff inclusiu => ens quedem amb < cutoff_ts
        return df[df["match_date"] < cutoff_ts].copy()

    # --- Aplica segons mode ---
    before_n = len(matches_train)

    if CUTOFF_MODE == "Tournament-level":
        matches_train = apply_tournament_cutoff(matches_train, cutoff_tour_ts)

    elif CUTOFF_MODE == "Match-level":
        matches_train = apply_match_cutoff(matches_train, cutoff_match_ts)

    elif CUTOFF_MODE == "Both (stricter)":
        matches_train = apply_tournament_cutoff(matches_train, cutoff_tour_ts)
        matches_train = apply_match_cutoff(matches_train, cutoff_match_ts)

    # Mantén consistència de points contra match_id
    if "match_id" in matches_train.columns and "match_id" in points_train.columns:
        allowed_ids = set(matches_train["match_id"].astype(str))
        points_train["match_id"] = points_train["match_id"].astype(str)
        points_train = points_train[points_train["match_id"].isin(allowed_ids)].copy()
    else:
        st.warning("Could not filter points_sets_games by match_id (missing match_id column).")

    removed = before_n - len(matches_train)

    # Debug/Info
    if "date" in matches_train.columns and not matches_train["date"].isna().all():
        st.write("Tournament date range | min:", matches_train["date"].min(), "| max:", matches_train["date"].max())

    if "match_date" in matches_train.columns and not matches_train["match_date"].isna().all():
        st.write("Match date range | min:", matches_train["match_date"].min(), "| max:", matches_train["match_date"].max())

    st.info(
        f"Cutoffs | mode={CUTOFF_MODE} | "
        f"tournament_cutoff={'ON' if TRAIN_CUTOFF_ENABLED else 'OFF'}"
        f"{'' if not TRAIN_CUTOFF_ENABLED else f' (<{cutoff_tour_ts.date()})'} | "
        f"match_cutoff={'ON' if MATCH_CUTOFF_ENABLED else 'OFF'}"
        f"{'' if not MATCH_CUTOFF_ENABLED else f' (<{cutoff_match_ts.date()})'} | "
        f"removed={removed:,} | remaining matches={len(matches_train):,} | points rows={len(points_train):,}"
    )

    # Deixa rastre per altres tabs / debug
    st.session_state["TRAIN_CUTOFF_ENABLED"] = bool(TRAIN_CUTOFF_ENABLED)
    st.session_state["TRAIN_CUTOFF_DATE"] = str(TRAIN_CUTOFF_DATE)
    st.session_state["MATCH_CUTOFF_ENABLED"] = bool(MATCH_CUTOFF_ENABLED)
    st.session_state["MATCH_CUTOFF_DATE"] = str(MATCH_CUTOFF_DATE)
    st.session_state["CUTOFF_MODE"] = str(CUTOFF_MODE)

    print("[build_matches_from_tml] matches columns:", list(matches.columns))
    print("[build_matches_from_tml] match_date non-null:", matches["match_date"].notna().mean())
    print(matches[["match_id", "date", "match_date", "round"]].head(5))

    # ============================================================
    # 3) FEATURE BUILDING (+ META FEATURES) — USA matches_train / points_train
    # ============================================================
    st.subheader("3) Feature building (pipeline v2 + meta)")
    if not PIPE_OK:
        st.error("Could not import `tennis_model_pipeline_v2`. Put the .py next to this app.")
        st.stop()

    with st.status("Computing pre-match features & dataset…", expanded=False):
        # Enriquir DOMÈSTIC abans de fer features
        matches_enr = enrich_matches_domestic(matches_train)

        # Base features del pipeline
        feats = compute_pre_match_features_v2(matches_enr, points_train)
        dataset_raw, model_cols_base = make_match_features(feats, matches_enr)

        # Meta (alineada amb Tab2)
        pmeta, tmeta = load_meta_tables()
        dataset_final, added_cols = add_meta_features_v2(dataset_raw, matches_enr, pmeta, tmeta)

        # Columnes d'entrenament finals (dedup + anti-leak)
        model_cols = list(model_cols_base) + list(added_cols or [])
        seen = set()
        model_cols = [c for c in model_cols if not (c in seen or seen.add(c))]

        _leak_words = ("odds", "edge", "kelly", "stake", "best_side", "decision", "hint")
        model_cols = [c for c in model_cols if all(w not in c.lower() for w in _leak_words)]

        # Guarda fitxers
        os.makedirs(OUT_DIR, exist_ok=True)
        feats_path = os.path.join(OUT_DIR, "features_player_pre.csv")
        ds_raw_path = os.path.join(OUT_DIR, "dataset_match_level_raw.csv")
        ds_final_path = os.path.join(OUT_DIR, "dataset_match_level.csv")

        feats.to_csv(feats_path, index=False)
        dataset_raw.to_csv(ds_raw_path, index=False)
        dataset_final.to_csv(ds_final_path, index=False)

        # Guarda també l'ordre exacte de les columnes d'entrenament finals
        with open(os.path.join(OUT_DIR, "model_columns.txt"), "w") as f:
            for c in model_cols:
                f.write(str(c) + "\n")

        # Session state perquè Tab3/Tab4/Tab5 ho reutilitzin
        st.session_state["matches_enr_latest"] = matches_enr
        st.session_state["dataset_latest"] = dataset_final
        st.session_state["model_cols_latest"] = model_cols

        st.success(
            f"Features built. Raw rows={len(dataset_raw):,}, Final rows={len(dataset_final):,}, features={len(model_cols)}"
        )

    # ─────────────────────────────────────────────────────────────
    # 4) Train model + Calibrate + Save (+ offline strategy backtest)
    # ─────────────────────────────────────────────────────────────
    with st.status("Training…", expanded=False) as status:
        # ENTRENAR sobre dataset_final
        model, scaler, iso, metrics, splits = train_models(
            dataset_final,
            model_cols,
            use_lgb=USE_LGB
        )

        # guarda calibrador si existeix
        if iso:
            joblib.dump(iso, os.path.join(OUT_DIR, "isotonic.pkl"))

        # ===== Helpers per avaluació offline (iguals als que ja tenies) =====
        def _kelly_full_fraction(pv, ov):
            if not (isinstance(ov, (int, float, np.floating)) and np.isfinite(ov) and ov > 1.0):
                return 0.0
            b = ov - 1.0
            full_k = (b * pv - (1.0 - pv)) / b
            if not np.isfinite(full_k):
                full_k = 0.0
            return max(full_k, 0.0)

        def _edge(prob, odd):
            if isinstance(odd, (int, float, np.floating)) and np.isfinite(odd) and odd > 0:
                return prob - (1.0 / odd)
            return np.nan

        def _prep_eval_predictions(dataset_df, matches_df, model, scaler, iso, model_cols_for_eval):
            Xdf = dataset_df.reindex(columns=model_cols_for_eval, fill_value=0.0)
            Xdf = Xdf.apply(pd.to_numeric, errors="coerce")
            Xdf = Xdf.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            X_eval = Xdf.values
            if scaler is not None and hasattr(scaler, "transform"):
                X_eval = scaler.transform(X_eval)
            X_eval = np.nan_to_num(X_eval, nan=0.0, posinf=0.0, neginf=0.0)

            if hasattr(model, "predict_proba"):
                p_raw = model.predict_proba(X_eval)[:, 1]
            else:
                try:
                    p_raw = model.predict(X_eval, num_iteration=getattr(model, "best_iteration", None))
                except TypeError:
                    p_raw = model.predict(X_eval)

            if iso is not None:
                p_raw = iso.transform(p_raw)

            p_clip = np.clip(p_raw, 1e-6, 1 - 1e-6)

            eval_df = dataset_df[["match_id", "date", "y_home_win"]].copy()
            eval_df["p_home_win"] = p_clip

            merge_cols = [c for c in ["match_id", "odds_home", "odds_away", "best_of_5"] if c in matches_df.columns]
            eval_df = eval_df.merge(matches_df[merge_cols], on="match_id", how="left")
            return eval_df

        def _hi_conf_accuracy(eval_df, p_hi=0.60, p_lo=0.40):
            df = eval_df.copy()
            y = pd.to_numeric(df["y_home_win"], errors="coerce")
            bin_ok = y.isin([0, 1])
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

        def _backtest_hit_rate(eval_df):
            rows = []
            baseline_correct_mask = (
                ((eval_df["p_home_win"] >= 0.5) & (eval_df["y_home_win"] == 1)) |
                ((eval_df["p_home_win"] < 0.5) & (eval_df["y_home_win"] == 0))
            )
            baseline_acc = float(baseline_correct_mask.mean()) if len(eval_df) else np.nan

            for _, rr in eval_df.iterrows():
                p_h = float(rr["p_home_win"])
                p_a = 1.0 - p_h
                oh = float(rr["odds_home"]) if pd.notnull(rr.get("odds_home")) else np.nan
                oa = float(rr["odds_away"]) if pd.notnull(rr.get("odds_away")) else np.nan
                eh = _edge(p_h, oh)
                ea = _edge(p_a, oa)
                kh = _kelly_full_fraction(p_h, oh)
                ka = _kelly_full_fraction(p_a, oa)

                bs, be, stake, reason = decide_with_filters(
                    p_h, oh, p_a, oa, kh, ka, eh, ea,
                    fav_p_min=FAV_P_MIN, dog_p_max=DOG_P_MAX, edge_min=EDGE_MIN,
                    margin_min=MKT_MARGIN_MIN, kelly_min=KELLY_MIN, prob_gap_min=PROB_GAP_MIN,
                    cap_fav=CAP_FAV, cap_dog=CAP_DOG, cap_mid=CAP_MID, cap_global=CAP_GLOBAL,
                    risk_fav=RISK_FAV, risk_mid=RISK_MID, risk_dog=RISK_DOG, risk_book_dog=RISK_BOOK_DOG,
                )
                if bs in ("home(A)", "away(B)"):
                    real_home_won = int(rr["y_home_win"]) if pd.notnull(rr.get("y_home_win")) else None
                    if real_home_won in (0, 1):
                        bet_hit = 1 if ((bs == "home(A)" and real_home_won == 1) or (bs == "away(B)" and real_home_won == 0)) else 0
                        rows.append({"bet_hit": bet_hit, "stake_pct": stake, "reason": reason})

            if not rows:
                return {"bt_n_bets": 0, "bt_hit_rate": np.nan, "bt_avg_stake_pct": 0.0,
                        "bt_coverage_pct": 0.0, "bt_baseline_acc": baseline_acc}

            bt = pd.DataFrame(rows)
            n_bets = len(bt)
            hit_rate = float(bt["bet_hit"].mean())
            avg_stake_pct = float(bt["stake_pct"].mean())
            coverage_pct = 100.0 * n_bets / len(eval_df)
            return {"bt_n_bets": int(n_bets), "bt_hit_rate": hit_rate, "bt_avg_stake_pct": avg_stake_pct,
                    "bt_coverage_pct": coverage_pct, "bt_baseline_acc": baseline_acc}

        # PREP eval_df amb dataset_final + matches_enr
        eval_df = _prep_eval_predictions(dataset_final, matches_enr, model, scaler, iso, model_cols)

        # (A) Hi-conf sense quotes
        hc_stats = _hi_conf_accuracy(eval_df, p_hi=0.60, p_lo=0.40)
        # (B) Backtest estratègia amb quotes
        bt_stats = _backtest_hit_rate(eval_df)

        # GUARDA totes les mètriques
        metrics_to_save = metrics.copy()
        metrics_to_save["timestamp"] = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
        metrics_to_save["model_type"] = model.__class__.__name__
        metrics_to_save.update({
            "hc_n_picks": hc_stats["hc_n_picks"],
            "hc_hit_rate": hc_stats["hc_hit_rate"],
            "hc_cov_pct": hc_stats["hc_cov_pct"],
            "bt_n_bets": bt_stats["bt_n_bets"],
            "bt_hit_rate": bt_stats["bt_hit_rate"],
            "bt_avg_stake_pct": bt_stats["bt_avg_stake_pct"],
            "bt_coverage_pct": bt_stats["bt_coverage_pct"],
            "bt_baseline_acc": bt_stats["bt_baseline_acc"],
        })

        with open(os.path.join(OUT_DIR, "train_metrics.json"), "w") as f:
            json.dump(metrics_to_save, f, indent=2)

        # MOSTRAR resultats
        st.json(metrics_to_save)

        st.markdown("#### High-confidence picks (no odds needed)")
        cH1, cH2, cH3 = st.columns(3)
        cH1.metric("Hit-rate hi-conf", ("—" if (hc_stats["hc_n_picks"] == 0 or not np.isfinite(hc_stats["hc_hit_rate"])) else f"{hc_stats['hc_hit_rate']*100:.1f}%"))
        cH2.metric("Cobertura hi-conf", f"{hc_stats['hc_cov_pct']:.1f}%")
        cH3.metric("#picks hi-conf", f"{hc_stats['hc_n_picks']}")

        st.markdown("---")
        st.markdown("#### Strategy picks (ML A/B only, filtered)")
        cA, cB, cC = st.columns(3)
        cA.metric("Hit-rate picks ML filtrades", ("—" if (bt_stats["bt_n_bets"] == 0 or not np.isfinite(bt_stats["bt_hit_rate"])) else f"{bt_stats['bt_hit_rate']*100:.1f}%"))
        cB.metric("Cobertura ML filtrada", f"{bt_stats['bt_coverage_pct']:.1f}%")
        cC.metric("Stake mig recomanat", f"{bt_stats['bt_avg_stake_pct']:.2f}%")
        st.caption("Baseline naive (p>=0.5 → home, sinó away): " + ("—" if not np.isfinite(bt_stats["bt_baseline_acc"]) else f"{bt_stats['bt_baseline_acc']*100:.1f}%"))

        status.update(label="Done. Metrics computed & saved.", state="complete")

    # ─────────────────────────────────────────────────────────────
    # 5) Predict upcoming (if any) — usa el dataset_final i matches_train (consistent)
    # ─────────────────────────────────────────────────────────────
    preds = predict_upcoming(model, scaler, iso, dataset_final, model_cols, matches_train)
    if len(preds):
        st.write(preds.head(20))
        preds.to_csv(os.path.join(OUT_DIR, "predictions_upcoming.csv"), index=False)
        st.success(f"Saved outputs/predictions_upcoming.csv ({len(preds)} rows)")
    else:
        st.info("No upcoming/unknown-winner matches detected in (filtered) TML data.")





#with :
#    st.subheader("Preview saved data")
#    def show_csv(path, n=200):
#        if os.path.exists(path):
#            st.caption(path)
#            st.dataframe(pd.read_csv(path).head(n))
#        else:
#            st.warning(f"Missing: {path}")
#    colA, colB = st.columns(2)
#    with colA:
#        show_csv(os.path.join(DATA_DIR, "matches.csv"))
#        show_csv(os.path.join(OUT_DIR, "features_player_pre.csv"))
#    with colB:
#        show_csv(os.path.join(DATA_DIR, "points_sets_games.csv"))
#        show_csv(os.path.join(OUT_DIR, "dataset_match_level.csv"))

with tab2:
    import os, re, pandas as pd, numpy as np, joblib, html
    import streamlit as st
    from datetime import datetime
    import requests

    # =========================
    # CONSTANTS / PATHS
    # =========================
    OUT_DIR = OUT_DIR  # assumim que ja existeix globalment
    DATA_DIR = DATA_DIR
    LOG_PATH = os.path.join(OUT_DIR, "predictions_log.csv")
    PENDING_PATH = os.path.join(OUT_DIR, "pending_queue.csv")

    # =========================
    # DATE HELPERS
    # =========================
    def _canonicalize_date(val, match_id=""):
        """
        Retorna 'YYYY-MM-DD':
          1) parseja val
          2) si falla: extreu YYYY-MM-DD del match_id
          3) si falla: avui (UTC)
        """
        dt = pd.to_datetime(val, errors="coerce")
        if pd.isna(dt):
            m = re.search(r"(\d{4}-\d{2}-\d{2})", str(match_id))
            if m:
                dt = pd.to_datetime(m.group(1), errors="coerce")
        if pd.isna(dt):
            dt = pd.Timestamp.utcnow().normalize()
        return dt.strftime("%Y-%m-%d")

    def _now_utc_str():
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")

    def _repair_dates_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Neteja robusta de 'date' per evitar 'None':
         - si 'date' és buida/NaN/None → busca YYYY-MM-DD al match_id
         - si no hi és → usa pred_time_utc
         - si encara falla → avui UTC
         - ALWAYS escriu 'date' com a string YYYY-MM-DD
        """
        d = df.copy()
        # original
        dates = pd.to_datetime(d.get("date"), errors="coerce")
        # regex al match_id
        rx = d.get("match_id", "").astype(str).str.extract(r"(\d{4}-\d{2}-\d{2})", expand=False)
        rx_dt = pd.to_datetime(rx, errors="coerce")
        # pred_time_utc
        ptu_dt = pd.to_datetime(d.get("pred_time_utc"), errors="coerce")
        # today
        today = pd.Timestamp.utcnow().normalize()

        # emplena
        filled = dates.copy()
        filled = filled.where(~filled.isna(), rx_dt)
        filled = filled.where(~filled.isna(), ptu_dt)
        filled = filled.where(~filled.isna(), today)

        d["date"] = filled.dt.strftime("%Y-%m-%d")
        return d

    # =========================
    # META HELPERS (UPDATED)
    # =========================
    def _meta_paths():
        pmeta_path = os.path.join(DATA_DIR, "players_meta.csv")
        tmeta_path = os.path.join(DATA_DIR, "tournaments_meta.csv")
        return pmeta_path, tmeta_path

    def load_meta_tables():
        """
        Carrega players_meta.csv i tournaments_meta.csv.
        Normalitza columnes i crea camps mínims:
          - pmeta: player_id(str), country_id(str), is_lefty(int)
          - tmeta: tournament(str), country_id(str), level(str), is_neutral(int)
        """
        pmeta_path, tmeta_path = _meta_paths()
        pmeta = pd.read_csv(pmeta_path) if os.path.exists(pmeta_path) else pd.DataFrame()
        tmeta = pd.read_csv(tmeta_path) if os.path.exists(tmeta_path) else pd.DataFrame()

        # ---- Players meta ----
        if not pmeta.empty:
            pm = pmeta.copy()
            pm.columns = [c.strip().lower() for c in pm.columns]

            # id
            if "player_id" not in pm.columns:
                cand = [c for c in pm.columns if c in ("id", "atp_id")]
                if cand:
                    pm = pm.rename(columns={cand[0]: "player_id"})
            if "player_id" in pm.columns:
                pm["player_id"] = pm["player_id"].astype(str)
            else:
                pm["player_id"] = ""

            # country
            if "country_id" not in pm.columns:
                pm["country_id"] = "??"
            pm["country_id"] = pm["country_id"].astype(str)

            # lefty
            if "is_lefty" not in pm.columns:
                if "handedness" in pm.columns:
                    pm["is_lefty"] = pm["handedness"].astype(str).str.upper().str.startswith("L").astype(int)
                else:
                    pm["is_lefty"] = 0
            pm["is_lefty"] = pd.to_numeric(pm["is_lefty"], errors="coerce").fillna(0).astype(int)

            # (compat) aquests camps poden existir al teu pipeline; els mantenim però NO afecten home calc
            for c in ["serve_score", "return_score"]:
                if c not in pm.columns:
                    pm[c] = np.nan

            pmeta = pm

        # ---- Tournaments meta ----
        if not tmeta.empty:
            tm = tmeta.copy()
            tm.columns = [c.strip().lower() for c in tm.columns]

            # tournament name column
            if "tournament" not in tm.columns:
                for cand in ["name", "tournament_name", "tourney", "tournament_title"]:
                    if cand in tm.columns:
                        tm = tm.rename(columns={cand: "tournament"})
                        break
            if "tournament" in tm.columns:
                tm["tournament"] = tm["tournament"].astype(str)
            else:
                tm["tournament"] = ""

            # country + level
            if "country_id" not in tm.columns:
                tm["country_id"] = "N/A"
            tm["country_id"] = tm["country_id"].astype(str)

            if "level" not in tm.columns:
                tm["level"] = "ATP"
            tm["level"] = tm["level"].astype(str)

            # neutral flag (només per anul·lar home; NO és feature del model)
            if "is_neutral" not in tm.columns:
                if "neutral" in tm.columns:
                    tm["is_neutral"] = (
                        (tm["neutral"].astype(str).str.lower().isin(["1", "true", "yes", "neutral"])) |
                        (tm["country_id"].astype(str).str.upper().isin(["N/A", "NA", "??", "NEUTRAL", ""]))
                    ).astype(int)
                else:
                    tm["is_neutral"] = (tm["country_id"].astype(str).str.upper().isin(["N/A", "NA", "??", "NEUTRAL", ""])).astype(int)
            tm["is_neutral"] = pd.to_numeric(tm["is_neutral"], errors="coerce").fillna(0).astype(int)

            tmeta = tm

        return pmeta, tmeta

    def add_meta_features_v2(dataset_df: pd.DataFrame,
                             matches_df: pd.DataFrame,
                             pmeta: pd.DataFrame,
                             tmeta: pd.DataFrame):
        """
        VERSIÓ ACTUALITZADA:
        - Per "home" NOMÉS calcula un únic senyal numèric: home_advantage = A_home - B_home
        - A_home/B_home només són 1 si:
            * el torneig NO és neutral
            * torneig té country_id usable
            * el country_id del jugador coincideix exactament amb el del torneig (comparació normalitzada)
        - NO afegeix home_A/home_B/is_neutral com a features del model
        """
        df = dataset_df.copy()

        # 0) Porta camps clau des de matches (si hi són)
        mcols = [c for c in ["match_id", "player_a_id", "player_b_id", "tournament", "surface", "best_of_5"]
                 if c in matches_df.columns]
        if "match_id" not in df.columns:
            raise ValueError("add_meta_features_v2: falta 'match_id' a dataset_raw")
        if mcols:
            df = df.merge(matches_df[mcols].drop_duplicates("match_id"), on="match_id", how="left")

        # garanteix existència
        for c, default in [("player_a_id", ""), ("player_b_id", ""), ("tournament", ""), ("best_of_5", 0)]:
            if c not in df.columns:
                df[c] = default

        # 1) Players meta → A/B (country_id + is_lefty)
        if not pmeta.empty:
            pm = pmeta.copy()
            pm.columns = [c.strip().lower() for c in pm.columns]

            if "player_id" not in pm.columns:
                cand = [c for c in pm.columns if c in ("id", "atp_id")]
                if cand:
                    pm = pm.rename(columns={cand[0]: "player_id"})

            pm["player_id"] = pm["player_id"].astype(str) if "player_id" in pm.columns else ""
            if "country_id" not in pm.columns:
                pm["country_id"] = "??"
            if "is_lefty" not in pm.columns:
                if "handedness" in pm.columns:
                    pm["is_lefty"] = pm["handedness"].astype(str).str.upper().str.startswith("L").astype(int)
                else:
                    pm["is_lefty"] = 0

            pm["country_id"] = pm["country_id"].astype(str)
            pm["is_lefty"] = pd.to_numeric(pm["is_lefty"], errors="coerce").fillna(0).astype(int)

            pA = pm.rename(columns={
                "player_id": "player_a_id",
                "country_id": "A_country_id",
                "is_lefty": "A_is_lefty"
            })[["player_a_id", "A_country_id", "A_is_lefty"]].drop_duplicates("player_a_id")

            pB = pm.rename(columns={
                "player_id": "player_b_id",
                "country_id": "B_country_id",
                "is_lefty": "B_is_lefty"
            })[["player_b_id", "B_country_id", "B_is_lefty"]].drop_duplicates("player_b_id")

            df["player_a_id"] = df["player_a_id"].astype(str)
            df["player_b_id"] = df["player_b_id"].astype(str)

            df = df.merge(pA, on="player_a_id", how="left")
            df = df.merge(pB, on="player_b_id", how="left")
        else:
            df["A_country_id"] = "??"
            df["B_country_id"] = "??"
            df["A_is_lefty"] = 0
            df["B_is_lefty"] = 0

        # 2) Tournament meta (country_id + level + is_neutral intern)
        if not tmeta.empty:
            tm = tmeta.copy()
            tm.columns = [c.strip().lower() for c in tm.columns]

            if "tournament" not in tm.columns:
                for cand in ["name", "tournament_name", "tourney", "tournament_title"]:
                    if cand in tm.columns:
                        tm = tm.rename(columns={cand: "tournament"})
                        break

            if "tournament" in tm.columns:
                keep = ["tournament"]
                if "country_id" in tm.columns:
                    keep.append("country_id")
                if "level" in tm.columns:
                    keep.append("level")
                if "is_neutral" in tm.columns:
                    keep.append("is_neutral")

                tm = tm[keep].drop_duplicates("tournament").rename(columns={
                    "country_id": "tourney_country_id",
                    "level": "tourney_level"
                })

                if "is_neutral" not in tm.columns:
                    tm["is_neutral"] = 0

                df["tournament"] = df["tournament"].astype(str)
                tm["tournament"] = tm["tournament"].astype(str)
                df = df.merge(tm, on="tournament", how="left")

        # defaults torneo
        if "tourney_country_id" not in df.columns:
            df["tourney_country_id"] = "N/A"
        if "tourney_level" not in df.columns:
            df["tourney_level"] = "ATP"
        if "is_neutral" not in df.columns:
            df["is_neutral"] = 0

        # 3) Derivades NUMÈRIQUES (HOME: només home_advantage)
        HOME_NEUTRAL_CODES = {"", "N/A", "NA", "??", "NEUTRAL", "NONE", "NULL"}

        def _clean_code(s: pd.Series) -> pd.Series:
            x = s.astype(str).str.strip().str.upper()
            x = x.where(~x.isin(HOME_NEUTRAL_CODES), "")
            return x

        A_c = _clean_code(df["A_country_id"].fillna("??"))
        B_c = _clean_code(df["B_country_id"].fillna("??"))
        T_c = _clean_code(df["tourney_country_id"].fillna("N/A"))

        # si el torneig és neutral, anul·la qualsevol "home"
        neu_flag = pd.to_numeric(df.get("is_neutral", 0), errors="coerce").fillna(0).astype(int)
        T_c = T_c.where(neu_flag.eq(0), "")

        A_home = (T_c != "") & (A_c != "") & (A_c == T_c)
        B_home = (T_c != "") & (B_c != "") & (B_c == T_c)

        df["home_advantage"] = (A_home.astype(int) - B_home.astype(int)).astype(int)

        # Lefty flags numèrics
        df["A_is_lefty"] = pd.to_numeric(df.get("A_is_lefty", 0), errors="coerce").fillna(0).astype(int)
        df["B_is_lefty"] = pd.to_numeric(df.get("B_is_lefty", 0), errors="coerce").fillna(0).astype(int)

        # Bo5 flag numèric
        df["best_of_5"] = pd.to_numeric(df.get("best_of_5", 0), errors="coerce").fillna(0).astype(int)

        # Level code (mateixa idea que abans; si vols mapping estable, fes-ho també a train)
        df["tourney_level_code"] = pd.factorize(df["tourney_level"].fillna("ATP").astype(str))[0].astype(int)

        # Features numèriques que afegirem al model (inclou home_advantage, NO inclou is_neutral/home_A/home_B)
        added_cols = [
            "A_is_lefty", "B_is_lefty",
            "home_advantage",
            "best_of_5", "tourney_level_code"
        ]

        for c in added_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).replace([np.inf, -np.inf], 0.0)

        return df, added_cols

    # ---------------- Models/helpers ----------------
    def _read_model_columns(out_dir: str):
        fp = os.path.join(out_dir, "model_columns.txt")
        if os.path.exists(fp):
            with open(fp, "r") as f:
                cols = [ln.strip() for ln in f if ln.strip()]
            return cols
        return None

    def _pick_latest_model_path(out_dir: str):
        cand = []
        for name in ("model_lightgbm.pkl", "model_logistic.pkl"):
            p = os.path.join(out_dir, name)
            if os.path.exists(p):
                cand.append((os.path.getmtime(p), p))
        if not cand:
            return None
        cand.sort(reverse=True)
        return cand[0][1]

    def _load_model_bundle():
        model = scaler = iso = None
        mp = _pick_latest_model_path(OUT_DIR)
        if mp is not None:
            model = joblib.load(mp)
        sc_path = os.path.join(OUT_DIR, "scaler.pkl")
        if os.path.exists(sc_path):
            scaler = joblib.load(sc_path)
        for ip in ("calibrator_isotonic.pkl", "isotonic.pkl"):
            cp = os.path.join(OUT_DIR, ip)
            if os.path.exists(cp):
                iso = joblib.load(cp)
                break
        train_cols = _read_model_columns(OUT_DIR)
        return model, scaler, iso, train_cols

    def _prep_X_for_model(df_feats: pd.DataFrame, train_cols: list, scaler):
        if train_cols is None:
            st.warning("No s'ha trobat model_columns.txt — faré servir les columnes actuals (pot desalinear).")
            train_cols = [c for c in df_feats.columns if c not in ("match_id", "date", "surface", "y_home_win")]
        Xdf = df_feats.reindex(columns=train_cols, fill_value=0.0)
        Xdf = Xdf.apply(pd.to_numeric, errors="coerce")
        Xdf = Xdf.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X = Xdf.values
        if scaler is not None and hasattr(scaler, "transform"):
            X = scaler.transform(X)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X, train_cols

    def _alt_hint_row(r):
        try:
            if str(r.get("best_side", "")).lower() != "no bet":
                return ""
            p_h = float(r["p_home_win"])
            p_a = 1.0 - p_h
            eh = pd.to_numeric(r.get("edge_home"), errors="coerce")
            ea = pd.to_numeric(r.get("edge_away"), errors="coerce")
            oh = pd.to_numeric(r.get("odds_home"), errors="coerce")
            oa = pd.to_numeric(r.get("odds_away"), errors="coerce")
            bo5 = bool(r.get("best_of_5", 0))
            return alt_market_hint(
                p_h, oh, p_a, oa, eh, ea, best_of_5=bo5,
                margin_min=MKT_MARGIN_MIN, edge_min=EDGE_MIN, alt_edge_soft=ALT_EDGE_SOFT,
                fav_p_min=FAV_P_MIN, dog_p_max=DOG_P_MAX, c_edge_soft=C_EDGE_SOFT,
            )
        except Exception:
            return ""

    try:
        _enrich_dom = enrich_matches_domestic
    except NameError:
        _enrich_dom = lambda df: df

    # ---------------- Telegram helpers ----------------
    def _normalize_chat_id(val: str) -> str:
        if not val:
            return ""
        s = str(val).strip()
        if s.startswith("https://t.me/"):
            s = s[len("https://t.me/"):]
        if s.startswith("+"):
            return s
        if s and not s.startswith("@") and not s.startswith("-100"):
            if all(ch not in s for ch in (" ", "/", "\\")):
                s = "@" + s
        return s

    def _valid_chat_id(s: str) -> bool:
        return bool(s) and (s.startswith("@") or s.startswith("-100"))

    def _safe_float(x, default=np.nan):
        try:
            xx = float(x)
            return xx if np.isfinite(xx) else default
        except Exception:
            return default

    def _units_from_half_kelly(stake_pct):
        s = _safe_float(stake_pct, 0.0)
        if s < 1.00:
            return 0.5
        if s <= 1.50:
            return 1.0
        if s <= 4.00:
            return 2.0
        return 3.0

    def _compute_tip_fields(row):
        def _sf(v, d=np.nan):
            try:
                vv = float(v)
                return vv if np.isfinite(vv) else d
            except Exception:
                return d

        side = str(row.get("best_side", "")).lower()
        stake_pct = _sf(row.get("stake_% (half kelly)"), 0.0)

        p_h = _sf(row.get("p_home_win"))
        p_a = _sf(row.get("p_away_win"))
        oh = _sf(row.get("odds_home"))
        oa = _sf(row.get("odds_away"))

        has_price_home = np.isfinite(oh) and (oh > 1.0)
        has_price_away = np.isfinite(oa) and (oa > 1.0)

        who = ""
        price = prob = fair = value = np.nan
        ml = "no bet"

        if side.startswith("home"):
            who = row.get("player_a_name") or row.get("player_a_id") or ""
            price = oh
            prob = p_h
            fair = (1.0 / prob) if (np.isfinite(prob) and prob > 0) else np.nan
            value = (prob - (1.0 / price)) if (np.isfinite(prob) and np.isfinite(price) and price > 0) else np.nan
            ml = "home(A)"
        elif side.startswith("away"):
            who = row.get("player_b_name") or row.get("player_b_id") or ""
            price = oa
            prob = p_a
            fair = (1.0 / prob) if (np.isfinite(prob) and prob > 0) else np.nan
            value = (prob - (1.0 / price)) if (np.isfinite(prob) and np.isfinite(price) and price > 0) else np.nan
            ml = "away(B)"

        units = _units_from_half_kelly(stake_pct)
        reason = str(row.get("decision_reason", "")).strip().lower()

        no_bet = (
            (not side.startswith(("home", "away"))) or
            (stake_pct <= 0.0) or
            (not np.isfinite(price)) or
            (price <= 1.0)
        )

        if (not has_price_home and not has_price_away) and reason == "":
            reason = "no_odds"

        return {
            "bet_on_player": who,
            "bet_market": "Moneyline",
            "bet_price": price,
            "bet_prob": prob,
            "bet_fair": fair,
            "bet_value": value,
            "bet_value_pct": (value * 100.0 if np.isfinite(value) else np.nan),
            "bet_units": units,
            "half_kelly_pct": stake_pct,
            "pick_side": ml,
            "prob_A": p_h,
            "prob_B": p_a,
            "no_bet": bool(no_bet),
            "reason": reason,
        }

    def _format_tip_message(row, tier_label="PREMIUM", allow_markdown=True):
        import html
        tname = str(row.get("tournament") or row.get("city") or "—")
        pA = str(row.get("player_a_name") or row.get("player_a_id") or "A")
        pB = str(row.get("player_b_name") or row.get("player_b_id") or "B")
        surface = str(row.get("surface") or "—")
        bo5 = int(_safe_float(row.get("best_of_5"), 0)) == 1
        book = str(row.get("bookmaker") or "").strip()

        f = _compute_tip_fields(row)

        probA_txt = "—" if not np.isfinite(f["prob_A"]) else f"{f['prob_A'] * 100:.1f}%"
        probB_txt = "—" if not np.isfinite(f["prob_B"]) else f"{f['prob_B'] * 100:.1f}%"
        fairA_txt = "—" if not np.isfinite(f["prob_A"]) else f"{1.0 / max(f['prob_A'], 1e-9):.2f}"
        fairB_txt = "—" if not np.isfinite(f["prob_B"]) else f"{1.0 / max(f['prob_B'], 1e-9):.2f}"

        def iso2_to_flag(code2: str) -> str:
            if not code2:
                return ""
            c = str(code2).strip().upper()
            if len(c) != 2 or not c.isalpha():
                return ""
            # Regional Indicator Symbols
            return chr(ord(c[0]) + 127397) + chr(ord(c[1]) + 127397)

        def _norm_tournament_name(s: str) -> str:
            s = str(s or "").strip().lower()
            s = re.sub(r"\s+", " ", s)
            s = re.sub(r"[^\w\s]", "", s)  # treu puntuació
            return s
        
        @st.cache_data(show_spinner=False)
        def load_tournament_country_map(data_dir: str) -> dict:
            path = os.path.join(data_dir, "tournaments_meta.csv")
            if not os.path.exists(path):
                return {}
        
            tm = pd.read_csv(path)
            tm.columns = [c.strip().lower() for c in tm.columns]
        
            # assegura columna tournament
            if "tournament" not in tm.columns:
                for cand in ["name", "tournament_name", "tourney", "tournament_title"]:
                    if cand in tm.columns:
                        tm = tm.rename(columns={cand: "tournament"})
                        break
        
            if "tournament" not in tm.columns:
                return {}
        
            # assegura country_id (ISO2)
            if "country_id" not in tm.columns:
                return {}
        
            out = {}
            for _, r in tm.iterrows():
                t = _norm_tournament_name(r.get("tournament", ""))
                cc = str(r.get("country_id", "") or "").strip().upper()[:2]
                if t and len(cc) == 2 and cc.isalpha():
                    out[t] = cc
        
            return out

        TOURNAMENT_COUNTRY_OVERRIDES = {
            # claus normalitzades
            _norm_tournament_name("Brisbane"): "AU",
            _norm_tournament_name("Barcelona Open"): "ES",
            _norm_tournament_name("Roland Garros"): "FR",
            _norm_tournament_name("Paris Masters"): "FR",
            # afegeix-ne els que calgui
        }

        def tournament_to_flag(tournament_name: str, data_dir: str) -> str:
            key = _norm_tournament_name(tournament_name)
            if not key:
                return ""
        
            # 1) override
            cc = TOURNAMENT_COUNTRY_OVERRIDES.get(key, "")
        
            # 2) lookup CSV
            if not cc:
                mp = load_tournament_country_map(data_dir)
                cc = mp.get(key, "")
        
            return iso2_to_flag(cc)

        flag = tournament_to_flag(tname, DATA_DIR)
        header = f"{flag} {html.escape(tname)}" if flag else f"{html.escape(tname)}"
        title = f"🎾 {html.escape(pA)} vs {html.escape(pB)}"
        meta = f"{'Bo5' if bo5 else 'Bo3'} · {html.escape(surface.capitalize())}"

        if f["no_bet"]:
            reason_map = {
                "no_odds": "No market odds provided.",
                "filtered_out": "Filters blocked the bet.",
                "cap": "Stake capped to 0 by risk caps.",
                "dog_value_min": f"Underdog value below +{MIN_DOG_VALUE_PCT:.0f}% threshold.",
            }
            rtxt = reason_map.get(f["reason"], f["reason"].capitalize() if f["reason"] else "No edge.")
            model = f"<b>Model:</b> {probA_txt} / {probB_txt} (fair {fairA_txt} / {fairB_txt})"
            msg = f"{title}\n🚫 <b>NO BET</b>\n{model}\n<b>Reason:</b> {html.escape(rtxt)}\n\n— This is not financial advice."
            return msg

        price_txt = "—" if not np.isfinite(f["bet_price"]) else f"{f['bet_price']:.2f}"
        fair_txt = "—" if not np.isfinite(f["bet_fair"]) else f"{f['bet_fair']:.2f}"
        prob_txt = "—" if not np.isfinite(f["bet_prob"]) else f"{f['bet_prob'] * 100:.1f}%"
        val_txt = "—" if not np.isfinite(f["bet_value_pct"]) else f"{f['bet_value_pct']:+.1f}%"
        units_txt = f"{f['bet_units']:.1f}u"

        line = f"<b>Bet:</b> {html.escape(str(f['bet_on_player']))} ML @ {price_txt}"
        if book and np.isfinite(f["bet_price"]):
            line += f"\n<b>Book:</b> {html.escape(book)}"
            
        stake = f"<b>Stake:</b> {units_txt}"
        stats = f"<b>Model:</b> {prob_txt} • <b>Value:</b> {val_txt}"

        msg = f"{header}\n{title}\n{line} • {stake}\n{stats}"
        return msg



    def _send_telegram_channel(token: str, chat_id: str, msg_html: str):
        if not token:
            return False, "BOT TOKEN missing", None
        if not chat_id:
            return False, "chat_id missing", None
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            resp = requests.post(
                url,
                data={"chat_id": str(chat_id), "text": msg_html, "parse_mode": "HTML", "disable_web_page_preview": True},
                timeout=15,
            )
            j = resp.json()
            if j.get("ok"):
                mid = str(j["result"]["message_id"])
                return True, "Sent", mid
            else:
                return False, f"Telegram error: {j}", None
        except Exception as e:
            return False, f"{type(e).__name__}: {e}", None

    def _append_to_log(rows_df: pd.DataFrame):
        os.makedirs(OUT_DIR, exist_ok=True)
        if os.path.exists(LOG_PATH):
            try:
                old = pd.read_csv(LOG_PATH)
            except Exception:
                old = pd.DataFrame()
        else:
            old = pd.DataFrame()
        allc = sorted(set(old.columns) | set(rows_df.columns))
        newdf = pd.concat([old.reindex(columns=allc), rows_df.reindex(columns=allc)], ignore_index=True)

        # dedup
        if "channel" in newdf.columns:
            newdf = newdf.drop_duplicates(subset=["match_id", "channel"], keep="last")
        else:
            newdf = newdf.drop_duplicates(subset=["match_id"], keep="last")

        # 🔒 arregla i força 'date' abans d'escriure
        newdf = _repair_dates_df(newdf)

        newdf.to_csv(LOG_PATH, index=False)

    def _publish_and_log(base_df: pd.DataFrame, channels=("premium",), source="manual", model_name=""):
        token = st.session_state.get("TG_TOKEN", "")
        premiumID = st.session_state.get("TG_PREMIUM", "")
        freeID = st.session_state.get("TG_FREE", "")

        id_map = {
            "premium": _normalize_chat_id(premiumID),
            "free": _normalize_chat_id(freeID),
        }
        results = {}
        row0 = base_df.iloc[0].to_dict()
        msg_p = _format_tip_message(row0, "PREMIUM")
        msg_f = _format_tip_message(row0, "FREE")

        rows_to_log = []
        for ch in channels:
            cid = id_map["premium"] if ch == "premium" else id_map["free"]
            msg = msg_p if ch == "premium" else msg_f

            ok = False
            info = "skipped"
            mid = ""
            if not _valid_chat_id(cid):
                info = f"Invalid chat_id '{cid}'. Usa @handle o -100..."
            else:
                ok, info, mid = _send_telegram_channel(token, cid, msg)

            now_utc = _now_utc_str()
            r = base_df.copy()

            # força data i pred_time_utc per cada fila
            r.loc[:, "date"] = r.apply(lambda rr: _canonicalize_date(rr.get("date"), rr.get("match_id")), axis=1)
            if "pred_time_utc" not in r.columns or r["pred_time_utc"].isna().any():
                r.loc[:, "pred_time_utc"] = now_utc

            r.loc[:, "channel"] = ch
            r.loc[:, "published_to"] = ch
            r.loc[:, "tg_status"] = ("sent" if ok else f"error: {info}")
            r.loc[:, "tg_message_id"] = mid or ""
            r.loc[:, "published_at_utc"] = now_utc
            r.loc[:, "source"] = source
            r.loc[:, "model_name"] = model_name
            rows_to_log.append(r)
            results[ch] = (ok, info, mid)

        if rows_to_log:
            _append_to_log(pd.concat(rows_to_log, ignore_index=True))

        return results

    # ---------------- Manual fixture UI ----------------
    st.markdown("---")
    st.subheader("Manual fixture (single match)")
    st.caption("Introdueix un partit manualment. Si no poses odds, es mostren igualment probabilitats i fair odds.")

    tmeta_path = os.path.join(DATA_DIR, "tournaments_meta.csv")
    tmeta = pd.read_csv(tmeta_path) if os.path.exists(tmeta_path) else pd.DataFrame()
    if "tournament" not in tmeta.columns:
        for cand in ["name", "tournament_name", "tourney", "tournament_title"]:
            if cand in tmeta.columns:
                tmeta = tmeta.rename(columns={cand: "tournament"})
                break

    _tour_options = ["(custom)"]
    if not tmeta.empty and "tournament" in tmeta.columns:
        _tour_options += sorted(tmeta["tournament"].dropna().astype(str).unique().tolist(), key=lambda s: s.lower())

    BOOKMAKERS_EU = [
        "(choose bookmaker)",
        "Bet365", "Sportium", "Winamax", "1xbet"
    ]

    with st.form("manual_fixture_form"):
        r1c1, r1c2, r1c3 = st.columns([1, 1, 1])
        m_date_ts = r1c1.date_input("Date", value=pd.Timestamp.today())
        m_surface = r1c2.selectbox("Surface", ["hard", "clay", "grass", "indoor-hard"], index=0)
        m_best5 = r1c3.checkbox("Best of 5?", value=False)

        # data NORMALITZADA des del principi
        event_date_str = pd.to_datetime(m_date_ts, errors="coerce").strftime("%Y-%m-%d")

        r1b = st.columns([1, 1])
        m_tour_sel = r1b[0].selectbox("Tournament", _tour_options, index=0, help="Origen: data/tournaments_meta.csv")
        if m_tour_sel == "(custom)":
            m_tournament = r1b[1].text_input("(custom)", "", placeholder="Ex: Barcelona Open")
        else:
            m_tournament = r1b[1].text_input("(custom)", m_tour_sel)

        r2c1, r2c2 = st.columns([1, 1])
        m_player_a = r2c1.text_input("Player A (surname, full name or ATP ID)", "", key="manual_player_a")
        m_player_b = r2c2.text_input("Player B (surname, full name or ATP ID)", "", key="manual_player_b")

        r3c1, r3c2 = st.columns([1, 1])
        m_odds_home = r3c1.text_input("Odds for Player A (optional)", "", key="manual_odds_home")
        m_odds_away = r3c2.text_input("Odds for Player B (optional)", "", key="manual_odds_away")

        m_bookmaker = st.selectbox("Bookmaker", BOOKMAKERS_EU, index=1)

        submitted = st.form_submit_button("Predict manual match", use_container_width=True)

    # -------- RUN PREDICTION on submit, store in session --------
    if submitted:
        MAP_FULL = st.session_state.get("MAP_FULL", {})
        MAP_ALIAS = st.session_state.get("MAP_ALIAS", {})
        INDEX_LAST = st.session_state.get("INDEX_LAST", {})

        def _as_id(x):
            xs = str(x).strip()
            return xs if xs.isdigit() else xs

        def _pid_key(pid: str):
            s = str(pid).strip()
            return (0, int(s)) if s.isdigit() else (1, s)
        
        def _canonicalize_ab(a_id, b_id, a_name, b_name, odds_a, odds_b):
            # retorna A/B canònic i si s'ha fet swap
            if _pid_key(a_id) <= _pid_key(b_id):
                return a_id, b_id, a_name, b_name, odds_a, odds_b, False
            return b_id, a_id, b_name, a_name, odds_b, odds_a, True


        a_pid, a_mode, a_conf = resolve_player_name(m_player_a, MAP_FULL, MAP_ALIAS, INDEX_LAST) if (MAP_FULL or MAP_ALIAS or INDEX_LAST) else (_as_id(m_player_a), "RAW", 0.0)
        b_pid, b_mode, b_conf = resolve_player_name(m_player_b, MAP_FULL, MAP_ALIAS, INDEX_LAST) if (MAP_FULL or MAP_ALIAS or INDEX_LAST) else (_as_id(m_player_b), "RAW", 0.0)

        # odds com floats (o np.nan)
        oh = pd.to_numeric(m_odds_home, errors="coerce") if m_odds_home else np.nan
        oa = pd.to_numeric(m_odds_away, errors="coerce") if m_odds_away else np.nan
        
        A_id, B_id, A_name, B_name, oh_c, oa_c, swapped = _canonicalize_ab(
            str(a_pid), str(b_pid),
            str(m_player_a), str(m_player_b),
            oh, oa
        )


        if not a_pid or not b_pid or str(a_pid) == str(b_pid):
            st.error("Calen dos jugadors diferents (A i B).")
        else:
            m_indoor = 1 if m_surface == "indoor-hard" else 0

            # ⚠️ match_id amb data ISO segura
            mid = f"manual_{event_date_str}_{_norm_name(m_player_a)}_vs_{_norm_name(m_player_b)}"

            manual_row = pd.DataFrame([{
                "match_id": mid,
                "date": event_date_str,  # <-- sempre YYYY-MM-DD
                "tournament": str(m_tournament).strip() if m_tournament else "Manual",
                "city": "Manual",
                "country": "",
                "level": "A",
                "round": "",
                "best_of_5": int(m_best5),
                "surface": m_surface,
                "indoor": int(m_indoor),
                "player_a_id": A_id,
                "player_b_id": B_id,
                "winner_id": "",
                "duration_minutes": np.nan,
                "odds_home": oh_c,
                "odds_away": oa_c,
                "player_a_name": A_name,
                "player_b_name": B_name,
                "bookmaker": ("" if m_bookmaker == "(choose bookmaker)" else str(m_bookmaker))
            }])

            base_matches_path = os.path.join(DATA_DIR, "matches.csv")
            base = pd.read_csv(base_matches_path) if os.path.exists(base_matches_path) else pd.DataFrame(columns=list(manual_row.columns))
            merged = pd.concat([base, manual_row], ignore_index=True).drop_duplicates("match_id")
            merged.to_csv(base_matches_path, index=False)

            merged_enr = _enrich_dom(merged)
            psg_path = os.path.join(DATA_DIR, "points_sets_games.csv")
            psg = pd.read_csv(psg_path) if os.path.exists(psg_path) else None
            feats = compute_pre_match_features_v2(merged_enr, psg)
            dataset_base, _gen_cols = make_match_features(feats, merged_enr)

            # ✅ META FEATURES (UPDATED): només home_advantage per "home"
            pmeta_tbl, tmeta_tbl = load_meta_tables()
            dataset_enh, _added_cols = add_meta_features_v2(dataset_base, merged_enr, pmeta_tbl, tmeta_tbl)

            model, scaler, iso, train_cols = _load_model_bundle()
            if model is None:
                st.error("No s'ha trobat cap model a outputs/. Entrena primer a 'Refresh & Train'.")
            else:
                ds_fx = dataset_enh[dataset_enh["match_id"] == mid].copy()
                if not len(ds_fx):
                    st.error("No s'han generat features per al partit manual. Revisa IDs/noms i la data.")
                else:
                    X, _ = _prep_X_for_model(ds_fx, train_cols, scaler)
                    if hasattr(model, "predict_proba"):
                        p = model.predict_proba(X)[:, 1]
                    else:
                        try:
                            p = model.predict(X, num_iteration=getattr(model, "best_iteration", None))
                        except TypeError:
                            p = model.predict(X)
                    if iso is not None:
                        p = iso.transform(p)

                    p_h = float(np.clip(p[0], 1e-6, 1 - 1e-6))
                    p_a = 1.0 - p_h

                    # construeix 'out' amb data ISO + pred_time_utc
                    out = pd.DataFrame([{
                        "match_id": mid,
                        "date": event_date_str,
                        "pred_time_utc": _now_utc_str(),
                    }])

                    out.loc[0, "player_a_id"] = A_id
                    out.loc[0, "player_b_id"] = B_id
                    out.loc[0, "player_a_name"] = A_name
                    out.loc[0, "player_b_name"] = B_name
                    out.loc[0, "tournament"] = str(m_tournament).strip() if m_tournament else "Manual"
                    out.loc[0, "p_home_win"] = p_h
                    out.loc[0, "p_away_win"] = p_a
                    out.loc[0, "fair_odds_home"] = 1.0 / p_h
                    out.loc[0, "fair_odds_away"] = 1.0 / p_a
                    out.loc[0, "best_of_5"] = int(m_best5)
                    out.loc[0, "surface"] = m_surface
                    out.loc[0, "bookmaker"] = ("" if m_bookmaker == "(choose bookmaker)" else str(m_bookmaker))

                    def _to_float_or_nan(x):
                        v = pd.to_numeric(x, errors="coerce")
                        try:
                            v = float(v)
                        except Exception:
                            return np.nan
                        return v if np.isfinite(v) else np.nan
                    
                    def _valid_odd(v):
                        v = _to_float_or_nan(v)
                        return np.isfinite(v) and v > 1.0

                    oh = _to_float_or_nan(manual_row["odds_home"].values[0])
                    oa = _to_float_or_nan(manual_row["odds_away"].values[0])
                    
                    no_odds = (not _valid_odd(oh)) and (not _valid_odd(oa))

                    #oh = pd.to_numeric(manual_row["odds_home"], errors="coerce").values[0]
                    #oa = pd.to_numeric(manual_row["odds_away"], errors="coerce").values[0]
                    #no_odds = (not (isinstance(oh, (int, float)) and np.isfinite(oh) and oh > 1.0)) and \
                    #          (not (isinstance(oa, (int, float)) and np.isfinite(oa) and oa > 1.0))
                    
                    if no_odds:
                        out.loc[0, ["edge_home", "edge_away", "kelly_home", "kelly_away"]] = [np.nan, np.nan, 0.0, 0.0]
                        out.loc[0, "best_side"] = "by_prob: " + ("home(A)" if p_h >= 0.5 else "away(B)")
                        out.loc[0, ["best_edge", "stake_% (half kelly)", "decision_reason"]] = [np.nan, 0.0, "no_odds"]
                        out["alt_markets_hint"] = ""
                    else:
                        edge_h = p_h - (1.0 / oh if _valid_odd(oh) else np.nan)
                        edge_a = p_a - (1.0 / oa if _valid_odd(oa) else np.nan)
                        
                        def _k(pv, ov):
                            return max(((ov - 1.0) * pv - (1.0 - pv)) / (ov - 1.0), 0.0) if _valid_odd(ov) else 0.0

                        #def _k(pv, ov):
                        #    if not (isinstance(ov, (int, float)) and np.isfinite(ov) and ov > 1.0):
                        #        return 0.0
                        #    b = ov - 1.0
                        #    return max((b * pv - (1 - pv)) / b, 0.0)

                        kh = _k(p_h, oh)
                        ka = _k(p_a, oa)
                        out.loc[0, "odds_home"] = oh
                        out.loc[0, "odds_away"] = oa
                        out.loc[0, "edge_home"] = edge_h
                        out.loc[0, "edge_away"] = edge_a
                        out.loc[0, "kelly_home"] = kh
                        out.loc[0, "kelly_away"] = ka

                        bs, be, stake, reason = decide_with_filters(
                            p_h, oh, p_a, oa, kh, ka, edge_h, edge_a,
                            fav_p_min=FAV_P_MIN,
                            dog_p_max=DOG_P_MAX,
                            edge_min=EDGE_MIN,
                            margin_min=MKT_MARGIN_MIN,
                            kelly_min=KELLY_MIN,
                            prob_gap_min=PROB_GAP_MIN,
                            cap_fav=CAP_FAV,
                            cap_dog=CAP_DOG,
                            cap_mid=CAP_MID,
                            cap_global=CAP_GLOBAL,
                            risk_fav=RISK_FAV,
                            risk_mid=RISK_MID,
                            risk_dog=RISK_DOG,
                            risk_book_dog=RISK_BOOK_DOG,
                        )

                        # sota-dog: value mínim
                        val_pct = (be * 100.0) if (be is not None and np.isfinite(be)) else np.nan
                        is_dog_pick = ((str(bs).lower().startswith("home") and p_h < 0.5) or
                                       (str(bs).lower().startswith("away") and p_a < 0.5))
                        if is_dog_pick and (not np.isfinite(val_pct) or val_pct < MIN_DOG_VALUE_PCT):
                            bs = "no bet"
                            stake = 0.0
                            reason = (str(reason).strip() + "|dog_value_min").strip("|")

                        out.loc[0, "best_side"] = bs
                        out.loc[0, "best_edge"] = be
                        out.loc[0, "stake_% (half kelly)"] = stake
                        out.loc[0, "decision_reason"] = reason
                        out["alt_markets_hint"] = out.apply(_alt_hint_row, axis=1)

                    # segura a sessió
                    out.loc[0, "date"] = _canonicalize_date(out.loc[0, "date"], mid)

                    st.session_state["last_manual_tip"] = out.copy()
                    st.session_state["last_model_name"] = getattr(model, "__name__", getattr(model, "__class__", type("x", (object,), {})).__name__)
                    st.success("Predicció preparada. Revisa el preview i usa els botons d'enviament de sota.")

    # -------------- PUBLISHING AREA --------------
    st.markdown("---")
    st.subheader("Approval & Telegram send")

    if "TG_TOKEN" not in st.session_state:
        st.session_state["TG_TOKEN"] = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
        st.session_state["TG_PREMIUM"] = st.secrets.get("TELEGRAM_CHAT_ID_PREMIUM", "")
        st.session_state["TG_FREE"] = st.secrets.get("TELEGRAM_CHAT_ID_FREE", "")

    c1, c2 = st.columns(2)
    st.session_state["TG_TOKEN"] = c1.text_input("BOT TOKEN", st.session_state["TG_TOKEN"], type="password")
    st.session_state["TG_PREMIUM"] = c2.text_input("CHAT_ID PREMIUM", st.session_state["TG_PREMIUM"], help="Ex. @tennispicksIA o -100xxxxxxxxxx")
    st.session_state["TG_FREE"] = c2.text_input("CHAT_ID FREE", st.session_state["TG_FREE"], help="Ex. @tennispicksIA_free o -100xxxxxxxxxx", key="tg_free2")

    for lab, v in [("Premium", st.session_state["TG_PREMIUM"]), ("Free", st.session_state["TG_FREE"])]:
        if str(v).startswith("https://t.me/+"):
            st.warning(f"{lab}: l’enllaç d’invitació no és vàlid per enviar. Necessites el @handle del canal o l’ID -100...")

    if "last_manual_tip" in st.session_state and isinstance(st.session_state["last_manual_tip"], pd.DataFrame):
        out = st.session_state["last_manual_tip"].copy()
        model_name = st.session_state.get("last_model_name", "")

        extra = out.apply(lambda r: pd.Series(_compute_tip_fields(r)), axis=1)
        out_tip = pd.concat([out, extra], axis=1)

        st.markdown("##### Tip preview (Premium)")
        st.code(_format_tip_message(out_tip.iloc[0].to_dict(), "PREMIUM"), language="markdown")

        b1, b2, b3 = st.columns([1, 1, 1])
        if b1.button("Approve & send → Premium", use_container_width=True):
            results = _publish_and_log(
                out_tip, channels=("premium",),
                source="manual", model_name=model_name
            )
            ok_p, info_p, _ = results.get("premium", (False, "", None))
            if ok_p:
                st.success("Enviat a Premium i guardat al log.")
            else:
                st.warning(f"Guardat al log. Telegram Premium NO enviat: {info_p}")

        if b2.button("Approve & send → Premium + Free", use_container_width=True):
            results = _publish_and_log(
                out_tip, channels=("premium", "free"),
                source="manual", model_name=model_name
            )
            ok_p, info_p, _ = results.get("premium", (False, "", None))
            ok_f, info_f, _ = results.get("free", (False, "", None))
            if ok_p and ok_f:
                st.success("Enviat a Premium i Free, i guardat al log.")
            elif ok_p and not ok_f:
                st.warning(f"Premium OK · Free KO: {info_f}. Igualment, ambdós registres guardats al log.")
            elif not ok_p and ok_f:
                st.warning(f"Free OK · Premium KO: {info_p}. Igualment, ambdós registres guardats al log.")
            else:
                st.error(f"Cap enviament s’ha pogut fer. Els dos registres s’han guardat al log amb l’error corresponent.")

        if b3.button("Descartar aquesta predicció (netejar preview)", use_container_width=True):
            st.session_state.pop("last_manual_tip", None)
            st.session_state.pop("last_model_name", None)
            st.info("Preview netejat.")
    else:
        st.info("Fes una predicció manual a dalt per obtenir el preview i poder enviar/guardar.")




with tab3:
    import os, json
    import numpy as np
    import pandas as pd
    import streamlit as st

    st.subheader("Predictions log & Results")


    # ─────────────────────────────────────────────
    # Helpers generals
    # ─────────────────────────────────────────────
    def show_current_model_kpis():
        p = os.path.join(OUT_DIR, "train_metrics.json")
        if os.path.exists(p):
            m = json.load(open(p, "r"))

            def _fmt(x):
                try:
                    return f"{float(x):.3f}"
                except Exception:
                    return str(x)

            cols = [
                ("Valid AUC", _fmt(m.get("valid_auc"))),
                ("Valid logloss", _fmt(m.get("valid_logloss"))),
                ("Test AUC", _fmt(m.get("test_auc"))),
                ("Test logloss", _fmt(m.get("test_logloss"))),
                ("Test Brier", _fmt(m.get("test_brier"))),
                ("Train/Valid/Test", f'{m.get("n_train")}/{m.get("n_valid")}/{m.get("n_test")}'),
            ]
            st.success(
                "Current model "
                f'({m.get("model_type","?")}, {m.get("timestamp","")}) · ' +
                " · ".join([f"{k}: {v}" for k,v in cols])
            )
        else:
            st.info("Encara no hi ha `train_metrics.json`. Executa “Refresh & Train` a Tab 2`.")

    def _model_mtime_str():
        cands = [
            os.path.join(OUT_DIR, "model_lightgbm.pkl"),
            os.path.join(OUT_DIR, "model_logistic.pkl")
        ]
        p = next((c for c in cands if os.path.exists(c)), None)
        if not p:
            return ("-", "-")
        ts = os.path.getmtime(p)
        return (
            os.path.basename(p),
            pd.to_datetime(ts, unit="s").strftime("%Y-%m-%d %H:%M")
        )

    def _load_train_metrics():
        # mira json, pkl, txt
        for fn in ["train_metrics.json", "train_metrics.pkl", "train_metrics.txt"]:
            p = os.path.join(OUT_DIR, fn)
            if os.path.exists(p):
                try:
                    if fn.endswith(".json"):
                        return json.load(open(p,"r"))
                    elif fn.endswith(".pkl"):
                        import joblib
                        return joblib.load(p)
                    else:
                        out = {}
                        with open(p,"r") as f:
                            for line in f:
                                if "=" in line:
                                    k,v = line.strip().split("=",1)
                                    out[k.strip()] = v.strip()
                        return out
                except Exception:
                    pass
        return {}

    def enforce_decided_tolerance(df: pd.DataFrame, date_tol_days: int = 15) -> pd.DataFrame:
        """
        decided = True només si:
          - y_home_win ∈ {0,1}
          - hi ha decided_src_date i date
          - |decided_src_date - date| ≤ date_tol_days
          - decided_src_date ≤ avui
        """
        df = df.copy()

        # date
        if "date" in df.columns:
            date_ser = pd.to_datetime(df["date"], errors="coerce")
        else:
            date_ser = pd.Series(pd.NaT, index=df.index)

        # decided_src_date
        if "decided_src_date" in df.columns:
            dsrc = pd.to_datetime(df["decided_src_date"], errors="coerce")
        else:
            dsrc = pd.Series(pd.NaT, index=df.index)

        # y_home_win
        if "y_home_win" in df.columns:
            y = pd.to_numeric(df["y_home_win"], errors="coerce")
        else:
            y = pd.Series(np.nan, index=df.index)

        today = pd.Timestamp.today().normalize()
        within = (
            y.isin([0, 1]) &
            dsrc.notna() & date_ser.notna() &
            ((dsrc - date_ser).abs() <= pd.Timedelta(days=date_tol_days)) &
            (dsrc <= today)
        )
        df["decided"] = within.astype(bool)
        return df

    def repair_missing_dates(df: pd.DataFrame) -> pd.DataFrame:
        """
        Omple 'date' si està buida:
          1) prova extreure YYYY-MM-DD del match_id
          2) si no, usa 'pred_time_utc'
        """
        df = df.copy()

        # date original
        if "date" in df.columns:
            date_raw = df["date"].astype(str)
        else:
            date_raw = pd.Series("", index=df.index, dtype=str)

        # 1) regex al match_id
        rx = df.get("match_id", "").astype(str).str.extract(
            r"(\d{4}-\d{2}-\d{2})",
            expand=False
        )

        # 2) 'pred_time_utc'
        ptu = pd.to_datetime(
            df.get("pred_time_utc"),
            errors="coerce"
        ).dt.date.astype("string")

        # primera passada: si buit, intenta rx
        date_filled_str = date_raw.where(~date_raw.isin(["", "None", "nan"]), rx)
        # segona passada: si encara buit, intenta ptu
        date_filled_str = date_filled_str.where(
            ~date_filled_str.isna() & (date_filled_str != ""),
            ptu
        )

        date_dt = pd.to_datetime(date_filled_str, errors="coerce")
        df["date"] = date_dt
        return df

    def _bucket_stake(x):
        try:
            x = float(x)
        except Exception:
            return 0.0
        if not np.isfinite(x):
            return 0.0
        if x < 1.0:
            return 0.5
        elif x <= 1.5:
            return 1.0
        elif x <= 4.0:
            return 2.0
        else:
            return 3.0

    def _derive_channel_col(df: pd.DataFrame) -> pd.Series:
        """
        Deriva el camp 'channel' com:
          1) Si ja existeix 'channel' amb valors vàlids (premium/free), el fem servir.
          2) Si no, prova 'published_to'.
          3) Fallback: mira 'source' per paraules 'premium' o 'free'.
          4) Altrament '-'.
        Retorna un pd.Series alineat amb df.index.
        """
        def _lower_series(colname: str) -> pd.Series:
            if colname in df.columns:
                return df[colname].astype(str).str.lower()
            return pd.Series([""] * len(df), index=df.index, dtype="object")

        ch_base = _lower_series("channel")
        pub     = _lower_series("published_to")
        src     = _lower_series("source")

        # Si 'channel' ja té premium/free, respecta-ho
        valid_base = ch_base.where(ch_base.isin(["premium", "free"]), "")

        # Si no, mira 'published_to'
        from_pub = pub.where(pub.isin(["premium", "free"]), "")

        # Últim recurs: dedueix-ho de 'source'
        from_src = np.where(
            src.str.contains("premium", na=False), "premium",
            np.where(src.str.contains("free", na=False), "free", "")
        )

        # Prioritat: channel > published_to > source > '-'
        ch = valid_base
        ch = np.where(ch == "", from_pub, ch)
        ch = np.where((ch == "") & (from_src != ""), from_src, ch)
        ch = np.where(ch == "", "-", ch)

        return pd.Series(ch, index=df.index)

    def collapse_log_per_match(df: pd.DataFrame) -> pd.DataFrame:
        """
        Col·lapsa el log a 1 fila per match_id:
          - Escull la fila 'representativa' (la darrera publicada → 'published_at_utc',
            si no hi és, per 'pred_time_utc').
          - Crea 'sent_to' que resumeix on s'ha enviat realment: premium, free, premium+free o none.
          - Recalcula decided/derivades i assegura la 'date'.
        """
        if "match_id" not in df.columns:
            return df.copy()
    
        d = df.copy()
        d = repair_missing_dates(clean_log_df(d))
        d = add_outcome_columns(d)
        d = enforce_decided_tolerance(d, date_tol_days=15)
    
        # estat d'enviament
        ch = d.get("channel", pd.Series([""] * len(d), index=d.index)).astype(str).str.lower()
        stt = d.get("tg_status", pd.Series([""] * len(d), index=d.index)).astype(str)
        sent_mask = stt.str.startswith("sent")
    
        # 'sent_to' per match
        def _sent_to(gr: pd.DataFrame) -> str:
            s = set(gr.loc[sent_mask.reindex(gr.index, fill_value=False), "channel"].str.lower())
            if "premium" in s and "free" in s:
                return "premium+free"
            if "premium" in s:
                return "premium"
            if "free" in s:
                return "free"
            return "none"
    
        sent_map = d.groupby("match_id", as_index=True).apply(_sent_to)
    
        # tria la fila representativa: última publicada > última pred
        d["__pub"] = pd.to_datetime(d.get("published_at_utc"), errors="coerce")
        d["__pred"] = pd.to_datetime(d.get("pred_time_utc"), errors="coerce")
        d_sorted = d.sort_values(["__pub", "__pred"], ascending=[False, False])
        base = d_sorted.drop_duplicates(subset=["match_id"], keep="first").copy()
        base["sent_to"] = base["match_id"].map(sent_map).fillna("none")
    
        # neteja
        base = base.drop(columns=["__pub", "__pred"], errors="ignore")
        base["date"] = pd.to_datetime(base.get("date"), errors="coerce")
        return base


    def _kpis_header_from_log(log_df: pd.DataFrame, channel_filter: str = "All"):
        """
        KPIs 'lifetime' sobre el log real (respectant el filtre de canal).
        """
        df = clean_log_df(log_df)
        df = add_outcome_columns(df)
        df = df.sort_values("date")

        # canal
        df["channel"] = _derive_channel_col(df)
        if channel_filter in {"Premium","Free","None"}:
            key = channel_filter.lower()
            df = df[df["channel"] == key]

        # stake decimal per agregats "legacy"
        df["stake"] = pd.to_numeric(df.get("stake_% (half kelly)"), errors="coerce")/100.0
        bets = df[(df["stake"] > 0) & (df["decided"])].copy()

        total_bets = int(len(bets))
        total_wins = int(
            pd.to_numeric(bets["bet_correct"], errors="coerce").fillna(0).sum()
        )
        lifetime_acc = (
            (total_wins / total_bets * 100) if total_bets else float("nan")
        )

        base_sample = bets.head(100)
        baseline_acc = (
            base_sample["bet_correct"].mean() * 100 if len(base_sample) else float("nan")
        )

        curr_sample = bets.tail(100)
        current_acc = (
            curr_sample["bet_correct"].mean() * 100 if len(curr_sample) else float("nan")
        )

        improvement = (
            current_acc - baseline_acc
            if (np.isfinite(current_acc) and np.isfinite(baseline_acc))
            else float("nan")
        )

        cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=60)
        recent = bets[bets["date"] >= cutoff]
        total_stake_60 = float(recent["stake"].sum())
        units_60 = float(
            pd.to_numeric(recent["unit_return_calc"], errors="coerce").fillna(0).sum()
        )
        roi_60 = (
            (units_60 / total_stake_60 * 100) if total_stake_60 > 0 else float("nan")
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Baseline acc. (primeres 100 apostes)",
            f"{baseline_acc:0.1f}%" if np.isfinite(baseline_acc) else "—"
        )
        c2.metric(
            "Acc. actual (últimes 100 apostes)",
            f"{current_acc:0.1f}%" if np.isfinite(current_acc) else "—",
            delta=(
                f"{improvement:+0.1f} pp"
                if np.isfinite(improvement) else None
            )
        )
        c3.metric("Apostes totals", f"{total_bets:,}")
        c4.metric("Encerts totals", f"{total_wins:,}")

        c5, c6 = st.columns(2)
        c5.metric(
            "Lifetime acc. apostes",
            f"{lifetime_acc:0.1f}%" if np.isfinite(lifetime_acc) else "—"
        )
        c6.metric(
            "ROI últims 60 dies",
            f"{roi_60:0.2f}%" if np.isfinite(roi_60) else "—"
        )

    def summarize_monthly_log(log_raw: pd.DataFrame):
        """
        Resum mensual usant *unitats bucketitzades* per a stake i P/L, i també per canal.
        """
        # Neteja bàsica + columnes derivades
        df = clean_log_df(log_raw)
        df = add_outcome_columns(df)
        df = repair_missing_dates(df)
        df = enforce_decided_tolerance(df, date_tol_days=15)
        df["channel"] = _derive_channel_col(df)

        # Dates per fer el month
        if not np.issubdtype(df["date"].dtype, np.datetime64):
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["month"] = df["date"].dt.to_period("M").astype(str).fillna("unknown")

        # ── Bucket d'unitats a partir del % stake (half kelly)
        stake_pct = pd.to_numeric(df.get("stake_% (half kelly)"), errors="coerce")
        df["stake_units"] = stake_pct.apply(_bucket_stake)

        # ── Info necessària per P/L
        odds_home = pd.to_numeric(df.get("odds_home"), errors="coerce")
        odds_away = pd.to_numeric(df.get("odds_away"), errors="coerce")
        y         = pd.to_numeric(df.get("y_home_win"), errors="coerce")

        side = df.get("best_side", "").astype(str).str.lower()
        is_home_pick = side.str.startswith("home")
        is_away_pick = side.str.startswith("away")

        # Pick ML vàlid (i decidit)
        decided = df.get("decided", False).astype(bool)
        ml_pick = decided & (df["stake_units"] > 0) & (is_home_pick | is_away_pick)

        # Bet correct per ML picks
        bet_correct_ml = np.where(
            is_home_pick, (y == 1).astype(float),
            np.where(is_away_pick, (y == 0).astype(float), np.nan)
        )

        # P/L en unitats bucketitzades
        pl_units = (
            np.where(is_home_pick & (y == 1), df["stake_units"] * (odds_home - 1.0), 0.0)
            + np.where(is_away_pick & (y == 0), df["stake_units"] * (odds_away - 1.0), 0.0)
            - np.where((is_home_pick & (y == 0)) | (is_away_pick & (y == 1)), df["stake_units"], 0.0)
        )
        df["pl_units_ml"] = np.where(ml_pick, pl_units, 0.0)

        # Accuracy de predicció global (sobre decidides)
        if "pred_correct" not in df.columns:
            p = pd.to_numeric(df.get("p_home_win"), errors="coerce")
            df["pred_correct"] = np.where(
                (p >= 0.5) & (y == 1), 1,
                np.where((p < 0.5) & (y == 0), 1, np.nan)
            )

        # Sèries per agregats
        df["ml_pick"]        = ml_pick
        df["bet_correct_ml"] = np.where(ml_pick, bet_correct_ml, np.nan)
        df["stake_units_ml"] = np.where(ml_pick, df["stake_units"], 0.0)

        # Aggregats principals (overall)
        g = df.groupby("month", dropna=False).agg(
            n=("match_id", "count"),
            decided=("decided", "sum"),
            bets=("ml_pick", "sum"),
            total_stake=("stake_units_ml", "sum"),
            units=("pl_units_ml", "sum"),
        ).reset_index()
        acc_bets = df[df["ml_pick"]].groupby("month")["bet_correct_ml"].mean()
        acc_pred = df[df["decided"]].groupby("month")["pred_correct"].mean()
        g["acc_bets_%"] = (g["month"].map(acc_bets) * 100).round(1)
        g["acc_pred_%"] = (g["month"].map(acc_pred) * 100).round(1)
        g["roi_%"] = np.where(
            g["total_stake"] > 0,
            (g["units"] / g["total_stake"]) * 100,
            np.nan
        ).round(1)

        # Aggregats per canal (més robust amb merge)
        gc = df.groupby(["month","channel"], dropna=False).agg(
            bets=("ml_pick","sum"),
            total_stake=("stake_units_ml","sum"),
            units=("pl_units_ml","sum"),
        ).reset_index()
        acc_bets_c = (
            df[df["ml_pick"]]
            .groupby(["month","channel"])["bet_correct_ml"]
            .mean()
            .to_frame("acc_bets")
            .reset_index()
        )
        gc = gc.merge(acc_bets_c, on=["month","channel"], how="left")
        gc["acc_bets_%"] = (gc["acc_bets"] * 100).round(1)
        gc["roi_%"] = np.where(
            gc["total_stake"] > 0,
            (gc["units"] / gc["total_stake"]) * 100,
            np.nan
        ).round(1)

        return df, g, gc

    # --- Manual restore of predictions_log.csv to keep history across sessions ---
    st.markdown("##### Restore saved log (to keep history across sessions)")
    uploaded_log = st.file_uploader(
        "Upload a previous predictions_log.csv (optional)",
        type=["csv"],
        key="restore_predictions_log"
    )
    if uploaded_log is not None:
        restored_df = pd.read_csv(uploaded_log)

        # mateixa neteja que fem després normalment
        restored_df = clean_log_df(restored_df)
        restored_df = repair_missing_dates(restored_df)          # ✅ ara ja està definida
        restored_df = add_outcome_columns(restored_df)
        restored_df = enforce_decided_tolerance(restored_df, date_tol_days=15)  # ✅ definida

        # assegurem carpeta on viu el log
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        restored_df.to_csv(LOG_PATH, index=False)

        st.success("Log restaurat des del fitxer pujat ✅ (guardat per aquesta sessió).")


    # ─────────────────────────────────────────────
    # (A) KPIs del model actual (from train_metrics.json)
    # ─────────────────────────────────────────────
    show_current_model_kpis()

    st.markdown("### 📊 Current algorithm performance")

    tm = _load_train_metrics()

    # bloc mètriques offline de model pur
    colT1, colT2, colT3, colT4 = st.columns(4)
    if tm:
        colT1.metric(
            "Offline AUC (test)",
            f"{float(tm.get('test_auc', float('nan'))):.3f}"
            if tm.get('test_auc') is not None else "—"
        )
        colT2.metric(
            "Offline LogLoss",
            f"{float(tm.get('test_logloss', float('nan'))):.3f}"
            if tm.get('test_logloss') is not None else "—"
        )
        colT3.metric(
            "Offline Brier",
            f"{float(tm.get('test_brier', float('nan'))):.3f}"
            if tm.get('test_brier') is not None else "—"
        )
        colT4.metric(
            "n_train / n_valid / n_test",
            f"{int(tm.get('n_train',0))} / {int(tm.get('n_valid',0))} / {int(tm.get('n_test',0))}"
        )
        st.caption(
            f"Offline (test) metrics from last training · timestamp: {tm.get('timestamp','—')}"
        )
    else:
        colT1.metric("Offline AUC (test)", "—")
        colT2.metric("Offline LogLoss", "—")
        colT3.metric("Offline Brier", "—")
        colT4.metric("n_train / n_valid / n_test", "—")
        st.caption("No s'han trobat mètriques d'entrenament guardades.")

    # bloc mètriques HIGH-CONFIDENCE sense quotes (hc_*)
    st.markdown("#### 🔥 High-confidence picks (no odds needed)")
    colH1, colH2, colH3 = st.columns(3)
    if tm:
        hc_hit_rate = tm.get("hc_hit_rate", None)
        hc_cov      = tm.get("hc_cov_pct", None)
        hc_n        = tm.get("hc_n_picks", None)

        # Hit-rate hi-conf
        if (hc_hit_rate is None) or (
            isinstance(hc_hit_rate, float) and not np.isfinite(hc_hit_rate)
        ) or (hc_n == 0):
            hc_hit_txt = "—"
        else:
            hc_hit_txt = f"{hc_hit_rate*100:.1f}%"

        # Cobertura hi-conf
        if (hc_cov is None) or (
            isinstance(hc_cov, float) and not np.isfinite(hc_cov)
        ):
            hc_cov_txt = "—"
        else:
            hc_cov_txt = f"{hc_cov:.1f}%"

        colH1.metric(
            "Hit-rate hi-conf",
            hc_hit_txt,
            help="Accuracy quan el model diu 'clar favorit' (≥60%) o 'dog amb opció' (≤40%), sense mirar quotes."
        )
        colH2.metric(
            "Cobertura hi-conf",
            hc_cov_txt,
            help="% de partits històrics que entren en aquesta categoria forta (≥60% o ≤40%)."
        )
        colH3.metric(
            "# picks hi-conf",
            str(int(hc_n)) if hc_n is not None else "—",
            help="Quants partits històrics han entrat a aquesta categoria."
        )

        st.caption(
            "Això és pur model: només mirem la probabilitat que donem a A o B, "
            "i comptem encert si diem 'A guanya' i realment guanya A. "
            "No depèn de quotes de mercat."
        )
    else:
        colH1.metric("Hit-rate hi-conf", "—")
        colH2.metric("Cobertura hi-conf", "—")
        colH3.metric("# picks hi-conf", "—")
        st.caption("Sense mètriques hi-conf disponibles encara.")

    st.markdown("---")

    # ─────────────────────────────────────────────
    # (B) LIVE LOG (predictions_log.csv)
    # ─────────────────────────────────────────────
    mf, mtime = _model_mtime_str()
    st.caption(f"Model file: `{mf}` — last trained: {mtime}")

    if os.path.exists(LOG_PATH):
        raw0 = pd.read_csv(LOG_PATH)
        log0 = clean_log_df(raw0)
        log0 = add_outcome_columns(log0)

        # assegura 'date' consistent
        log0 = repair_missing_dates(log0)
        # aplica tolerància per marcar 'decided'
        log0 = enforce_decided_tolerance(log0, date_tol_days=15)
        # canal derivat
        log0["channel"] = _derive_channel_col(log0)
        # persistim la versió neta
        log0.to_csv(LOG_PATH, index=False)

        # --- filtres de finestra temporal + canal
        colw1, colw2, colw3 = st.columns([1,1,1])
        win = colw1.selectbox(
            "Window",
            ["7d", "30d", "90d", "365d", "All"],
            index=1
        )
        channel_filter = colw2.selectbox(
            "Channel",
            ["All", "Premium", "Free", "None"],
            index=0,
            help="Filtra mètriques per canal de publicació."
        )
        only_bets = colw3.checkbox(
            "Only settled bets",
            value=True,
            help="Stake > 0 i resultat decidit."
        )

        dfw = log0.copy()
        # NEW: dedup per match (evita comptar 2 cops si s'ha enviat a premium+free)
        dfw_unique = collapse_log_per_match(dfw)
        
        # numèrics bàsics
        dfw_unique["stake"]      = pd.to_numeric(dfw_unique.get("stake_% (half kelly)"), errors="coerce").fillna(0.0) / 100.0
        dfw_unique["y_home_win"] = pd.to_numeric(dfw_unique.get("y_home_win"), errors="coerce")
        dfw_unique["p_home_win"] = pd.to_numeric(dfw_unique.get("p_home_win"), errors="coerce").clip(1e-6, 1-1e-6)
        
        mask_decided = dfw_unique.get("decided")
        if mask_decided is None:
            mask_decided = dfw_unique["y_home_win"].isin([0,1])
        else:
            mask_decided = mask_decided.astype(bool)
        
        # KPIs de 'bets' dins la finestra (ja deduplicat)
        if only_bets:
            bets = dfw_unique[mask_decided & (dfw_unique["stake"] > 0)].copy()
        else:
            bets = dfw_unique[mask_decided].copy()
        
        n_bets = int(len(bets))
        total_stake = float(pd.to_numeric(bets.get("stake"), errors="coerce").sum()) if n_bets else 0.0
        units = float(pd.to_numeric(bets.get("unit_return_calc"), errors="coerce").sum()) if n_bets else float("nan")
        hit = float(pd.to_numeric(bets.get("bet_correct"), errors="coerce").mean()*100.0) if n_bets else float("nan")
        roi = (units/total_stake*100.0) if (n_bets and total_stake>0) else float("nan")
        avg_stake = (float(bets["stake"].mean()*100.0) if n_bets else 0.0)
        
        # Mètriques de probabilitat (live, sobre decidides)
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
        preds_live = dfw_unique[mask_decided & dfw_unique["p_home_win"].notna()].copy()
        auc = logloss = brier = float("nan")
        if len(preds_live):
            y = pd.to_numeric(preds_live["y_home_win"], errors="coerce")
            p = pd.to_numeric(preds_live["p_home_win"], errors="coerce").clip(1e-6,1-1e-6)
            if y.notna().sum() and p.notna().sum():
                if len(np.unique(y.dropna())) > 1:
                    try:    auc = float(roc_auc_score(y, p))
                    except: pass
                    try:    logloss = float(log_loss(y, p))
                    except: pass
                try:        brier = float(brier_score_loss(y, p))
                except:     pass


        # pinta KPIs live (respecten el filtre de canal i finestra)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Bets (window)", f"{n_bets}")
        m2.metric("Hit rate (bets)", "—" if np.isnan(hit) else f"{hit:.1f}%")
        m3.metric("ROI (bets)", "—" if np.isnan(roi) else f"{roi:.1f}%")
        m4.metric("Avg stake", f"{avg_stake:.1f}%")

        m5, m6, m7 = st.columns(3)
        m5.metric("AUC (live preds)", "—" if np.isnan(auc) else f"{auc:.3f}")
        m6.metric("LogLoss", "—" if np.isnan(logloss) else f"{logloss:.3f}")
        m7.metric("Brier", "—" if np.isnan(brier) else f"{brier:.3f}")

        # === Desglossament per canal (finestra actual) amb unitats bucketitzades ===
        st.markdown("#### Breakdown per canal (window)")

        bw = dfw.copy()
        bw["stake_raw_pct"] = pd.to_numeric(bw.get("stake_% (half kelly)"), errors="coerce")
        bw["stake_units"]   = bw["stake_raw_pct"].apply(_bucket_stake)

        # odds i resultat real
        bw["odds_home"] = pd.to_numeric(bw.get("odds_home"), errors="coerce")
        bw["odds_away"] = pd.to_numeric(bw.get("odds_away"), errors="coerce")
        y               = pd.to_numeric(bw.get("y_home_win"), errors="coerce")

        side = bw.get("best_side", "").astype(str).str.lower()
        is_home_pick = side.str.startswith("home")
        is_away_pick = side.str.startswith("away")
        win_home  = is_home_pick & (y == 1)
        loss_home = is_home_pick & (y == 0)
        win_away  = is_away_pick & (y == 0)
        loss_away = is_away_pick & (y == 1)

        bw["ml_pick"] = (bw["stake_units"] > 0) & (is_home_pick | is_away_pick) & bw["decided"].astype(bool)
        bw["unit_pl"] = (
            np.where(win_home, bw["stake_units"] * (bw["odds_home"] - 1.0), 0.0)
            + np.where(win_away, bw["stake_units"] * (bw["odds_away"] - 1.0), 0.0)
            - np.where(loss_home | loss_away, bw["stake_units"], 0.0)
        )
        bw["bet_correct_ml"] = np.where(is_home_pick, (y == 1).astype(float),
                                  np.where(is_away_pick, (y == 0).astype(float), np.nan))

        br = bw[bw["ml_pick"]].groupby("channel").agg(
            bets=("ml_pick","sum"),
            stake_units=("stake_units","sum"),
            pl_units=("unit_pl","sum"),
            acc=("bet_correct_ml","mean"),
        ).reset_index()
        if len(br):
            br["roi_%"] = np.where(
                br["stake_units"] > 0,
                (br["pl_units"] / br["stake_units"]) * 100.0,
                np.nan
            ).round(1)
            br["acc_%"] = (br["acc"] * 100.0).round(1)
            st.dataframe(br[["channel","bets","stake_units","pl_units","roi_%","acc_%"]])
        else:
            st.caption("No hi ha apostes decidides en aquesta finestra per calcular breakdown.")

        st.markdown("---")

        # ====== HEAD KPIs lifetime / ROI última finestra gran (respectant filtre de canal)
        _kpis_header_from_log(log0, channel_filter=channel_filter)

        # ====== Resum mensual (overall i per canal)
        # NEW: dedup abans de resum mensual
        full_unique = collapse_log_per_match(log0)
        full_u, monthly_u, _unused = summarize_monthly_log(full_unique)
        
        st.subheader("Resum mensual (picks únics)")
        st.dataframe(monthly_u)
        # JA NO mostrem “per canal”


        # ====== Últimes entrades
        st.subheader("Últimes entrades")
        disp = full_unique.copy()  # deduplicat (una fila per match)
        
        disp["_pred_dt"] = pd.to_datetime(disp.get("pred_time_utc"), errors="coerce")
        disp = disp.sort_values(["date", "_pred_dt"], ascending=[False, False]).drop(columns=["_pred_dt"]).head(200)
        
        # stake decimal a partir del % half kelly
        stake_num   = pd.to_numeric(disp.get("stake_% (half kelly)"), errors="coerce").fillna(0.0) / 100.0
        side_s      = disp.get("best_side", "").astype(str).str.lower()
        
        is_bet = (stake_num > 0) & (side_s.str.startswith("home") | side_s.str.startswith("away"))
        is_bet_and_decided = is_bet & disp.get("decided", False)
        
        # Emoji resultat (només per files amb aposta i decidides)
        if "result_emoji" in disp.columns:
            base_emoji = disp["result_emoji"].astype(str)
        else:
            bc = pd.to_numeric(disp.get("bet_correct"), errors="coerce")
            base_emoji = np.where(bc == 1, "✅", np.where(bc == 0, "❌", ""))
        
        disp["result_emoji_display"] = np.where(is_bet_and_decided, base_emoji, "-")
        disp["bet_correct_display"]  = np.where(is_bet_and_decided, pd.to_numeric(disp.get("bet_correct"), errors="coerce"), np.nan)
        
        best_side_display = np.where(is_bet, disp.get("best_side"), "-")
        stake_display     = np.where(is_bet, disp.get("stake_% (half kelly)"), "-")
        
        # Calcula unitats i P/L unitats per la vista
        disp_num_stake = pd.to_numeric(disp.get("stake_% (half kelly)"), errors="coerce")
        disp_units = disp_num_stake.apply(_bucket_stake)
        
        d_odds_home = pd.to_numeric(disp.get("odds_home"), errors="coerce")
        d_odds_away = pd.to_numeric(disp.get("odds_away"), errors="coerce")
        d_y         = pd.to_numeric(disp.get("y_home_win"), errors="coerce")
        d_is_home   = side_s.str.startswith("home")
        d_is_away   = side_s.str.startswith("away")
        d_loss = (d_is_home & (d_y == 0)) | (d_is_away & (d_y == 1))
        
        disp_pl_units = (
            np.where(d_is_home & (d_y == 1), disp_units * (d_odds_home - 1.0), 0.0)
            + np.where(d_is_away & (d_y == 0), disp_units * (d_odds_away - 1.0), 0.0)
            - np.where(d_loss, disp_units, 0.0)
        )
        
        # Format de la data per evitar 'None'
        date_fmt = pd.to_datetime(disp.get("date"), errors="coerce").dt.strftime("%Y-%m-%d")
        
        disp_show = pd.DataFrame({
            "date": date_fmt,
            "player_a_name": disp.get("player_a_name"),
            "player_b_name": disp.get("player_b_name"),
            "tournament": disp.get("tournament"),
            "p_home_win": disp.get("p_home_win"),
            "odds_home": disp.get("odds_home"),
            "odds_away": disp.get("odds_away"),
            "y_home_win": disp.get("y_home_win"),
            "pred_correct": disp.get("pred_correct"),
        })
        
        disp_show["best_side"]            = best_side_display
        disp_show["stake_% (half kelly)"] = stake_display
        disp_show["bet_correct"]          = disp["bet_correct_display"]
        disp_show["result"]               = np.where(is_bet_and_decided, disp["result_emoji_display"], "-")
        disp_show["stake_units"]          = np.where(is_bet, disp_units, "-")
        disp_show["PL_units"]             = np.where(is_bet, disp_pl_units, "-")
        
        # NEW: on s'ha enviat (resum)
        if "sent_to" in disp.columns:
            disp_show["sent_to"] = disp["sent_to"]
        else:
            # compatibilitat: dedueix a partir de columnes clàssiques si no hi ha 'sent_to'
            pub = disp.get("published_to", "").astype(str).str.lower()
            chn = disp.get("channel", "").astype(str).str.lower()
            stt = disp.get("tg_status", "").astype(str)
            sent_mask = stt.str.startswith("sent")
            sent_to = np.where(sent_mask & (chn=="premium"), "premium",
                        np.where(sent_mask & (chn=="free"), "free", "none"))
            disp_show["sent_to"] = sent_to
        
        # opcional: mostra 'bookmaker' si el tens
        if "bookmaker" in disp.columns:
            disp_show["bookmaker"] = disp["bookmaker"]
        
        st.dataframe(disp_show)


        # botons utilitat (backfill / debug)
        c1, c2 = st.columns([1, 1])
        if c1.button("Backfill results from TML (last 90 days)", key="btn_backfill_90"):
            ok, msg = backfill_results_from_tml(days_back=90, date_tol_days=15)
            st.info(msg)

        if c2.button("Debug pending backfill (sample)", key="btn_debug_backfill"):
            debug_backfill_pending(sample_n=30, days_back=90, date_tol_days=15)

        # ─────────────────────────────────────────
        # NOVA SECCIÓ: entrada manual de resultats
        # ─────────────────────────────────────────
        st.markdown("---")
        st.subheader("Manual result entry (si el backfill no troba el guanyador)")

        # tornem a llegir el CSV brut per editar-lo després
        log_edit = pd.read_csv(LOG_PATH)
        log_edit = repair_missing_dates(clean_log_df(log_edit))
        log_edit = add_outcome_columns(log_edit)
        log_edit = enforce_decided_tolerance(log_edit, date_tol_days=15)

        # candidats = files no decidides encara
        cand = log_edit[~log_edit["decided"].fillna(False)].copy()

        if len(cand) == 0:
            st.caption("No hi ha partits pendents de marcar manualment 🎉")
        else:
            # fem una etiqueta maca per seleccionar
            cand["display_label"] = cand.apply(
                lambda r: f"{str(r.get('date'))[:10]} · {r.get('player_a_name','?')} vs {r.get('player_b_name','?')} · match_id={r.get('match_id','?')}",
                axis=1
            )

            # fem un selectbox per triar quin partit vols marcar
            chosen_label = st.selectbox(
                "Quin partit vols tancar manualment?",
                ["-- escull un partit --"] + cand["display_label"].tolist()
            )

            if chosen_label != "-- escull un partit --":
                row_sel = cand[cand["display_label"] == chosen_label].iloc[0]
                st.write(f"Has triat: {row_sel['player_a_name']} (home/A) vs {row_sel['player_b_name']} (away/B) a {row_sel['date']}")

                winner_choice = st.radio(
                    "Qui va guanyar realment?",
                    ["home(A)", "away(B)"],
                    horizontal=True
                )

                if st.button("Guardar resultat manualment", key="btn_save_manual_result"):
                    # actualitzem log_edit per aquest match_id
                    mid = row_sel["match_id"]
                    idx = log_edit[log_edit["match_id"] == mid].index

                    if winner_choice == "home(A)":
                        y_val = 1
                    else:
                        y_val = 0

                    today_str = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")

                    # marcar resultat i decided
                    log_edit.loc[idx, "y_home_win"] = y_val
                    log_edit.loc[idx, "decided_src_date"] = today_str
                    log_edit.loc[idx, "decided"] = True

                    # recomputem outcome helpers (bet_correct, etc.) per coherència
                    log_edit = clean_log_df(log_edit)
                    log_edit = add_outcome_columns(log_edit)
                    log_edit = repair_missing_dates(log_edit)
                    log_edit = enforce_decided_tolerance(log_edit, date_tol_days=15)

                    # guardem al csv
                    log_edit.to_csv(LOG_PATH, index=False)

                    st.success("Resultat manual guardat i comptarà a les mètriques. ✅")

    else:
        st.info("Encara no hi ha `predictions_log.csv` — fes alguna predicció per començar.")
        st.markdown("---")

    # --- descarregar log complet per guardar-lo tu offline ---
    st.markdown("---")
    st.markdown("##### Download full history backup")

    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "rb") as f:
            st.download_button(
                "Download predictions_log.csv (backup)",
                f,
                file_name="predictions_log.csv",
                mime="text/csv",
                use_container_width=True
            )
        st.caption("Descarrega això quan acabis sessió. La propera vegada que obris l'app, puja'l a 'Restore saved log'.")
    else:
        st.info("Encara no hi ha cap log per descarregar (encara no has guardat res aquesta sessió).")

    st.caption(f"📁 Log path: `{LOG_PATH}`")


