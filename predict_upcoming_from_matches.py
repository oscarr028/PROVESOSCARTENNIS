#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, pandas as pd, numpy as np, joblib
from tennis_model_pipeline_v2 import compute_pre_match_features_v2, make_match_features

def load_artifacts(out_dir: str):
    model = None; scaler = None
    lgb_path = os.path.join(out_dir, "model_lightgbm.pkl")
    lr_path  = os.path.join(out_dir, "model_logistic.pkl")
    sc_path  = os.path.join(out_dir, "scaler.pkl")
    if os.path.exists(lgb_path):
        model = joblib.load(lgb_path)
    elif os.path.exists(lr_path):
        model = joblib.load(lr_path)
    if os.path.exists(sc_path):
        scaler = joblib.load(sc_path)
    return model, scaler

def main(args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--out_dir", type=str, default="outputs")
    ap.add_argument("--from_date", type=str, default=None, help="YYYY-MM-DD; include matches on/after this date")
    ap.add_argument("--calibrated", type=int, default=0, help="if 1, apply isotonic.pkl if present")
    ap.add_argument("--odds_col", type=str, default="odds_home", help="column in matches to compute value")
    cfg = ap.parse_args(args)

    matches = pd.read_csv(os.path.join(cfg.data_dir, "matches.csv"))
    points_path = os.path.join(cfg.data_dir, "points_sets_games.csv")
    points = pd.read_csv(points_path) if os.path.exists(points_path) else None

    matches['date'] = pd.to_datetime(matches['date'], errors='coerce')

    cand = matches.copy()
    if cfg.from_date:
        cand = cand[cand['date'] >= pd.to_datetime(cfg.from_date)]
    mask_unknown = cand['winner_id'].isna() | (cand['winner_id'].astype(str)=="")
    cand = cand[mask_unknown | (cand['date'] >= (pd.to_datetime(cfg.from_date) if cfg.from_date else cand['date'].min()))]

    if len(cand)==0:
        print("No candidate matches found."); return

    feats = compute_pre_match_features_v2(matches, points)
    dataset, model_cols = make_match_features(feats, matches)

    ds_cand = dataset[dataset['match_id'].isin(cand['match_id'].astype(str))].copy()
    if len(ds_cand)==0:
        print("No feature rows for candidates. Are player IDs consistent?"); return

    model, scaler = load_artifacts(cfg.out_dir)
    if model is None:
        raise FileNotFoundError("No trained model found in outputs/. Train first and save .pkl.")

    X = ds_cand[model_cols].fillna(0.0).values
    if hasattr(model, "predict_proba"):
        try:
            p = model.predict_proba(X, num_iteration=getattr(model, "best_iteration_", None))[:,1]
        except TypeError:
            p = model.predict_proba(X)[:,1]
    else:
        if scaler is None:
            raise FileNotFoundError("Scaler required for logistic model is missing.")
        p = model.predict_proba(scaler.transform(ds_cand[model_cols].fillna(0.0)))[:,1]

    iso_path = os.path.join(cfg.out_dir, "isotonic.pkl")
    if cfg.calibrated and os.path.exists(iso_path):
        iso = joblib.load(iso_path)
        p = iso.transform(p)

    out = ds_cand[['match_id','date']].copy()
    out['p_home_win'] = np.clip(p, 1e-6, 1-1e-6)

    if cfg.odds_col in matches.columns:
        mx = matches[['match_id', cfg.odds_col]].copy()
        out = out.merge(mx, on='match_id', how='left')
        if cfg.odds_col in out.columns:
            o = pd.to_numeric(out[cfg.odds_col], errors='coerce')
            out['edge'] = out['p_home_win'] - 1.0/o

    os.makedirs(cfg.out_dir, exist_ok=True)
    pred_path = os.path.join(cfg.out_dir, "predictions_upcoming.csv")
    out.to_csv(pred_path, index=False)
    print(f"[Saved] {pred_path} ({len(out)} rows)")

if __name__ == "__main__":
    main()
