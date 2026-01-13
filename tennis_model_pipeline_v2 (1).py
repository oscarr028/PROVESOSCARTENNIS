#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, os, math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import joblib

SURFACES = ["hard","clay","grass","indoor-hard"]

def _safe_div(num, den, default=np.nan):
    try:
        num = float(num); den = float(den)
        return num/den if den else default
    except Exception:
        return default

def exponential_decay_weights(deltas_days: np.ndarray, tau: float = 60.0) -> np.ndarray:
    deltas_days = np.asarray(deltas_days, dtype=float)
    return np.exp(-deltas_days / float(tau))

def bayes_shrink(successes: float, trials: float, prior_mean: float, alpha: float = 30.0) -> float:
    return (float(successes) + alpha * prior_mean) / (float(trials) + alpha) if (trials is not None and not math.isnan(trials)) else np.nan

@dataclass
class EloConfig:
    k_global: float = 24.0
    k_surface: float = 28.0
    base: float = 1500.0
    lambda_blend: float = 0.75
    surfaces: Tuple[str,...] = tuple(SURFACES)

class EloEngine:
    def __init__(self, cfg: EloConfig):
        self.cfg = cfg
        self.elo_global: Dict[str,float] = {}
        self.elo_surface: Dict[Tuple[str,str], float] = {}

    @staticmethod
    def _expected(ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10 ** (-(ra - rb) / 400.0))

    def get_pre_match(self, pid: str, surface: str) -> Tuple[float,float]:
        eg = self.elo_global.get(pid, self.cfg.base)
        es = self.elo_surface.get((pid, surface), self.cfg.base)
        return eg, es

    def update_after_match(self, a_id: str, b_id: str, surface: str, a_won: bool):
        eg_a, es_a = self.get_pre_match(a_id, surface)
        eg_b, es_b = self.get_pre_match(b_id, surface)

        exp_a_g = self._expected(eg_a, eg_b)
        r_a = 1.0 if a_won else 0.0
        r_b = 1.0 - r_a
        eg_a_new = eg_a + self.cfg.k_global * (r_a - exp_a_g)
        eg_b_new = eg_b + self.cfg.k_global * (r_b - (1.0 - exp_a_g))
        self.elo_global[a_id] = eg_a_new
        self.elo_global[b_id] = eg_b_new

        exp_a_s = self._expected(es_a, es_b)
        es_a_new = es_a + self.cfg.k_surface * (r_a - exp_a_s)
        es_b_new = es_b + self.cfg.k_surface * (r_b - (1.0 - exp_a_s))

        es_a_new = self.cfg.lambda_blend * es_a_new + (1 - self.cfg.lambda_blend) * eg_a_new
        es_b_new = self.cfg.lambda_blend * es_b_new + (1 - self.cfg.lambda_blend) * eg_b_new
        self.elo_surface[(a_id, surface)] = es_a_new
        self.elo_surface[(b_id, surface)] = es_b_new

def compute_pre_match_features_v2(matches: pd.DataFrame,
                                  points: Optional[pd.DataFrame] = None,
                                  tau_days: float = 60.0,
                                  alpha_pct: float = 30.0) -> pd.DataFrame:
    m = matches.copy()
    m['date'] = pd.to_datetime(m['date'], errors='coerce')
    m = m.dropna(subset=['date']).sort_values('date')

    pts_idx = {}
    if points is not None and len(points):
        for _, r in points.iterrows():
            pts_idx[(str(r['match_id']), str(r['player_id']))] = r

    hist_state: Dict[str, List[dict]] = {}
    elo = EloEngine(EloConfig())
    out_rows = []

    for _, r in m.iterrows():
        mid = str(r['match_id']); dt = r['date']
        surf = r.get('surface', 'hard'); indoor = int(r.get('indoor', 0))
        best5 = int(r.get('best_of_5', 0))
        a_id = str(r['player_a_id']); b_id = str(r['player_b_id']); w_id = str(r.get('winner_id',''))

        for pid, opp in [(a_id, b_id), (b_id, a_id)]:
            hist = hist_state.get(pid, [])
            hist_before = [h for h in hist if h['date'] < dt]

            if len(hist_before):
                deltas = np.array([(dt - h['date']).days for h in hist_before], dtype=float)
                _ = exponential_decay_weights(deltas, tau=tau_days)
            wins = np.array([h['won'] for h in hist_before], dtype=float) if len(hist_before) else np.array([])
            winrate10 = np.nan if len(wins)==0 else wins[-10:].mean()
            winrate25 = np.nan if len(wins)==0 else wins[-25:].mean()

            sos_recent = np.nan
            if len(hist_before):
                opp_elos = np.array([h.get('opp_elo_surface', np.nan) for h in hist_before], dtype=float)
                if np.any(~np.isnan(opp_elos)):
                    sos_recent = np.nanmean(opp_elos[-25:])

            def _sum_last(key, n=25):
                if not len(hist_before): return np.nan
                arr = np.array([h.get(key, np.nan) for h in hist_before], dtype=float)[-n:]
                return np.nansum(arr) if len(arr) else np.nan

            hold_succ = _sum_last('hold_games_won')
            hold_trials = _sum_last('service_games')
            brk_succ = _sum_last('break_games_won')
            brk_trials = _sum_last('return_games')
            hold_pct = bayes_shrink(hold_succ, hold_trials, prior_mean=0.80, alpha=alpha_pct) if hold_trials==hold_trials and hold_trials>0 else np.nan
            break_pct= bayes_shrink(brk_succ, brk_trials, prior_mean=0.20, alpha=alpha_pct) if brk_trials==brk_trials and brk_trials>0 else np.nan

            aces = _sum_last('aces', 25); dfs = _sum_last('double_faults', 25)
            srv_games = _sum_last('service_games', 25); ret_games = _sum_last('return_games', 25)
            aces_pg = _safe_div(aces, srv_games); dfs_pg  = _safe_div(dfs, srv_games)

            first_in   = bayes_shrink(_sum_last('first_sv_in'), srv_games, prior_mean=0.60, alpha=alpha_pct) if srv_games==srv_games and srv_games>0 else np.nan
            first_pts  = bayes_shrink(_sum_last('first_sv_pts_won'), _sum_last('first_sv_in'), prior_mean=0.72, alpha=alpha_pct)
            second_pts = bayes_shrink(_sum_last('second_sv_pts_won'), _sum_last('second_sv_attempts'), prior_mean=0.52, alpha=alpha_pct)
            tb_winrate = np.nan

            if len(hist_before):
                last_date = hist_before[-1]['date']
                rest_days = (dt - last_date).days
                matches_14d = sum(1 for h in hist_before if (dt - h['date']).days <= 14)
            else:
                rest_days = np.nan; matches_14d = 0

            eg_pre, es_pre = elo.get_pre_match(pid, surf)
            opp_es_pre = elo.get_pre_match(opp, surf)[1]

            out_rows.append(dict(
                match_id=mid, date=dt, surface=surf, indoor=indoor, best_of_5=best5,
                player_id=pid, opponent_id=opp,
                elo_global_pre=eg_pre, elo_surface_pre=es_pre, opp_elo_surface_pre=opp_es_pre,
                winrate10_pre=winrate10, winrate25_pre=winrate25, sos_elo_recent_pre=sos_recent,
                hold_pre=hold_pct, break_pre=break_pct, serve_return_sum_pre=(hold_pct if hold_pct==hold_pct else 0)+(break_pct if break_pct==break_pct else 0),
                rest_days_pre=rest_days, matches_14d_pre=matches_14d,
                aces_pg_pre=aces_pg, dfs_pg_pre=dfs_pg,
                tb_winrate_pre=tb_winrate, first_in_pre=first_in, first_pts_pre=first_pts, second_pts_pre=second_pts
            ))

        a_won = (w_id == a_id)
        if w_id in (a_id, b_id):
            elo.update_after_match(a_id, b_id, surf, a_won=a_won)

        for pid, opp in [(a_id, b_id), (b_id, a_id)]:
            pr = pts_idx.get((mid, pid), {})
            rec = dict(
                date=dt,
                won=1 if w_id==pid else 0,
                service_games=pd.to_numeric(pr.get('service_games', np.nan), errors='coerce'),
                return_games=pd.to_numeric(pr.get('return_games', np.nan), errors='coerce'),
                hold_games_won=pd.to_numeric(pr.get('hold_games_won', np.nan), errors='coerce'),
                break_games_won=pd.to_numeric(pr.get('break_games_won', np.nan), errors='coerce'),
                aces=pd.to_numeric(pr.get('aces', np.nan), errors='coerce'),
                double_faults=pd.to_numeric(pr.get('double_faults', np.nan), errors='coerce'),
                first_sv_in=pd.to_numeric(pr.get('first_sv_in', np.nan), errors='coerce'),
                first_sv_pts_won=pd.to_numeric(pr.get('first_sv_pts_won', np.nan), errors='coerce'),
                second_sv_pts_won=pd.to_numeric(pr.get('second_sv_pts_won', np.nan), errors='coerce'),
                second_sv_attempts=np.nan,
                tb_played=np.nan, tb_won=np.nan,
                opp_elo_surface=elo.get_pre_match(opp, surf)[1],
            )
            hist_state.setdefault(pid, []).append(rec)

    return pd.DataFrame(out_rows)

def make_match_features(feats_player_pre: pd.DataFrame, matches: pd.DataFrame):
    a = feats_player_pre.rename(columns=lambda c: c if c in ['match_id','date','surface','indoor','best_of_5'] else f"A_{c}")
    b = feats_player_pre.rename(columns=lambda c: c if c in ['match_id','date','surface','indoor','best_of_5'] else f"B_{c}")
    merged_a = matches[['match_id','date','surface','indoor','best_of_5','player_a_id','player_b_id','winner_id']]\
        .merge(a, left_on=['match_id','player_a_id'], right_on=['match_id','A_player_id'], how='left')
    merged_ab = merged_a.merge(b, left_on=['match_id','player_b_id'], right_on=['match_id','B_player_id'], how='left')

    base_cols = [
        'elo_global_pre','elo_surface_pre','winrate10_pre','winrate25_pre','sos_elo_recent_pre',
        'hold_pre','break_pre','serve_return_sum_pre','rest_days_pre','matches_14d_pre',
        'aces_pg_pre','dfs_pg_pre','tb_winrate_pre','first_in_pre','first_pts_pre','second_pts_pre'
    ]
    for c in base_cols:
        merged_ab[f"{c}_diff"] = merged_ab[f"A_{c}"] - merged_ab[f"B_{c}"]

    merged_ab['y_home_win'] = (merged_ab['winner_id'] == merged_ab['player_a_id']).astype(int)
    merged_ab['is_indoor'] = merged_ab['indoor'].astype(int)
    merged_ab['is_best_of_5'] = merged_ab['best_of_5'].astype(int)
    for s in SURFACES:
        merged_ab[f"surface_{s}"] = (merged_ab['surface'] == s).astype(int)

    diff_cols = [f"{c}_diff" for c in base_cols]
    dummies = ['is_indoor','is_best_of_5'] + [f"surface_{s}" for s in SURFACES]
    model_cols = diff_cols + dummies

    dataset = merged_ab[['match_id','date','surface'] + model_cols + ['y_home_win']].copy()
    return dataset, model_cols

def temporal_split(df: pd.DataFrame, train_end: str, valid_end: str):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    train = df[df['date'] <= pd.to_datetime(train_end)]
    valid = df[(df['date'] > pd.to_datetime(train_end)) & (df['date'] <= pd.to_datetime(valid_end))]
    test  = df[df['date'] > pd.to_datetime(valid_end)]
    return train, valid, test

def train_logistic_baseline(train: pd.DataFrame, valid: pd.DataFrame, model_cols: List[str], C: float = 1.0, max_iter: int = 200):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train[model_cols].fillna(0.0)); y_tr = train['y_home_win'].values
    X_va = scaler.transform(valid[model_cols].fillna(0.0));     y_va = valid['y_home_win'].values
    lr = LogisticRegression(C=C, max_iter=max_iter); lr.fit(X_tr, y_tr)
    p_tr = lr.predict_proba(X_tr)[:,1]; p_va = lr.predict_proba(X_va)[:,1]
    metrics = {
        'train_logloss': float(log_loss(y_tr, p_tr)),
        'valid_logloss': float(log_loss(y_va, p_va)),
        'valid_auc': float(roc_auc_score(y_va, p_va)),
        'valid_brier': float(brier_score_loss(y_va, p_va)),
    }
    return lr, scaler, metrics, p_va

def evaluate_and_plot_calibration(y_true: np.ndarray, y_prob: np.ndarray, out_dir: str, tag: str):
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(); plt.plot([0,1],[0,1],'--'); plt.plot(mean_pred, frac_pos, marker='o')
    plt.title(f'Calibration — {tag}'); plt.xlabel('Mean predicted probability'); plt.ylabel('Fraction of positives')
    fig_path = os.path.join(out_dir, f'calibration_{tag}.png')
    plt.savefig(fig_path, bbox_inches='tight', dpi=160); plt.close()
    return fig_path

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

def main(args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--out_dir", type=str, default="outputs")
    ap.add_argument("--train_end", type=str, default="2024-12-31")
    ap.add_argument("--valid_end", type=str, default="2025-08-31")
    ap.add_argument("--tau_days", type=float, default=60.0)
    ap.add_argument("--alpha_pct", type=float, default=30.0)
    cfg = ap.parse_args(args)

    os.makedirs(cfg.out_dir, exist_ok=True)
    matches = load_csv(os.path.join(cfg.data_dir, "matches.csv"))
    points_path = os.path.join(cfg.data_dir, "points_sets_games.csv")
    points = load_csv(points_path) if os.path.exists(points_path) else None

    feats = compute_pre_match_features_v2(matches, points, tau_days=cfg.tau_days, alpha_pct=cfg.alpha_pct)
    feats.to_csv(os.path.join(cfg.out_dir, "features_player_pre.csv"), index=False)

    dataset, model_cols = make_match_features(feats, matches)
    dataset.to_csv(os.path.join(cfg.out_dir, "dataset_match_level.csv"), index=False)

    train, valid, test = temporal_split(dataset, cfg.train_end, cfg.valid_end)
    lr, scaler, metrics, p_va = train_logistic_baseline(train, valid, model_cols)
    print("[Metrics]", metrics)
    fig_path = evaluate_and_plot_calibration(valid['y_home_win'].values, p_va, cfg.out_dir, "valid")
    print(f"[Saved] {fig_path}")

    X_te = scaler.transform(test[model_cols].fillna(0.0))
    y_te = test['y_home_win'].values
    p_te = lr.predict_proba(X_te)[:,1]
    test_metrics = {
        'test_logloss': float(log_loss(y_te, p_te)) if len(np.unique(y_te))>1 else None,
        'test_auc': float(roc_auc_score(y_te, p_te)) if len(np.unique(y_te))>1 else None,
        'test_brier': float(brier_score_loss(y_te, p_te)),
    }
    print("[Test]", test_metrics)

    out_preds = test[['match_id','date']].copy()
    out_preds['p_home_win'] = p_te; out_preds['y'] = y_te
    out_preds.to_csv(os.path.join(cfg.out_dir, "preds_test.csv"), index=False)

    joblib.dump(lr, os.path.join(cfg.out_dir, "model_logistic.pkl"))
    joblib.dump(scaler, os.path.join(cfg.out_dir, "scaler.pkl"))
    with open(os.path.join(cfg.out_dir, "model_columns.txt"), "w") as f:
        for c in model_cols:
            f.write(str(c) + "\n")
    print("[Saved] model_logistic.pkl, scaler.pkl and model_columns.txt")

if __name__ == "__main__":
    main()
