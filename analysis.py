"""
Enhanced analysis utilities:
- Bootstrap confidence intervals
- Per-team performance analysis
- Time-based performance trends
- Statistical significance tests
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import sys
sys.path.append('..')

from sklearn.metrics import log_loss
from src.model import PoissonStrengthModel, poisson_match_probs
from src.evaluation import multiclass_brier, outcome_label

def bootstrap_confidence_intervals(train_df: pd.DataFrame, test_df: pd.DataFrame,
                                   n_bootstrap: int = 100, confidence: float = 0.95) -> Dict:
    """
    Generate bootstrap confidence intervals for model performance.
    
    Resamples training data, retrains model, evaluates on test set.
    Returns confidence intervals for log loss and Brier score.
    """
    print(f"Running {n_bootstrap} bootstrap iterations...")
    log_losses = []
    brier_scores = []
    
    y_true = test_df.apply(outcome_label, axis=1).values
    
    for i in range(n_bootstrap):
        if (i + 1) % 20 == 0:
            print(f"  Bootstrap iteration {i+1}/{n_bootstrap}")
        
        # Resample training data with replacement
        train_boot = train_df.sample(n=len(train_df), replace=True)
        
        # Train model
        try:
            model = PoissonStrengthModel(reg=1.0, home_adv=0.10).fit(train_boot)
            
            # Predict on test set
            probs = []
            for r in test_df.itertuples(index=False):
                try:
                    lam_h, lam_a = model.predict_lambdas(r.home_team, r.away_team)
                    probs.append(poisson_match_probs(lam_h, lam_a, max_goals=10))
                except KeyError:
                    # Team not in bootstrap sample, skip
                    probs.append([np.nan, np.nan, np.nan])
            
            probs = np.vstack(probs)
            
            # Remove NaN predictions
            valid_idx = ~np.isnan(probs).any(axis=1)
            if valid_idx.sum() > 0:
                probs_valid = probs[valid_idx]
                y_true_valid = y_true[valid_idx]
                
                ll = log_loss(y_true_valid, probs_valid, labels=[0,1,2])
                br = multiclass_brier(probs_valid, y_true_valid)
                
                log_losses.append(ll)
                brier_scores.append(br)
        except:
            # Optimization failed, skip this iteration
            continue
    
    log_losses = np.array(log_losses)
    brier_scores = np.array(brier_scores)
    
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    results = {
        'log_loss': {
            'mean': float(np.mean(log_losses)),
            'std': float(np.std(log_losses)),
            'ci_lower': float(np.percentile(log_losses, lower_percentile)),
            'ci_upper': float(np.percentile(log_losses, upper_percentile)),
        },
        'brier': {
            'mean': float(np.mean(brier_scores)),
            'std': float(np.std(brier_scores)),
            'ci_lower': float(np.percentile(brier_scores, lower_percentile)),
            'ci_upper': float(np.percentile(brier_scores, upper_percentile)),
        },
        'n_successful': len(log_losses),
    }
    
    print(f"✓ Bootstrap complete ({results['n_successful']}/{n_bootstrap} successful)")
    return results

def per_team_analysis(test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze model performance for each team.
    
    Returns DataFrame with metrics per team.
    """
    teams = sorted(set(test_df['home_team']).union(set(test_df['away_team'])))
    
    team_stats = []
    for team in teams:
        # Matches involving this team
        team_matches = test_df[
            (test_df['home_team'] == team) | (test_df['away_team'] == team)
        ].copy()
        
        if len(team_matches) == 0:
            continue
        
        # Outcomes
        team_matches['team_result'] = team_matches.apply(
            lambda r: 'W' if (
                (r['home_team'] == team and r['home_goals'] > r['away_goals']) or
                (r['away_team'] == team and r['away_goals'] > r['home_goals'])
            ) else ('D' if r['home_goals'] == r['away_goals'] else 'L'),
            axis=1
        )
        
        wins = (team_matches['team_result'] == 'W').sum()
        draws = (team_matches['team_result'] == 'D').sum()
        losses = (team_matches['team_result'] == 'L').sum()
        
        # Average model probability for team outcomes
        team_probs = []
        for _, r in team_matches.iterrows():
            if r['home_team'] == team:
                # Team is home
                prob_win = r['p_home_model']
                prob_draw = r['p_draw_model']
                prob_loss = r['p_away_model']
            else:
                # Team is away
                prob_win = r['p_away_model']
                prob_draw = r['p_draw_model']
                prob_loss = r['p_home_model']
            
            team_probs.append({
                'prob_win': prob_win,
                'prob_draw': prob_draw,
                'prob_loss': prob_loss,
                'actual': r['team_result']
            })
        
        team_probs_df = pd.DataFrame(team_probs)
        
        avg_prob_win = team_probs_df['prob_win'].mean()
        avg_prob_draw = team_probs_df['prob_draw'].mean()
        avg_prob_loss = team_probs_df['prob_loss'].mean()
        
        # Expected vs actual
        expected_wins = team_probs_df['prob_win'].sum()
        expected_draws = team_probs_df['prob_draw'].sum()
        expected_losses = team_probs_df['prob_loss'].sum()
        
        team_stats.append({
            'team': team,
            'matches': len(team_matches),
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'win_pct': wins / len(team_matches),
            'expected_wins': expected_wins,
            'expected_win_pct': expected_wins / len(team_matches),
            'expected_draws': expected_draws,
            'expected_losses': expected_losses,
            'win_diff': wins - expected_wins,
            'avg_prob_win': avg_prob_win,
            'avg_prob_draw': avg_prob_draw,
            'avg_prob_loss': avg_prob_loss,
            'avg_ev': team_matches['best_ev'].mean(),
        })
    
    team_df = pd.DataFrame(team_stats).sort_values('win_pct', ascending=False)
    return team_df

def analyze_prediction_errors(test_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Find matches with largest prediction errors.
    
    Returns matches where model was most confident but wrong.
    """
    test_df = test_df.copy()
    
    # Model confidence = max probability
    model_probs = test_df[['p_home_model', 'p_draw_model', 'p_away_model']].values
    test_df['model_confidence'] = model_probs.max(axis=1)
    test_df['model_prediction'] = model_probs.argmax(axis=1)
    
    # Prediction error
    test_df['prediction_correct'] = (test_df['model_prediction'] == test_df['y_true'])
    
    # Biggest confident mistakes
    mistakes = test_df[~test_df['prediction_correct']].copy()
    mistakes = mistakes.sort_values('model_confidence', ascending=False).head(top_n)
    
    outcome_map = {0: 'Home', 1: 'Draw', 2: 'Away'}
    mistakes['predicted_outcome'] = mistakes['model_prediction'].map(outcome_map)
    mistakes['actual_outcome'] = mistakes['y_true'].map(outcome_map)
    
    return mistakes[[
        'date', 'home_team', 'away_team', 'home_goals', 'away_goals',
        'model_confidence', 'predicted_outcome', 'actual_outcome',
        'p_home_model', 'p_draw_model', 'p_away_model'
    ]]

def time_series_performance(test_df: pd.DataFrame, window: int = 38) -> pd.DataFrame:
    """
    Calculate rolling performance metrics over time.
    
    Args:
        window: Number of matches for rolling window (38 = one full round)
    """
    test_df = test_df.copy().sort_values('date').reset_index(drop=True)
    
    # Calculate per-match log loss
    match_log_losses = []
    for _, r in test_df.iterrows():
        probs = np.array([r['p_home_model'], r['p_draw_model'], r['p_away_model']])
        true_idx = int(r['y_true'])
        # Log loss for single sample
        ll = -np.log(probs[true_idx] + 1e-10)
        match_log_losses.append(ll)
    
    test_df['match_log_loss'] = match_log_losses
    
    # Rolling metrics
    test_df['rolling_log_loss'] = test_df['match_log_loss'].rolling(window=window).mean()
    test_df['rolling_avg_ev'] = test_df['best_ev'].rolling(window=window).mean()
    
    return test_df

def compare_to_baseline(test_df: pd.DataFrame) -> Dict:
    """
    Compare model to simple baselines.
    
    Baselines:
    1. Always predict home win (home advantage only)
    2. Historical outcome frequencies
    3. Uniform probabilities [0.33, 0.33, 0.33]
    """
    y_true = test_df['y_true'].values
    n = len(y_true)
    
    # Model performance
    model_probs = test_df[['p_home_model', 'p_draw_model', 'p_away_model']].values
    model_ll = log_loss(y_true, model_probs, labels=[0,1,2])
    model_br = multiclass_brier(model_probs, y_true)
    
    # Baseline 1: Historical frequencies
    freq_home = (y_true == 0).mean()
    freq_draw = (y_true == 1).mean()
    freq_away = (y_true == 2).mean()
    baseline_freq = np.tile([freq_home, freq_draw, freq_away], (n, 1))
    
    baseline_freq_ll = log_loss(y_true, baseline_freq, labels=[0,1,2])
    baseline_freq_br = multiclass_brier(baseline_freq, y_true)
    
    # Baseline 2: Uniform
    baseline_uniform = np.tile([1/3, 1/3, 1/3], (n, 1))
    baseline_uniform_ll = log_loss(y_true, baseline_uniform, labels=[0,1,2])
    baseline_uniform_br = multiclass_brier(baseline_uniform, y_true)
    
    # Baseline 3: Always predict home
    baseline_home = np.tile([0.8, 0.15, 0.05], (n, 1))  # Strong home bias
    baseline_home_ll = log_loss(y_true, baseline_home, labels=[0,1,2])
    baseline_home_br = multiclass_brier(baseline_home, y_true)
    
    return {
        'model': {'log_loss': model_ll, 'brier': model_br},
        'baseline_frequency': {'log_loss': baseline_freq_ll, 'brier': baseline_freq_br},
        'baseline_uniform': {'log_loss': baseline_uniform_ll, 'brier': baseline_uniform_br},
        'baseline_home_bias': {'log_loss': baseline_home_ll, 'brier': baseline_home_br},
    }

def export_team_strengths(model, output_path: str = 'team_strengths.csv'):
    """Export team strengths to CSV for easy viewing."""
    if not model.fitted_:
        raise ValueError("Model must be fitted first")
    
    teams = model.teams_
    n_teams = len(teams)
    mu, ha, attack, defense = model._unpack(model.params_, n_teams)
    
    df = pd.DataFrame({
        'team': teams,
        'attack_strength': attack,
        'defense_strength': -defense,  # Flip sign
        'expected_goals_scored': np.exp(mu + ha + attack),  # At home
        'expected_goals_conceded': np.exp(mu - defense),  # At home
    }).sort_values('attack_strength', ascending=False)
    
    df.to_csv(output_path, index=False)
    print(f"✓ Team strengths exported to {output_path}")
    return df
