"""
End-to-end pipeline for La Liga prediction system.

Training: Seasons 2019/20 through 2022/23 (4 seasons)
Validation: Season 2023/24
Testing: Season 2024/25 (with actual results for comparison)
Prediction: Season 2025/26 (ongoing season, partial data)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .data_loader import load_la_liga_matches, LoadSpec
from .odds import add_market_probs
from .model import PoissonStrengthModel, poisson_match_probs
from .evaluation import add_true_outcome, summarize_probs, best_ev_bet

def split_data(df: pd.DataFrame) -> dict:
    """
    Split data into train/val/test/future.
    
    Train: 1920, 2021, 2122, 2223 (4 seasons)
    Val: 2324 (1 season) 
    Test: 2425 (1 season) - for evaluation
    Future: 2526 (ongoing) - for predictions
    """
    train = df[df["season"].isin(["1920", "2021", "2122", "2223"])].copy()
    val = df[df["season"] == "2324"].copy()
    test = df[df["season"] == "2425"].copy()
    
    # Future predictions (2025/26 season - ongoing, may have partial data)
    future = df[df["season"] == "2526"].copy() if "2526" in df["season"].values else pd.DataFrame()
    
    return {
        "train": train,
        "val": val, 
        "test": test,
        "future": future
    }

def predict_matches(model: PoissonStrengthModel, matches_df: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions for a set of matches."""
    probs = []
    for r in matches_df.itertuples(index=False):
        try:
            lam_h, lam_a = model.predict_lambdas(r.home_team, r.away_team)
            probs.append(poisson_match_probs(lam_h, lam_a, max_goals=10))
        except KeyError:
            # Team not in training data
            probs.append([np.nan, np.nan, np.nan])
    
    probs = np.vstack(probs)
    
    result_df = matches_df.copy()
    result_df["p_home_model"] = probs[:, 0]
    result_df["p_draw_model"] = probs[:, 1]
    result_df["p_away_model"] = probs[:, 2]
    
    # Calculate EV for completed matches
    if "odds_home" in result_df.columns:
        evs = []
        for i, r in enumerate(result_df.itertuples(index=False)):
            if pd.notna(r.odds_home):
                odds_triplet = np.array([r.odds_home, r.odds_draw, r.odds_away])
                bet_idx, ev = best_ev_bet(probs[i], odds_triplet)
                evs.append(ev)
            else:
                evs.append(np.nan)
        result_df["best_ev"] = evs
    
    return result_df

def run() -> dict:
    """
    Run complete analysis pipeline.
    
    Returns dict with:
    - model: Trained model
    - train_df, val_df, test_df: Historical data splits
    - test_metrics: Performance on 2024/25 season  
    - future_df: Predictions for 2025/26 season (if available)
    """
    print("Loading La Liga data...")
    df = load_la_liga_matches(LoadSpec(league="la_liga"))
    df = add_market_probs(df)
    df = add_true_outcome(df)
    
    print(f"Loaded {len(df)} matches from seasons: {sorted(df['season'].unique())}")
    
    # Split data
    splits = split_data(df)
    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]
    future_df = splits["future"]
    
    print(f"\nData splits:")
    print(f"  Train: {len(train_df)} matches (2019-2023)")
    print(f"  Val:   {len(val_df)} matches (2023/24)")
    print(f"  Test:  {len(test_df)} matches (2024/25)")
    if len(future_df) > 0:
        print(f"  Future: {len(future_df)} matches (2025/26 ongoing)")
    
    # Train model on train + val
    print("\nTraining Poisson team-strength model...")
    train_combined = pd.concat([train_df, val_df], ignore_index=True)
    model = PoissonStrengthModel(reg=1.0, home_adv=0.10).fit(train_combined)
    print("✓ Model trained")
    
    # Predict on test set (2024/25 - has actual results)
    print("\nGenerating predictions for 2024/25 season...")
    test_df = predict_matches(model, test_df)
    
    # Evaluate test performance
    test_df_complete = test_df.dropna(subset=["p_home_model"])
    if len(test_df_complete) > 0:
        model_probs = test_df_complete[["p_home_model", "p_draw_model", "p_away_model"]].values
        market_probs = test_df_complete[["market_p_home", "market_p_draw", "market_p_away"]].values
        y_true = test_df_complete["y_true"].values
        
        model_metrics = summarize_probs(model_probs, y_true)
        market_metrics = summarize_probs(market_probs, y_true)
    else:
        model_metrics = market_metrics = None
    
    # Predict on future matches (2025/26 ongoing)
    future_results = None
    if len(future_df) > 0:
        print(f"\nGenerating predictions for 2025/26 season ({len(future_df)} matches)...")
        future_results = predict_matches(model, future_df)
    
    return {
        "model": model,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "test_metrics": {
            "model": model_metrics,
            "market": market_metrics
        },
        "future_df": future_results
    }

if __name__ == "__main__":
    results = run()
    
    print("\n" + "="*60)
    print("RESULTS - 2024/25 Season (Test Set)")
    print("="*60)
    
    if results["test_metrics"]["model"]:
        print(f"\nModel Performance:")
        print(f"  Log Loss:    {results['test_metrics']['model']['log_loss']:.4f}")
        print(f"  Brier Score: {results['test_metrics']['model']['brier']:.4f}")
        
        print(f"\nMarket Performance:")
        print(f"  Log Loss:    {results['test_metrics']['market']['log_loss']:.4f}")
        print(f"  Brier Score: {results['test_metrics']['market']['brier']:.4f}")
        
        avg_ev = results["test_df"]["best_ev"].mean()
        print(f"\nAvg EV per match: {avg_ev:.4f}")
    
    if results["future_df"] is not None and len(results["future_df"]) > 0:
        print("\n" + "="*60)
        print("PREDICTIONS - 2025/26 Season (Ongoing)")
        print("="*60)
        print(f"\n{len(results['future_df'])} upcoming/recent matches predicted")
        print("\nSample predictions:")
        sample = results["future_df"][["date", "home_team", "away_team", 
                                        "p_home_model", "p_draw_model", "p_away_model"]].head(5)
        print(sample.to_string(index=False))
