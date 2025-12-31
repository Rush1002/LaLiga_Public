
"""
Odds utilities: implied probabilities and de-vig normalization.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

def implied_probs_from_odds(odds_home: float, odds_draw: float, odds_away: float) -> np.ndarray:
    """Return raw implied probs (not de-vigged) for [home, draw, away]."""
    o = np.array([odds_home, odds_draw, odds_away], dtype=float)
    p = 1.0 / o
    return p

def devig_normalize(p_raw: np.ndarray) -> np.ndarray:
    """Normalize implied probabilities to sum to 1."""
    s = np.sum(p_raw)
    if not np.isfinite(s) or s <= 0:
        return np.array([np.nan, np.nan, np.nan])
    return p_raw / s

def add_market_probs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - market_p_home, market_p_draw, market_p_away (de-vigged)
      - market_overround (sum(raw implied probs) - 1)
    """
    p_raw = np.vstack([
        implied_probs_from_odds(h,d,a)
        for h,d,a in zip(df["odds_home"], df["odds_draw"], df["odds_away"])
    ])
    overround = p_raw.sum(axis=1) - 1.0
    p = p_raw / p_raw.sum(axis=1, keepdims=True)
    df = df.copy()
    df["market_p_home"] = p[:,0]
    df["market_p_draw"] = p[:,1]
    df["market_p_away"] = p[:,2]
    df["market_overround"] = overround
    return df
