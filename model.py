
"""
Interpretable team-strength Poisson goal model.

We fit attack/defense strengths with a log-link:
    log(lambda_home) = mu + home_adv + attack[home] + defense[away]
    log(lambda_away) = mu + attack[away] + defense[home]

Note: defense terms are typically negative for strong defenses; we learn them freely.
Identifiability: we impose sum(attack)=0 and sum(defense)=0 via constraints by
re-centering after optimization.

Fitting: penalized maximum likelihood (L2 regularization).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

@dataclass
class PoissonStrengthModel:
    reg: float = 1.0          # L2 penalty strength
    home_adv: float = 0.10    # initial value on log scale
    fitted_: bool = False

    teams_: List[str] = None
    params_: np.ndarray = None  # [mu, home_adv, attacks..., defenses...]

    def _pack(self, mu: float, home_adv: float, attack: np.ndarray, defense: np.ndarray) -> np.ndarray:
        return np.concatenate([[mu, home_adv], attack, defense])

    def _unpack(self, x: np.ndarray, n_teams: int) -> Tuple[float, float, np.ndarray, np.ndarray]:
        mu = x[0]
        ha = x[1]
        attack = x[2:2+n_teams]
        defense = x[2+n_teams:2+2*n_teams]
        # enforce identifiability by centering
        attack = attack - attack.mean()
        defense = defense - defense.mean()
        return mu, ha, attack, defense

    def fit(self, df: pd.DataFrame) -> "PoissonStrengthModel":
        df = df.copy().sort_values("date")
        teams = sorted(set(df["home_team"]).union(set(df["away_team"])))
        team_to_idx = {t:i for i,t in enumerate(teams)}
        n = len(teams)

        home_idx = df["home_team"].map(team_to_idx).to_numpy()
        away_idx = df["away_team"].map(team_to_idx).to_numpy()
        y_home = df["home_goals"].to_numpy().astype(int)
        y_away = df["away_goals"].to_numpy().astype(int)

        # init
        mu0 = np.log(np.maximum(df[["home_goals","away_goals"]].to_numpy().mean(), 1e-6))
        a0 = np.zeros(n)
        d0 = np.zeros(n)
        x0 = self._pack(mu0, self.home_adv, a0, d0)

        def nll(x: np.ndarray) -> float:
            mu, ha, a, d = self._unpack(x, n)
            lam_home = np.exp(mu + ha + a[home_idx] + d[away_idx])
            lam_away = np.exp(mu + a[away_idx] + d[home_idx])

            # Poisson negative log-likelihood (up to constant): lam - y*log(lam) + log(y!)
            ll = (
                -lam_home + y_home * np.log(lam_home) - gammaln(y_home + 1) +
                -lam_away + y_away * np.log(lam_away) - gammaln(y_away + 1)
            ).sum()
            # L2 regularization on team params (exclude mu and home_adv)
            reg = self.reg * (np.sum(a*a) + np.sum(d*d))
            return -(ll) + reg

        res = minimize(nll, x0, method="L-BFGS-B")
        if not res.success:
            raise RuntimeError(f"Optimization failed: {res.message}")

        self.teams_ = teams
        self.params_ = res.x
        self.fitted_ = True
        return self

    def predict_lambdas(self, home_team: str, away_team: str) -> Tuple[float, float]:
        if not self.fitted_:
            raise ValueError("Model not fitted")
        teams = self.teams_
        n = len(teams)
        idx = {t:i for i,t in enumerate(teams)}
        mu, ha, a, d = self._unpack(self.params_, n)
        hi = idx[home_team]; ai = idx[away_team]
        lam_home = float(np.exp(mu + ha + a[hi] + d[ai]))
        lam_away = float(np.exp(mu + a[ai] + d[hi]))
        return lam_home, lam_away

def poisson_match_probs(lam_home: float, lam_away: float, max_goals: int = 10) -> np.ndarray:
    """
    Compute [P(home), P(draw), P(away)] by summing scorelines 0..max_goals.
    """
    from scipy.stats import poisson
    ph = 0.0; pd = 0.0; pa = 0.0
    for i in range(max_goals+1):
        pi = poisson.pmf(i, lam_home)
        for j in range(max_goals+1):
            p = pi * poisson.pmf(j, lam_away)
            if i > j: ph += p
            elif i == j: pd += p
            else: pa += p
    s = ph+pd+pa
    return np.array([ph/s, pd/s, pa/s], dtype=float)
