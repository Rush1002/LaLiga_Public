"""
Enhanced visualization utilities for La Liga pricing study.

Includes:
- Calibration curves (reliability diagrams)
- Team strength visualization
- EV distribution analysis
- Model vs market comparison plots
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

import sys
sys.path.append('..')
from src.evaluation import calibration_data_binary

def plot_calibration_curves(model_probs: np.ndarray, market_probs: np.ndarray, 
                            y_true: np.ndarray, outcome_names: list = None,
                            figsize: tuple = (15, 5)):
    """
    Plot calibration curves for each outcome (Home/Draw/Away).
    
    A well-calibrated model: when it predicts 40%, the outcome happens ~40% of the time.
    """
    if outcome_names is None:
        outcome_names = ['Home Win', 'Draw', 'Away Win']
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    for outcome_idx in range(3):
        ax = axes[outcome_idx]
        
        # Binary labels: 1 if this outcome happened, 0 otherwise
        y_binary = (y_true == outcome_idx).astype(int)
        
        # Model calibration
        model_p = model_probs[:, outcome_idx]
        try:
            mean_pred_model, frac_pos_model = calibration_data_binary(model_p, y_binary, n_bins=10)
            ax.plot(mean_pred_model, frac_pos_model, 'o-', linewidth=2, 
                   label='Model', color='#2E86AB', markersize=8)
        except:
            pass
        
        # Market calibration
        market_p = market_probs[:, outcome_idx]
        try:
            mean_pred_market, frac_pos_market = calibration_data_binary(market_p, y_binary, n_bins=10)
            ax.plot(mean_pred_market, frac_pos_market, 's-', linewidth=2,
                   label='Market', color='#A23B72', markersize=8)
        except:
            pass
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Perfect Calibration')
        
        ax.set_xlabel('Predicted Probability', fontsize=11, fontweight='bold')
        ax.set_ylabel('Observed Frequency', fontsize=11, fontweight='bold')
        ax.set_title(f'{outcome_names[outcome_idx]} Calibration', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    return fig

def plot_team_strengths(model, figsize: tuple = (12, 8)):
    """
    Visualize team attack and defense strengths.
    """
    if not model.fitted_:
        raise ValueError("Model must be fitted first")
    
    teams = model.teams_
    n_teams = len(teams)
    mu, ha, attack, defense = model._unpack(model.params_, n_teams)
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        'Team': teams,
        'Attack': attack,
        'Defense': -defense  # Flip sign so positive = good defense
    }).sort_values('Attack', ascending=False)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Attack strengths
    colors_attack = ['#27ae60' if x > 0 else '#e74c3c' for x in df['Attack']]
    axes[0].barh(df['Team'], df['Attack'], color=colors_attack, alpha=0.8, edgecolor='black')
    axes[0].axvline(x=0, color='black', linewidth=2)
    axes[0].set_xlabel('Attack Strength (log scale)', fontsize=11, fontweight='bold')
    axes[0].set_title('Team Attack Strengths', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Defense strengths (flipped so positive = good)
    df_def = df.sort_values('Defense', ascending=False)
    colors_def = ['#27ae60' if x > 0 else '#e74c3c' for x in df_def['Defense']]
    axes[1].barh(df_def['Team'], df_def['Defense'], color=colors_def, alpha=0.8, edgecolor='black')
    axes[1].axvline(x=0, color='black', linewidth=2)
    axes[1].set_xlabel('Defense Strength (log scale, flipped)', fontsize=11, fontweight='bold')
    axes[1].set_title('Team Defense Strengths', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig

def plot_ev_distribution(test_df: pd.DataFrame, figsize: tuple = (14, 5)):
    """
    Plot distribution of expected values and cumulative analysis.
    """
    evs = test_df['best_ev'].values
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 1. Histogram of EVs
    axes[0].hist(evs, bins=30, alpha=0.7, color='#3498db', edgecolor='black')
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero EV')
    axes[0].axvline(x=np.mean(evs), color='green', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(evs):.4f}')
    axes[0].set_xlabel('Expected Value', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0].set_title('Distribution of Best EV per Match', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 2. Cumulative EV over time
    cumulative_ev = np.cumsum(evs)
    axes[1].plot(cumulative_ev, linewidth=2, color='#9b59b6')
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    axes[1].set_xlabel('Match Number', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Cumulative EV', fontsize=11, fontweight='bold')
    axes[1].set_title('Cumulative Expected Value', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Positive vs Negative EV
    positive_ev = evs[evs > 0]
    negative_ev = evs[evs <= 0]
    
    data = [len(positive_ev), len(negative_ev)]
    labels = [f'Positive EV\n({len(positive_ev)} matches)', 
             f'Non-positive EV\n({len(negative_ev)} matches)']
    colors = ['#27ae60', '#e74c3c']
    
    axes[2].pie(data, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
    axes[2].set_title('Positive vs Non-positive EV', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_model_vs_market(test_df: pd.DataFrame, figsize: tuple = (15, 5)):
    """
    Scatter plots comparing model probabilities to market probabilities.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    outcomes = ['home', 'draw', 'away']
    titles = ['Home Win', 'Draw', 'Away Win']
    colors = ['#3498db', '#f39c12', '#e74c3c']
    
    for i, (outcome, title, color) in enumerate(zip(outcomes, titles, colors)):
        model_col = f'p_{outcome}_model'
        market_col = f'market_p_{outcome}'
        
        x = test_df[market_col].values
        y = test_df[model_col].values
        
        axes[i].scatter(x, y, alpha=0.5, s=30, color=color, edgecolor='black', linewidth=0.5)
        axes[i].plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='y=x')
        
        # Add correlation
        corr = np.corrcoef(x, y)[0, 1]
        axes[i].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                    transform=axes[i].transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        axes[i].set_xlabel('Market Probability', fontsize=11, fontweight='bold')
        axes[i].set_ylabel('Model Probability', fontsize=11, fontweight='bold')
        axes[i].set_title(f'{title} Probability Comparison', fontsize=12, fontweight='bold')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim([0, 1])
        axes[i].set_ylim([0, 1])
    
    plt.tight_layout()
    return fig

def plot_performance_by_favorite(test_df: pd.DataFrame, figsize: tuple = (12, 5)):
    """
    Analyze model performance on favorites vs underdogs.
    """
    # Identify favorite (lowest odds = highest implied probability)
    test_df = test_df.copy()
    odds_array = test_df[['odds_home', 'odds_draw', 'odds_away']].values
    favorite_idx = np.argmin(odds_array, axis=1)
    test_df['favorite'] = favorite_idx
    
    # Calculate if model agrees with market on favorite
    model_probs = test_df[['p_home_model', 'p_draw_model', 'p_away_model']].values
    model_favorite = np.argmax(model_probs, axis=1)
    test_df['model_agrees'] = (favorite_idx == model_favorite)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 1. Agreement rate
    agree_rate = test_df['model_agrees'].mean()
    data = [agree_rate * 100, (1 - agree_rate) * 100]
    labels = [f'Agree\n({agree_rate:.1%})', f'Disagree\n({1-agree_rate:.1%})']
    colors = ['#27ae60', '#e74c3c']
    
    axes[0].pie(data, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    axes[0].set_title('Model-Market Agreement on Favorite', fontsize=12, fontweight='bold')
    
    # 2. Performance when agreeing vs disagreeing
    agree_subset = test_df[test_df['model_agrees']]
    disagree_subset = test_df[~test_df['model_agrees']]
    
    # Check if favorite won
    agree_subset['favorite_won'] = (agree_subset['y_true'] == agree_subset['favorite'])
    disagree_subset['favorite_won'] = (disagree_subset['y_true'] == disagree_subset['favorite'])
    
    agree_win_rate = agree_subset['favorite_won'].mean() if len(agree_subset) > 0 else 0
    disagree_win_rate = disagree_subset['favorite_won'].mean() if len(disagree_subset) > 0 else 0
    
    categories = ['Model Agrees\nwith Market', 'Model Disagrees\nwith Market']
    win_rates = [agree_win_rate * 100, disagree_win_rate * 100]
    colors = ['#3498db', '#9b59b6']
    
    bars = axes[1].bar(categories, win_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    axes[1].set_ylabel('Favorite Win Rate (%)', fontsize=11, fontweight='bold')
    axes[1].set_title('Favorite Win Rate by Agreement', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, win_rates):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_summary_report(results: dict, save_path: Optional[str] = None):
    """
    Generate comprehensive visual summary report.
    """
    test_df = results['test_df']
    model = results['model']
    
    model_probs = test_df[['p_home_model', 'p_draw_model', 'p_away_model']].values
    market_probs = test_df[['market_p_home', 'market_p_draw', 'market_p_away']].values
    y_true = test_df['y_true'].values
    
    # Create all plots
    print("Creating calibration curves...")
    fig1 = plot_calibration_curves(model_probs, market_probs, y_true)
    
    print("Creating team strength visualization...")
    fig2 = plot_team_strengths(model)
    
    print("Creating EV distribution analysis...")
    fig3 = plot_ev_distribution(test_df)
    
    print("Creating model vs market comparison...")
    fig4 = plot_model_vs_market(test_df)
    
    print("Creating performance by favorite analysis...")
    fig5 = plot_performance_by_favorite(test_df)
    
    if save_path:
        fig1.savefig(f'{save_path}_calibration.png', dpi=300, bbox_inches='tight')
        fig2.savefig(f'{save_path}_strengths.png', dpi=300, bbox_inches='tight')
        fig3.savefig(f'{save_path}_ev_dist.png', dpi=300, bbox_inches='tight')
        fig4.savefig(f'{save_path}_model_vs_market.png', dpi=300, bbox_inches='tight')
        fig5.savefig(f'{save_path}_performance.png', dpi=300, bbox_inches='tight')
        print(f"Figures saved to {save_path}_*.png")
    
    return [fig1, fig2, fig3, fig4, fig5]
