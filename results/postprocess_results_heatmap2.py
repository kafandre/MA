import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import SymLogNorm

def generate_drift_performance_heatmap():
    input_file = 'results.csv'
    output_file = 'relative_drift_heatmap.png'
        
    # Load data
    cols_to_keep = [
        'scenario', 'method', 'use_flooding',
        'mse_meaningful_weak_best', 'mse_meaningful_strong_best',
        'mse_noise_weak_best', 'mse_noise_strong_best'
    ]
    
    try:
        df = pd.read_csv(input_file, usecols=cols_to_keep)
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    # Filter out real datasets
    df = df[~df['scenario'].astype(str).str.lower().str.startswith('real')]

    # Parse 'scenario' column into signal and deviation
    df[['raw_sig', 'raw_scen']] = df['scenario'].str.split('_', n=1, expand=True)
    
    signal_map = {
        'linear': 'Linear',
        'smooth': 'Smooth',
        'sine': 'Sine',
        'step': 'Step',
        'mixed': 'Mixed'
    }
    scenario_map = {
        'base': 'Baseline',
        'corr': 'Multicollinearity',
        'highdim': 'High-dimensionality',
        'highnoise': 'High noise',
        'all': 'All deviations'
    }
    
    df['Signal'] = df['raw_sig'].str.lower().map(signal_map)
    df['Scenario'] = df['raw_scen'].str.lower().map(scenario_map)

    # Construct Full_Method names
    df['is_flooded'] = df['use_flooding'].astype(str).str.lower() == 'true'
    df['Full_Method'] = df['method']
    df.loc[df['is_flooded'], 'Full_Method'] = df['method'] + '+Flood'
    
    df['Full_Method'] = df['Full_Method'].str.replace('Momentum', 'Mom')

    # Average MSE values
    group_cols = ['Signal', 'Scenario', 'Full_Method']
    df_mean = df.groupby(group_cols)[
        ['mse_meaningful_weak_best', 'mse_meaningful_strong_best',
         'mse_noise_weak_best', 'mse_noise_strong_best']
    ].mean().reset_index()

    # Naming drift metrics
    drift_mapping = {
        'mse_meaningful_weak_best': 'Weak Concept',
        'mse_meaningful_strong_best': 'Strong Concept',
        'mse_noise_weak_best': 'Weak Noise',
        'mse_noise_strong_best': 'Strong Noise'
    }
    
    df_melted = df_mean.melt(
        id_vars=['Signal', 'Scenario', 'Full_Method'], 
        value_vars=list(drift_mapping.keys()), 
        var_name='raw_drift', 
        value_name='mse'
    )
    df_melted['Drift'] = df_melted['raw_drift'].map(drift_mapping)

    # Extract Vanilla baselines
    vanilla_mask = df_melted['Full_Method'] == 'Vanilla'
    df_vanilla = df_melted[vanilla_mask][['Signal', 'Scenario', 'Drift', 'mse']]
    df_vanilla = df_vanilla.rename(columns={'mse': 'mse_vanilla'})

    # Merge and calculate relative change
    df_merged = pd.merge(df_melted, df_vanilla, on=['Signal', 'Scenario', 'Drift'])
    df_merged['rel_change'] = (df_merged['mse'] - df_merged['mse_vanilla']) / df_merged['mse_vanilla']

    df_final = df_merged[df_merged['Full_Method'] != 'Vanilla'].copy()

    # Axes definitions
    method_order = [
        'Vanilla+Flood', 'Mom', 'Mom+Flood', 'TopK', 
        'TopK+Flood', 'TopK+Mom', 'TopK+Mom+Flood'
    ]
    drift_order = ['Weak Concept', 'Strong Concept', 'Weak Noise', 'Strong Noise']
    
    df_final['X_Label'] = df_final['Full_Method'] + " | " + df_final['Drift']
    x_categories = [f"{m} | {d}" for m in method_order for d in drift_order]
    df_final['X_Label'] = pd.Categorical(df_final['X_Label'], categories=x_categories, ordered=True)

    signal_order = ['Linear', 'Smooth', 'Sine', 'Step', 'Mixed']
    scenario_order = ['Baseline', 'Multicollinearity', 'High-dimensionality', 'High noise', 'All deviations']
    df_final['Signal'] = pd.Categorical(df_final['Signal'], categories=signal_order, ordered=True)
    df_final['Scenario'] = pd.Categorical(df_final['Scenario'], categories=scenario_order, ordered=True)
    
    heatmap_data = df_final.pivot_table(
        index=['Signal', 'Scenario'], 
        columns='X_Label', 
        values='rel_change', 
        dropna=False,
        observed=False
    )
    
    heatmap_data.index = [f"{sig}: {scen}" for sig, scen in heatmap_data.index]
    
    # Plotting 
    fig, ax = plt.subplots(figsize=(16, 12))
    
    sns.heatmap(
        heatmap_data, 
        annot=False,  
        cmap="vlag", 
        norm=SymLogNorm(linthresh=0.02, vmin=-0.40, vmax=0.40, base=10),
        cbar_kws={
            'label': 'Relative MSE Change vs. Vanilla CWB (Drifted Sets)', 
            'shrink': 0.8,
            'extend': 'both',
            'ticks': [-0.4, -0.1, -0.02, 0, 0.02, 0.1, 0.4]
        },
        linewidths=0.5,
        linecolor='lightgray',
        ax=ax
    )
    
    # Format Axes
    ax.set_ylabel('Signal Profile & Scenario', fontsize=14, labelpad=15)
    ax.set_xlabel('') 
    
    ax.xaxis.tick_top()
    
    drift_labels = [label.get_text().split(" | ")[1].replace(" ", "\n") for label in ax.get_xticklabels()]
    ax.set_xticklabels(drift_labels, rotation=90, ha='center', fontsize=10)
    plt.yticks(fontsize=11)

    # Add Custom Group Headers for Methods
    for i, method in enumerate(method_order):
        center_x = i * 4 + 2 
        
        if method == 'TopK+Mom+Flood':
            display_method = 'TopK+Mom\n+Flood'
        else:
            display_method = method
            
        ax.text(center_x, -2.5, display_method, ha='center', va='bottom', 
                fontsize=13, fontweight='bold', clip_on=False)

    # Add vertical lines 
    for i in range(4, 28, 4):
        ax.axvline(i, color='black', lw=1.5)
        
    # Add horizontal lines 
    for i in range(5, 25, 5):
        ax.axhline(i, color='black', lw=1.5)
    
    # top margin to 0.80
    plt.subplots_adjust(top=0.80)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    generate_drift_performance_heatmap()