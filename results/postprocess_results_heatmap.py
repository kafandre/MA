import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import SymLogNorm

def generate_performance_heatmap():
    input_file = 'significance_summary.csv'
    output_file = 'relative_performance_heatmap.png'
    
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Run postprocess_results_tests.py first.")
        return

    # Combine 'method' and 'use_flooding'
    df['Full_Method'] = df['method']
    df.loc[df['use_flooding'] == True, 'Full_Method'] = df['method'] + '+Flood'
    df.loc[df['Full_Method'] == 'Vanilla+Flood', 'Full_Method'] = 'Vanilla+Flood'

    # Logical Sorting
    signal_order = ['Linear', 'Smooth', 'Sine', 'Step', 'Mixed', 'Real']
    scenario_order = [
        'Baseline', 'Multicollinearity', 'High-dimensionality', 
        'High noise', 'All deviations', 'Bodyfat', 'Diabetes', 'Riboflavin'
    ]
    method_order = [
        'Vanilla+Flood', 'Momentum', 'Momentum+Flood', 'TopK', 
        'TopK+Flood', 'TopK+Momentum', 'TopK+Momentum+Flood'
    ]

    df['signal'] = pd.Categorical(df['signal'], categories=signal_order, ordered=True)
    df['scenario'] = pd.Categorical(df['scenario'], categories=scenario_order, ordered=True)
    df['Full_Method'] = pd.Categorical(df['Full_Method'], categories=method_order, ordered=True)

    # Pivot data
    heatmap_data = df.pivot_table(index=['signal', 'scenario'], columns='Full_Method', values='mse_clean_rel_change', dropna=False)
    p_data = df.pivot_table(index=['signal', 'scenario'], columns='Full_Method', values='mse_clean_p', dropna=False)

    heatmap_data = heatmap_data.dropna(how='all')
    p_data = p_data.dropna(how='all')

    # Generate Text Annotations (Percentage + Stars)
    def get_annotation(rel_change, p_val):
        if pd.isna(rel_change) or pd.isna(p_val):
            return ""
        
        stars = ""
        if p_val < 0.001:
            stars = "***"
        elif p_val < 0.01:
            stars = "**"
        elif p_val < 0.05:
            stars = "*"
            
        val_str = f"{rel_change * 100:+.1f}%"
        if stars:
            return f"{val_str}\n{stars}"
        return val_str

    annot_data = pd.DataFrame(index=heatmap_data.index, columns=heatmap_data.columns)
    for col in heatmap_data.columns:
        for row in heatmap_data.index:
            annot_data.loc[row, col] = get_annotation(heatmap_data.loc[row, col], p_data.loc[row, col])

    heatmap_data.index = [f"{sig}: {scen}" for sig, scen in heatmap_data.index]

    # Plotting
    plt.figure(figsize=(14, 16))
    
    ax = sns.heatmap(
        heatmap_data, 
        annot=annot_data, 
        fmt="", 
        cmap="vlag", 
        norm=SymLogNorm(linthresh=0.02, vmin=-0.40, vmax=0.40, base=10),
        cbar_kws={
            'label': 'Relative MSE Change vs. Vanilla CWB', 
            'shrink': 0.8,
            'extend': 'both',
            'ticks': [-0.4, -0.1, -0.02, 0, 0.02, 0.1, 0.4]
        },
        linewidths=0.5,
        linecolor='lightgray'
    )
    
    # Formatting
    plt.ylabel('Signal Profile & Scenario', fontsize=14, labelpad=15)
    plt.xlabel('Method Configuration', fontsize=14, labelpad=15)
    
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.xticks(rotation=45, ha='left', fontsize=12)
    plt.yticks(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    generate_performance_heatmap()