import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

SCENARIO_METADATA = {
    # LINEAR
    "linear_base":      {"noise_std": 2.0},
    "linear_highdim":   {"noise_std": 2.0},
    "linear_highnoise": {"noise_std": 5.0},
    "linear_corr":      {"noise_std": 2.0},
    "linear_all":       {"noise_std": 5.0},

    # SMOOTH
    "smooth_base":      {"noise_std": 2.0},
    "smooth_highdim":   {"noise_std": 2.0},
    "smooth_highnoise": {"noise_std": 5.0},
    "smooth_corr":      {"noise_std": 2.0},
    "smooth_all":       {"noise_std": 5.0},

    # STEP
    "step_base":        {"noise_std": 1.0},
    "step_highdim":     {"noise_std": 1.0},
    "step_highnoise":   {"noise_std": 2.0},
    "step_corr":        {"noise_std": 1.0},
    "step_all":         {"noise_std": 2.0},

    # SINE
    "sine_base":        {"noise_std": 2.0},
    "sine_highdim":     {"noise_std": 2.0},
    "sine_highnoise":   {"noise_std": 5.0},
    "sine_corr":        {"noise_std": 2.0},
    "sine_all":         {"noise_std": 5.0},
}

SIGNALS = ["linear", "smooth", "step", "sine"]
VARIATIONS = ["base", "highnoise", "highdim", "corr", "all"]
VARIATION_LABELS = ["Base", "High Noise", "High Dim", "High Corr", "All"]

METHODS_ORIGINAL = {
    "Vanilla":              ("Vanilla", False),
    "Vanilla+Flood":        ("Vanilla", True),
    "TopK":                 ("TopK", False),
    "TopK+Flood":           ("TopK", True),
    "Momentum":             ("Momentum", False),
    "Momentum+Flood":       ("Momentum", True),
    "TopK+Momentum":        ("TopK+Momentum", False),
    "TopK+Momentum+Flood":  ("TopK+Momentum", True),
}

# New Method Order Mapping
METHODS_NEW_ORDER = [
    ("Vanilla",             ("Vanilla", False)),
    ("Vanilla+Flood",       ("Vanilla", True)),
    ("Mom",                 ("Momentum", False)),
    ("TopK",                ("TopK", False)),
    ("Mom+TopK",            ("TopK+Momentum", False)),
    ("Mom+Flood",           ("Momentum", True)),
    ("TopK+Flood",          ("TopK", True)),
    ("Mom+TopK+Flood",      ("TopK+Momentum", True)),
]

SEEDS = range(1000, 1030)
RESULTS_DIR = "histories"
OUTPUT_DIR = "feature_selection_plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Helper Functions

def get_history_filename(scenario, method_name, seed, use_flooding, noise_std):
    base = f"{scenario}_{method_name}_s{seed}"
    if use_flooding:
        flood_level = noise_std ** 2
        suffix = f"_flood{flood_level:.4f}"
    else:
        suffix = "_clean"
    return f"Hist_{base}{suffix}.pkl"

def compute_ratios_for_scenario(scenario_key, method_conf, seed):
    method_name, use_flooding = method_conf
    meta = SCENARIO_METADATA.get(scenario_key)

    filename = get_history_filename(scenario_key, method_name, seed, use_flooding, meta['noise_std'])
    filepath = os.path.join(RESULTS_DIR, filename)

    if not os.path.exists(filepath):
        return None

    try:
        with open(filepath, 'rb') as f:
            history = pickle.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

    # Determine Best Iteration
    val_losses = history.get('val_loss', [])
    if not val_losses:
        return None
    
    best_iter_idx = np.argmin(val_losses) 
    
    # Get selections up to best iteration
    selected = history['selected_features'][:best_iter_idx+1]
    
    if len(selected) == 0:
        return [0, 0, 0, 0]

    # Count x0, x1, x2, noise
    counts = [0, 0, 0, 0]
    
    for feat_idx in selected:
        if feat_idx == 0:
            counts[0] += 1
        elif feat_idx == 1:
            counts[1] += 1
        elif feat_idx == 2:
            counts[2] += 1
        else:
            counts[3] += 1
            
    # Normalize
    total = len(selected)
    ratios = [c / total for c in counts]
    return ratios

def create_stacked_bar_plot(plot_data, x_labels, title, filename_suffix, x_axis_label):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ["#8AD3FB", "#8AC2FB", "#8AB2FB", "#FB7171"] 
    legend_labels = ['x0', 'x1', 'x2', 'Noise']
    
    bottoms = np.zeros(len(x_labels))
    x_positions = np.arange(len(x_labels))
    
    for i in range(4): # Loop through x0, x1, x2, Noise
        heights = plot_data[:, i]
        bars = ax.bar(x_positions, heights, bottom=bottoms, label=legend_labels[i], color=colors[i], edgecolor='white', width=0.6)
        
        # Add text inside bars
        for bar, val in zip(bars, heights):
            if val > 4: 
                y_center = bar.get_y() + bar.get_height() / 2
                ax.text(bar.get_x() + bar.get_width()/2, y_center, f"{val:.1f}%", 
                        ha='center', va='center', color='black', fontsize=10, fontweight='bold')
        
        bottoms += heights

    # Styling
    ax.set_ylabel("Selection Percentage (%)", fontsize=13)
    ax.set_xlabel(x_axis_label, fontsize=13)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=12, rotation=45 if len(x_labels) > 5 else 0)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_ylim(0, 100)
    
    # Legend
    handles, leg_labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], leg_labels[::-1], loc='upper left', bbox_to_anchor=(1, 1), title="Features", fontsize=12, title_fontsize=13)
    
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, f"FeatureSelection_{filename_suffix}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

# Main Loops

def generate_method_plots():
    
    for method_display_name, method_conf in METHODS_ORIGINAL.items():        
        plot_data = np.zeros((len(VARIATIONS), 4)) 
        
        for v_idx, variation in enumerate(VARIATIONS):
            accumulated_ratios = []
            
            # Aggregate over Signals and Seeds
            for signal in SIGNALS:
                scenario_key = f"{signal}_{variation}"
                for seed in SEEDS:
                    ratios = compute_ratios_for_scenario(scenario_key, method_conf, seed)
                    if ratios is not None:
                        accumulated_ratios.append(ratios)
            
            if accumulated_ratios:
                avg_ratios = np.mean(accumulated_ratios, axis=0)
                plot_data[v_idx] = avg_ratios * 100
        
        safe_name = method_display_name.replace(" ", "_").replace("+", "_plus_")
        create_stacked_bar_plot(
            plot_data=plot_data,
            x_labels=VARIATION_LABELS,
            title=f"Average Feature Selections for {method_display_name}",
            filename_suffix=f"Method_{safe_name}",
            x_axis_label="Scenario Variation"
        )

def generate_variation_plots():
    
    method_labels = [m[0] for m in METHODS_NEW_ORDER]
    
    for v_idx, variation in enumerate(VARIATIONS):
        variation_display = VARIATION_LABELS[v_idx]
        print(f"Processing Variation: {variation_display}")
        
        plot_data = np.zeros((len(METHODS_NEW_ORDER), 4))
        
        for m_idx, (method_label, method_conf) in enumerate(METHODS_NEW_ORDER):
            accumulated_ratios = []
            
            # Aggregate over Signals and Seeds
            for signal in SIGNALS:
                scenario_key = f"{signal}_{variation}"
                for seed in SEEDS:
                    ratios = compute_ratios_for_scenario(scenario_key, method_conf, seed)
                    if ratios is not None:
                        accumulated_ratios.append(ratios)
            
            if accumulated_ratios:
                avg_ratios = np.mean(accumulated_ratios, axis=0)
                plot_data[m_idx] = avg_ratios * 100
        
        safe_name = variation.replace(" ", "_")
        create_stacked_bar_plot(
            plot_data=plot_data,
            x_labels=method_labels,
            title=f"Average Feature Selections for {variation_display} Scenarios",
            filename_suffix=f"Variation_{safe_name}",
            x_axis_label="Method"
        )

if __name__ == "__main__":
    generate_method_plots()
    generate_variation_plots()