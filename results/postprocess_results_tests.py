import pandas as pd
import numpy as np
import scipy.stats as stats
import warnings

def run_significance_test():
    input_file = 'results.csv'
    output_file = 'significance_summary.csv'
    
    print(f"Reading {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    results = []
    
    # Suppress warnings
    warnings.filterwarnings('ignore')

    print("Calculating Statistics")

    scenarios = df['scenario'].unique()

    for scen in scenarios:
        # Split scenario string into 'signal' and 'scenario'
        parts = scen.split('_', 1)
        signal_val = parts[0]
        scenario_val = parts[1] if len(parts) > 1 else scen

        # Get Baseline (Vanilla)
        baseline_df = df[
            (df['scenario'] == scen) & 
            (df['method'] == 'Vanilla') & 
            (df['use_flooding'] == False)
        ].set_index('seed')

        if baseline_df.empty:
            continue

        # Get deviation setups
        current_setups = df[df['scenario'] == scen][['method', 'use_flooding']].drop_duplicates()

        for _, setup in current_setups.iterrows():
            method = setup['method']
            flood = setup['use_flooding']

            if method == 'Vanilla' and flood == False:
                continue

            treatment_df = df[
                (df['scenario'] == scen) & 
                (df['method'] == method) & 
                (df['use_flooding'] == flood)
            ].set_index('seed')

            # Common Seeds
            common_seeds = baseline_df.index.intersection(treatment_df.index)
            if len(common_seeds) < 3: 
                continue

            row = {
                'signal': signal_val,
                'scenario': scenario_val,
                'method': method,
                'use_flooding': flood
            }

            if 'mse_clean' not in treatment_df.columns or 'mse_clean' not in baseline_df.columns:
                row['mse_clean_p'] = np.nan
                row['mse_clean_rel_change'] = np.nan
                row['mse_clean_test'] = np.nan
                results.append(row)
                continue

            # CALCULATION
            vals_t = treatment_df.loc[common_seeds, 'mse_clean']
            vals_b = baseline_df.loc[common_seeds, 'mse_clean']
            
            # Calculate Differences
            differences = vals_t - vals_b
            
            # Check for Identical Data
            if np.allclose(differences, differences.iloc[0]):
                if np.allclose(differences, 0):
                    p_value = 1.0
                    test_name = 'identical'
                else:
                    try:
                        stat, p_value = stats.ttest_rel(vals_t, vals_b)
                        test_name = 't-test (const)'
                    except:
                        p_value = 1.0
                        test_name = 'error'
            else:
                # Normality Test (Shapiro-Wilk)
                try:
                    shapiro_stat, shapiro_p = stats.shapiro(differences)
                    is_normal = shapiro_p > 0.05
                except:
                    is_normal = False

                # Statistical Test
                if is_normal:
                    test_name = 't-test'
                    try:
                        stat, p_value = stats.ttest_rel(vals_t, vals_b)
                    except:
                        p_value = 1.0
                else:
                    test_name = 'Wilcoxon'
                    try:
                        stat, p_value = stats.wilcoxon(vals_t, vals_b)
                    except ValueError:
                        p_value = 1.0

            # Calculate relative change & round
            baseline_mean = vals_b.mean()
            if baseline_mean != 0:
                rel_change = round(differences.mean() / baseline_mean, 4)
            else:
                rel_change = np.nan  

            # Store Results
            row['mse_clean_p'] = round(p_value, 5)
            row['mse_clean_rel_change'] = rel_change
            row['mse_clean_test'] = test_name

            results.append(row)

    # SAVE AND FORMAT
    if not results:
        print("No results generated.")
        return

    results_df = pd.DataFrame(results)
    
    # Custom Sorting
    signal_order = ['linear', 'smooth', 'sine', 'step', 'mixed', 'real']
    scenario_order = ['base', 'corr', 'highdim', 'highnoise', 'all', 'bodyfat', 'diabetes', 'riboflavin']
    
    results_df['signal'] = pd.Categorical(results_df['signal'], categories=signal_order, ordered=True)
    results_df['scenario'] = pd.Categorical(results_df['scenario'], categories=scenario_order, ordered=True)

    # Apply the sort
    results_df.sort_values(by=['method', 'use_flooding', 'signal', 'scenario'], inplace=True)
    
    # Rename Values
    results_df['signal'] = results_df['signal'].astype(str).str.capitalize()
    
    scenario_mapping = {
        'base': 'Baseline',
        'corr': 'Multicollinearity',
        'highdim': 'High-dimensionality',
        'highnoise': 'High noise',
        'all': 'All deviations'
    }
    
    # Map the custom strings
    results_df['scenario'] = results_df['scenario'].astype(str).map(lambda x: scenario_mapping.get(x, x.capitalize()))
    
    print(f"Computed statistics for {len(results_df)} setups.")
    results_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    run_significance_test()