import pandas as pd

def process_csv():
    input_file = 'results.csv'
    output_file = 'avg_summary.csv'

    group_cols = ['scenario', 'method', 'use_flooding']
    
    mean_cols = [
        'mse_clean', 
        'mse_meaningful_weak_best', 
        'mse_meaningful_strong_best',
        'mse_noise_weak_best', 
        'mse_noise_strong_best'
    ]

    print(f"Reading {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}")
        return

    # Filter Columns (Drop all except relevant ones)
    cols_to_keep = group_cols + mean_cols
    
    missing_cols = [col for col in cols_to_keep if col not in df.columns]
    if missing_cols:
        print(f"Error: The following required columns are missing from the input: {missing_cols}")
        return

    df = df[cols_to_keep]
    
    print("Grouping and aggregating...")
    result = df.groupby(group_cols)[mean_cols].mean().reset_index()

    result[mean_cols] = result[mean_cols].round(4)
    print("Applying structural changes...")


    result.loc[result['use_flooding'] == True, 'method'] = result['method'].astype(str) + '+Flood'

    # Delete 'use_flooding' column
    result.drop(columns=['use_flooding'], inplace=True)

    # Split 'scenario' column
    # Split at the underscore
    split_scenario = result['scenario'].str.split('_', n=1, expand=True)
    
    # Update original 'scenario' to be the first part
    result['scenario'] = split_scenario[0]
    
    # Insert 'scenario2' right beside 'scenario' (index 1)
    result.insert(1, 'scenario2', split_scenario[1])

    # Sorting
    print("Sorting data...")
    result.sort_values(by=['scenario', 'scenario2', 'method'], inplace=True)

    # Save the file
    print(f"Saving to {output_file}...")
    result.to_csv(output_file, index=False)
    print("Done.")

if __name__ == "__main__":
    process_csv()