# CNN/plotter.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_results(arch, output_dir='results'):
    """Loads CSV files and generates plots comparing experiments."""
    print("--- Generating Plots ---")
    
    # Find all relevant CSV files
    results_files = [f for f in os.listdir(output_dir) if f.startswith(f'results_{arch}') and f.endswith('.csv')]
    if not results_files:
        print(f"No result files found for architecture '{arch}' in '{output_dir}'.")
        return

    # Load and combine data
    df_list = []
    for f in results_files:
        df = pd.read_csv(os.path.join(output_dir, f))
        df_list.append(df)
    full_df = pd.concat(df_list)

    # Plotting
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    
    plot = sns.lineplot(
        data=full_df,
        x='weights_remaining_pct',
        y='test_accuracy',
        hue='run_type',
        marker='o',
        errorbar='sd' # Show standard deviation if multiple trials are run
    )
    
    plot.set_title(f'Test Accuracy vs. Sparsity for {arch}')
    plot.set_xlabel('Percentage of Weights Remaining')
    plot.set_ylabel('Test Accuracy (%)')
    plt.xscale('log')
    plt.gca().invert_xaxis() # Invert x-axis to match paper's plots
    plt.legend(title='Experiment Type')
    
    # Save the plot
    plot_filename = os.path.join(output_dir, f'plot_{arch}.png')
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    plt.show()