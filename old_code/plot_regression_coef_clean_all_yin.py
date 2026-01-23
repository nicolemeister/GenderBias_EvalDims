import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import gaussian_kde

# ----------------------------
# Helpers
# ----------------------------
def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)

# ----------------------------
# Main
# ----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run plotting for new data format (All models on one plot).")
    parser.add_argument("--input_file", type=str, default="/nlp/scr/nmeist/EvalDims/results/yin_sampling_downsampling_analysis_original_prev.csv", help="Path to input CSV.")
    parser.add_argument("--output_dir", type=str, default="results/figs/yin/all", help="Directory to save figures.")
    args = parser.parse_args()

    # 1. Load Data
    if not os.path.exists(args.input_file):
        print(f"Error: File {args.input_file} not found.")
        return

    df = pd.read_csv(args.input_file)
    
    # 2. Filter Data (Focus on 'W')
    working_df = df[df["gender"] == "W"].copy()

    # 3. Define Metrics and Models
    metrics_to_plot = ["selection_rate", "disparate_impact_ratio"]
    unique_models = working_df["model"].unique()
    
    ensure_dirs(args.output_dir)

    print(f"Found models: {unique_models}")
    print(f"Plotting metrics: {metrics_to_plot}")

    # Generate distinct colors for the models
    # 'tab10' is good for up to 10 models; use 'tab20' if you have more
    cmap = cm.get_cmap('tab10') 
    
    # 4. Loop over Metrics (One plot per metric)
    for metric in metrics_to_plot:
        
        # Initialize the figure for this metric
        plt.figure(figsize=(12, 7))
        
        models_plotted = 0

        # Loop over Models (All on the same plot)
        for i, model_name in enumerate(unique_models):
            
            # Extract values
            model_data = working_df[working_df["model"] == model_name][metric].dropna()
            values = model_data.values.tolist()

            if not values:
                print(f"No data found for Model: {model_name}, Metric: {metric}")
                continue

            # Clean outliers
            clean_values = [v for v in values if -5 <= v <= 5]
            
            if not clean_values:
                continue

            # Assign color
            color = cmap(i % 10)

            # ----------------------------
            # Plotting Logic (KDE)
            # ----------------------------
            # We strictly prefer KDE for multi-model plots to avoid messy histogram overlap
            if len(clean_values) > 1 and np.std(clean_values) > 1e-9:
                try:
                    kde = gaussian_kde(clean_values)
                    
                    x_min, x_max = min(clean_values), max(clean_values)
                    x_range = x_max - x_min
                    if x_range == 0: x_range = 1.0
                    
                    # Compute grid
                    x_grid = np.linspace(x_min - 0.2 * x_range, x_max + 0.2 * x_range, 500)
                    y_grid = kde(x_grid)

                    # Plot Line (No Fill to keep it readable)
                    plt.plot(x_grid, y_grid, color=color, lw=2.5, label=model_name)
                    models_plotted += 1
                    
                except Exception as e:
                    print(f"KDE failed for {model_name}: {e}. Skipping.")
            else:
                # If variance is too low (e.g., all values are 1.0), plotting a line is hard.
                # We can plot a vertical line or a single bin histogram.
                print(f"Not enough variance for {model_name}. Plotting as vertical line.")
                plt.axvline(np.mean(clean_values), color=color, linestyle="--", lw=2, label=f"{model_name} (const)")
                models_plotted += 1

        # ----------------------------
        # Formatting & Saving
        # ----------------------------
        if models_plotted > 0:
            plt.title(f'Comparison of {metric} Across Models (Gender: W)')
            plt.xlabel(metric)
            plt.ylabel("Density")
            plt.grid(axis='y', alpha=0.3)
            plt.legend(title="Models", loc='best')
            plt.tight_layout()
            
            out_path = f"{args.output_dir}/{metric}_all_models_combined.png"
            print(f"Saving combined plot to {out_path}")
            plt.savefig(out_path)
        else:
            print(f"No valid data plotted for metric {metric}")
            
        plt.close()

if __name__ == "__main__":
    main()