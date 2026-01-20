import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    parser = argparse.ArgumentParser(description="Run plotting for new data format (Aggregated models).")
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

    # 3. Define Metrics
    metrics_to_plot = ["selection_rate", "disparate_impact_ratio"]
    ensure_dirs(args.output_dir)

    print(f"Plotting metrics: {metrics_to_plot} (Aggregating all models)")

    # 4. Loop over Metrics (One plot per metric, combining all models)
    for metric in metrics_to_plot:
        
        # Initialize figure
        plt.figure(figsize=(10, 6))

        # --- Aggregate Data ---
        # We take the column for the metric across ALL rows (all models)
        all_data = working_df[metric].dropna()
        values = all_data.values.tolist()

        if not values:
            print(f"No data found for Metric: {metric}")
            plt.close()
            continue

        # Clean outliers (optional, keeping consistent with previous logic)
        clean_values = [v for v in values if -5 <= v <= 5]
        
        if not clean_values:
            print(f"No valid data after cleaning for {metric}")
            plt.close()
            continue

        # ----------------------------
        # Plotting Logic (Single Global Density)
        # ----------------------------
        color = "purple" # Unified color for the aggregate

        if len(clean_values) > 1 and np.std(clean_values) > 1e-9:
            try:
                kde = gaussian_kde(clean_values)
                
                x_min, x_max = min(clean_values), max(clean_values)
                x_range = x_max - x_min
                if x_range == 0: x_range = 1.0
                
                # Compute grid
                x_grid = np.linspace(x_min - 0.2 * x_range, x_max + 0.2 * x_range, 500)
                y_grid = kde(x_grid)

                # Plot single line
                plt.plot(x_grid, y_grid, color=color, lw=3, label="All Models Combined")
                plt.fill_between(x_grid, y_grid, color=color, alpha=0.3)
                plt.ylabel("Density")
                
            except Exception as e:
                print(f"KDE failed for {metric}: {e}. Plotting Histogram.")
                plt.hist(clean_values, bins=30, color=color, alpha=0.6, density=True, label="All Models Combined")
                plt.ylabel("Density")
        else:
            # Low variance fallback
            plt.hist(clean_values, bins=30, color=color, alpha=0.6, density=True, label="All Models Combined")
            plt.ylabel("Density")

        # ----------------------------
        # Formatting & Saving
        # ----------------------------
        plt.title(f'Global Distribution of {metric} (All Models Aggregated)\nGender: W')
        plt.xlabel(metric)
        plt.grid(axis='y', alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        
        out_path = f"{args.output_dir}/{metric}_global_distribution.png"
        print(f"Saving global plot to {out_path}")
        plt.savefig(out_path)
        plt.close()

if __name__ == "__main__":
    main()
