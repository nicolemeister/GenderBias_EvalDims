import argparse
import ast
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde

# ----------------------------
# Parsing helpers
# ----------------------------
def _safe_literal_eval(x: Any) -> Any:
    """ast.literal_eval with guardrails."""
    if pd.isna(x):
        return None
    if isinstance(x, (list, dict, tuple, int, float)):
        return x
    s = str(x).strip()
    if s == "":
        return None
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def parse_float_list(x: Any) -> Optional[List[float]]:
    """
    Parses things like:
      "[np.float64(0.1), np.float64(0.2)]"
      "[0.1, 0.2]"
      "0.1"
    into a list[float] when possible.
    """
    if pd.isna(x):
        return None

    if isinstance(x, list):
        try:
            return [float(v) for v in x]
        except Exception:
            return None

    s = str(x)
    s = s.replace("np.float64(", "").replace(")", "")
    val = _safe_literal_eval(s)
    if val is None:
        # last-ditch: try float
        try:
            return [float(s)]
        except Exception:
            return None

    if isinstance(val, list):
        try:
            return [float(v) for v in val]
        except Exception:
            return None

    # scalar
    try:
        return [float(val)]
    except Exception:
        return None


def parse_str_list(x: Any) -> Optional[List[str]]:
    val = _safe_literal_eval(x)
    if val is None:
        return None
    if isinstance(val, list):
        return [str(v) for v in val]
    return None


def parse_p_values(x: Any) -> Optional[List[float]]:
    """
    Your P_Value_Vector appears to be stored like "[0.123]" (string).
    Parse into list[float]. If it's a scalar, return [scalar].
    """
    if pd.isna(x):
        return None
    if isinstance(x, list):
        try:
            return [float(v) for v in x]
        except Exception:
            return None

    val = _safe_literal_eval(x)
    if val is None:
        # try stripping brackets manually
        s = str(x).strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1].strip()
        try:
            return [float(s)]
        except Exception:
            return None

    if isinstance(val, list):
        try:
            return [float(v) for v in val]
        except Exception:
            return None

    try:
        return [float(val)]
    except Exception:
        return None


# ----------------------------
# Domain helpers
# ----------------------------
def woman_index(groups: Optional[List[str]]) -> int:
    """Index of 'Woman' in Sensitive_Attribute_Vector; fallback 0."""
    if not groups:
        return 0
    try:
        return groups.index("Woman")
    except ValueError:
        return 0


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def pad_ylim(values: List[float], frac: float = 0.05) -> Tuple[float, float]:
    if not values:
        return (0.0, 1.0)
    ymin, ymax = min(values), max(values)
    span = ymax - ymin
    if span == 0:
        span = 1.0
    return (ymin - frac * span, ymax + frac * span)


def label_for_cell(name: str, job: str) -> str:
    return f"name_{name}_job_{job}"


def clean_values(values: List[float], remove_zeros: bool = True, remove_outliers: bool = True) -> List[float]:
    """Clean values by removing zeros and/or outliers."""
    cleaned = values.copy()
    if remove_zeros:
        cleaned = [v for v in cleaned if v != 0]
    if remove_outliers:
        cleaned = [v for v in cleaned if -5 <= v <= 5]
    return cleaned


# ----------------------------
# Plotting functions
# ----------------------------
def plot_global_density(values: List[float], metric: str, out_path: str) -> None:
    """Plot a single global density plot for all values."""
    clean_vals = clean_values(values)
    
    if len(clean_vals) > 1 and np.std(clean_vals) > 1e-9:
        plt.figure(figsize=(10, 6))
        kde = gaussian_kde(clean_vals)
        x_min, x_max = min(clean_vals), max(clean_vals)
        x_range = x_max - x_min
        x_grid = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 500)
        y_grid = kde(x_grid)
        
        plt.plot(x_grid, y_grid, color="purple", lw=2, label="Density")
        plt.fill_between(x_grid, y_grid, color="purple", alpha=0.3)
        plt.title(f'Density Plot of {metric} (All Data)')
        plt.xlabel(metric)
        plt.ylabel("Density")
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
    else:
        # Fallback to histogram
        plt.figure(figsize=(10, 6))
        plt.hist(clean_vals, bins=100, alpha=0.6, color="purple", edgecolor='black')
        plt.title(f'Distribution of {metric} (All Data)')
        plt.xlabel(metric)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(out_path.replace('density', 'histogram'))
        plt.close()


def plot_global_histogram(values: List[float], metric: str, out_path: str) -> None:
    clean_vals = clean_values(values)
    
    # Calculate the same limits as the density plot
    x_min, x_max = min(clean_vals), max(clean_vals)
    x_range = x_max - x_min
    limit_min = x_min - 0.1 * x_range
    limit_max = x_max + 0.1 * x_range

    plt.figure(figsize=(10, 6))
    plt.hist(clean_vals, bins=100, alpha=0.6, color="purple", edgecolor='black')
    
    # Explicitly set the x-axis limits
    plt.xlim(limit_min, limit_max)
    
    plt.title(f'Distribution of {metric} (All Data)')
    plt.xlabel(metric)
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_split_density_cdf(
    split_dict: Dict[str, List[float]], 
    metric: str, 
    split_type: str,
    out_root: str,
    bls_values: Optional[Dict[str, float]] = None
) -> None:
    """Plot density, CDF, and histogram plots split by model/job/name."""
    if not split_dict:
        print(f"No data to plot for {split_type} split.")
        return
    
    # Initialize figures
    fig_den, ax_den = plt.subplots(figsize=(12, 7))
    fig_cdf, ax_cdf = plt.subplots(figsize=(12, 7))
    fig_hist, ax_hist = plt.subplots(figsize=(12, 7))
    
    # Use colormap
    n_items = len(split_dict)
    if n_items <= 10:
        colors_list = plt.cm.tab10(np.linspace(0, 1, n_items))
    else:
        colors_list = plt.cm.tab20(np.linspace(0, 1, n_items))
    
    has_plotted = False
    
    # Determine global x range for histogram bins
    all_clean_vals = []
    for values in split_dict.values():
        clean_vals = clean_values(values)
        if len(clean_vals) > 0:
            all_clean_vals.extend(clean_vals)
    
    # Set histogram bins based on all data
    if all_clean_vals:
        hist_bins = np.linspace(min(all_clean_vals), max(all_clean_vals), 30)
    else:
        hist_bins = 30
    
    for idx, (key, values) in enumerate(split_dict.items()):
        clean_vals = clean_values(values)
        
        if len(clean_vals) > 0:
            # Determine label
            if bls_values and key in bls_values:
                label_text = f"{key} (BLS: {bls_values[key]})"
            else:
                label_text = str(key)
            
            # Plot Histogram
            ax_hist.hist(clean_vals, 
                        bins=hist_bins if isinstance(hist_bins, np.ndarray) else 30,
                        alpha=0.5, 
                        color=colors_list[idx], 
                        edgecolor='black',
                        label=label_text,
                        density=False)  # Use frequency, not density
            
            if len(clean_vals) > 1 and np.std(clean_vals) > 1e-9:
                # Plot Density
                try:
                    kde = gaussian_kde(clean_vals)
                    x_min, x_max = min(clean_vals), max(clean_vals)
                    x_range = x_max - x_min
                    if x_range == 0:
                        x_range = 1.0
                    x_grid = np.linspace(x_min - 0.2 * x_range, x_max + 0.2 * x_range, 200)
                    
                    ax_den.plot(x_grid, kde(x_grid), 
                               color=colors_list[idx], 
                               lw=2, 
                               alpha=0.8, 
                               label=label_text)
                except np.linalg.LinAlgError:
                    print(f"Skipping density for {key}: Singular matrix (no variance).")
                    # Still plot histogram and CDF even if density fails
                
                # Plot CDF
                sorted_data = np.sort(clean_vals)
                yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                
                ax_cdf.step(sorted_data, yvals, 
                           where='post', 
                           color=colors_list[idx], 
                           lw=2, 
                           alpha=0.8, 
                           label=label_text)
                
                has_plotted = True
            else:
                print(f"Skipping density/CDF for {key}: Not enough variance (but histogram will be plotted).")
                # Still plot histogram even if variance is low
                has_plotted = True
    
    if has_plotted:
        # Save Histogram Plot
        ax_hist.set_title(f'Comparison of {metric} Histogram by {split_type.capitalize()}')
        ax_hist.set_xlabel(metric)
        ax_hist.set_ylabel("Frequency")
        ax_hist.grid(True, linestyle="--", alpha=0.3, axis='y')
        ax_hist.legend(title=split_type.capitalize(), bbox_to_anchor=(1.05, 1), loc='upper left')
        fig_hist.tight_layout()
        
        hist_path = f"{out_root}/{metric}_combined_histogram_by_{split_type}.png"
        print(f"Saving combined histogram plot to {hist_path}")
        fig_hist.savefig(hist_path)
        
        # Save Density Plot
        ax_den.set_title(f'Comparison of {metric} Density by {split_type.capitalize()}')
        ax_den.set_xlabel(metric)
        ax_den.set_ylabel("Density")
        ax_den.grid(True, linestyle="--", alpha=0.3)
        ax_den.legend(title=split_type.capitalize(), bbox_to_anchor=(1.05, 1), loc='upper left')
        fig_den.tight_layout()
        
        den_path = f"{out_root}/{metric}_combined_density_by_{split_type}.png"
        print(f"Saving combined density plot to {den_path}")
        fig_den.savefig(den_path)
        
        # Save CDF Plot
        ax_cdf.set_title(f'Comparison of {metric} CDF by {split_type.capitalize()}')
        ax_cdf.set_xlabel(metric)
        ax_cdf.set_ylabel("Cumulative Probability")
        ax_cdf.grid(True, linestyle="--", alpha=0.3)
        ax_cdf.legend(title=split_type.capitalize(), bbox_to_anchor=(1.05, 1), loc='upper left')
        fig_cdf.tight_layout()
        
        cdf_path = f"{out_root}/{metric}_combined_cdf_by_{split_type}.png"
        print(f"Saving combined CDF plot to {cdf_path}")
        fig_cdf.savefig(cdf_path)
    else:
        print(f"No {split_type}s had enough data to plot.")
    
    plt.close(fig_den)
    plt.close(fig_cdf)
    plt.close(fig_hist)


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Unified plotting script for regression coefficients.")
    parser.add_argument("--sampling", type=str, default="downsampling", help="Sampling method to use.")
    parser.add_argument("--analysis", type=str, default="original", help="Analysis to perform.")
    parser.add_argument("--exp_framework", type=str, default="armstrong", help="Experiment framework to use.")
    parser.add_argument("--model", type=str, default="all", help="Model to use (or 'all').")
    parser.add_argument("--metric", type=str, default="regression_coefficients", help="Metric to plot.")
    parser.add_argument("--split_by", type=str, default="all", 
                       choices=["all", "model", "job", "name"], 
                       help="How to split the data: 'all' (no split), 'model', 'job', or 'name'.")
    args = parser.parse_args()

    # BLS values for job legend (optional)
    bls_values = {
        'armstrong': 57.564,
        'rozado': 47.47,
        'wen': 57.88,
        'wang': 28.71,
        'karvonen': 67.98,
        'zollo': 48.77,
        'yin': 46.5
    }

    # Load data
    in_path = f"results/{args.exp_framework}_sampling_{args.sampling}_analysis_{args.analysis}.csv"
    if not os.path.exists(in_path):
        print(f"Error: File {in_path} not found.")
        return
    
    df = pd.read_csv(in_path)

    # Parse columns
    if "Result_Vector" in df.columns:
        df["Result_Vector"] = df["Result_Vector"].apply(parse_float_list)
        df["Std_Error_Vector"] = df["Std_Error_Vector"].apply(parse_float_list)
        df["Sensitive_Attribute_Vector"] = df["Sensitive_Attribute_Vector"].apply(parse_str_list)
        df["P_Value_Vector"] = df["P_Value_Vector"].apply(parse_p_values)
        data_format = "armstrong"
    else:
        # Assume yin format or other format
        data_format = "yin"
        print("Warning: Data format not recognized. Attempting to use direct column access.")

    # Filter by model if specified
    if args.model != "all" and "Model" in df.columns:
        working_df = df[df["Model"] == args.model].copy()
    else:
        working_df = df.copy()

    # Grid definition (10×7)
    row_labels = ["armstrong", "rozado", "wen", "wang", "gaeb", "lippens", "seshadri", "karvonen", "zollo", "yin"]
    col_labels = ["armstrong", "rozado", "wen", "wang", "karvonen", "zollo", "yin"]

    # Output dirs
    out_root = f"results/figs/{args.exp_framework}/{args.model}"
    ensure_dirs(f"results/figs/{args.exp_framework}", out_root)

    # Build lookup table for grid logic
    if data_format == "armstrong":
        df_keyed = (
            working_df.sort_values(["Names", "Jobs", "Metric"])
              .drop_duplicates(subset=["Names", "Jobs", "Metric"], keep="first")
              .set_index(["Names", "Jobs", "Metric"])
        )

        def get_cell_row(name: str, job: str, metric: str) -> Optional[pd.Series]:
            try:
                return df_keyed.loc[(name, job, metric)]
            except KeyError:
                return None

        def get_woman_stats(row: Optional[pd.Series]) -> Tuple[float, Optional[float], Optional[float]]:
            """Returns (value_for_woman, stderr_for_woman, p_value_scalar_or_none)"""
            if row is None:
                return (0.0, None, None)

            groups = row.get("Sensitive_Attribute_Vector", None)
            idx = woman_index(groups)

            rv = row.get("Result_Vector", None)
            sv = row.get("Std_Error_Vector", None)
            pv = row.get("P_Value_Vector", None)

            val = 0.0
            err = None
            p = None

            if isinstance(rv, list) and len(rv) > idx:
                val = float(rv[idx])

            if isinstance(sv, list) and len(sv) > idx:
                err = float(sv[idx])

            if isinstance(pv, list) and len(pv) > 0:
                p = float(pv[0])

            return (val, err, p)
    else:
        # For yin format or other formats, define simpler accessors
        df_keyed = None
        
        def get_cell_row(name: str, job: str, metric: str) -> Optional[pd.Series]:
            return None  # Not supported for non-armstrong format
        
        def get_woman_stats(row: pd.Series) -> Tuple[float, Optional[float], Optional[float]]:
            metric_col = args.metric
            if metric_col in row:
                val = float(row[metric_col]) if pd.notna(row[metric_col]) else 0.0
            else:
                val = 0.0
            return (val, None, None)

    # Colors for job groups
    label_to_color = {
        "armstrong": "tab:blue",
        "rozado": "tab:orange",
        "wen": "tab:green",
        "wang": "tab:red",
        "karvonen": "tab:purple",
        "zollo": "tab:brown",
        "yin": "tab:pink",
        "lippens": "tab:gray",
        "seshadri": "tab:olive",
        "gaeb": "tab:cyan",
    }

    metric = args.metric

    # ----------------------------
    # Main plotting logic based on split_by
    # ----------------------------
    if args.split_by == "all":
        # Collect all values across all models
        print(f"Collecting all values for {metric}...")
        all_collected_values = []
        
        if data_format == "armstrong":
            for _, row in df.iterrows():
                if row.get("Metric") == metric:
                    val, _, _ = get_woman_stats(row)
                    all_collected_values.append(val)
        else:
            # For yin format
            if "gender" in df.columns:
                working_subset = df[df["gender"] == "W"]
            else:
                working_subset = df
            if metric in working_subset.columns:
                all_collected_values = working_subset[metric].dropna().tolist()
        
        if all_collected_values:
            plot_global_density(
                all_collected_values, 
                metric, 
                f"{out_root}/{metric}_global_density_all.png"
            )
            plot_global_histogram(
                all_collected_values,
                metric,
                f"{out_root}/{metric}_global_histogram_all.png"
            )

    elif args.split_by == "model":
        # Split by model
        print(f"Generating plots split by model for {metric}...")
        if "Model" not in df.columns:
            print("Error: 'Model' column not found in data.")
            return
        
        unique_models = df["Model"].unique()
        split_dict = {}
        
        for model_name in unique_models:
            model_subset = df[(df["Model"] == model_name) & (df["Metric"] == metric)]
            model_values = []
            for _, row in model_subset.iterrows():
                val, _, _ = get_woman_stats(row)
                model_values.append(val)
            if model_values:
                split_dict[model_name] = model_values
        
        plot_split_density_cdf(split_dict, metric, "model", out_root)

    elif args.split_by == "job":
        # Split by job
        print(f"Generating plots split by job for {metric}...")
        if "Jobs" not in df.columns:
            print("Error: 'Jobs' column not found in data.")
            return
        
        unique_jobs = df["Jobs"].unique()
        split_dict = {}
        
        for job_val in unique_jobs:
            job_subset = df[(df["Jobs"] == job_val) & (df["Metric"] == metric)]
            job_values = []
            for _, row in job_subset.iterrows():
                val, _, _ = get_woman_stats(row)
                job_values.append(val)
            if job_values:
                split_dict[job_val] = job_values
        
        plot_split_density_cdf(split_dict, metric, "job", out_root, bls_values)

    elif args.split_by == "name":
        # Split by name
        print(f"Generating plots split by name for {metric}...")
        if "Names" not in df.columns:
            print("Error: 'Names' column not found in data.")
            return
        
        unique_names = df["Names"].unique()
        split_dict = {}
        
        for name_val in unique_names:
            name_subset = df[(df["Names"] == name_val) & (df["Metric"] == metric)]
            name_values = []
            for _, row in name_subset.iterrows():
                val, _, _ = get_woman_stats(row)
                name_values.append(val)
            if name_values:
                split_dict[name_val] = name_values
        
        plot_split_density_cdf(split_dict, metric, "name", out_root)

    # ----------------------------
    # Grid Generation (Heatmap/Bar Chart logic) - Only for armstrong format
    # ----------------------------
    if data_format == "armstrong" and "Names" in df.columns and "Jobs" in df.columns:
        values_mat = np.zeros((len(row_labels), len(col_labels)), dtype=float)
        errors_mat: List[List[Optional[float]]] = [[None] * len(col_labels) for _ in row_labels]
        pvals_mat: List[List[Optional[float]]] = [[None] * len(col_labels) for _ in row_labels]

        for i, name in enumerate(row_labels):
            for j, job in enumerate(col_labels):
                row = get_cell_row(name, job, metric)
                v, e, p = get_woman_stats(row)
                values_mat[i, j] = v
                errors_mat[i][j] = e
                pvals_mat[i][j] = p

        values = values_mat.reshape(-1).tolist()
        errors = [errors_mat[i][j] for i in range(len(row_labels)) for j in range(len(col_labels))]
        p_values = [pvals_mat[i][j] for i in range(len(row_labels)) for j in range(len(col_labels))]

        labels = [label_for_cell(n, j) for n in row_labels for j in col_labels]
        colors = [label_to_color.get(job, "tab:blue") for _name in row_labels for job in col_labels]

        nonzero_values = [v for v in values if v != 0]
        ylims = pad_ylim(nonzero_values if nonzero_values else values)

        # Grid-Specific Histogram
        plt.figure(figsize=(10, 6))
        plt.hist(nonzero_values, bins=30, alpha=0.5, color="blue")
        plt.title(f'Histogram of {metric} for "Woman" (Grid Subset 10x7)')
        plt.xlabel(metric)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"{out_root}/{metric}_sampling_{args.sampling}_analysis_{args.analysis}_grid_histogram_nonzero.png")
        plt.close()

        # Heatmap
        plt.figure(figsize=(7, 10))
        plt.imshow(values_mat, cmap="viridis", aspect="auto")
        plt.colorbar(label="Value")
        plt.xticks(ticks=range(len(col_labels)), labels=col_labels)
        plt.yticks(ticks=range(len(row_labels)), labels=row_labels)
        plt.title("Names × Jobs Heatmap")
        plt.xlabel("Jobs")
        plt.ylabel("Names")
        plt.tight_layout()
        plt.savefig(f"{out_root}/{metric}_sampling_{args.sampling}_analysis_{args.analysis}_all_heatmap.png")
        plt.close()

        # Save heatmap data
        heatmap_df = pd.DataFrame(values_mat, columns=col_labels, index=row_labels)
        heatmap_df.to_csv(
            f"{out_root}/{metric}_sampling_{args.sampling}_analysis_{args.analysis}_all_heatmap_data.csv",
            index=True,
        )

        # Bar Chart
        yerr = None
        if any(e is not None for e in errors):
            non_none = [e for e in errors if e is not None]
            avg = float(np.mean(non_none)) if non_none else 0.0
            yerr = []
            for v, e in zip(values, errors):
                if v == 0 or np.isnan(v):
                    yerr.append(0)
                else:
                    yerr.append(avg if e is None else e)

        plt.figure(figsize=(40, 6))
        bars = plt.bar(labels, values, yerr=yerr, capsize=4, align="center", color=colors)
        
        legend_handles = [mpatches.Patch(color=c, label=lab) for lab, c in label_to_color.items()]
        plt.legend(handles=legend_handles, title="Job Group")

        alpha = 0.05 / (len(row_labels) * len(col_labels))
        for bar, p in zip(bars, p_values):
            if p is not None and p < alpha:
                h = bar.get_height()
                plt.annotate(
                    "*",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=18,
                    color="red",
                )

        plt.ylabel(metric)
        plt.ylim(ylims)
        plt.title(f'{metric} for "Woman" ({args.sampling}, {args.analysis}, {args.exp_framework})')
        plt.xticks(rotation=90)
        plt.tight_layout()

        out_path = f"{out_root}/{metric}_sampling_{args.sampling}_analysis_{args.analysis}_all.png"
        print(f"Saving figure to {out_path}")
        plt.savefig(out_path)
        plt.close()


if __name__ == "__main__":
    main()


'''
# Plot all data together
python plot_armstrong.py --split_by all --metric regression_coefficients

# Split by model
python plot_armstrong.py --split_by model --metric regression_coefficients

# Split by job
python plot_armstrong.py --split_by job --metric regression_coefficients

# Split by name
python plot_armstrong.py --split_by name --metric regression_coefficients

'''