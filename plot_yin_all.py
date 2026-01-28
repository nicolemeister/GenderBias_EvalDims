import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import ast


# ----------------------------
# Helpers
# ----------------------------
def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def add_pval_suffix_to_path(path: str, p_value_threshold: Optional[float]) -> str:
    """Add p-value threshold suffix to file path if threshold is provided."""
    if p_value_threshold is not None:
        base, ext = os.path.splitext(path)
        return f"{base}_pval_{p_value_threshold}{ext}"
    return path


def parse_p_values(x: Any) -> Optional[List[float]]:
    """
    Parse P_Value_Vector which may be stored as string like "[0.123]" or as list.
    """
    if pd.isna(x):
        return None
    if isinstance(x, list):
        try:
            return [float(v) for v in x]
        except Exception:
            return None
    
    # Try to parse as string representation of list
    try:
        val = ast.literal_eval(str(x))
        if isinstance(val, list):
            return [float(v) for v in val]
        return [float(val)]
    except Exception:
        # Try stripping brackets manually
        s = str(x).strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1].strip()
        try:
            return [float(s)]
        except Exception:
            return None


def clean_values(values: List[float], remove_zeros: bool = False, remove_outliers: bool = False) -> List[float]:
    """Clean values by removing zeros and/or outliers.
    
    Handles multi-indexed dataframes and various pandas types (Index, Series, scalars).
    """
    if not values:
        return []
    
    # First, extract actual scalar values from any pandas types (Index, Series, etc.)
    extracted_values = []
    for v in values:
        # Handle pandas Index objects
        if isinstance(v, pd.Index):
            extracted_values.extend(v.tolist())
        # Handle pandas Series
        elif isinstance(v, pd.Series):
            extracted_values.extend(v.tolist())
        # Handle numpy arrays
        elif isinstance(v, np.ndarray):
            extracted_values.extend(v.tolist())
        # Handle scalar values (including pandas scalars)
        else:
            # Use .item() for pandas scalars, otherwise just use the value
            if hasattr(v, 'item'):
                try:
                    extracted_values.append(v.item())
                except (ValueError, AttributeError):
                    extracted_values.append(v)
            else:
                extracted_values.append(v)
    
    # Now convert to numeric using pandas (handles strings, mixed types, etc.)
    series = pd.Series(extracted_values)
    numeric_series = pd.to_numeric(series, errors='coerce')
    
    # Get only finite, non-null values
    cleaned = numeric_series[numeric_series.notna() & np.isfinite(numeric_series)].tolist()
    
    if remove_zeros:
        cleaned = [v for v in cleaned if v != 0]
    if remove_outliers:
        cleaned = [v for v in cleaned if -5 <= v <= 5]
    return cleaned


# ----------------------------
# Plotting functions
# ----------------------------
def plot_global_density(clean_vals: List[float], metric: str, out_path: str, yin_values: Optional[List[float]] = None, vlines: Optional[List[Tuple[float, str]]] = None, sig_values: Optional[List[Tuple[float, float]]] = None, p_value_threshold: Optional[float] = None) -> None:
    """Plot a single global density plot for all values."""
    
    if len(clean_vals) > 1 and np.std(clean_vals) > 1e-9:
        plt.figure(figsize=(10, 6))
        kde = gaussian_kde(clean_vals)
        # kde.set_bandwidth(bw_method=0.1)
        x_min, x_max = min(clean_vals), max(clean_vals)
        x_range = x_max - x_min
        x_grid = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 500)
        y_grid = kde(x_grid)
        
        plt.plot(x_grid, y_grid, color="purple", lw=2, label="Density")
        plt.fill_between(x_grid, y_grid, color="purple", alpha=0.3)
        
        # Add vertical lines if provided
        if vlines:
            for x_val, label in vlines:
                plt.axvline(x=x_val, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=label)
        
        # Plot significant values if provided (just the values, no extra text)
        if sig_values:
            for x_val, _ in sig_values:
                plt.axvline(x=x_val, color='green', linestyle=':', linewidth=1, alpha=0.5)
        
        plt.title(f'Density Plot of {metric} (All Data)\nGender: W')
        plt.xlabel(metric)
        plt.ylabel("Density")
        plt.grid(axis='y', alpha=0.3)
        if vlines:
            plt.legend(loc='best')
        plt.tight_layout()
        final_path = add_pval_suffix_to_path(out_path, p_value_threshold)
        plt.savefig(final_path)
        plt.close()
    else:
        # Fallback to histogram
        plt.figure(figsize=(10, 6))
        plt.hist(clean_vals, bins=100, alpha=0.6, color="purple", edgecolor='black')
        
        # Add vertical lines if provided
        if vlines:
            for x_val, label in vlines:
                plt.axvline(x=x_val, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=label)
        
        # Plot significant values if provided
        if sig_values:
            for x_val, _ in sig_values:
                plt.axvline(x=x_val, color='green', linestyle=':', linewidth=1, alpha=0.5)
        
        plt.title(f'Distribution of {metric} (All Data)\nGender: W')
        plt.xlabel(metric)
        plt.ylabel("Frequency")
        if vlines:
            plt.legend(loc='best')
        plt.tight_layout()
        hist_path = out_path.replace('density', 'histogram')
        final_path = add_pval_suffix_to_path(hist_path, p_value_threshold)
        plt.savefig(final_path)
        plt.close()


def plot_global_histogram(clean_vals: List[float], metric: str, out_path: str, yin_values: Optional[List[float]] = None, vlines: Optional[List[Tuple[float, str]]] = None, sig_values: Optional[List[Tuple[float, float]]] = None, p_value_threshold: Optional[float] = None) -> None:
    # Calculate the same limits as the density plot
    x_min, x_max = min(clean_vals), max(clean_vals)
    x_range = x_max - x_min
    limit_min = x_min - 0.1 * x_range
    limit_max = x_max + 0.1 * x_range

    plt.figure(figsize=(10, 6))
    plt.hist(clean_vals, bins=100, alpha=0.6, color="purple", edgecolor='black')
    
    # Add vertical lines if provided
    if vlines:
        for x_val, label in vlines:
            plt.axvline(x=x_val, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=label)
    
    # Plot significant values if provided
    if sig_values:
        for x_val, _ in sig_values:
            plt.axvline(x=x_val, color='green', linestyle=':', linewidth=1, alpha=0.5)
    
    # Explicitly set the x-axis limits
    plt.xlim(limit_min, limit_max)
    
    plt.title(f'Distribution of {metric} (All Data)\nGender: W')
    plt.xlabel(metric)
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.3)
    if vlines:
        plt.legend(loc='best')
    plt.tight_layout()
    final_path = add_pval_suffix_to_path(out_path, p_value_threshold)
    plt.savefig(final_path)
    plt.close()


def plot_split_density_cdf(
    split_dict: Dict[str, List[float]], 
    metric: str, 
    split_type: str,
    out_root: str,
    bls_values: Optional[Dict[str, float]] = None,
    yin_values: Optional[List[float]] = None,
    yin_model: Optional[str] = None,
    vlines: Optional[List[Tuple[float, str]]] = None,
    model_vlines: Optional[Dict[str, float]] = None,
    sig_values_dict: Optional[Dict[str, List[Tuple[float, float]]]] = None,
    p_value_threshold: Optional[float] = None
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
                    # kde.set_bandwidth(bw_method=0.1)
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
        # Add vertical lines to all plots if provided
        if vlines:
            for x_val, label in vlines:
                ax_hist.axvline(x=x_val, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=label)
                ax_den.axvline(x=x_val, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=label)
                ax_cdf.axvline(x=x_val, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=label)
        
        # Add model-specific vertical lines for model split
        if split_type == "model" and model_vlines:
            for model_name, x_val in model_vlines.items():
                if model_name in split_dict:
                    ax_hist.axvline(x=x_val, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label=model_name)
                    ax_den.axvline(x=x_val, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label=model_name)
                    ax_cdf.axvline(x=x_val, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label=model_name)
        
        # Add significant values if provided (just the values, no extra text)
        if sig_values_dict:
            for key, sig_vals in sig_values_dict.items():
                if key in split_dict:
                    for x_val, _ in sig_vals:
                        ax_hist.axvline(x=x_val, color='green', linestyle=':', linewidth=1, alpha=0.5)
                        ax_den.axvline(x=x_val, color='green', linestyle=':', linewidth=1, alpha=0.5)
                        ax_cdf.axvline(x=x_val, color='green', linestyle=':', linewidth=1, alpha=0.5)
        
        # Save Histogram Plot
        ax_hist.set_title(f'Comparison of {metric} Histogram by {split_type.capitalize()}\nGender: W')
        ax_hist.set_xlabel(metric)
        ax_hist.set_ylabel("Frequency")
        ax_hist.grid(True, linestyle="--", alpha=0.3, axis='y')
        ax_hist.legend(title=split_type.capitalize(), bbox_to_anchor=(1.05, 1), loc='upper left')
        fig_hist.tight_layout()
        
        hist_path = f"{out_root}/{metric}_combined_histogram_by_{split_type}.png"
        final_hist_path = add_pval_suffix_to_path(hist_path, p_value_threshold)
        print(f"Saving combined histogram plot to {final_hist_path}")
        fig_hist.savefig(final_hist_path)
        
        # Save Density Plot
        ax_den.set_title(f'Comparison of {metric} Density by {split_type.capitalize()}\nGender: W')
        ax_den.set_xlabel(metric)
        ax_den.set_ylabel("Density")
        ax_den.grid(True, linestyle="--", alpha=0.3)
        ax_den.legend(title=split_type.capitalize(), bbox_to_anchor=(1.05, 1), loc='upper left')
        fig_den.tight_layout()
        
        den_path = f"{out_root}/{metric}_combined_density_by_{split_type}.png"
        final_den_path = add_pval_suffix_to_path(den_path, p_value_threshold)
        print(f"Saving combined density plot to {final_den_path}")
        fig_den.savefig(final_den_path)
        
        # Save CDF Plot
        ax_cdf.set_title(f'Comparison of {metric} CDF by {split_type.capitalize()}\nGender: W')
        ax_cdf.set_xlabel(metric)
        ax_cdf.set_ylabel("Cumulative Probability")
        ax_cdf.grid(True, linestyle="--", alpha=0.3)
        ax_cdf.legend(title=split_type.capitalize(), bbox_to_anchor=(1.05, 1), loc='upper left')
        fig_cdf.tight_layout()
        
        cdf_path = f"{out_root}/{metric}_combined_cdf_by_{split_type}.png"
        final_cdf_path = add_pval_suffix_to_path(cdf_path, p_value_threshold)
        print(f"Saving combined CDF plot to {final_cdf_path}")
        fig_cdf.savefig(final_cdf_path)
    else:
        print(f"No {split_type}s had enough data to plot.")
    
    plt.close(fig_den)
    plt.close(fig_cdf)
    plt.close(fig_hist)


def plot_split_density_cdf_with_bundle(
    split_dict: Dict[str, List[float]], 
    metric: str, 
    split_type: str,
    out_root: str,
    job_bundle: str,
    bls_values: Optional[Dict[str, float]] = None,
    yin_values: Optional[List[float]] = None,
    yin_model: Optional[str] = None,
    vlines: Optional[List[Tuple[float, str]]] = None,
    model_vlines: Optional[Dict[str, float]] = None,
    sig_values_dict: Optional[Dict[str, List[Tuple[float, float]]]] = None,
    p_value_threshold: Optional[float] = None
) -> None:
    """Plot density, CDF, and histogram plots split by job_name within a job_bundle.
    
    Similar to plot_split_density_cdf but includes job_bundle name in filenames.
    """
    if not split_dict:
        print(f"No data to plot for {split_type} split in job_bundle {job_bundle}.")
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
                    # kde.set_bandwidth(bw_method=0.1)
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
        # Add vertical lines to all plots if provided
        if vlines:
            for x_val, label in vlines:
                ax_hist.axvline(x=x_val, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=label)
                ax_den.axvline(x=x_val, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=label)
                ax_cdf.axvline(x=x_val, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=label)
        
        # Add model-specific vertical lines for model split
        if split_type == "model" and model_vlines:
            for model_name, x_val in model_vlines.items():
                if model_name in split_dict:
                    ax_hist.axvline(x=x_val, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label=model_name)
                    ax_den.axvline(x=x_val, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label=model_name)
                    ax_cdf.axvline(x=x_val, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label=model_name)
        
        # Add significant values if provided (just the values, no extra text)
        if sig_values_dict:
            for key, sig_vals in sig_values_dict.items():
                if key in split_dict:
                    for x_val, _ in sig_vals:
                        ax_hist.axvline(x=x_val, color='green', linestyle=':', linewidth=1, alpha=0.5)
                        ax_den.axvline(x=x_val, color='green', linestyle=':', linewidth=1, alpha=0.5)
                        ax_cdf.axvline(x=x_val, color='green', linestyle=':', linewidth=1, alpha=0.5)
        
        # Save Histogram Plot (with job_bundle in filename)
        ax_hist.set_title(f'Comparison of {metric} Histogram by {split_type.capitalize()}\nJob Bundle: {job_bundle}\nGender: W')
        ax_hist.set_xlabel(metric)
        ax_hist.set_ylabel("Frequency")
        ax_hist.grid(True, linestyle="--", alpha=0.3, axis='y')
        ax_hist.legend(title=split_type.capitalize(), bbox_to_anchor=(1.05, 1), loc='upper left')
        fig_hist.tight_layout()
        
        hist_path = f"{out_root}/{metric}_combined_histogram_by_{split_type}_job_bundle_{job_bundle}.png"
        final_hist_path = add_pval_suffix_to_path(hist_path, p_value_threshold)
        print(f"Saving combined histogram plot to {final_hist_path}")
        fig_hist.savefig(final_hist_path)
        
        # Save Density Plot (with job_bundle in filename)
        ax_den.set_title(f'Comparison of {metric} Density by {split_type.capitalize()}\nJob Bundle: {job_bundle}\nGender: W')
        ax_den.set_xlabel(metric)
        ax_den.set_ylabel("Density")
        ax_den.grid(True, linestyle="--", alpha=0.3)
        ax_den.legend(title=split_type.capitalize(), bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_den.set_xlim((0.4, 0.650))
        fig_den.tight_layout()
        
        den_path = f"{out_root}/{metric}_combined_density_by_{split_type}_job_bundle_{job_bundle}.png"
        final_den_path = add_pval_suffix_to_path(den_path, p_value_threshold)
        print(f"Saving combined density plot to {final_den_path}")
        fig_den.savefig(final_den_path)
        
        # Save CDF Plot (with job_bundle in filename)
        ax_cdf.set_title(f'Comparison of {metric} CDF by {split_type.capitalize()}\nJob Bundle: {job_bundle}\nGender: W')
        ax_cdf.set_xlabel(metric)
        ax_cdf.set_ylabel("Cumulative Probability")
        ax_cdf.grid(True, linestyle="--", alpha=0.3)
        ax_cdf.legend(title=split_type.capitalize(), bbox_to_anchor=(1.05, 1), loc='upper left')
        fig_cdf.tight_layout()
        
        cdf_path = f"{out_root}/{metric}_combined_cdf_by_{split_type}_job_bundle_{job_bundle}.png"
        final_cdf_path = add_pval_suffix_to_path(cdf_path, p_value_threshold)
        print(f"Saving combined CDF plot to {final_cdf_path}")
        fig_cdf.savefig(final_cdf_path)
    else:
        print(f"No {split_type}s had enough data to plot for job_bundle {job_bundle}.")
    
    plt.close(fig_den)
    plt.close(fig_cdf)
    plt.close(fig_hist)


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run plotting for yin data format with split options.")
    parser.add_argument("--input_file", type=str, default="/nlp/scr/nmeist/EvalDims/results/yin_sampling_downsampling_analysis_original.csv", help="Path to input CSV.")
    parser.add_argument("--output_dir", type=str, default="results/figs/yin/all", help="Directory to save figures.")
    parser.add_argument("--split_by", type=str, default="all", 
                       choices=["all", "model", "job", "job_name", "name", "job_bundle_job"], 
                       help="How to split the data: 'all' (no split), 'model', 'job' (by job_bundle), 'job_name' (by job), 'name', or 'job_bundle_job' (one plot per job_bundle, split by job name).")
    parser.add_argument("--metric", type=str, default="all",
                       help="Metric to plot: 'all', 'selection_rate', 'disparate_impact_ratio', or 'selection_rate'.")
    parser.add_argument("--p_value_threshold", type=float, default=None,
                       help="If provided, only plot values where P_Value_Vector < threshold. Filename will include threshold.")

    parser.add_argument("--bw_threshold", type=float, default=None,
                       help="If provided, only plot the density graph with the provided threshold for bandwidth selection.")

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

    # 1. Load Data
    if not os.path.exists(args.input_file):
        print(f"Error: File {args.input_file} not found.")
        return

    # 1. Load the data
    #    (We use index_col=[0, 1] so it correctly identifies the M/W and ID columns first)
    df = pd.read_csv(args.input_file, index_col=[0, 1])

    # 2. Flatten the table
    #    (This moves the Index levels into the dataframe)
    df = df.reset_index()

    # 3. Rename columns - handle variable number of columns
    #    Check if we have the standard 10 columns or more (with selection_rate, P_Value_Vector, etc.)
    num_cols = len(df.columns)
    if num_cols >= 10:
        # Standard columns
        base_cols = [
            'gender', 
            'top', 
            'top_og', 
            'selection_rate', 
            'disparate_impact_ratio', 
            'job', 
            'model', 
            'rank', 
            'name_bundle', 
            'job_bundle'
        ]
        # Add additional columns if they exist
        if num_cols > 10:
            additional_cols = df.columns[10:].tolist()
            df.columns = base_cols + additional_cols
        else:
            df.columns = base_cols
    else:
        # If fewer columns, try to map what we have
        print(f"Warning: Expected at least 10 columns, found {num_cols}. Attempting to map columns...")
        if 'gender' not in df.columns and num_cols > 0:
            df.columns = list(df.columns[:num_cols])
    
    # Parse P_Value_Vector if it exists
    if 'P_Value_Vector' in df.columns:
        df['P_Value_Vector'] = df['P_Value_Vector'].apply(parse_p_values)

    # 4. Filter for 'W'
    #    (Now works easily because 'gender' is a real column)
    if args.split_by == "job_name" or args.split_by == 'job_bundle_job':
        # For job_name split, filter for gender values ending with '_W', 
        # and then group by job, model, name_bundle, job_bundle, summing selection_rate and averaging disparate_impact_ratio
        filtered_df = df[df['gender'].str.endswith("_W")].copy()
        if not filtered_df.empty:
            group_cols = ['job', 'model', 'name_bundle', 'job_bundle']
            agg_dict = {
                'selection_rate': 'sum',
                'disparate_impact_ratio': 'mean',
                'gender': 'first',   # keep first gender value in each group
                'top': 'sum',
                'top_og': 'sum',
                'rank': 'first'      # or 'mean', depending on how you want to handle ranks
            }
            # Only use columns that exist in the DataFrame
            agg_dict = {k: v for k, v in agg_dict.items() if k in filtered_df.columns}
            working_df = filtered_df.groupby(group_cols, as_index=False).agg(agg_dict)
        else:
            working_df = filtered_df  # will be empty DataFrame
    else:
        # Otherwise, filter for gender == 'W'
        working_df = df[df['gender'] == 'W'].copy()

    # 3. Define Metrics
    if args.metric == "all":
        metrics_to_plot = ["selection_rate", "disparate_impact_ratio"]
        if "selection_rate" in df.columns:
            metrics_to_plot.append("selection_rate")
    else:
        metrics_to_plot = [args.metric]
    
    # Define vertical reference lines for selection_rate
    score_diff_vlines = None
    score_diff_model_vlines = None

    if "selection_rate" in metrics_to_plot:
        score_diff_vlines = [
            (0.5, "x=0.5"),
            (0.50275, "Our Reproduced Yin Results"),
            # (0.51125, "Yin Results")
        ]
        score_diff_model_vlines = {
            # "gpt-4o-2024-11-20": 0.50275,
            # "gpt-5-nano-2025-08-07": 0.49525,
            # "mistral-small-24b": 0.53625,
            # "meta-llama-3.1-8b-instruct-turbo": 0.545,
            # "meta-llama-3.3-70b-instruct-turbo": 0.5036496350364964,
            # "mistral-7b-v0.3,1.0": 0.515
        }
    
    ensure_dirs(args.output_dir)

    print(f"Plotting metrics: {metrics_to_plot} (Split by: {args.split_by})")

    # Extract yin values for each metric (where name_bundle=='yin' and job_bundle=='yin')
    yin_values_dict = {}
    yin_model_dict = {}
    for metric in metrics_to_plot:
        if metric in working_df.columns:
            yin_subset = working_df[(working_df['name_bundle'] == 'yin') & (working_df['job_bundle'] == 'yin')]
            yin_vals = yin_subset[metric].dropna().tolist()
            yin_values_dict[metric] = clean_values(yin_vals) if yin_vals else []
            # Get the model name(s) for yin data (use the most common one if multiple)
            if not yin_subset.empty and 'model' in yin_subset.columns:
                yin_models = yin_subset['model'].value_counts()
                if len(yin_models) > 0:
                    yin_model_dict[metric] = yin_models.index[0]  # Use most common model
            if yin_values_dict[metric]:
                print(f"Found {len(yin_values_dict[metric])} yin values for {metric}: {yin_values_dict[metric]}")
                if metric in yin_model_dict:
                    print(f"Yin data belongs to model: {yin_model_dict[metric]}")
            else:
                print(f"No yin values found for {metric}")

    # 4. Loop over Metrics
    for metric in metrics_to_plot:
        if metric not in working_df.columns:
            print(f"Warning: Metric '{metric}' not found in data. Skipping.")
            continue

        # ----------------------------
        # Main plotting logic based on split_by
        # ----------------------------
        if args.split_by == "all":
            # Collect all values across all models
            print(f"Collecting all values for {metric}...")
            
            # Check if working_df is empty after filtering
            if working_df.empty:
                print(f"Warning: No data found after filtering (gender == 'W'). Skipping {metric}.")
                continue
            
            # Get the metric column and check for valid values
            metric_series = working_df[metric]
            non_null_count = metric_series.notna().sum()
            print(f"Found {non_null_count} non-null values out of {len(metric_series)} total rows for {metric}")
            
            # Filter by p-value threshold if provided
            all_collected_values = []
            sig_values = []  # List of (x_val, value) tuples for significant p-values
            
            if args.p_value_threshold is not None and 'P_Value_Vector' in working_df.columns:
                # Only include values where p-value < threshold
                for idx, row in working_df.iterrows():
                    metric_val = row[metric]
                    p_vals = row.get('P_Value_Vector', None)
                    
                    if pd.notna(metric_val):
                        # Check if p-value is significant
                        is_sig = False
                        if p_vals is not None and isinstance(p_vals, list) and len(p_vals) > 0:
                            # Use first p-value if available
                            p_val = p_vals[0]
                            if pd.notna(p_val) and p_val < args.p_value_threshold:
                                is_sig = True
                        
                        if is_sig:
                            all_collected_values.append(metric_val)
                            sig_values.append((metric_val, metric_val))
            else:
                # No p-value filtering, collect all values
                all_collected_values = metric_series.dropna().values.tolist()
            
            # Clean the values to ensure they're numeric and handle any edge cases
            clean_vals = clean_values(all_collected_values)
            clean_sig_vals = clean_values([v[0] for v in sig_values]) if sig_values else []
            clean_sig_values = [(v, v) for v in clean_sig_vals] if clean_sig_vals else None

            if clean_vals:
                yin_vals = yin_values_dict.get(metric, [])
                vlines = score_diff_vlines if metric == "selection_rate" else None

                plot_global_density(
                    clean_vals, 
                    metric, 
                    f"{args.output_dir}/{metric}_global_density_all.png",
                    yin_vals,
                    vlines=vlines,
                    sig_values=clean_sig_values,
                    p_value_threshold=args.p_value_threshold
                )
                plot_global_histogram(
                    clean_vals,
                    metric,
                    f"{args.output_dir}/{metric}_global_histogram_all.png",
                    yin_vals,
                    vlines=vlines,
                    sig_values=clean_sig_values,
                    p_value_threshold=args.p_value_threshold
                )
            else:
                print(f"Warning: No valid numeric values found for metric {metric} after cleaning.")

        elif args.split_by == "model":
            # Split by model
            print(f"Generating plots split by model for {metric}...")
            if "model" not in working_df.columns:
                print("Error: 'model' column not found in data.")
                continue
            
            unique_models = working_df["model"].unique()
            split_dict = {}
            sig_values_dict = {}  # Dict mapping model_name -> list of (x_val, value) tuples
            
            for model_name in unique_models:
                model_subset = working_df[working_df["model"] == model_name]
                model_values = []
                model_sig_values = []
                
                for idx, row in model_subset.iterrows():
                    metric_val = row[metric]
                    if pd.notna(metric_val):
                        # Check p-value if threshold is set
                        if args.p_value_threshold is not None and 'P_Value_Vector' in row:
                            p_vals = row.get('P_Value_Vector', None)
                            is_sig = False
                            if p_vals is not None and isinstance(p_vals, list) and len(p_vals) > 0:
                                p_val = p_vals[0]
                                if pd.notna(p_val) and p_val < args.p_value_threshold:
                                    is_sig = True
                            
                            if is_sig:
                                model_values.append(metric_val)
                                model_sig_values.append((metric_val, metric_val))
                        else:
                            # No p-value filtering - include all values
                            model_values.append(metric_val)
                
                if model_values:
                    split_dict[model_name] = model_values
                    if model_sig_values:
                        sig_values_dict[model_name] = model_sig_values
            
            yin_vals = yin_values_dict.get(metric, [])
            yin_model = yin_model_dict.get(metric, None)
            vlines = score_diff_vlines if metric == "selection_rate" else None
            model_vlines = score_diff_model_vlines if metric == "selection_rate" else None
            clean_sig_values_dict = {}
            for key, sig_vals in sig_values_dict.items():
                clean_vals_list = clean_values([v[0] for v in sig_vals])
                clean_sig_values_dict[key] = [(v, v) for v in clean_vals_list] if clean_vals_list else []
            plot_split_density_cdf(
                split_dict, metric, "model", args.output_dir, 
                yin_values=yin_vals, yin_model=yin_model,
                vlines=vlines, model_vlines=model_vlines,
                sig_values_dict=clean_sig_values_dict if clean_sig_values_dict else None,
                p_value_threshold=args.p_value_threshold
            )

        elif args.split_by == "job":
            # Split by job (use job_bundle if available, otherwise job)
            print(f"Generating plots split by job for {metric}...")
            job_col = "job_bundle" if "job_bundle" in working_df.columns else "job"
            if job_col not in working_df.columns:
                print(f"Error: '{job_col}' column not found in data.")
                continue
            
            unique_jobs = working_df[job_col].unique()
            split_dict = {}
            sig_values_dict = {}
            
            for job_val in unique_jobs:
                job_subset = working_df[working_df[job_col] == job_val]
                job_values = []
                job_sig_values = []
                
                for idx, row in job_subset.iterrows():
                    metric_val = row[metric]
                    if pd.notna(metric_val):
                        # Check p-value if threshold is set
                        if args.p_value_threshold is not None and 'P_Value_Vector' in row:
                            p_vals = row.get('P_Value_Vector', None)
                            is_sig = False
                            if p_vals is not None and isinstance(p_vals, list) and len(p_vals) > 0:
                                p_val = p_vals[0]
                                if pd.notna(p_val) and p_val < args.p_value_threshold:
                                    is_sig = True
                            
                            if is_sig:
                                job_values.append(metric_val)
                                job_sig_values.append((metric_val, metric_val))
                        else:
                            # No p-value filtering
                            job_values.append(metric_val)
                
                if job_values:
                    split_dict[job_val] = job_values
                    if job_sig_values:
                        sig_values_dict[job_val] = job_sig_values
            
            yin_vals = yin_values_dict.get(metric, [])
            yin_model = yin_model_dict.get(metric, None)
            vlines = score_diff_vlines if metric == "selection_rate" else None
            clean_sig_values_dict = {}
            for key, sig_vals in sig_values_dict.items():
                clean_vals_list = clean_values([v[0] for v in sig_vals])
                clean_sig_values_dict[key] = [(v, v) for v in clean_vals_list] if clean_vals_list else []
            plot_split_density_cdf(
                split_dict, metric, "job", args.output_dir, bls_values, yin_vals, yin_model,
                vlines=vlines, sig_values_dict=clean_sig_values_dict if clean_sig_values_dict else None,
                p_value_threshold=args.p_value_threshold
            )

        elif args.split_by == "job_name":
            # Split by job_name (actual job names like "cashier", "administrative assistant", etc.)
            print(f"Generating plots split by job_name for {metric}...")
            if "job" not in working_df.columns:
                print("Error: 'job' column not found in data.")
                continue
            
            # Clean job names: strip whitespace and lowercase
            unique_job_names = working_df["job"].str.strip().str.lower().unique()
            split_dict = {}
            sig_values_dict = {}

            for job_name in unique_job_names:
                job_subset = working_df[working_df["job"].str.strip().str.lower() == job_name]
                job_values = []
                job_sig_values = []
                
                for idx, row in job_subset.iterrows():
                    metric_val = row[metric]
                    if pd.notna(metric_val):
                        # Check p-value if threshold is set
                        if args.p_value_threshold is not None and 'P_Value_Vector' in row:
                            p_vals = row.get('P_Value_Vector', None)
                            is_sig = False
                            if p_vals is not None and isinstance(p_vals, list) and len(p_vals) > 0:
                                p_val = p_vals[0]
                                if pd.notna(p_val) and p_val < args.p_value_threshold:
                                    is_sig = True
                            
                            if is_sig:
                                job_values.append(metric_val)
                                job_sig_values.append((metric_val, metric_val))
                        else:
                            # No p-value filtering - include all values
                            job_values.append(metric_val)
                
                if job_values:
                    split_dict[job_name] = job_values
                    if job_sig_values:
                        sig_values_dict[job_name] = job_sig_values
            
            yin_vals = yin_values_dict.get(metric, [])
            yin_model = yin_model_dict.get(metric, None)
            vlines = score_diff_vlines if metric == "selection_rate" else None
            clean_sig_values_dict = {}
            for key, sig_vals in sig_values_dict.items():
                clean_vals_list = clean_values([v[0] for v in sig_vals])
                clean_sig_values_dict[key] = [(v, v) for v in clean_vals_list] if clean_vals_list else []
            plot_split_density_cdf(
                split_dict, metric, "job_name", args.output_dir, yin_values=yin_vals, yin_model=yin_model,
                vlines=vlines, sig_values_dict=clean_sig_values_dict if clean_sig_values_dict else None,
                p_value_threshold=args.p_value_threshold
            )

        elif args.split_by == "name":
            # Split by name
            print(f"Generating plots split by name for {metric}...")
            if "name_bundle" not in working_df.columns:
                print("Error: 'name_bundle' column not found in data.")
                continue
            
            unique_names = working_df["name_bundle"].unique()
            split_dict = {}
            sig_values_dict = {}
            
            for name_val in unique_names:
                name_subset = working_df[working_df["name_bundle"] == name_val]
                name_values = []
                name_sig_values = []
                
                for idx, row in name_subset.iterrows():
                    metric_val = row[metric]
                    if pd.notna(metric_val):
                        # Check p-value if threshold is set
                        if args.p_value_threshold is not None and 'P_Value_Vector' in row:
                            p_vals = row.get('P_Value_Vector', None)
                            is_sig = False
                            if p_vals is not None and isinstance(p_vals, list) and len(p_vals) > 0:
                                p_val = p_vals[0]
                                if pd.notna(p_val) and p_val < args.p_value_threshold:
                                    is_sig = True
                            
                            if is_sig:
                                name_values.append(metric_val)
                                name_sig_values.append((metric_val, metric_val))
                        else:
                            # No p-value filtering - include all values
                            name_values.append(metric_val)
                
                if name_values:
                    split_dict[name_val] = name_values
                    if name_sig_values:
                        sig_values_dict[name_val] = name_sig_values
            
            yin_vals = yin_values_dict.get(metric, [])
            yin_model = yin_model_dict.get(metric, None)
            vlines = score_diff_vlines if metric == "selection_rate" else None
            clean_sig_values_dict = {}
            for key, sig_vals in sig_values_dict.items():
                clean_vals_list = clean_values([v[0] for v in sig_vals])
                clean_sig_values_dict[key] = [(v, v) for v in clean_vals_list] if clean_vals_list else []
            plot_split_density_cdf(
                split_dict, metric, "name", args.output_dir, yin_values=yin_vals, yin_model=yin_model,
                vlines=vlines, sig_values_dict=clean_sig_values_dict if clean_sig_values_dict else None,
                p_value_threshold=args.p_value_threshold
            )

        elif args.split_by == "job_bundle_job":
            # Split by job_bundle, then by job name within each bundle
            # Creates one plot per job_bundle, with each plot showing different job names
            print(f"Generating plots split by job_bundle and job_name for {metric}...")
            if "job_bundle" not in working_df.columns:
                print("Error: 'job_bundle' column not found in data.")
                continue
            if "job" not in working_df.columns:
                print("Error: 'job' column not found in data.")
                continue
            
            unique_job_bundles = working_df["job_bundle"].unique()
            print(f"Found {len(unique_job_bundles)} unique job bundles: {unique_job_bundles}")
            
            for job_bundle in unique_job_bundles:
                print(f"\nProcessing job_bundle: '{job_bundle}'...")
                # Filter for this job bundle
                bundle_subset = working_df[working_df["job_bundle"] == job_bundle].copy()
                
                if bundle_subset.empty:
                    print(f"No data found for job_bundle '{job_bundle}'. Skipping.")
                    continue
                
                print(f"Found {len(bundle_subset)} rows for job_bundle '{job_bundle}'")
                
                # Get unique job names within this bundle (clean them)
                unique_job_names = bundle_subset["job"].str.strip().str.lower().unique()
                print(f"Found {len(unique_job_names)} unique job names in bundle '{job_bundle}': {unique_job_names}")
                split_dict = {}
                sig_values_dict = {}
                
                for job_name in unique_job_names:
                    job_subset = bundle_subset[bundle_subset["job"].str.strip().str.lower() == job_name]
                    job_values = []
                    job_sig_values = []
                    
                    for idx, row in job_subset.iterrows():
                        metric_val = row[metric]
                        if pd.notna(metric_val):
                            # Check p-value if threshold is set
                            if args.p_value_threshold is not None and 'P_Value_Vector' in row:
                                p_vals = row.get('P_Value_Vector', None)
                                is_sig = False
                                if p_vals is not None and isinstance(p_vals, list) and len(p_vals) > 0:
                                    p_val = p_vals[0]
                                    if pd.notna(p_val) and p_val < args.p_value_threshold:
                                        is_sig = True
                                
                                if is_sig:
                                    job_values.append(metric_val)
                                    job_sig_values.append((metric_val, metric_val))
                            else:
                                # No p-value filtering - include all values
                                job_values.append(metric_val)
                    
                    if job_values:
                        split_dict[job_name] = job_values
                        if job_sig_values:
                            sig_values_dict[job_name] = job_sig_values
                
                if split_dict:
                    print(f"Creating plots for job_bundle '{job_bundle}' with {len(split_dict)} job names...")
                    # Create output directory for this job bundle
                    bundle_output_dir = os.path.join(args.output_dir, f"job_bundle_{job_bundle}")
                    ensure_dirs(bundle_output_dir)
                    print(f"Output directory: {bundle_output_dir}")
                    
                    yin_vals = yin_values_dict.get(metric, [])
                    yin_model = yin_model_dict.get(metric, None)
                    vlines = score_diff_vlines if metric == "selection_rate" else None
                    clean_sig_values_dict = {}
                    for key, sig_vals in sig_values_dict.items():
                        clean_vals_list = clean_values([v[0] for v in sig_vals])
                        clean_sig_values_dict[key] = [(v, v) for v in clean_vals_list] if clean_vals_list else []
                    
                    # Use a modified version of plot_split_density_cdf that includes job_bundle in filename
                    plot_split_density_cdf_with_bundle(
                        split_dict, metric, "job_name", bundle_output_dir, 
                        job_bundle=job_bundle,
                        yin_values=yin_vals, yin_model=yin_model,
                        vlines=vlines, sig_values_dict=clean_sig_values_dict if clean_sig_values_dict else None,
                        p_value_threshold=args.p_value_threshold
                    )
                    print(f"Completed plots for job_bundle '{job_bundle}'.\n")
                else:
                    print(f"No data to plot for job_bundle '{job_bundle}'.")

if __name__ == "__main__":
    main()



'''

python plot_yin_all.py --split_by model; python plot_yin_all.py --split_by name; python plot_yin_all.py --split_by job

python plot_yin_all.py --split_by job_bundle_job


'''