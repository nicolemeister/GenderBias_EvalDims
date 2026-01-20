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
    Parse into list[float]. If it’s a scalar, return [scalar].
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


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    # BLS values for the legend
    bls_values = {
        'armstrong': 57.564,
        'rozado': 47.47,
        'wen': 57.88,
        'wang': 28.71,
        'karvonen': 67.98,
        'zollo': 48.77,
        'yin': 46.5
    }

    parser = argparse.ArgumentParser(description="Run perturbation experiments analysis.")
    parser.add_argument("--sampling", type=str, default="downsampling", help="Sampling method to use.")
    parser.add_argument("--analysis", type=str, default="original", help="Analysis to perform.")
    parser.add_argument("--exp_framework", type=str, default="armstrong", help="Experiment framework to use.")
    parser.add_argument("--model", type=str, default="all", help="Model to use.")
    args = parser.parse_args()

    in_path = f"results/{args.exp_framework}_sampling_{args.sampling}_analysis_{args.analysis}.csv"
    df = pd.read_csv(in_path)

    # Parse columns
    df["Result_Vector"] = df["Result_Vector"].apply(parse_float_list)
    df["Std_Error_Vector"] = df["Std_Error_Vector"].apply(parse_float_list)
    df["Sensitive_Attribute_Vector"] = df["Sensitive_Attribute_Vector"].apply(parse_str_list)
    df["P_Value_Vector"] = df["P_Value_Vector"].apply(parse_p_values)

    # NOTE: We do NOT filter df by args.model here yet if we want to loop over all models later.
    if args.model == "all":
        working_df = df.copy()
    else:
        working_df = df[df["Model"] == args.model].copy()

    # Your fixed grid definition (10×7)
    row_labels = ["armstrong", "rozado", "wen", "wang", "gaeb", "lippens", "seshadri", "karvonen", "zollo", "yin"]
    col_labels = ["armstrong", "rozado", "wen", "wang", "karvonen", "zollo", "yin"]

    # Output dirs
    out_root = f"results/figs/{args.exp_framework}/{args.model}"
    ensure_dirs(f"results/figs/{args.exp_framework}", out_root)

    # Build a fast lookup table for the grid logic: (Names, Jobs, Metric) -> row (Series)
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
        """
        Returns (value_for_woman, stderr_for_woman, p_value_scalar_or_none)
        """
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

    # Metric(s) you want to analyze
    metrics_to_plot = ["regression_coefficients"]

    # Colors for job groups (by column label)
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

    for metric in metrics_to_plot:

        # ----------------------------
        # GLOBAL DENSITY & CDF COMPARISON
        # ----------------------------
        if args.model == 'all':
            print(f"Generating combined density and CDF plots for {metric} across all unique Jobs...")
            
            # 1. Identify all unique Jobs
            unique_jobs = df["Jobs"].unique()
            
            # Initialize separate figures for Density and CDF
            fig_den, ax_den = plt.subplots(figsize=(12, 7))
            fig_cdf, ax_cdf = plt.subplots(figsize=(12, 7))
            
            # Use tab20 colormap to handle potentially many unique jobs distinctively
            colors_list = plt.cm.tab20(np.linspace(0, 1, len(unique_jobs)))
            
            has_plotted_anything = False
            
            # 2. Iterate through each Job
            for idx, job_val in enumerate(unique_jobs):
                # Filter rows for this specific job and metric
                job_subset = df[(df["Jobs"] == job_val) & (df["Metric"] == metric)]
                
                job_values = []
                # Iterate over every row found for this job
                for _, row in job_subset.iterrows():
                    val, _, _ = get_woman_stats(row)
                    job_values.append(val)
                
                # Filter out zeroes and outliers
                nonzero_job_vals = [v for v in job_values if v != 0]
                clean_values = [v for v in nonzero_job_vals if v != 0 and -5 <= v <= 5]
                nonzero_job_vals = clean_values.copy()

                if len(nonzero_job_vals) > 1 and np.std(nonzero_job_vals) > 1e-9:
                    
                    # Print Mean
                    mean_val = np.mean(nonzero_job_vals)
                    print(f"Job: {job_val:<15} | Mean {metric}: {mean_val:.4f}")

                    # Determine Label
                    job_str = str(job_val)
                    if job_str in bls_values:
                        label_text = f"{job_str} (BLS: {bls_values[job_str]})"
                    else:
                        label_text = job_str

                    # --- Plot Density ---
                    try:
                        kde = gaussian_kde(nonzero_job_vals)
                        x_min, x_max = min(nonzero_job_vals), max(nonzero_job_vals)
                        x_range = x_max - x_min
                        x_grid = np.linspace(x_min - 0.2 * x_range, x_max + 0.2 * x_range, 200)

                        ax_den.plot(x_grid, kde(x_grid), 
                                    color=colors_list[idx], 
                                    lw=2, 
                                    alpha=0.8, 
                                    label=label_text)
                    except np.linalg.LinAlgError:
                        print(f"Skipping density for {job_val}: Singular matrix (no variance).")

                    # --- Plot CDF ---
                    sorted_data = np.sort(nonzero_job_vals)
                    # Y-axis: probability from 1/N to 1
                    yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                    
                    ax_cdf.step(sorted_data, yvals, 
                                where='post', 
                                color=colors_list[idx], 
                                lw=2, 
                                alpha=0.8, 
                                label=label_text)
                    
                    has_plotted_anything = True
                else:
                    print(f"Skipping {job_val}: Not enough data points or variance.")

            if has_plotted_anything:
                # --- Save Density Plot ---
                ax_den.set_title(f'Comparison of {metric} Density by Job')
                ax_den.set_xlabel(metric)
                ax_den.set_ylabel("Density")
                ax_den.grid(True, linestyle="--", alpha=0.3)
                ax_den.legend(title="Job", bbox_to_anchor=(1.05, 1), loc='upper left')
                fig_den.tight_layout()
                
                den_path = f"{out_root}/{metric}_combined_density_by_job.png"
                print(f"Saving combined density plot to {den_path}")
                fig_den.savefig(den_path)
                
                # --- Save CDF Plot ---
                ax_cdf.set_title(f'Comparison of {metric} CDF by Job')
                ax_cdf.set_xlabel(metric)
                ax_cdf.set_ylabel("Cumulative Probability")
                ax_cdf.grid(True, linestyle="--", alpha=0.3)
                ax_cdf.legend(title="Job", bbox_to_anchor=(1.05, 1), loc='upper left')
                fig_cdf.tight_layout()
                
                cdf_path = f"{out_root}/{metric}_combined_cdf_by_job.png"
                print(f"Saving combined CDF plot to {cdf_path}")
                fig_cdf.savefig(cdf_path)

            else:
                print("No jobs had enough data to plot.")
            
            plt.close(fig_den)
            plt.close(fig_cdf)

        # ----------------------------
        # Grid Generation (Heatmap/Bar Chart logic) - Unchanged
        # ----------------------------
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

        # ----------------------------
        # Grid-Specific Histogram (Subset)
        # ----------------------------
        plt.figure(figsize=(10, 6))
        plt.hist(nonzero_values, bins=30, alpha=0.5, color="blue")
        plt.title(f'Histogram of {metric} for "Woman" (Grid Subset 10x7)')
        plt.xlabel(metric)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"{out_root}/{metric}_sampling_{args.sampling}_analysis_{args.analysis}_grid_histogram_nonzero.png")
        plt.close()

        # ----------------------------
        # Heatmap
        # ----------------------------
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

        # ----------------------------
        # Bar Chart
        # ----------------------------
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