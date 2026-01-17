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
# BLS Values Mapping
# ----------------------------
bls_values = {
    'armstrong': 57.564,
    'rozado': 47.47,
    'wen': 57.88,
    'karvonen': 67.98,
    'zollo': 48.77,
    'yin': 46.5
    # Note: 'wang' is excluded per instructions
}

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


def parse_str_list(x: Any) -> Optional[List[str]]:
    val = _safe_literal_eval(x)
    if val is None:
        return None
    if isinstance(val, list):
        return [str(v) for v in val]
    return None


def parse_p_values(x: Any) -> Optional[List[float]]:
    if pd.isna(x):
        return None
    if isinstance(x, list):
        try:
            return [float(v) for v in x]
        except Exception:
            return None
    val = _safe_literal_eval(x)
    if val is None:
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
    valid = [v for v in values if not np.isnan(v)]
    if not valid:
        return (0.0, 1.0)
    ymin, ymax = min(valid), max(valid)
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

    if args.model == "all":
        working_df = df.copy()
    else:
        working_df = df[df["Model"] == args.model].copy()

    # Grid definition
    row_labels = ["armstrong", "rozado", "wen", "wang", "gaeb", "lippens", "seshadri", "karvonen", "zollo", "yin"]
    col_labels = ["armstrong", "rozado", "wen", "wang", "karvonen", "zollo", "yin"]

    # Output dirs
    out_root = f"results/figs/{args.exp_framework}/{args.model}"
    ensure_dirs(f"results/figs/{args.exp_framework}", out_root)

    # Build lookup table
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

    # Metric to plot
    metrics_to_plot = ["regression_coefficients"]

    # Color Mapping
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
        # NEW: Scatter Plot (Coeffs vs BLS Values)
        # ----------------------------
        if args.model == 'all':
            print(f"Generating scatter plot: {metric} vs BLS Values...")
            
            plt.figure(figsize=(10, 8))
            
            # Iterate only through jobs defined in bls_values (this implicitly filters 'wang' if it's not in the dict)
            # However, we must ensure we iterate through the dataset logic correctly.
            unique_jobs = df["Jobs"].unique()
            
            # Helper to track if we added a label to the legend already
            plotted_labels = set()

            for job_val in unique_jobs:
                # 1. Skip if job is 'wang' or not in our BLS dictionary
                if job_val == 'wang' or job_val not in bls_values:
                    continue

                # 2. Get the X-axis value
                x_val = bls_values[job_val]
                
                # 3. Filter rows: Job == job_val AND Names != 'wang'
                job_subset = df[
                    (df["Jobs"] == job_val) & 
                    (df["Names"] != "wang") & 
                    (df["Metric"] == metric)
                ]

                # 4. Extract values
                y_values = []
                for _, row in job_subset.iterrows():
                    val, _, _ = get_woman_stats(row)
                    y_values.append(val)
                
                # 5. Filter: Non-zero AND Outliers (-5 to 5)
                clean_y = [y for y in y_values if y != 0 and -5 <= y <= 5]
                
                if not clean_y:
                    continue

                # 6. Plot Scatter
                color = label_to_color.get(job_val, "black")
                
                # Create X array of same length
                x_data = [x_val] * len(clean_y)
                
                # Add jitter to X slightly to see overlapping points better? 
                # (Optional: remove `+ np.random...` if you want a straight vertical line)
                # For now, keeping it straight vertical as per typical "values vs X" plots, 
                # or adding very slight jitter if preferred. Let's do straight line for precision.
                
                plt.scatter(
                    x_data, 
                    clean_y, 
                    color=color, 
                    alpha=0.7, 
                    s=50, 
                    label=job_val if job_val not in plotted_labels else ""
                )
                plotted_labels.add(job_val)

            plt.title('Regression Coefficients vs BLS Values')
            plt.xlabel('BLS Value')
            plt.ylabel('Regression Coefficient')
            plt.grid(True, linestyle="--", alpha=0.3)
            plt.legend(title="Job", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            plot_path = f"{out_root}/{metric}_scatter_vs_bls.png"
            print(f"Saving scatter plot to {plot_path}")
            plt.savefig(plot_path)
            plt.close()

        # ----------------------------
        # Grid Generation (Heatmap/Bar Chart logic) - With Wang Exclusion in visualization if desired
        # Note: The user asked to exclude Wang from the *Scatter Plot*. 
        # I will leave the Grid logic as is (filtering outliers only), 
        # unless you want Wang excluded from the heatmap too. 
        # Below preserves original grid logic but adds outlier filtering.
        # ----------------------------
        values_mat = np.zeros((len(row_labels), len(col_labels)), dtype=float)
        errors_mat: List[List[Optional[float]]] = [[None] * len(col_labels) for _ in row_labels]
        pvals_mat: List[List[Optional[float]]] = [[None] * len(col_labels) for _ in row_labels]

        for i, name in enumerate(row_labels):
            for j, job in enumerate(col_labels):
                row = get_cell_row(name, job, metric)
                v, e, p = get_woman_stats(row)
                
                # Filter Outliers for Heatmap
                if v < -5 or v > 5:
                    values_mat[i, j] = np.nan
                else:
                    values_mat[i, j] = v
                
                errors_mat[i][j] = e
                pvals_mat[i][j] = p

        values = values_mat.reshape(-1).tolist()
        errors = [errors_mat[i][j] for i in range(len(row_labels)) for j in range(len(col_labels))]
        p_values = [pvals_mat[i][j] for i in range(len(row_labels)) for j in range(len(col_labels))]

        labels = [label_for_cell(n, j) for n in row_labels for j in col_labels]
        colors = [label_to_color.get(job, "tab:blue") for _name in row_labels for job in col_labels]

        nonzero_values = [v for v in values if not np.isnan(v) and v != 0]
        ylims = pad_ylim(values)

        # Histogram
        plt.figure(figsize=(10, 6))
        plt.hist(nonzero_values, bins=30, alpha=0.5, color="blue")
        plt.title(f'Histogram of {metric} (Grid Subset, Outliers Excluded)')
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
                if np.isnan(v) or v == 0:
                    yerr.append(0)
                else:
                    yerr.append(avg if e is None else e)

        plt.figure(figsize=(40, 6))
        plot_values = [0 if np.isnan(v) else v for v in values]
        bars = plt.bar(labels, plot_values, yerr=yerr, capsize=4, align="center", color=colors)
        
        legend_handles = [mpatches.Patch(color=c, label=lab) for lab, c in label_to_color.items()]
        plt.legend(handles=legend_handles, title="Job Group")

        alpha = 0.05 / (len(row_labels) * len(col_labels))
        for bar, p, val in zip(bars, p_values, values):
            if not np.isnan(val) and p is not None and p < alpha:
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
        plt.savefig(f"{out_root}/{metric}_sampling_{args.sampling}_analysis_{args.analysis}_all.png")
        plt.close()


if __name__ == "__main__":
    main()