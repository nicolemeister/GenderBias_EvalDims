import pandas as pd
import numpy as np

def find_consistent_drivers(df):
    """
    Analyzes selection rate flips across the 0.5 threshold.
    """
    
    def check_flip(sr1, sr2):
        # Returns True if one is > 0.5 and the other is < 0.5
        return (sr1 - 0.5) * (sr2 - 0.5) < 0

    # --- 1. NAME BUNDLE FLIPS (Consistent across ALL models for a fixed Job) ---
    print("--- 1. Name Bundle Flips (Across all models per Job) ---")
    job_bundles = df['job_bundle'].unique()
    name_bundles = df['name_bundle'].unique()
    models = df['model'].unique()
    
    for job in job_bundles:
        sub_df = df[df['job_bundle'] == job]
        # Compare pairs of names
        for i, n1 in enumerate(name_bundles):
            for n2 in name_bundles[i+1:]:
                # Check if this name change flips the result for every model and gender
                data1 = sub_df[sub_df['name_bundle'] == n1].set_index(['model', 'gender'])['selection_rate']
                data2 = sub_df[sub_df['name_bundle'] == n2].set_index(['model', 'gender'])['selection_rate']
                
                common_idx = data1.index.intersection(data2.index)
                if len(common_idx) == 0: continue
                
                flips = check_flip(data1.loc[common_idx], data2.loc[common_idx])
                # If it flips for almost every combination of (Model, Gender) available
                # (allow at most one non-flip). Require at least one true for very small samples.
                min_required = max(1, len(flips) - 1)
                if (flips.sum() if hasattr(flips, 'sum') else int(flips)) >= min_required and len(common_idx) >= (len(models) * 2):
                    print(f"Consistent Flip found for Job: {job}")
                    print(f"  Names: {n1} vs {n2} (Flipped in all {len(common_idx)} contexts)")
                    print("-" * 30)

    # --- 2. JOB BUNDLE FLIPS (Consistent across ALL models for a fixed Name) ---
    print("\n--- 2. Job Bundle Flips (Across all models per Name) ---")
    for name in name_bundles:
        sub_df = df[df['name_bundle'] == name]
        for i, j1 in enumerate(job_bundles):
            for j2 in job_bundles[i+1:]:
                data1 = sub_df[sub_df['job_bundle'] == j1].set_index(['model', 'gender'])['selection_rate']
                data2 = sub_df[sub_df['job_bundle'] == j2].set_index(['model', 'gender'])['selection_rate']
                
                common_idx = data1.index.intersection(data2.index)
                if len(common_idx) == 0: continue
                
                flips = check_flip(data1.loc[common_idx], data2.loc[common_idx])
                min_required = max(1, len(flips) - 1)
                if (flips.sum() if hasattr(flips, 'sum') else int(flips)) >= min_required:
                    print(f"Consistent Flip found for Name: {name}")
                    print(f"  Jobs: {j1} vs {j2}")
                    print("-" * 30)

    # --- 3. MODEL FLIPS (Consistent across ALL Name/Job combinations) ---
    print("\n--- 3. Model Flips (Across all Name/Job context) ---")
    for i, m1 in enumerate(models):
        for m2 in models[i+1:]:
            # Compare Model 1 vs Model 2
            data1 = df[df['model'] == m1].set_index(['name_bundle', 'job_bundle', 'gender'])['selection_rate']
            data2 = df[df['model'] == m2].set_index(['name_bundle', 'job_bundle', 'gender'])['selection_rate']
            
            common_idx = data1.index.intersection(data2.index)
            if len(common_idx) == 0: continue
            
            flips = check_flip(data1.loc[common_idx], data2.loc[common_idx])
            
            # Check consistency (Percentage of contexts where they differ in preference)
            consistency = flips.mean()
            if consistency > 0.9: # 90% consistency or higher
                print(f"High Consistency Flip between Models: {m1} vs {m2}")
                print(f"  Consistency: {consistency:.1%} across {len(common_idx)} contexts")
                print("-" * 30)

df = pd.read_csv('/nlp/scr/nmeist/EvalDims/results/yin_sampling_downsampling_analysis_original.csv')

df = df[df['gender'] == 'W']
breakpoint()

find_consistent_drivers(df)