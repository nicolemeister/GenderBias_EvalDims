import pandas as pd
import io

# Example data loading (replace this with reading your actual file)
df = pd.read_csv('/nlp/scr/nmeist/EvalDims/results/yin_sampling_downsampling_analysis_original_exact_matches.csv')

# 1. Filter the dataset where the ratio is > 50%
filtered_df = df[(df['number_of_instances'] / df['total_instances']) > 0.5].copy()

# Count of occurrences
total_over_50 = len(filtered_df)

# 2. Get the breakdown of values for specific columns
model_breakdown = filtered_df['model'].value_counts()
name_bundle_breakdown = filtered_df['name_bundle'].value_counts()
job_bundle_breakdown = filtered_df['job_bundle'].value_counts()

# Print results
print(f"Number of times ratio > 50%: {total_over_50} / {len(df)}" )
print("\n--- Model Breakdown ---")
print(model_breakdown)
print("\n--- Name Bundle Breakdown ---")
print(name_bundle_breakdown)
print("\n--- Job Bundle Breakdown ---")
print(job_bundle_breakdown)