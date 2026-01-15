
import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("results/perturbation_results.csv")

# ---- Helpers ----
def parse_list(s):
    """Turn strings like "[np.float64(0.1), ...]" into real Python lists of floats."""
    if pd.isna(s):
        return None
    return ast.literal_eval(s.replace("np.float64", ""))

df["Result_Vector"] = df["Result_Vector"].apply(parse_list)
df["Std_Error_Vector"] = df["Std_Error_Vector"].apply(parse_list)
df["Sensitive_Attribute_Vector"] = df["Sensitive_Attribute_Vector"].apply(ast.literal_eval)

# ---- Select the two conditions ----
cond_gpt4o_10_names_armstrong_jobs_armstrong = (
    (df["Author"]=="armstrong") & (df["Names"]=="armstrong") & (df["Jobs"]=="armstrong") &
    (df["Num_Trials"]==10) & (df["Model"]=="gpt-4o-2024-11-20") & (df["Temperature"]==1.0)
)

cond_gpt4o_1_names_armstrong_jobs_armstrong = (
    (df["Author"]=="armstrong") & (df["Names"]=="armstrong") & (df["Jobs"]=="armstrong") &
    (df["Num_Trials"]==1) & (df["Model"]=="gpt-4o-2024-11-20") & (df["Temperature"]==1.0)
)

cond_gpt4o_1_names_armstrong_jobs_rozado = (
    (df["Author"]=="armstrong") & (df["Names"]=="armstrong") & (df["Jobs"]=="rozado") &
    (df["Num_Trials"]==1) & (df["Model"]=="gpt-4o-2024-11-20") & (df["Temperature"]==1.0)
)

# cond_gpt4o_1_names_armstrong_jobs_wen = (
#     (df["Author"]=="armstrong") & (df["Names"]=="rozado") & (df["Jobs"]=="wen") &
#     (df["Num_Trials"]==1) & (df["Model"]=="gpt-4o-2024-11-20") & (df["Temperature"]==1.0)
# )

cond_gpt4o_1_names_armstrong_jobs_wang = (
    (df["Author"]=="armstrong") & (df["Names"]=="armstrong") & (df["Jobs"]=="wang") &
    (df["Num_Trials"]==1) & (df["Model"]=="gpt-4o-2024-11-20") & (df["Temperature"]==1.0)
)



cond_gpt4o_1_names_wang_jobs_armstrong = (
    (df["Author"]=="armstrong") & (df["Names"]=="wang") & (df["Jobs"]=="armstrong") &
    (df["Num_Trials"]==1) & (df["Model"]=="gpt-4o-2024-11-20") & (df["Temperature"]==1.0)
)


cond_gpt4o_1_names_wang_jobs_rozado = (
    (df["Author"]=="armstrong") & (df["Names"]=="wang") & (df["Jobs"]=="rozado") &
    (df["Num_Trials"]==1) & (df["Model"]=="gpt-4o-2024-11-20") & (df["Temperature"]==1.0)
)


cond_gpt4o_1_names_wang_jobs_wen = (
    (df["Author"]=="armstrong") & (df["Names"]=="wang") & (df["Jobs"]=="wen") &
    (df["Num_Trials"]==1) & (df["Model"]=="gpt-4o-2024-11-20") & (df["Temperature"]==1.0)
)


cond_gpt4o_1_names_wang_jobs_wang = (
    (df["Author"]=="armstrong") & (df["Names"]=="wang") & (df["Jobs"]=="wang") &
    (df["Num_Trials"]==1) & (df["Model"]=="gpt-4o-2024-11-20") & (df["Temperature"]==1.0)
)




# df_gpt4o_10_names_armstrong_jobs_armstrong = df[cond_gpt4o_10_names_armstrong_jobs_armstrong]
# df_gpt4o_1_names_armstrong_jobs_armstrong = df[cond_gpt4o_1_names_armstrong_jobs_armstrong]
# df_gpt4o_1_names_armstrong_jobs_rozado = df[cond_gpt4o_1_names_armstrong_jobs_rozado]
# df_gpt4o_1_names_armstrong_jobs_wang = df[cond_gpt4o_1_names_armstrong_jobs_wang]






df_gpt4o_1_names_wang_jobs_armstrong = df[cond_gpt4o_1_names_wang_jobs_armstrong]
df_gpt4o_1_names_wang_jobs_rozado = df[cond_gpt4o_1_names_wang_jobs_rozado]
df_gpt4o_1_names_wang_jobs_wen = df[cond_gpt4o_1_names_wang_jobs_wen]
df_gpt4o_1_names_wang_jobs_wang = df[cond_gpt4o_1_names_wang_jobs_wang]


# We’ll plot one bar chart per metric; y = metric value for "Woman", x = model (gpt4o vs gpt3.5)
metrics = sorted(set(df_gpt4o_1_names_wang_jobs_wang["Metric"]).intersection(df_gpt4o_1_names_wang_jobs_armstrong["Metric"]))

# Index of "Woman" in the sensitive attribute vector
def get_woman_index(row):
    groups = row["Sensitive_Attribute_Vector"]
    try:
        return groups.index("Woman")
    except ValueError:
        # Fallback: user said "Woman is the first value", so default to 0 if missing
        return 0
    

# condition_pairs = {"Model": (df_gpt35, df_gpt4o), "Temp": (df_temp03, df_temp10), "Names": (df_names_armstrong, df_names_rozado, df_names_gaeb), "Jobs": (df_jobs_armstrong, df_jobs_rozado)}
# condition_pair_names = {"Model": ["gpt3.5", "gpt-4o-2024-11-20"], "Temp": ["Temp=0.3", "Temp=1.0"], "Names": ["Names=armstrong", "Names=rozado", "Names=gaeb"], "Jobs": ["Jobs=armstrong", "Jobs=rozado"]}
# condition_pair_ylims = {"regression_coefficients": (-0.5, 1.5), "wasserstein_distance": (0, 1.0), "score_difference": (-0.001, 0.005)}

condition_pairs = { "Wang_Name": (df_gpt4o_1_names_wang_jobs_armstrong, df_gpt4o_1_names_wang_jobs_rozado, df_gpt4o_1_names_wang_jobs_wen, df_gpt4o_1_names_wang_jobs_wang)}
condition_pair_names = {"Wang_Name": ['armstrong_1', "rozado_1", "wang_1", 'wen_1']}
condition_pair_ylims = {"regression_coefficients": (-1.5, 2), "ttest": (0, 10), "wasserstein_distance": (0, 1.0), "score_difference": (-0.001, 0.005)}


for metric in metrics:
    for condition_pair_key in condition_pairs.keys():
        condition_pair = condition_pairs[condition_pair_key]
        df1, df2, df3, df4 = condition_pair
        r1 = df1[df1["Metric"]==metric].iloc[0]
        r2 = df2[df2["Metric"]==metric].iloc[0]
        r3 = df3[df3["Metric"]==metric].iloc[0]
        try: r4 = df4[df4["Metric"]==metric].iloc[0]
        except: breakpoint()
        # r5 = df5[df5["Metric"]==metric].iloc[0]
        idx = get_woman_index(r2)


        # Pull single values for "Woman"
        try: v1 = float(r1["Result_Vector"][idx])
        except: breakpoint()
        e1 = float(r1["Std_Error_Vector"][idx]) if r1["Std_Error_Vector"] is not None else None

        v2 = float(r2["Result_Vector"][idx])
        e2 = float(r2["Std_Error_Vector"][idx]) if r2["Std_Error_Vector"] is not None else None
        
        v3 = float(r3["Result_Vector"][idx])
        e3 = float(r3["Std_Error_Vector"][idx]) if r3["Std_Error_Vector"] is not None else None

        v4 = float(r4["Result_Vector"][idx])
        e4 = float(r4["Std_Error_Vector"][idx]) if r4["Std_Error_Vector"] is not None else None
        
        if metric == 'regression_coefficients' or metric == 'ttest':
            # get the p_value vector P_Value_Vector
            p1 = float(r1["P_Value_Vector"][2:-2])
            p2 = float(r2["P_Value_Vector"][2:-2])
            p3 = float(r3["P_Value_Vector"][2:-2])
            p4 = float(r4["P_Value_Vector"][2:-2])
            p_values = [p1, p2, p3, p4]
            
        models = condition_pair_names[condition_pair_key]
        values = [v1, v2, v3, v4]
        errors = [e1, e2, e3, e4]

        # Build error list only if at least one error exists; Matplotlib expects None or arraylike
        yerr = None if all(err is None for err in errors) else errors

        # ---- Plot: one chart per metric ----
        plt.figure(figsize=(6,4))
        bars = plt.bar(models, values, yerr=yerr, capsize=4, align='center')
        # Add a star if p-value is below 0.05
        if metric == 'regression_coefficients' or metric == 'ttest':
            for i, (bar, pval) in enumerate(zip(bars, p_values)):
                if pval < 0.05:
                    # get bar height for annotation
                    height = bar.get_height()
                    plt.annotate('*',
                                 xy=(bar.get_x() + bar.get_width() / 2, height),
                                 xytext=(0, 6),  # 6 points vertical offset
                                 textcoords="offset points",
                                 ha='center', va='bottom', fontsize=18, color='red')
        plt.ylabel(metric)
        if metric in condition_pair_ylims.keys():
            plt.ylim(condition_pair_ylims[metric])
        else:
            plt.ylim(0.5, 1.2)
        plt.title(f'{metric} for "Woman"')
        # Optional: rotate x labels if they wrap
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f'results/figs/module_swapping/{metric}_{condition_pair_key}_Woman.png')
        plt.close()