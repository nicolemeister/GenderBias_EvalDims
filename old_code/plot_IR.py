
import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser(description='Run perturbation experiments.')
parser.add_argument('--sampling', type=str, default='downsampling', help='Sampling method to use.') 
parser.add_argument('--analysis', type=str, default='original', help='Analysis to perform.') # original or modified metric 
parser.add_argument('--exp_framework', type=str, default='armstrong', help='Experiment framework to use.') # all or specific author 

# parser.add_argument('--random_state', type=int, default=42, help='Random state for the experiment.')
args = parser.parse_args()


# Load the dataset
df = pd.read_csv(f"results/{args.exp_framework}_sampling_{args.sampling}_analysis_{args.analysis}.csv")

# ---- Helpers ----
def parse_list(s):
    """Turn strings like "[np.float64(0.1), ...]" into real Python lists of floats."""
    if pd.isna(s):
        return None
    return ast.literal_eval(s.replace("np.float64", ""))

df["Result_Vector"] = df["Result_Vector"].apply(parse_list)
df["Std_Error_Vector"] = df["Std_Error_Vector"].apply(parse_list)
df["Sensitive_Attribute_Vector"] = df["Sensitive_Attribute_Vector"].apply(ast.literal_eval)

# ---- define the conditions ----

exp_framework = args.exp_framework

# df_names_armstrong_jobs_armstrong = df[(df["Names"]=="armstrong") & (df["Jobs"]=="armstrong")]
# df_names_armstrong_jobs_rozado = df[(df["Names"]=="armstrong") & (df["Jobs"]=="rozado") & (df["Random_State"]==42)]
# df_names_armstrong_jobs_rozado_1 = df[(df["Names"]=="armstrong") & (df["Jobs"]=="rozado") & (df["Random_State"]==2)]
# df_names_armstrong_jobs_rozado_2 = df[(df["Names"]=="armstrong") & (df["Jobs"]=="rozado") & (df["Random_State"]==3)]
# df_names_armstrong_jobs_rozado_3 = df[(df["Names"]=="armstrong") & (df["Jobs"]=="rozado") & (df["Random_State"]==4)]
# df_names_armstrong_jobs_rozado_4 = df[(df["Names"]=="armstrong") & (df["Jobs"]=="rozado") & (df["Random_State"]==5)]
# df_names_armstrong_jobs_rozado_5 = df[(df["Names"]=="armstrong") & (df["Jobs"]=="rozado") & (df["Random_State"]==6)]
# df_names_armstrong_jobs_rozado_6 = df[(df["Names"]=="armstrong") & (df["Jobs"]=="rozado") & (df["Random_State"]==7)]
# df_names_armstrong_jobs_rozado_7 = df[(df["Names"]=="armstrong") & (df["Jobs"]=="rozado") & (df["Random_State"]==8)]


# name_rozado, job_armstrong
# df_names_armstrong_jobs_wang = df[(df["Names"]=="armstrong") & (df["Jobs"]=="wang") & (df["Random_State"]==42)]
# df_names_armstrong_jobs_wang_1 = df[(df["Names"]=="armstrong") & (df["Jobs"]=="wang") & (df["Random_State"]==1)]
# df_names_armstrong_jobs_wang_2 = df[(df["Names"]=="armstrong") & (df["Jobs"]=="wang") & (df["Random_State"]==2)]
# df_names_armstrong_jobs_wang_3 = df[(df["Names"]=="armstrong") & (df["Jobs"]=="wang") & (df["Random_State"]==3)]
# df_names_armstrong_jobs_wang_4 = df[(df["Names"]=="armstrong") & (df["Jobs"]=="wang") & (df["Random_State"]==4)]

df_names_armstrong_jobs_armstrong = df[(df["Names"]=="armstrong") & (df["Jobs"]=="armstrong")]
df_names_armstrong_jobs_rozado = df[(df["Names"]=="armstrong") & (df["Jobs"]=="rozado")]
df_names_armstrong_jobs_wen = df[(df["Names"]=="armstrong") & (df["Jobs"]=="wen")]
df_names_armstrong_jobs_wang = df[(df["Names"]=="armstrong") & (df["Jobs"]=="wang")]
df_names_armstrong_jobs_karvonen = df[(df["Names"]=="armstrong") & (df["Jobs"]=="karvonen")]
df_names_armstrong_jobs_zollo = df[(df["Names"]=="armstrong") & (df["Jobs"]=="zollo")]
df_names_armstrong_jobs_yin = df[(df["Names"]=="armstrong") & (df["Jobs"]=="yin")]

df_names_rozado_jobs_armstrong = df[(df["Names"]=="rozado") & (df["Jobs"]=="armstrong")]
df_names_rozado_jobs_rozado = df[(df["Names"]=="rozado") & (df["Jobs"]=="rozado")]
df_names_rozado_jobs_wen = df[(df["Names"]=="rozado") & (df["Jobs"]=="wen")]
df_names_rozado_jobs_wang = df[(df["Names"]=="rozado") & (df["Jobs"]=="wang")]
df_names_rozado_jobs_karvonen = df[(df["Names"]=="rozado") & (df["Jobs"]=="karvonen")]
df_names_rozado_jobs_zollo = df[(df["Names"]=="rozado") & (df["Jobs"]=="zollo")]
df_names_rozado_jobs_yin = df[(df["Names"]=="rozado") & (df["Jobs"]=="yin")]


df_names_wen_jobs_armstrong = df[(df["Names"]=="wen") & (df["Jobs"]=="armstrong")]
df_names_wen_jobs_rozado = df[(df["Names"]=="wen") & (df["Jobs"]=="rozado")]
df_names_wen_jobs_wen = df[(df["Names"]=="wen") & (df["Jobs"]=="wen")]
df_names_wen_jobs_wang = df[(df["Names"]=="wen") & (df["Jobs"]=="wang")]
df_names_wen_jobs_karvonen = df[(df["Names"]=="wen") & (df["Jobs"]=="karvonen")]
df_names_wen_jobs_zollo = df[(df["Names"]=="wen") & (df["Jobs"]=="zollo")]
df_names_wen_jobs_yin = df[(df["Names"]=="wen") & (df["Jobs"]=="yin")]

df_names_wang_jobs_armstrong = df[(df["Names"]=="wang") & (df["Jobs"]=="armstrong")]
df_names_wang_jobs_rozado = df[(df["Names"]=="wang") & (df["Jobs"]=="rozado")]
df_names_wang_jobs_wen = df[(df["Names"]=="wang") & (df["Jobs"]=="wen")]
df_names_wang_jobs_wang = df[(df["Names"]=="wang") & (df["Jobs"]=="wang")]
df_names_wang_jobs_karvonen = df[(df["Names"]=="wang") & (df["Jobs"]=="karvonen")]
df_names_wang_jobs_zollo = df[(df["Names"]=="wang") & (df["Jobs"]=="zollo")]
df_names_wang_jobs_yin = df[(df["Names"]=="wang") & (df["Jobs"]=="yin")]


df_names_gaeb_jobs_armstrong = df[(df["Names"]=="gaeb") & (df["Jobs"]=="armstrong")]
df_names_gaeb_jobs_rozado = df[(df["Names"]=="gaeb") & (df["Jobs"]=="rozado")]
df_names_gaeb_jobs_wen = df[(df["Names"]=="gaeb") & (df["Jobs"]=="wen")]
df_names_gaeb_jobs_wang = df[(df["Names"]=="gaeb") & (df["Jobs"]=="wang")]
df_names_gaeb_jobs_karvonen = df[(df["Names"]=="gaeb") & (df["Jobs"]=="karvonen")]
df_names_gaeb_jobs_zollo = df[(df["Names"]=="gaeb") & (df["Jobs"]=="zollo")]
df_names_gaeb_jobs_yin = df[(df["Names"]=="gaeb") & (df["Jobs"]=="yin")]


df_names_lippens_jobs_armstrong = df[(df["Names"]=="lippens") & (df["Jobs"]=="armstrong")]
df_names_lippens_jobs_rozado = df[(df["Names"]=="lippens") & (df["Jobs"]=="rozado")]
df_names_lippens_jobs_wen = df[(df["Names"]=="lippens") & (df["Jobs"]=="wen")]
df_names_lippens_jobs_wang = df[(df["Names"]=="lippens") & (df["Jobs"]=="wang")]
df_names_lippens_jobs_karvonen = df[(df["Names"]=="lippens") & (df["Jobs"]=="karvonen")]
df_names_lippens_jobs_zollo = df[(df["Names"]=="lippens") & (df["Jobs"]=="zollo")]
df_names_lippens_jobs_yin = df[(df["Names"]=="lippens") & (df["Jobs"]=="yin")]


df_names_seshadri_jobs_armstrong = df[(df["Names"]=="seshadri") & (df["Jobs"]=="armstrong")]
df_names_seshadri_jobs_rozado = df[(df["Names"]=="seshadri") & (df["Jobs"]=="rozado")]
df_names_seshadri_jobs_wen = df[(df["Names"]=="seshadri") & (df["Jobs"]=="wen")]
df_names_seshadri_jobs_wang = df[(df["Names"]=="seshadri") & (df["Jobs"]=="wang")]
df_names_seshadri_jobs_karvonen = df[(df["Names"]=="seshadri") & (df["Jobs"]=="karvonen")]
df_names_seshadri_jobs_zollo = df[(df["Names"]=="seshadri") & (df["Jobs"]=="zollo")]
df_names_seshadri_jobs_yin = df[(df["Names"]=="seshadri") & (df["Jobs"]=="yin")]

df_names_karvonen_jobs_armstrong = df[(df["Names"]=="karvonen") & (df["Jobs"]=="armstrong")]
df_names_karvonen_jobs_rozado = df[(df["Names"]=="karvonen") & (df["Jobs"]=="rozado")]
df_names_karvonen_jobs_wen = df[(df["Names"]=="karvonen") & (df["Jobs"]=="wen")]
df_names_karvonen_jobs_wang = df[(df["Names"]=="karvonen") & (df["Jobs"]=="wang")]
df_names_karvonen_jobs_karvonen = df[(df["Names"]=="karvonen") & (df["Jobs"]=="karvonen")]
df_names_karvonen_jobs_zollo = df[(df["Names"]=="karvonen") & (df["Jobs"]=="zollo")]
df_names_karvonen_jobs_yin = df[(df["Names"]=="karvonen") & (df["Jobs"]=="yin")]

df_names_zollo_jobs_armstrong = df[(df["Names"]=="zollo") & (df["Jobs"]=="armstrong")]
df_names_zollo_jobs_rozado = df[(df["Names"]=="zollo") & (df["Jobs"]=="rozado")]
df_names_zollo_jobs_wen = df[(df["Names"]=="zollo") & (df["Jobs"]=="wen")]
df_names_zollo_jobs_wang = df[(df["Names"]=="zollo") & (df["Jobs"]=="wang")]
df_names_zollo_jobs_karvonen = df[(df["Names"]=="zollo") & (df["Jobs"]=="karvonen")]
df_names_zollo_jobs_zollo = df[(df["Names"]=="zollo") & (df["Jobs"]=="zollo")]
df_names_zollo_jobs_yin = df[(df["Names"]=="zollo") & (df["Jobs"]=="yin")]

df_names_yin_jobs_armstrong = df[(df["Names"]=="yin") & (df["Jobs"]=="armstrong")]
df_names_yin_jobs_rozado = df[(df["Names"]=="yin") & (df["Jobs"]=="rozado")]
df_names_yin_jobs_wen = df[(df["Names"]=="yin") & (df["Jobs"]=="wen")]
df_names_yin_jobs_wang = df[(df["Names"]=="yin") & (df["Jobs"]=="wang")]
df_names_yin_jobs_karvonen = df[(df["Names"]=="yin") & (df["Jobs"]=="karvonen")]
df_names_yin_jobs_zollo = df[(df["Names"]=="yin") & (df["Jobs"]=="zollo")]
df_names_yin_jobs_yin = df[(df["Names"]=="yin") & (df["Jobs"]=="yin")]



# We’ll plot one bar chart per metric; y = metric value for "Woman", x = model (gpt4o vs gpt3.5)
metrics = sorted(set(df_names_armstrong_jobs_wang["Metric"]).intersection(df_names_armstrong_jobs_wang["Metric"]))

# Index of "Woman" in the sensitive attribute vector
def get_woman_index(row):
    groups = row["Sensitive_Attribute_Vector"]
    try:
        return groups.index("Woman")
    except ValueError:
        # Fallback: user said "Woman is the first value", so default to 0 if missing
        return 0


# the names will be next to each other in the bar chart so color the jobs with a unique color 

condition_pairs = { "Armstrong_Name": (df_names_armstrong_jobs_armstrong, df_names_armstrong_jobs_rozado, df_names_armstrong_jobs_wen, df_names_armstrong_jobs_wang, df_names_armstrong_jobs_karvonen, df_names_armstrong_jobs_zollo, df_names_armstrong_jobs_yin, 
df_names_rozado_jobs_armstrong, df_names_rozado_jobs_rozado, df_names_rozado_jobs_wen, df_names_rozado_jobs_wang, df_names_rozado_jobs_karvonen, df_names_rozado_jobs_zollo, df_names_rozado_jobs_yin, 
df_names_wen_jobs_armstrong, df_names_wen_jobs_rozado, df_names_wen_jobs_wen, df_names_wen_jobs_wang, df_names_wen_jobs_karvonen, df_names_wen_jobs_zollo, df_names_wen_jobs_yin, 
df_names_wang_jobs_armstrong, df_names_wang_jobs_rozado, df_names_wang_jobs_wen, df_names_wang_jobs_wang, df_names_wang_jobs_karvonen, df_names_wang_jobs_zollo, df_names_wang_jobs_yin, 
df_names_gaeb_jobs_armstrong, df_names_gaeb_jobs_rozado, df_names_gaeb_jobs_wen, df_names_gaeb_jobs_wang, df_names_gaeb_jobs_karvonen, df_names_gaeb_jobs_zollo, df_names_gaeb_jobs_yin, 
df_names_lippens_jobs_armstrong, df_names_lippens_jobs_rozado, df_names_lippens_jobs_wen, df_names_lippens_jobs_wang, df_names_lippens_jobs_karvonen, df_names_lippens_jobs_zollo, df_names_lippens_jobs_yin, 
df_names_seshadri_jobs_armstrong, df_names_seshadri_jobs_rozado, df_names_seshadri_jobs_wen, df_names_seshadri_jobs_wang, df_names_seshadri_jobs_karvonen, df_names_seshadri_jobs_zollo, df_names_seshadri_jobs_yin, 
df_names_karvonen_jobs_armstrong, df_names_karvonen_jobs_rozado, df_names_karvonen_jobs_wen, df_names_karvonen_jobs_wang, df_names_karvonen_jobs_karvonen, df_names_karvonen_jobs_zollo, df_names_karvonen_jobs_yin,
df_names_zollo_jobs_armstrong, df_names_zollo_jobs_rozado, df_names_zollo_jobs_wen, df_names_zollo_jobs_wang, df_names_zollo_jobs_karvonen, df_names_zollo_jobs_zollo, df_names_zollo_jobs_yin, 
df_names_yin_jobs_armstrong, df_names_yin_jobs_rozado, df_names_yin_jobs_wen, df_names_yin_jobs_wang, df_names_yin_jobs_karvonen, df_names_yin_jobs_zollo, df_names_yin_jobs_yin)}


condition_pair_names = {"Armstrong_Name": ['name_armstrong_job_armstrong', "name_armstrong_job_rozado", "name_armstrong_job_wen", "name_armstrong_job_wang", "name_armstrong_job_karvonen", "name_armstrong_job_zollo", "name_armstrong_job_yin", 
"name_rozado_job_armstrong", "name_rozado_job_rozado", "name_rozado_job_wen", "name_rozado_job_wang", "name_rozado_job_karvonen", "name_rozado_job_zollo", "name_rozado_job_yin", 
"name_wen_job_armstrong", "name_wen_job_rozado", "name_wen_job_wen", "name_wen_jobs_wang", "name_wen_job_karvonen", "name_wen_job_zollo", "name_wen_job_yin", 
"name_wang_job_armstrong", "name_wang_job_rozado", "name_wang_job_wen", "name_wang_job_wang", "name_wang_job_karvonen", "name_wang_job_zollo", "name_wang_job_yin", 
"name_gaeb_job_armstrong", "name_gaeb_job_rozado", "name_gaeb_job_wen", "name_gaeb_job_wang", "name_gaeb_job_karvonen", "name_gaeb_job_zollo", "name_gaeb_job_yin", 
"name_lippens_job_armstrong", "name_lippens_job_rozado", "name_lippens_job_wen", "name_lippens_job_wang", "name_lippens_job_karvonen", "name_lippens_job_zollo", "name_lippens_job_yin", 
"name_seshadri_job_armstrong", "name_seshadri_job_rozado", "name_seshadri_job_wen", "name_seshadri_job_wang", "name_seshadri_job_karvonen", "name_seshadri_job_zollo", "name_seshadri_job_yin", 
"name_karvonen_job_armstrong", "name_karvonen_job_rozado", "name_karvonen_job_wen", "name_karvonen_job_wang", "name_karvonen_job_karvonen", "name_karvonen_job_zollo", "name_karvonen_job_yin",
"name_zollo_job_armstrong", "name_zollo_job_rozado", "name_zollo_job_wen", "name_zollo_job_wang", "name_zollo_job_karvonen", "name_zollo_job_zollo", "name_zollo_job_yin",
"name_yin_job_armstrong", "name_yin_job_rozado", "name_yin_job_wen", "name_yin_job_wang", "name_yin_job_karvonen", "name_yin_job_zollo", "name_yin_job_yin"]}


# condition_pairs = { "Armstrong_Name": (df_names_armstrong_jobs_wang, df_names_armstrong_jobs_wang_1, df_names_armstrong_jobs_wang_2, df_names_armstrong_jobs_wang_3, df_names_armstrong_jobs_wang_4)}
# condition_pair_names = {"Armstrong_Name": ['name_armstrong_job_wang', "name_armstrong_job_wang_1", "name_armstrong_job_wang_2", "name_armstrong_job_wang_3", "name_armstrong_job_wang_4"]}

condition_pair_ylims = {}
# for metric in ["regression_coefficients", "ttest", "score_difference"]:
for metric in ["impact_ratio_0.75", "impact_ratio_0.90"]:
# for metric in ["ttest"]:
    metric_df = df[df["Metric"] == metric]
    # Flatten all values in Result_Vector columns into a single list, ignoring errors
    all_results = []
    for vals in metric_df["Result_Vector"]:
        try:
            if isinstance(vals, list):
                all_results.extend([float(x) for x in vals])
            elif pd.notnull(vals):
                v = ast.literal_eval(str(vals))
                if isinstance(v, list):
                    all_results.extend([float(x) for x in v])
                else:
                    all_results.append(float(v))
        except Exception:
            continue
    if all_results:
        ymin, ymax = min(all_results), max(all_results)
        # Add a little padding
        span = ymax - ymin
        ymin -= 0.05 * span
        ymax += 0.05 * span
        condition_pair_ylims[metric] = (ymin, ymax)
    else:
        condition_pair_ylims[metric] = (0, 1)  # fallback if empty

# make a directory for the exp framework
os.makedirs(f'results/figs/{args.exp_framework}', exist_ok=True)

def calculate_difference(impact_ratio, threshold = 0.75, total_applicants=3200):
    impact_ratio = impact_ratio['Result_Vector'][0]
    threshold = float(threshold)
    # Assume half of the applicants are women and half are men
    men_applicants = total_applicants / 2
    women_applicants = total_applicants / 2

    # Top 25% of applicants are selected
    men_selected = men_applicants * (1-threshold) 
    women_selected = women_applicants * (1-threshold) * float(impact_ratio)

    # Calculate the difference in the number of women and men selected
    difference = women_selected - men_selected
    return difference




for metric in ["impact_ratio_0.75", "impact_ratio_0.90"]:
    for condition_pair_key in condition_pairs.keys():
        condition_pair = condition_pairs[condition_pair_key]
        df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16, df17, df18, df19, df20, df21, df22, df23, df24, df25, df26, df27, df28, df29, df30, df31, df32, df33, df34, df35, df36, df37, df38, df39, df40, df41, df42, df43, df44, df45, df46, df47, df48, df49, df50, df51, df52, df53, df54, df55, df56, df57, df58, df59, df60, df61, df62, df63, df64, df65, df66, df67, df68, df69, df70 = condition_pair
        # df1, df2, df3, df4, df5  = condition_pair
        # df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12  = condition_pair
        
        
        r1 = calculate_difference(df1[df1["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r2 = calculate_difference(df2[df2["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r3 = calculate_difference(df3[df3["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r4 = calculate_difference(df4[df4["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r5 = calculate_difference(df5[df5["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r6 = calculate_difference(df6[df6["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r7 = calculate_difference(df7[df7["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r8 = calculate_difference(df8[df8["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r9 = calculate_difference(df9[df9["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r10 = calculate_difference(df10[df10["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r11 = calculate_difference(df11[df11["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r12 = calculate_difference(df12[df12["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r13 = calculate_difference(df13[df13["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r14 = calculate_difference(df14[df14["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r15 = calculate_difference(df15[df15["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r16 = calculate_difference(df16[df16["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r17 = calculate_difference(df17[df17["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r18 = calculate_difference(df18[df18["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r19 = calculate_difference(df19[df19["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r20 = calculate_difference(df20[df20["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r21 = calculate_difference(df21[df21["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r22 = calculate_difference(df22[df22["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r23 = calculate_difference(df23[df23["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r24 = calculate_difference(df24[df24["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r25 = calculate_difference(df25[df25["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)  
        r26 = calculate_difference(df26[df26["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r27 = calculate_difference(df27[df27["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r28 = calculate_difference(df28[df28["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r29 = calculate_difference(df29[df29["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r30 = calculate_difference(df30[df30["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r31 = calculate_difference(df31[df31["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r32 = calculate_difference(df32[df32["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r33 = calculate_difference(df33[df33["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r34 = calculate_difference(df34[df34["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r35 = calculate_difference(df35[df35["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)  
        r36 = calculate_difference(df36[df36["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r37 = calculate_difference(df37[df37["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r38 = calculate_difference(df38[df38["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r39 = calculate_difference(df39[df39["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r40 = calculate_difference(df40[df40["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r41 = calculate_difference(df41[df41["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r42 = calculate_difference(df42[df42["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r43 = calculate_difference(df43[df43["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r44 = calculate_difference(df44[df44["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r45 = calculate_difference(df45[df45["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r46 = calculate_difference(df46[df46["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r47 = calculate_difference(df47[df47["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r48 = calculate_difference(df48[df48["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r49 = calculate_difference(df49[df49["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r50 = calculate_difference(df50[df50["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r51 = calculate_difference(df51[df51["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r52 = calculate_difference(df52[df52["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r53 = calculate_difference(df53[df53["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r54 = calculate_difference(df54[df54["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r55 = calculate_difference(df55[df55["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r56 = calculate_difference(df56[df56["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)  
        r57 = calculate_difference(df57[df57["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r58 = calculate_difference(df58[df58["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r59 = calculate_difference(df59[df59["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r60 = calculate_difference(df60[df60["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r61 = calculate_difference(df61[df61["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r62 = calculate_difference(df62[df62["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r63 = calculate_difference(df63[df63["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r64 = calculate_difference(df64[df64["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r65 = calculate_difference(df65[df65["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r66 = calculate_difference(df66[df66["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r67 = calculate_difference(df67[df67["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r68 = calculate_difference(df68[df68["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r69 = calculate_difference(df69[df69["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)
        r70 = calculate_difference(df70[df70["Metric"]==metric].iloc[0], threshold = metric.split('_')[-1], total_applicants=3200)

        idx = get_woman_index(r2)
        # Pull single values for "Woman"
        v1 = float(r1["Result_Vector"][idx])
        e1 = float(r1["Std_Error_Vector"][idx]) if r1["Std_Error_Vector"] is not None else None

        v2 = float(r2["Result_Vector"][idx])
        e2 = float(r2["Std_Error_Vector"][idx]) if r2["Std_Error_Vector"] is not None else None
        
        v3 = float(r3["Result_Vector"][idx])
        e3 = float(r3["Std_Error_Vector"][idx]) if r3["Std_Error_Vector"] is not None else None

        v4 = float(r4["Result_Vector"][idx])
        e4 = float(r4["Std_Error_Vector"][idx]) if r4["Std_Error_Vector"] is not None else None

        v5 = float(r5["Result_Vector"][idx])
        e5 = float(r5["Std_Error_Vector"][idx]) if r5["Std_Error_Vector"] is not None else None

        v6 = float(r6["Result_Vector"][idx])
        e6 = float(r6["Std_Error_Vector"][idx]) if r6["Std_Error_Vector"] is not None else None

        v7 = float(r7["Result_Vector"][idx])
        e7 = float(r7["Std_Error_Vector"][idx]) if r7["Std_Error_Vector"] is not None else None

        v8 = float(r8["Result_Vector"][idx])
        e8 = float(r8["Std_Error_Vector"][idx]) if r8["Std_Error_Vector"] is not None else None

        v9 = float(r9["Result_Vector"][idx])
        e9 = float(r9["Std_Error_Vector"][idx]) if r9["Std_Error_Vector"] is not None else None

        v10 = float(r10["Result_Vector"][idx])
        e10 = float(r10["Std_Error_Vector"][idx]) if r10["Std_Error_Vector"] is not None else None

        v11 = float(r11["Result_Vector"][idx])
        e11 = float(r11["Std_Error_Vector"][idx]) if r11["Std_Error_Vector"] is not None else None

        v12 = float(r12["Result_Vector"][idx])
        e12 = float(r12["Std_Error_Vector"][idx]) if r12["Std_Error_Vector"] is not None else None

        v13 = float(r13["Result_Vector"][idx])
        e13 = float(r13["Std_Error_Vector"][idx]) if r13["Std_Error_Vector"] is not None else None

        v14 = float(r14["Result_Vector"][idx])
        e14 = float(r14["Std_Error_Vector"][idx]) if r14["Std_Error_Vector"] is not None else None

        v15 = float(r15["Result_Vector"][idx])
        e15 = float(r15["Std_Error_Vector"][idx]) if r15["Std_Error_Vector"] is not None else None

        v16 = float(r16["Result_Vector"][idx])
        e16 = float(r16["Std_Error_Vector"][idx]) if r16["Std_Error_Vector"] is not None else None

        v17 = float(r17["Result_Vector"][idx])
        e17 = float(r17["Std_Error_Vector"][idx]) if r17["Std_Error_Vector"] is not None else None

        v18 = float(r18["Result_Vector"][idx])
        e18 = float(r18["Std_Error_Vector"][idx]) if r18["Std_Error_Vector"] is not None else None

        v19 = float(r19["Result_Vector"][idx])
        e19 = float(r19["Std_Error_Vector"][idx]) if r19["Std_Error_Vector"] is not None else None

        v20 = float(r20["Result_Vector"][idx])
        e20 = float(r20["Std_Error_Vector"][idx]) if r20["Std_Error_Vector"] is not None else None

        v21 = float(r21["Result_Vector"][idx])
        e21 = float(r21["Std_Error_Vector"][idx]) if r21["Std_Error_Vector"] is not None else None

        v22 = float(r22["Result_Vector"][idx])
        e22 = float(r22["Std_Error_Vector"][idx]) if r22["Std_Error_Vector"] is not None else None

        v23 = float(r23["Result_Vector"][idx])
        e23 = float(r23["Std_Error_Vector"][idx]) if r23["Std_Error_Vector"] is not None else None

        v24 = float(r24["Result_Vector"][idx])
        e24 = float(r24["Std_Error_Vector"][idx]) if r24["Std_Error_Vector"] is not None else None

        v25 = float(r25["Result_Vector"][idx])
        e25 = float(r25["Std_Error_Vector"][idx]) if r25["Std_Error_Vector"] is not None else None

        v26 = float(r26["Result_Vector"][idx])
        e26 = float(r26["Std_Error_Vector"][idx]) if r26["Std_Error_Vector"] is not None else None

        v27 = float(r27["Result_Vector"][idx])
        e27 = float(r27["Std_Error_Vector"][idx]) if r27["Std_Error_Vector"] is not None else None

        v28 = float(r28["Result_Vector"][idx])
        e28 = float(r28["Std_Error_Vector"][idx]) if r28["Std_Error_Vector"] is not None else None

        v29 = float(r29["Result_Vector"][idx])
        e29 = float(r29["Std_Error_Vector"][idx]) if r29["Std_Error_Vector"] is not None else None

        v30 = float(r30["Result_Vector"][idx])
        e30 = float(r30["Std_Error_Vector"][idx]) if r30["Std_Error_Vector"] is not None else None

        v31 = float(r31["Result_Vector"][idx])
        e31 = float(r31["Std_Error_Vector"][idx]) if r31["Std_Error_Vector"] is not None else None

        v32 = float(r32["Result_Vector"][idx])
        e32 = float(r32["Std_Error_Vector"][idx]) if r32["Std_Error_Vector"] is not None else None

        v33 = float(r33["Result_Vector"][idx])
        e33 = float(r33["Std_Error_Vector"][idx]) if r33["Std_Error_Vector"] is not None else None

        v34 = float(r34["Result_Vector"][idx])
        e34 = float(r34["Std_Error_Vector"][idx]) if r34["Std_Error_Vector"] is not None else None

        v35 = float(r35["Result_Vector"][idx])
        e35 = float(r35["Std_Error_Vector"][idx]) if r35["Std_Error_Vector"] is not None else None

        v36 = float(r36["Result_Vector"][idx])
        e36 = float(r36["Std_Error_Vector"][idx]) if r36["Std_Error_Vector"] is not None else None

        v37 = float(r37["Result_Vector"][idx])
        e37 = float(r37["Std_Error_Vector"][idx]) if r37["Std_Error_Vector"] is not None else None

        v38 = float(r38["Result_Vector"][idx])
        e38 = float(r38["Std_Error_Vector"][idx]) if r38["Std_Error_Vector"] is not None else None

        v39 = float(r39["Result_Vector"][idx])
        e39 = float(r39["Std_Error_Vector"][idx]) if r39["Std_Error_Vector"] is not None else None

        v40 = float(r40["Result_Vector"][idx])
        e40 = float(r40["Std_Error_Vector"][idx]) if r40["Std_Error_Vector"] is not None else None

        v41 = float(r41["Result_Vector"][idx])
        e41 = float(r41["Std_Error_Vector"][idx]) if r41["Std_Error_Vector"] is not None else None

        v42 = float(r42["Result_Vector"][idx])
        e42 = float(r42["Std_Error_Vector"][idx]) if r42["Std_Error_Vector"] is not None else None

        v43 = float(r43["Result_Vector"][idx])
        e43 = float(r43["Std_Error_Vector"][idx]) if r43["Std_Error_Vector"] is not None else None

        v44 = float(r44["Result_Vector"][idx])  
        e44 = float(r44["Std_Error_Vector"][idx]) if r44["Std_Error_Vector"] is not None else None

        v45 = float(r45["Result_Vector"][idx])
        e45 = float(r45["Std_Error_Vector"][idx]) if r45["Std_Error_Vector"] is not None else None

        v46 = float(r46["Result_Vector"][idx])
        e46 = float(r46["Std_Error_Vector"][idx]) if r46["Std_Error_Vector"] is not None else None
        
        v47 = float(r47["Result_Vector"][idx])
        e47 = float(r47["Std_Error_Vector"][idx]) if r47["Std_Error_Vector"] is not None else None

        v48 = float(r48["Result_Vector"][idx])
        e48 = float(r48["Std_Error_Vector"][idx]) if r48["Std_Error_Vector"] is not None else None

        v49 = float(r49["Result_Vector"][idx])
        e49 = float(r49["Std_Error_Vector"][idx]) if r49["Std_Error_Vector"] is not None else None
        
        v50 = float(r50["Result_Vector"][idx])
        e50 = float(r50["Std_Error_Vector"][idx]) if r50["Std_Error_Vector"] is not None else None

        v51 = float(r51["Result_Vector"][idx])
        e51 = float(r51["Std_Error_Vector"][idx]) if r51["Std_Error_Vector"] is not None else None

        v52 = float(r52["Result_Vector"][idx])
        e52 = float(r52["Std_Error_Vector"][idx]) if r52["Std_Error_Vector"] is not None else None

        v53 = float(r53["Result_Vector"][idx])
        e53 = float(r53["Std_Error_Vector"][idx]) if r53["Std_Error_Vector"] is not None else None

        v54 = float(r54["Result_Vector"][idx])
        e54 = float(r54["Std_Error_Vector"][idx]) if r54["Std_Error_Vector"] is not None else None

        v55 = float(r55["Result_Vector"][idx])
        e55 = float(r55["Std_Error_Vector"][idx]) if r55["Std_Error_Vector"] is not None else None

        v56 = float(r56["Result_Vector"][idx])
        e56 = float(r56["Std_Error_Vector"][idx]) if r56["Std_Error_Vector"] is not None else None

        v57 = float(r57["Result_Vector"][idx])
        e57 = float(r57["Std_Error_Vector"][idx]) if r57["Std_Error_Vector"] is not None else None

        v58 = float(r58["Result_Vector"][idx])
        e58 = float(r58["Std_Error_Vector"][idx]) if r58["Std_Error_Vector"] is not None else None

        v59 = float(r59["Result_Vector"][idx])
        e59 = float(r59["Std_Error_Vector"][idx]) if r59["Std_Error_Vector"] is not None else None

        v60 = float(r60["Result_Vector"][idx])  
        e60 = float(r60["Std_Error_Vector"][idx]) if r60["Std_Error_Vector"] is not None else None

        v61 = float(r61["Result_Vector"][idx])
        e61 = float(r61["Std_Error_Vector"][idx]) if r61["Std_Error_Vector"] is not None else None
        v62 = float(r62["Result_Vector"][idx])
        e62 = float(r62["Std_Error_Vector"][idx]) if r62["Std_Error_Vector"] is not None else None
        v63 = float(r63["Result_Vector"][idx])
        e63 = float(r63["Std_Error_Vector"][idx]) if r63["Std_Error_Vector"] is not None else None
        v64 = float(r64["Result_Vector"][idx])
        e64 = float(r64["Std_Error_Vector"][idx]) if r64["Std_Error_Vector"] is not None else None
        v65 = float(r65["Result_Vector"][idx])
        e65 = float(r65["Std_Error_Vector"][idx]) if r65["Std_Error_Vector"] is not None else None
        v66 = float(r66["Result_Vector"][idx])
        e66 = float(r66["Std_Error_Vector"][idx]) if r66["Std_Error_Vector"] is not None else None
        v67 = float(r67["Result_Vector"][idx])
        e67 = float(r67["Std_Error_Vector"][idx]) if r67["Std_Error_Vector"] is not None else None
        v68 = float(r68["Result_Vector"][idx])
        e68 = float(r68["Std_Error_Vector"][idx]) if r68["Std_Error_Vector"] is not None else None
        v69 = float(r69["Result_Vector"][idx])
        e69 = float(r69["Std_Error_Vector"][idx]) if r69["Std_Error_Vector"] is not None else None
        v70 = float(r70["Result_Vector"][idx])
        e70 = float(r70["Std_Error_Vector"][idx]) if r70["Std_Error_Vector"] is not None else None

        if metric == 'regression_coefficients' or metric == 'ttest':
            # get the p_value vector P_Value_Vector
            p1 = float(r1["P_Value_Vector"][1:-1])
            p2 = float(r2["P_Value_Vector"][1:-1])
            p3 = float(r3["P_Value_Vector"][1:-1])
            p4 = float(r4["P_Value_Vector"][1:-1])
            p5 = float(r5["P_Value_Vector"][1:-1])
            p6 = float(r6["P_Value_Vector"][1:-1])
            p7 = float(r7["P_Value_Vector"][1:-1])
            p8 = float(r8["P_Value_Vector"][1:-1])
            p6 = float(r6["P_Value_Vector"][1:-1])
            p7 = float(r7["P_Value_Vector"][1:-1])
            p8 = float(r8["P_Value_Vector"][1:-1])
            p9 = float(r9["P_Value_Vector"][1:-1])
            p10 = float(r10["P_Value_Vector"][1:-1])
            p11 = float(r11["P_Value_Vector"][1:-1])
            p12 = float(r12["P_Value_Vector"][1:-1])
            p13 = float(r13["P_Value_Vector"][1:-1])
            p14 = float(r14["P_Value_Vector"][1:-1])
            p15 = float(r15["P_Value_Vector"][1:-1])
            p16 = float(r16["P_Value_Vector"][1:-1])
            p17 = float(r17["P_Value_Vector"][1:-1])
            p18 = float(r18["P_Value_Vector"][1:-1])
            p19 = float(r19["P_Value_Vector"][1:-1])
            p20 = float(r20["P_Value_Vector"][1:-1])
            p21 = float(r21["P_Value_Vector"][1:-1])
            p22 = float(r22["P_Value_Vector"][1:-1])
            p23 = float(r23["P_Value_Vector"][1:-1])
            p24 = float(r24["P_Value_Vector"][1:-1])
            p25 = float(r25["P_Value_Vector"][1:-1])
            p26 = float(r26["P_Value_Vector"][1:-1])
            p27 = float(r27["P_Value_Vector"][1:-1])
            p28 = float(r28["P_Value_Vector"][1:-1])
            p29 = float(r29["P_Value_Vector"][1:-1])
            p30 = float(r30["P_Value_Vector"][1:-1])
            p31 = float(r31["P_Value_Vector"][1:-1])
            p32 = float(r32["P_Value_Vector"][1:-1])
            p33 = float(r33["P_Value_Vector"][1:-1])
            p34 = float(r34["P_Value_Vector"][1:-1])
            p35 = float(r35["P_Value_Vector"][1:-1])
            p36 = float(r36["P_Value_Vector"][1:-1])
            p37 = float(r37["P_Value_Vector"][1:-1])
            p38 = float(r38["P_Value_Vector"][1:-1])
            p39 = float(r39["P_Value_Vector"][1:-1])
            p40 = float(r40["P_Value_Vector"][1:-1])
            p41 = float(r41["P_Value_Vector"][1:-1])
            p42 = float(r42["P_Value_Vector"][1:-1])
            p43 = float(r43["P_Value_Vector"][1:-1])
            p44 = float(r44["P_Value_Vector"][1:-1])
            p45 = float(r45["P_Value_Vector"][1:-1])
            p46 = float(r46["P_Value_Vector"][1:-1])
            p47 = float(r47["P_Value_Vector"][1:-1])
            p48 = float(r48["P_Value_Vector"][1:-1])
            p49 = float(r49["P_Value_Vector"][1:-1])
            p50 = float(r50["P_Value_Vector"][1:-1])
            p51 = float(r51["P_Value_Vector"][1:-1])
            p52 = float(r52["P_Value_Vector"][1:-1])
            p53 = float(r53["P_Value_Vector"][1:-1])
            p54 = float(r54["P_Value_Vector"][1:-1])
            p55 = float(r55["P_Value_Vector"][1:-1])
            p56 = float(r56["P_Value_Vector"][1:-1])
            p57 = float(r57["P_Value_Vector"][1:-1])
            p58 = float(r58["P_Value_Vector"][1:-1])
            p59 = float(r59["P_Value_Vector"][1:-1])
            p60 = float(r60["P_Value_Vector"][1:-1])

            p61 = float(r61["P_Value_Vector"][1:-1])
            p62 = float(r62["P_Value_Vector"][1:-1])
            p63 = float(r63["P_Value_Vector"][1:-1])
            p64 = float(r64["P_Value_Vector"][1:-1])
            p65 = float(r65["P_Value_Vector"][1:-1])
            p66 = float(r66["P_Value_Vector"][1:-1])
            p67 = float(r67["P_Value_Vector"][1:-1])
            p68 = float(r68["P_Value_Vector"][1:-1])
            p69 = float(r69["P_Value_Vector"][1:-1])
            p70 = float(r70["P_Value_Vector"][1:-1])
            # p_values = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16]
            # p_values = [p1, p2, p3, p4, p5, p6, p7, p8]
            p_values = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43, p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57, p58, p59, p60, p61, p62, p63, p64, p65, p66, p67, p68, p69, p70]
            # p_values = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12]
            
        models = condition_pair_names[condition_pair_key]
        # values = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16]
        # errors = [e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15, e16]

        # values = [v1, v2, v3, v4, v5, v6, v7, v8]
        values = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55, v56, v57, v58, v59, v60, v61, v62, v63, v64, v65, v66, v67, v68, v69, v70]
        # errors = [e1, e2, e3, e4, e5, e6, e7, e8]
        errors = [e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15, e16, e17, e18, e19, e20, e21, e22, e23, e24, e25, e26, e27, e28, e29, e30, e31, e32, e33, e34, e35, e36, e37, e38, e39, e40, e41, e42, e43, e44, e45, e46, e47, e48, e49, e50, e51, e52, e53, e54, e55, e56, e57, e58, e59, e60, e61, e62, e63, e64, e65, e66, e67, e68, e69, e70]


        # set the wang values to be zero 
        # values[21:28] = np.zeros(7)
        # plot a histogram of the values 
        plt.figure(figsize=(10, 6))
        plt.hist(values, bins=30, alpha=0.5, color='blue')
        plt.title(f'Histogram of {metric} for "Woman" ({args.sampling}, {args.analysis} {args.exp_framework} framework)')
        plt.xlabel(metric)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(f'results/figs/{args.exp_framework}/{metric}_sampling_{args.sampling}_analysis_{args.analysis}_all_histogram.png')
        plt.close()

        # plot a 10x7 heatmap of the values. there should be 10 rows and every row should contain 7 values. iterate through the values list and add the values to the heatmap.

        # Plot
        row_labels = ["armstrong", "rozado", "wen", "wang", "gaeb", "lippens", "seshadri", "karvonen", "zollo", "yin"]
        col_labels = ["armstrong", "rozado", "wen", "wang", "karvonen", "zollo", "yin"]
        # Reshape to 10×7 matrix
        heatmap_data = np.array(values).reshape(10, 7)

        # Plot
        plt.figure(figsize=(7, 10))
        plt.imshow(heatmap_data, cmap='viridis', aspect='auto')

        # Add colorbar
        plt.colorbar(label='Value')

        # Set tick positions and labels
        plt.xticks(ticks=range(7), labels=col_labels)
        plt.yticks(ticks=range(10), labels=row_labels)

        plt.title("Names X Jobs Heatmap")
        plt.xlabel("Jobs")
        plt.ylabel("Names")

        plt.savefig(f'results/figs/{args.exp_framework}/{metric}_sampling_{args.sampling}_analysis_{args.analysis}_all_heatmap.png')
        plt.close()

        # Build error list only if at least one error exists; Matplotlib expects None or arraylike
        yerr = None if all(err is None for err in errors) else errors
        # if a value in the erros is None, replace it with the average of the other errors 
        if any(err is None for err in errors):
            non_none_errors = [err for err in errors if err is not None]
            avg_error = sum(non_none_errors) / len(non_none_errors)
            yerr = [avg_error if err is None else err for err in errors]


        # ---- Plot: one chart per metric ----
        plt.figure(figsize=(40, 6))

        # Define a set of 4 distinct colors to cycle through
        # Map each color to a label from [armstrong, rozado, wen, wang]
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

        # As the models list is like name_job_author_job... (e.g., name_armstrong_job_armstrong), extract the job part (last word)
        def extract_job_label(model_name):
            for label in label_to_color:
                if f"job_{label}" in model_name:
                    return label
            # fallback just in case
            return "armstrong"

        colors = [label_to_color[extract_job_label(model_name)] for model_name in models]
        bars = plt.bar(models, values, yerr=yerr, capsize=4, align='center', color=colors)
        # Add legend mapping color to label
        import matplotlib.patches as mpatches
        legend_handles = [mpatches.Patch(color=color, label=label) for label, color in label_to_color.items()]
        plt.legend(handles=legend_handles, title="Job Group")
        # Add a star if p-value is below 0.05
        if metric == 'regression_coefficients' or metric == 'ttest':
            for i, (bar, pval) in enumerate(zip(bars, p_values)):
                if pval < (0.05/70): # Bonferroni correction
                # if pval < 0.05:
                    # get bar height for annotation
                    height = bar.get_height()
                    plt.annotate('*',
                                 xy=(bar.get_x() + bar.get_width() / 2, height),
                                 xytext=(0, 6),  # 6 points vertical offset
                                 textcoords="offset points",
                                 ha='center', va='bottom', fontsize=18, color='red')
        plt.ylabel(metric + ' (# women selected - # men selected)')
        if metric in condition_pair_ylims.keys():
            plt.ylim(condition_pair_ylims[metric])
        else:
            plt.ylim(0.5, 1.2)
        plt.title(f'{metric} for "Woman" ({args.sampling}, {args.analysis} {args.exp_framework} framework)')
        
        # Optional: rotate x labels if they wrap
        plt.xticks(rotation=90)
        plt.tight_layout()
        print(f'Saving figure to results/figs/{args.exp_framework}/{metric}_sampling_{args.sampling}_analysis_{args.analysis}_all.png')
        plt.savefig(f'results/figs/{args.exp_framework}/{metric}_sampling_{args.sampling}_analysis_{args.analysis}_all.png')
        plt.close()


print(f'Saving figure to results/figs/{args.exp_framework}/{metric}_sampling_{args.sampling}_analysis_{args.analysis}_all.png')

# save heatmap data to a csv file
heatmap_data = pd.DataFrame(heatmap_data, columns=col_labels, index=row_labels)
heatmap_data.to_csv(f'results/figs/{args.exp_framework}/{metric}_sampling_{args.sampling}_analysis_{args.analysis}_all_heatmap_data.csv', index=False)
