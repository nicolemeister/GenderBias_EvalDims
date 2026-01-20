import yaml
import pandas as pd
import os
import numpy as np
from scipy.stats import wasserstein_distance
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
from utils.merge_config import merge_config
from scipy.stats import ttest_ind
import glob
from collections import Counter
from IPython.display import display, HTML
from tqdm import tqdm
import json
from utils.variables import MODELS


columns = [
    "Author",
    "Names",
    "Jobs",
    "Num_Trials",
    "Model",
    "Temperature",
    "Metric",
    "Sensitive_Attribute_Vector",
    "Result_Vector",
    "Std_Error_Vector",
    "P_Value_Vector"
]

def downsample_data(df, config):
    if config['experimental_framework'] == 'armstrong':

        # first ensure that the 



        if config['Name']['Bundle_Name'] == 'wang':  
            # for each unique job in df['role'] 
            unique_jobs = df['role'].unique()
            sampled_rows = []
            for job in unique_jobs:
                # sample 1 value from each unique name
                job_rows = df[df['role'] == job]
                unique_names = job_rows['name'].unique()
                for name in unique_names:
                    name_rows = job_rows[job_rows['name'] == name]
                    sampled_row = name_rows.sample(n=1, random_state=42)
                    sampled_rows.append(sampled_row)
            df = pd.concat(sampled_rows, ignore_index=True)
            return df
        else: 
            return df



def compute_ttest(df, sensitive_attribute_vector):
    results, std, pvals = [], [], []
    for attr in sensitive_attribute_vector:
        if attr == 'Woman':
            group1 = df[df['gender'] == 'women']['value']
            group2 = df[df['gender'] == 'men']['value']
            ttest = ttest_ind(group1, group2)
            results.append(ttest.statistic)
            pvals.append(ttest.pvalue)
            # Standard error of difference in means, Welch's formula
            n1 = len(group1)
            n2 = len(group2)
            s1_sq = np.var(group1, ddof=1)
            s2_sq = np.var(group2, ddof=1)
            std_err = np.sqrt(s1_sq / n1 + s2_sq / n2)
            std.append(std_err)

    return results, std, pvals


def score_diff(df, sensitive_attribute_vector):

    import numpy as np

    # Define levels
    gender_levels = ["men", "women"]
    race_levels = ["white", "hispanic", "asian", "black"]

    # Aggregate sum of 'value' by race and gender
    agg = df.groupby(['race', 'gender'])['value'].sum().reset_index()

    # Calculate overall sum of 'value'
    total_value = agg['value'].sum()

    # Calculate probability of being hired per group
    agg['prob'] = agg['value'] / total_value

    # Create a function to get probability for a group
    def get_prob(race, gender):
        row = agg[(agg['race'] == race) & (agg['gender'] == gender)]
        if not row.empty:
            return row['prob'].values[0]
        else:
            return 0

    # Reference groups
    ref_gender = 'men'
    ref_race = 'white'

    # Probability for reference groups
    prob_ref_gender = agg[agg['gender'] == ref_gender]['prob'].sum()
    prob_ref_race = agg[agg['race'] == ref_race]['prob'].sum()
    prob_ref_combined = get_prob(ref_race, ref_gender)

    # Compute relative probabilities

    # 1. Female vs Male (all races)
    prob_women = agg[agg['gender'] == 'women']['prob'].sum()
    prob_men = agg[agg['gender'] == 'men']['prob'].sum()
    rel_prob_female_vs_male = prob_women / prob_men if prob_men else None

    # 2. Hispanic vs White (all genders)
    prob_hispanic = agg[agg['race'] == 'hispanic']['prob'].sum()
    prob_white = agg[agg['race'] == 'white']['prob'].sum()
    rel_prob_hispanic_vs_white = prob_hispanic / prob_white if prob_white else None

    # 3. Asian vs White (all genders)
    prob_asian = agg[agg['race'] == 'asian']['prob'].sum()
    rel_prob_asian_vs_white = prob_asian / prob_white if prob_white else None

    # 4. Black vs White (all genders)
    prob_black = agg[agg['race'] == 'black']['prob'].sum()
    rel_prob_black_vs_white = prob_black / prob_white if prob_white else None

    # 5. Black women vs White men
    prob_black_women = get_prob('black', 'women')
    prob_white_men = get_prob('white', 'men')
    rel_prob_black_women_vs_white_men = prob_black_women / prob_white_men if prob_white_men else None

    # Print reference probabilities (rounded to 2 decimal places)
    print("Probability female:", round(prob_women, 4))
    print("Probability male:", round(prob_men, 4))
    print("Probability white:", round(prob_white, 4))
    print("Probability hispanic:", round(prob_hispanic, 4))
    print("Probability asian:", round(prob_asian, 4))
    print("Probability black:", round(prob_black, 4))


    results = []
    std_errors = []
    for attr in sensitive_attribute_vector:
        if attr == 'Woman':
            score_diff = prob_women - prob_men
            # Standard error for difference in means
            group1 = df[df['gender'] == 'women']['value']
            group2 = df[df['gender'] == 'men']['value']
            std_error = np.sqrt(group1.var(ddof=1)/len(group1) + group2.var(ddof=1)/len(group2)) if len(group1) > 0 and len(group2) > 0 else None

        elif attr == 'Asian':
            score_diff = prob_asian - prob_white
            group1 = df[df['race'] == 'asian']['value']
            group2 = df[df['race'] == 'white']['value']
            std_error = np.sqrt(group1.var(ddof=1)/len(group1) + group2.var(ddof=1)/len(group2)) if len(group1) > 0 and len(group2) > 0 else None

        elif attr == 'Black':
            score_diff = prob_black - prob_white
            group1 = df[df['race'] == 'black']['value']
            group2 = df[df['race'] == 'white']['value']
            std_error = np.sqrt(group1.var(ddof=1)/len(group1) + group2.var(ddof=1)/len(group2)) if len(group1) > 0 and len(group2) > 0 else None

        elif attr == 'Hispanic':   
            score_diff = prob_hispanic - prob_white
            group1 = df[df['race'] == 'hispanic']['value']
            group2 = df[df['race'] == 'white']['value']
            std_error = np.sqrt(group1.var(ddof=1)/len(group1) + group2.var(ddof=1)/len(group2)) if len(group1) > 0 and len(group2) > 0 else None

        results.append(score_diff)
        std_errors.append(std_error)

    return results, std_errors


def impact_ratio(df, threshold_value, sensitive_attribute_vector):
    # Select exactly the top (1-threshold_value) percent
    # threshold_value=0.75 means we want top 25%
    n_select = int(len(df) * (1 - threshold_value))
    # Sort by value descending and select top n_select
    df_sorted = df.sort_values('value', ascending=False)
    top_indices = df_sorted.head(n_select).index
    df['selected'] = 0
    df.loc[top_indices, 'selected'] = 1
    gender_rates = df.groupby('gender')['selected'].mean()
    race_rates = df.groupby('race')['selected'].mean()

    results = []
    try: 
        for attr in sensitive_attribute_vector:
            if attr == 'Woman':
                ir = float(gender_rates['women']) / float(gender_rates['men'])

            elif attr == 'Asian':
                ir = float(race_rates['asian']) /  float(race_rates['white'])

            elif attr == 'Black':
                ir = float(race_rates['black']) /  float(race_rates['white'])

            elif attr == 'Hispanic':   
                ir = float(race_rates['hispanic']) /  float(race_rates['white'])

            results.append(ir)
    except: 
        return None, None
    return results, None

    
def compute_wasserstein_distance(df, sensitive_attribute_vector):
    df = df.groupby(
        ["role", "prompt", "variable", "race", "gender", "raceBLS", "genderBLS", "name"],
        group_keys=False
    ).apply(lambda x: x.sample(n=10, random_state=42))

    df_filtered = df

    results = []
    for attr in sensitive_attribute_vector:
        if attr == 'Woman':
            # Compute Wasserstein distance between men and women
            values_men = df_filtered[df_filtered['gender'] == 'men']['value']
            values_women = df_filtered[df_filtered['gender'] == 'women']['value']
            wd = wasserstein_distance(values_men, values_women)

        elif attr == 'Asian':

            # Compute Wasserstein distance between white and asian
            values_white = df_filtered[df_filtered['race'] == 'white']['value']
            values_asian = df_filtered[df_filtered['race'] == 'asian']['value']
            wd = wasserstein_distance(values_white, values_asian)

        elif attr == 'Black':
            # Compute Wasserstein distance between white and black
            values_white = df_filtered[df_filtered['race'] == 'white']['value']
            values_black = df_filtered[df_filtered['race'] == 'black']['value']
            wd = wasserstein_distance(values_white, values_black)

        elif attr == 'Hispanic':
            # Compute Wasserstein distance between white and hispanic
            values_white = df_filtered[df_filtered['race'] == 'white']['value']
            values_hispanic = df_filtered[df_filtered['race'] == 'hispanic']['value']
            wd = wasserstein_distance(values_white, values_hispanic)

        results.append(wd)

    return results, None

def compute_regression_coefficients(df, sensitive_attribute_vector, analysis):
    # Drop index column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])


    # Convert 'value' to numeric, coercing bad strings like "NULL" to NaN
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Define levels (to match R's factor levels)
    gender_levels = ["men", "women"]
    race_levels = ["white", "hispanic", "asian", "black"]


    def prepare_subset(df):

        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        subset=df

        subset = subset[
            subset["gender"].isin(gender_levels) &
            subset["race"].isin(race_levels)
        ].copy()
        

        subset = subset.dropna(subset=["raceBLS", "genderBLS", "gender", "race", "role", "name", "value"])


        # make sure calendar of gender only 

        # Factor-like columns (preserve level order like in R)
        subset["gender"] = pd.Categorical(subset["gender"], categories=gender_levels, ordered=True)
        subset["race"] = pd.Categorical(subset["race"], categories=race_levels, ordered=True)

        subset = subset.reset_index(drop=True)
        return subset


    def fit_lme(df_subset):
        # formula = "value ~ gender * race + raceBLS + genderBLS + role"
        # formula = "value ~ gender * race + raceBLS  + role" # LinAlg Error 
        # formula = "value ~ gender * race  + role"
        has_race = "race" in df_subset and df_subset["race"].nunique() > 2


        if analysis == 'original' and has_race:
            formula = "value ~ gender * race  + role"
        else:
            formula = "value ~ gender + role"

        df_subset = df_subset.reset_index(drop=True)
        print(len(df_subset))
        model = mixedlm(formula, data=df_subset, groups=df_subset["name"])

        # model = mixedlm(formula, data=df_subset, groups=df_subset["name"])
        result = model.fit(reml=True, method="nm", maxiter=5000)
        return result
        
    # Interview
    df_interview = prepare_subset(df)
    try: coefs = fit_lme(df_interview)
    except: return None, None, None
    # extract the parameter values and standard errors

    results, std, pvals = [], [], [] 

    for attr in sensitive_attribute_vector:
        if attr == 'Woman':
            results.append(coefs.params["gender[T.women]"])
            std.append(coefs.bse["gender[T.women]"])   
            pvals.append(coefs.pvalues["gender[T.women]"])

        # elif attr == 'Asian':
        #     results.append(coefs.params["gender[T.women]"])
        #     std.append(coefs.bse["gender[T.women]"])    
        #     pvals.append(coefs.pvalues["gender[T.women]"])
        # elif attr == 'Black':
        #     results.append(coefs.params["gender[T.women]"])
        #     std.append(coefs.bse["gender[T.women]"])   
        #     pvals.append(coefs.pvalues["gender[T.women]"]) 
        # elif attr == 'Hispanic':
        #     results.append(coefs.params["gender[T.women]"])
        #     std.append(coefs.bse["gender[T.women]"])   
        #     pvals.append(coefs.pvalues["gender[T.women]"])
    return results, std, pvals

def preprocess_rozado(config):

    '''
    # process the score+decision folder csvs to update the chosen_candidate_first_name column
    # open the csv file, read it into a pandas dataframe, and then update the chosen_candidate_first_name column
    import pandas as pd
    import os
    import time
    import random
    from pydantic import BaseModel


    # Define the main folder and the subdirectories
    base_folder = "experiment_name"
    models = ["gpt-3.5-turbo", "gpt-4o-mini" , "gpt-4o"]

    # Loop through each model folder
    for model in models:
        
        print(f"Processing model: {model}")

        model = model+'.csv'
        file_path = os.path.join(base_folder, model)

        # 1. Read the CSV file for each model
        df = pd.read_csv('score_task_format_'+file_path)

        # go through every row in the dataframe. extract the value at the user_prompt column 
        for index, row in df.iterrows():
            
            # 2. grab the response from the 'model_response' column
            model_response = row['model_response']

            try: 
                cv1_score, cv2_score = model_response.split(', ')
                cv1_score = cv1_score.split('CV 1: ')[1].split('\n')[0].strip()
                # if the cv1_score contains [ ], remove the brackets
                if cv1_score.startswith('[') and cv1_score.endswith(']'):
                    cv1_score = cv1_score[1:-1].strip()

                cv2_score = cv2_score.split('CV 2: ')[1].split('\n')[0].strip()
                if cv2_score.startswith('[') and cv2_score.endswith(']'):
                    cv2_score = cv2_score[1:-1].strip()

                # remove the final period from cv1_score and cv2_score if it exists
                if cv1_score.endswith('.'):
                    cv1_score = cv1_score[:-1].strip()
                if cv2_score.endswith('.'):
                    cv2_score = cv2_score[:-1].strip()

                if cv1_score.startswith('[') and cv1_score.endswith(']'):
                    cv1_score = cv1_score[1:-1].strip()

                if cv2_score.startswith('[') and cv2_score.endswith(']'):
                    cv2_score = cv2_score[1:-1].strip()
                
                # print(cv1_score)
                # print(cv2_score)
            except: 
                cv1_score = '0'
                cv2_score = '0'
                print(f"Error extracting CV scores from {model_response}, setting to 0")

            # compare the two CV scores. if cv1 is greater than cv2, then set the cv_chosen to the value at row['cv1']. else set it to the value at row['cv2']
            try: cv1_score = float(cv1_score)
            except ValueError:
                cv1_score = float(cv1_score[0])
            
            try:  cv2_score = float(cv2_score)
            except ValueError:
                cv2_score = float(cv2_score[0])

            if float(cv1_score) > float(cv2_score):
                cv_chosen = row['cv1']
            else:
                cv_chosen = row['cv2']

            # grab the name from the cv_chosen variable by splitting on "Name: "
            cv_chosen_name = cv_chosen.split('Name: ')[1]
            candidate_first_name = cv_chosen_name.split(" ")[0]
            candidate_first_name = candidate_first_name.strip()

            # store the value at response.choices[0].message.content in the value at row['model_response']
            df.at[index, 'chosen_candidate_first_name'] = candidate_first_name
            # 5. identify if this first name matches the value in male_full_name or female_full_name 
            if candidate_first_name in row['male_full_name']:
                df.at[index, 'chosen_gender'] = 'Male'
            else: 
                df.at[index, 'chosen_gender'] = 'Female'
            print(candidate_first_name)
            print(df.at[index, 'chosen_gender'])

        # Save the updated DataFrame back to the CSV file
        df.to_csv('score_task_format_'+file_path, index=False)
    '''
    return 

def analyze_armstrong(args, config, all_together=False):

    perturbation_results_filepath = 'results/{}_sampling_{}_analysis_{}.csv'.format(config['experimental_framework'], args.sampling, args.analysis)
    
    # read the perturbation_results_filepath 
    if not os.path.exists(perturbation_results_filepath):
        perturbation_results_df = pd.DataFrame(columns=columns)
        perturbation_results_df.to_csv(perturbation_results_filepath, index=False)
    else:
        perturbation_results_df = pd.read_csv(perturbation_results_filepath)


    # save data into results_df

    author = config['experimental_framework']
    model = config['Model']['Model_Name']
    temp = config['Model']['Temperature']
    num_trials = config['Prompt']['Trials_Per_Query']
    name_bundle = config['Name']['Bundle_Name']
    job_bundle = config['Job']['Bundle_Name']
    random_state = config['Random_State']

    # if the row exists with the same Author, Names, Jobs,Num_Trials, and Model value in the perturbation_results_df, and Sensitive_Attribute_Vector = regression_coefficients; check if the value at Result_Vector[0] exists. if it does, then skip the computation and break
    # 1. Filter the DataFrame for the matching row
    mask = (
        (perturbation_results_df['Author'] == author) &
        (perturbation_results_df['Names'] == name_bundle) &
        (perturbation_results_df['Jobs'] == job_bundle) &
        (perturbation_results_df['Num_Trials'] == num_trials) &
        (perturbation_results_df['Model'] == model) &
        (perturbation_results_df['Metric'] == 'regression_coefficients')
    )

    # 2. Check if the row exists
    if not perturbation_results_df.loc[mask].empty:
        # 3. Get the value of Result_Vector for that row (taking the first match)
        existing_result = perturbation_results_df.loc[mask, 'Result_Vector'].iloc[0]

        # 4. Check if the value "exists" (is not None, NaN, or empty depending on your data type)
        # Using pd.notna checks for NaN/None. If it's a list, check if len > 0.
        if pd.notna(existing_result):
            # Optional: specific check if it is a list/array and you want index 0
            # if len(existing_result) > 0:
            print(f"Row already exists for {author}, {name_bundle}, {job_bundle}, {num_trials}, {model}")
            return

    if author == 'rozado':
        preprocess_rozado(config)
        
    keys = columns[:6]
    values = [author, name_bundle, job_bundle, num_trials, model, temp, random_state]

    new_row_template = dict(zip(keys, values))

    metrics_to_compute = config['Results']['Metrics_to_Compute']
    if args.name == 'karvonen' or args.name == 'seshadri':
        sensitive_attribute_vector = ['Woman', 'Black']
    else:
        sensitive_attribute_vector = config['Results']['Sensitive_Attribute_Vector'] 

    # read in the data
    # data_filepath = "output_data/{}/model_{}_temp_{}_numtrials_{}_jobs_{}_names_{}_random_state_{}.csv".format(author, model, temp, num_trials, job_bundle, name_bundle, random_state)
    if all_together:
        data_filepath = f"/nlp/scr/nmeist/EvalDims/output_data/armstrong/together/processed_batch_outputs/{model.replace('_', '-')}/name_{name_bundle}_job_{job_bundle}.csv"
    else:
        try: data_filepath = "output_data/{}/model_{}_temp_{}_numtrials_{}_jobresumes_{}_names_{}_random_state_{}.csv".format(author, model, temp, num_trials, job_bundle, name_bundle, random_state)
        except: breakpoint()
  
    # add a row of values to perturbation_results_df
    # breakpoint()

    # READ IN THE OUTPUT DATA AND COMPUTE THE METRICS 
    try: 
        df = pd.read_csv(data_filepath)
    except: 
        print(f"Error reading {data_filepath}")
        return

    prompt_type='interview'

    df = df[
        (df["prompt"] == prompt_type) &
        (df["value"].notnull()) &
        (df["value"] > 0) 
    ].copy()

    if args.sampling == 'downsampling':
        df = downsample_data(df, config)
    
    new_rows_list = []
    pvals=None
    for metric in metrics_to_compute:
        print(f"Computing {metric}...")
        if metric == 'regression_coefficients':
            values, std, pvals = compute_regression_coefficients(df, sensitive_attribute_vector, args.analysis)
            # save a row with the coefficients
            print(data_filepath, pvals)
        
        elif metric == 'wasserstein_distance':
            values, std = compute_wasserstein_distance(df, sensitive_attribute_vector)
            # save a row with the distances

        elif metric == 'score_difference':
            values, std = score_diff(df, sensitive_attribute_vector)

        elif metric == 'ttest':
            values, std, pvals = compute_ttest(df, sensitive_attribute_vector)
            
        elif 'impact_ratio' in metric:
            values, std = impact_ratio(df, float(metric.split('_')[-1]), sensitive_attribute_vector)

        new_row_dict = new_row_template.copy()
        new_row_dict['Metric'] = metric
        new_row_dict['Sensitive_Attribute_Vector'] = sensitive_attribute_vector
        new_row_dict['Result_Vector'] = values
        new_row_dict['Std_Error_Vector'] = std
        new_row_dict['P_Value_Vector'] = pvals
        new_row_dict['Random_State'] = random_state
        new_rows_list.append(new_row_dict)

    # eventually 
    new_rows = pd.DataFrame(new_rows_list)
    # Drop all-NA columns from new_rows to avoid FutureWarning before concatenation
    new_rows = new_rows.dropna(axis=1, how='all')
    # if perturbation_results_filepath doesnt exist, create a csv with the right columns
    


    perturbation_results_df = pd.concat([perturbation_results_df, new_rows], ignore_index=True)
    # save the results to the filepath 
    perturbation_results_df.to_csv(perturbation_results_filepath, index=False)
    return 

def analyze_yin(args, config, all_together=False):

    author = config['experimental_framework']
    model = config['Model']['Model_Name']
    temp = config['Model']['Temperature']
    name_bundle = config['Name']['Bundle_Name']
    job_bundle = config['Job']['Bundle_Name']
    perturbation_results_filepath = 'results/{}_sampling_{}_analysis_{}.csv'.format(config['experimental_framework'], args.sampling, args.analysis)
    
    # Outputs
    fn_ranking_detailed = perturbation_results_filepath.replace('.csv', '_detailed.csv')
    fn_ranking_gender = perturbation_results_filepath


    # read the perturbation_results_filepath 
    if not os.path.exists(fn_ranking_gender):
        results_gender = pd.DataFrame(columns=columns)
        results_gender.to_csv(fn_ranking_gender, index=False)


    # read the perturbation_results_filepath 
    if not os.path.exists(fn_ranking_detailed):
        # Ensure 'columns' variable is defined in your previous scope
        results_detailed = pd.DataFrame(columns=columns) 
        results_detailed.to_csv(fn_ranking_detailed, index=False)

    if all_together:
        model_fullname = MODELS[model]
        model_fullname = model_fullname.split('/')[-1]
    # Inputs
    if all_together:
        fn_gpt4 = f"/nlp/scr/nmeist/EvalDims/output_data/yin/together/processed_batch_outputs/{model_fullname}/temp_1.0_names_{name_bundle}_jobresumes_{job_bundle}/*/*.json"        
    else:
        fn_gpt4 = f"/nlp/scr/nmeist/EvalDims/output_data/{author}/{model.replace('_', '-')}/temp_{temp}_names_{name_bundle}_jobresumes_{job_bundle}/*/*.json"
    
    files = glob.glob(fn_gpt4)

    if all_together:
        jobs = os.listdir(f"/nlp/scr/nmeist/EvalDims/output_data/{author}/together/processed_batch_outputs/{model_fullname}/temp_1.0_names_{name_bundle}_jobresumes_{job_bundle}")
    else:
        jobs = [f for f in os.listdir(f"/nlp/scr/nmeist/EvalDims/output_data/{author}/{model.replace('_', '-')}/temp_{temp}_names_{name_bundle}_jobresumes_{job_bundle}") if not f.startswith('run_')]

    # --- Helper Function for Gender ---
    def get_gender(demo_str):
        """Parses 'Asian_W' to 'W', 'White_M' to 'M', etc."""
        s = str(demo_str).strip()
        if s.endswith('_W'):
            return 'W'
        elif s.endswith('_M'):
            return 'M'
        return 'Unknown'

    # --- Step 1: Pre-load and Parse Data (Optimization) ---
    # We read files once to extract the ranking order, rather than reading them 
    # repeatedly for every job/analysis loop.

    parsed_data = []

    for fn in tqdm(files, desc="Parsing Files"):
        try:
            records = json.load(open(fn))
            sentence = records['choices'][0]['message']['content'].lower()
            context = records['context']
            
            real_order = context['default_order']
            real_order = [_.lower() for _ in real_order]
            demo_order = context['demo_order']
            
            # Determine GPT Order based on text position
            name2len = {}
            for name in real_order:
                # Split sentence by name, take length of first part to find position
                name2len[name] = len(sentence.split(name)[0])
            name2len = dict(sorted(name2len.items(), key=lambda item: item[1]))
            gpt_order = list(name2len.keys())

        
            # Map back to demographics
            name2race = dict(zip(real_order, demo_order))
            gpt_race_order = [name2race.get(_) for _ in gpt_order]
            
            # Store processed record
            parsed_data.append({
                'model': model,
                'job': context['job'],
                'demo_order': demo_order,         # The ground truth list of demos
                'gpt_race_order': gpt_race_order, # The re-ranked list of demos
                'natural_first': demo_order[0]    # The demographic that was naturally 1st
            })
            
        except Exception as e:
            # print(f"Error parsing {fn}: {e}")
            continue


    # --- Step 2: Generate Original Detailed CSV (By Job, Specific Demos) ---
    data_detailed = []

    model_data = parsed_data
    for N_top in range(1, 1+1):
        # Calculate overall top-1 agreement (just for logging purposes)
        topistop = sum(1 for d in model_data if d['gpt_race_order'][0] == d['natural_first'])
        print(f"Model: {model} | Top {N_top} | Natural Order Agreement: {topistop / len(model_data):.4f}")

        for job in jobs:
            # Filter data for this specific job
            job_data = [d for d in model_data if d['job'] == job]
            
            top_og = Counter()
            top_gpt = Counter()
            c = len(job_data)
            
            if c == 0: continue

            for record in job_data:
                top_og.update(record['demo_order'][:N_top])
                top_gpt.update(record['gpt_race_order'][:N_top])
            
            # Create Stats DataFrame
            print(f"Processing Job: {job}")
            df = pd.DataFrame(top_gpt.most_common(), columns=['demo', 'top'])
            df_og = pd.DataFrame(top_og.most_common(), columns=['demo', 'top_og'])            
            df = df.merge(df_og, on='demo', how='outer').fillna(0)

            df['selection_rate'] = df['top'] / c
            df['disparate_impact_ratio'] = df['selection_rate'] / df['selection_rate'].max()
            
            df['job'] = job
            df['model'] = model
            df['rank'] = N_top
            df['name_bundle'] = name_bundle
            df['job_bundle'] = job_bundle
            
            
            data_detailed.extend(df.to_dict(orient='records'))


    # Save Detailed CSV
    if data_detailed:  # Only write if there is data
        results_detailed = pd.DataFrame(data_detailed)
        
        # mode='a' appends to the file
        # header=False prevents writing the column names again in the middle of the file
        results_detailed.to_csv(fn_ranking_detailed, mode='a', index=False, header=False)
        print(f"Appended detailed ranking to {fn_ranking_detailed}")
    else:
        print("No new data to append.")

    # --- Step 3: Generate Gender Aggregate CSV (All Jobs, W vs M) ---
    data_gender = []

    print("\n--- Generating Gender Aggregate Analysis ---")

    model_data = parsed_data
    
    for N_top in range(1, 1+1):
        # No job loop here - we use all model_data
        gender_top_og = Counter()
        gender_top_gpt = Counter()
        c = len(model_data)
        
        for record in model_data:
            # Convert specific demos to gender
            genders_og = [get_gender(d) for d in record['demo_order'][:N_top]]
            genders_gpt = [get_gender(d) for d in record['gpt_race_order'][:N_top]]
            
            gender_top_og.update(genders_og)
            gender_top_gpt.update(genders_gpt)
        
        # Create Stats DataFrame
        df = pd.DataFrame(gender_top_gpt.most_common(), columns=['gender', 'top'])
        df_og = pd.DataFrame(gender_top_og.most_common(), columns=['gender', 'top_og'])
        
        df = df.merge(df_og, on='gender', how='outer').fillna(0)
        
        df['selection_rate'] = df['top'] / c
        df['disparate_impact_ratio'] = df['selection_rate'] / df['selection_rate'].max()
        
        df['job'] = 'ALL' # Aggregated
        df['model'] = model
        df['rank'] = N_top

        df['name_bundle'] = name_bundle
        df['job_bundle'] = job_bundle
        
        print(f"Aggregated Results (Top {N_top}):")
        display(HTML(df.sort_values(by='disparate_impact_ratio', ascending=True).reset_index(drop=1).to_html()))
        
        data_gender.extend(df.to_dict(orient='records'))
        # Save Detailed CSV
    if data_gender:  # Only write if there is data
        results_gender = pd.DataFrame(data_gender)
        
        # mode='a' appends to the file
        # header=False prevents writing the column names again in the middle of the file
        results_gender.to_csv(fn_ranking_gender, mode='a', index=False, header=False)
        print(f"Appended gender ranking to {fn_ranking_gender}")
    else:
        print("No new data to append.")
        

    return 

    

def analyze(args, config, all_together=False):
    if config['experimental_framework'] == 'armstrong':
        analyze_armstrong(args, config, all_together)
    elif config['experimental_framework'] == 'yin':
        analyze_yin(args, config, all_together)
    else:
        raise ValueError(f"experimental_framework {config['experimental_framework']} not supported")
