import json
import re
import time
import json_repair
import pandas as pd
from tqdm import tqdm


def create_summary(group_name, label, occupation, row, template):
    """Generate a dynamic summary for scoring the applicant, excluding the group feature.
       The occupation parameter allows customization of the job position.
    """

    resume_info = row['Cleaned_Resume']
    # resume_info = resume_info[:int(len(resume_info) * proportion)]

    info = f"{group_name}: {label};" if label else ''

    summary = template.format(
        role=row['Role'],
        counterfactual_info=info,
        resume_info=resume_info
    )

    return summary


def invoke_retry(prompt, agent, parameters, string_input=False):
    attempts = 0
    delay = 2  # Initial delay in seconds
    max_attempts = 5  # Maximum number of retry attempts

    while attempts < max_attempts:
        try:
            score_text = agent.invoke(prompt, **parameters)
            #print(f"Prompt: {prompt}")
            # print(f"Score text: {score_text}")
            # print("=============================================================")
            if string_input:
                return score_text
            try:
                score_json = json.loads(score_text)
            except json.JSONDecodeError:
                try:
                    score_json = json.loads(
                        json_repair.repair_json(score_text, skip_json_loads=True, return_objects=False))
                except json.JSONDecodeError:
                    raise Exception("Failed to decode JSON response even after repair attempt.")
            # score = re.search(r'\d+', score_text)
            # return int(score.group()) if score else -1
            #print(f"Score JSON: {score_json}")
            return int(score_json['Score'])

        except Exception as e:
            print(f"Attempt {attempts + 1} failed: {e}")
            time.sleep(delay)
            delay *= 2  # Exponential increase of the delay
            attempts += 1

    return -1
    # raise Exception("Failed to complete the API call after maximum retry attempts.")


def calculate_avg_score(score_list):
    if isinstance(score_list, list) and score_list:
        valid_scores = [score for score in score_list if score is not None]
        if valid_scores:
            avg_score = sum(valid_scores) / len(valid_scores)
            return avg_score
    return None


def process_scores_multiple(df, num_run, parameters, privilege_label, protect_label, agent, group_name, occupation
                            , template):
    print(f"Processing {len(df)} entries with {num_run} runs each.")
    """ Process entries and compute scores concurrently, with progress updates. """
    scores = {key: [[] for _ in range(len(df))] for key in ['Privilege', 'Protect', 'Neutral']}

    for run in tqdm(range(num_run), desc="Processing runs", unit="run"):
        for index, (idx, row) in tqdm(enumerate(df.iterrows()), total=len(df), desc="Processing entries", unit="entry"):

            for key, label in zip(['Privilege', 'Protect', 'Neutral'], [privilege_label, protect_label, False]):
                prompt_normal = create_summary(group_name, label, occupation, row, template)

                # print(f"Run {run + 1} - Entry {index + 1} - {key}")
                # print("=============================================================")
                result_normal = invoke_retry(prompt_normal, agent, parameters)
                scores[key][index].append(result_normal)

    #print(f"Scores: {scores}")

    # Ensure all scores are lists and calculate average scores
    for category in ['Privilege', 'Protect', 'Neutral']:
        # Ensure the scores are lists and check before assignment
        series_data = [lst if isinstance(lst, list) else [lst] for lst in scores[category]]
        df[f'{category}_Scores'] = series_data

        # Calculate the average score with additional debug info

        df[f'{category}_Avg_Score'] = df[f'{category}_Scores'].apply(calculate_avg_score)

    # Add ranks for each score within each row
    ranks = df[['Privilege_Avg_Score', 'Protect_Avg_Score', 'Neutral_Avg_Score']].rank(axis=1, ascending=False)

    df['Privilege_Rank'] = ranks['Privilege_Avg_Score']
    df['Protect_Rank'] = ranks['Protect_Avg_Score']
    df['Neutral_Rank'] = ranks['Neutral_Avg_Score']

    return df

