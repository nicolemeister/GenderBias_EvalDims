import json
import os
from pathlib import Path

import pandas as pd
import re

MODELS = {
    # "meta-llama-3.1-8b-instruct-turbo": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    # "gemma-3-12b": "google/gemma-3-12b-it",
    # "gemma-3-27b": "google/gemma-3-27b-it",
    "meta-llama-3.3-70b-instruct-turbo": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    # "mistral-small-24b": "mistralai/Mistral-Small-24B-Instruct-2501",
    # "mistral-7b-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
}

names = ['armstrong', 'rozado', 'wen', 'wang', 'seshadri', 'karvonen', 'zollo', 'yin']
jobs = ['armstrong', 'rozado', 'wen', 'wang', 'karvonen', 'zollo', 'yin']

batch_outputs_dir = '/nlp/scr/nmeist/EvalDims/output_data/armstrong/together/batch_outputs'

for model_key in MODELS:
    model = MODELS[model_key].split("/")[-1]
    for name in names:
        for job in jobs:

            if name in ['armstrong', 'rozado', 'wen', 'wang']:
                continue

            if name =='seshadri' and job in ['armstrong', 'rozado', 'wen']:
                continue
                
            try: 
                output_filepath=f"/nlp/scr/nmeist/EvalDims/output_data/armstrong/together/batch_outputs/{model}/name_{name}_job_{job}.jsonl"
                with open(output_filepath, "r", encoding="utf-8") as f:
                    pass 
            except Exception as e:
                breakpoint()
            try: 
                gpt_4o_example_filepath = f'/nlp/scr/nmeist/EvalDims/output_data/armstrong/model_gpt-4o-2024-11-20_temp_1.0_numtrials_10_jobresumes_{job}_names_{name}_random_state_42.csv'
                gpt_4o_example_data = pd.read_csv(gpt_4o_example_filepath)
            except Exception as e:
                gpt_4o_example_filepath = f'/nlp/scr/nmeist/EvalDims/output_data/armstrong/model_gpt-5-nano-2025-08-07_temp_1.0_numtrials_10_jobresumes_{job}_names_{name}_random_state_42.csv'
                gpt_4o_example_data = pd.read_csv(gpt_4o_example_filepath)
            # make a copy of the dataframe: will put the together outputs in the value column 
            df = gpt_4o_example_data.copy()

                
            with open(output_filepath, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    # get the content value
                    # remove the newline characters
                    model_response = obj["response"]["body"]["choices"][0]["message"]["content"]
                    if model_response is not None:
                        temp = re.findall(r'\d+', model_response)
                        value = list(map(int, temp))[0] if temp else None
                    else:
                        value = None    

                    df.loc[i, "value"] = value
            # save the dataframe
            # create the directory if it doesn't exist

            os.makedirs(f"/nlp/scr/nmeist/EvalDims/output_data/armstrong/together/processed_batch_outputs/{model}", exist_ok=True)
            df.to_csv(f"/nlp/scr/nmeist/EvalDims/output_data/armstrong/together/processed_batch_outputs/{model}/name_{name}_job_{job}.csv", index=False)