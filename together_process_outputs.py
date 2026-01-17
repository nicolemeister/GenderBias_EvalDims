import json
import os
from pathlib import Path

import pandas as pd
import random
from tqdm import tqdm
import re

import argparse
from utils.variables import MODELS, NAMES, JOBS



def process_armstrong():
    batch_outputs_dir = '/nlp/scr/nmeist/EvalDims/output_data/armstrong/together/batch_outputs'

    for model_key in MODELS.keys():
        model = MODELS[model_key].split("/")[-1]
        for name in names:
            for job in jobs:
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

def process_yin():
    batch_outputs_dir = '/nlp/scr/nmeist/EvalDims/output_data/yin/together/batch_outputs'
    processed_batch_outputs_dir = '/nlp/scr/nmeist/EvalDims/output_data/yin/together/processed_batch_outputs'

    for model_key in MODELS.keys():
        model = MODELS[model_key].split("/")[-1]
        list_of_files = os.listdir(f"{batch_outputs_dir}/{model}")
        list_of_metadata_files = os.listdir(f"/nlp/scr/nmeist/EvalDims/output_data/yin/together/batch_inputs")
        for name_bundle in NAMES:
            for job_bundle in JOBS:

                ''' 

                TODO: IMPLEMENT THIS LOGIC -- the code below is wrong 
                1. check if the file starts with the name_{name_bundle}_job_{job_bundle} 
                2. if it does, check how many parts there are 
                3. go through the parts: 
                    a. read the jobs that fit into the file 
                    b. create the run_{i}.jso files and save the values 
                '''
                # grab the files in list_of_files that start with the name_{name_bundle}_job_{job_bundle}
                list_of_files_filtered = [file for file in list_of_files if file.startswith(f"name_{name_bundle}_job_{job_bundle}")]
                list_of_files_filtered_metadata = [file for file in list_of_metadata_files if file.startswith(f"metadata_{name_bundle}_{job_bundle}")]

                metadata = {}
                # read in the metadata but have each custom id be a key and the value be the metadata
                for file in list_of_files_filtered_metadata:
                    with open(f"/nlp/scr/nmeist/EvalDims/output_data/yin/together/batch_inputs/{file}", "r", encoding="utf-8") as f:
                        try:
                            data = json.load(f)
                            for obj in data:
                                metadata[obj["custom_id"]] = obj
                        except json.JSONDecodeError as e:
                            print(f"Error reading JSON from {file}: {e}")
                            continue

                jobs = set()

                # go through the list of files and grab the unique jobs in the value at the custom_id
                for file in list_of_files_filtered:
                    with open(f"{batch_outputs_dir}/{model}/{file}", "r", encoding="utf-8") as f:
                        for line in f:
                            obj = json.loads(line)
                            jobs.add(obj["custom_id"].split("_")[1])

                # conver the set to a list 
                jobs = list(jobs)

                for job in jobs:
                    dir_out = (processed_batch_outputs_dir + '/' + model +
                        '/' + 'temp_1.0' + 
                        '_names_' + name_bundle + 
                        '_jobresumes_' + job_bundle + '/' + job + '/')
                    os.makedirs(dir_out, exist_ok=True)

                # read in the meta data file 
                for file in list_of_files_filtered:
                    with open(f"{batch_outputs_dir}/{model}/{file}", "r", encoding="utf-8") as f:
                        for line in f:
                            obj = json.loads(line)
                            custom_id = obj["custom_id"]
                            records = {}
                            records['choices'] = []
                            records = {'choices': [{'message': {'content': obj['response']['body']['choices'][0]['message']['content']}}]}
                            records['context'] = {
                                'job': metadata[custom_id]["job"],
                                'default_order': metadata[custom_id]["default_order_names"],
                                'demo_order': metadata[custom_id]["demo_order"],
                            }
                            job = metadata[custom_id]["job"]
                            i = custom_id.split("_")[2]
                            # to do: grab i 
                            dir_out = (processed_batch_outputs_dir + '/' + model +
                                '/' + 'temp_1.0' + 
                                '_names_' + name_bundle + 
                                '_jobresumes_' + job_bundle + '/' + job + '/')
                            os.makedirs(dir_out, exist_ok=True)
                            fn_out = os.path.join(dir_out, f"run_{i}.json")
                            with open(fn_out, "w", encoding="utf-8") as f:
                                f.write(json.dumps(records))


    return 


def __main__():

    parser = argparse.ArgumentParser()
    parser.add_argument("--author", type=str, default="armstrong")
    args = parser.parse_args()

    if args.author == 'armstrong':
        process_armstrong()
    elif args.author == 'yin':
        process_yin()

    return 

if __name__ == "__main__":
    __main__()