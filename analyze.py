# read in the config file to figure out what to run
import os
import sys
import yaml
from utils import generate_results, analyze_results
import pandas as pd
import argparse

from utils.merge_config import merge_config
from pathlib import Path
import json
from together import Together
from utils.variables import MODELS, NAMES, JOBS

def main():
    # read in the config file path as an argument
    parser = argparse.ArgumentParser(description='Run perturbation experiments.')
    parser.add_argument('--config_path', type=str, default='configs/armstrong/base.yaml', help='Path to the config file.')
   # parser.add_argument('--overlay_config_path', type=str, default='configs/armstrong/overlays/name_armstrong_job_armstrong.yaml', help='Path to the overlay config file.')
    parser.add_argument('--all', action='store_true', help='Analyze all configs.')
    parser.add_argument('--all_together', action='store_true', help='Analyze all together batch results .')
    parser.add_argument('--sampling', type=str, default='downsampling', help='Sampling method to use.') 
    parser.add_argument('--random_state', type=int, default=42, help='Random state for the experiment.') # for sampling 
    parser.add_argument('--analysis', type=str, default='original', help='Analysis to perform.') # original or modified metric 
    parser.add_argument('--author', type=str, default='armstrong', help='Author to analyze.') # all or specific author 
    parser.add_argument('--overlay_type', type=str, default='overlays', help='Type of overlay to use.') # overlays or additional_overlays 
    parser.add_argument('--name', type=str, help='Name of the bundle.')
    parser.add_argument('--job', type=str, help='job of the bundle.')
    # parser.add_argument('--random_state', type=int, default=42, help='Random state for the experiment.')
    args = parser.parse_args()
    
    # config_path = args.config_path
    names = ['armstrong', 'karvonen', 'gaeb', 'lippens', 'rozado', 'wang', 'wen', 'seshadri', 'zollo', 'yin']
    jobs = ['armstrong', 'rozado', 'wen', 'wang', 'karvonen', 'zollo', 'yin']

    author = args.author
    base_config_path = os.path.join('configs', f'base_{author}.yaml')
    if args.all:

        for name in names:
            for job in jobs:
                config = yaml.load(open(base_config_path), Loader=yaml.FullLoader)
                
                config['Name']['Bundle_Name'] = name
                config['Job']['Bundle_Name'] = job

                analyze_results.analyze(args, config)


    if args.all_together:


        METADATA_PATH = Path(f"/nlp/scr/nmeist/EvalDims/output_data/{author}/together/batch_inputs/batch_info.json")

        client = Together(api_key=os.environ["TOGETHER_API_KEY"])

        data = json.loads(METADATA_PATH.read_text())

        for item in data:
            config = yaml.load(open(base_config_path), Loader=yaml.FullLoader)

            batch_id = item["batch_id"]
            # grab the name and job value from parsing the filepath (e.g., '/nlp/scr/nmeist/EvalDims/output_data/armstrong/together/batch_inputs/batch_input_armstrong_armstrong_meta_llama_Meta_Llama_3.1_8B_Instruct_Turbo.jsonl')

            name = item["file_path"].split("/")[-1].split("_")[2]
            job = item["file_path"].split("/")[-1].split("_")[3]
            model = item["model"].split("/")[-1]

            config['Model']['Model_Name'] = model
            config['Name']['Bundle_Name'] = name
            config['Job']['Bundle_Name'] = job 
            print(f"Analyzing {model} {name} {job}")
            analyze_results.analyze(args, config, all_together=True)

    if not args.all:
        # config = merge_config(args.config_path, args.overlay_config_path)
        config = yaml.load(open(args.config_path), Loader=yaml.FullLoader)
        config['Name']['Bundle_Name'] = args.name
        config['Job']['Bundle_Name'] = args.job
        analyze_results.analyze(args, config)

if __name__ == "__main__":
    main()