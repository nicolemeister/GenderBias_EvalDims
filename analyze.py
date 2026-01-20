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
from utils.variables import MODELS, NAMES, JOBS, MODELS_GPT

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

    author = args.author
    base_config_path = os.path.join('configs', f'base_{author}.yaml')
    if args.all:
        for model in MODELS_GPT.keys():
            for name in NAMES:
                for job in JOBS:
                    config = yaml.load(open(base_config_path), Loader=yaml.FullLoader)
                    config['Model']['Model_Name'] = model
                    config['Name']['Bundle_Name'] = name
                    config['Job']['Bundle_Name'] = job
                    config['experimental_framework'] = author
                    print(f"Analyzing {model} {name} {job}")
                    analyze_results.analyze(args, config)
                    
    if args.all_together:
        for model in MODELS.keys():
            for name in NAMES:
                for job in JOBS:
                    config = yaml.load(open(base_config_path), Loader=yaml.FullLoader)
                    config['Model']['Model_Name'] = model
                    config['Name']['Bundle_Name'] = name
                    config['Job']['Bundle_Name'] = job
                    config['experimental_framework'] = author
                    print(f"Analyzing {model} {name} {job}")
                    analyze_results.analyze(args, config, all_together=True)

    if not args.all:
        # config = merge_config(args.config_path, args.overlay_config_path)
        config = yaml.load(open(args.config_path), Loader=yaml.FullLoader)
        config['Name']['Bundle_Name'] = args.name
        config['Job']['Bundle_Name'] = args.job
        config['experimental_framework'] = author
        analyze_results.analyze(args, config)

if __name__ == "__main__":
    main()