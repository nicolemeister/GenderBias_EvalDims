"""
Script to generate final_data.csv using GPT-4 (2023-11-06)
Based on pages/1_Injection.py with Streamlit dependencies removed.
"""

import pandas as pd
import os
from dotenv import load_dotenv
from util.injection import process_scores_multiple
from util.model import GPTAgent
from util.prompt import PROMPT_TEMPLATE

# Load environment variables
load_dotenv()

# Configuration
CONFIG = {
    # Model settings
    "api_key": os.getenv("OPENAI_API_KEY", ""),
    "endpoint_url": os.getenv("AZURE_ENDPOINT", "https://api.openai.com/v1"),
    "deployment_name": "gpt-4-1106-preview",  # GPT-4 (2023-11-06)
    "api_version": "2024-02-15-preview",

    # Generation parameters
    "temperature": 0.0,
    "max_tokens": 300,

    # Experiment settings
    "group_name": "Gender",
    "privilege_label": "Male",
    "protect_label": "Female",
    "num_run": 1,
    "sample_size": None,  # None means use all data
    "occupation": None,  # None means process all occupations

    # File paths
    "input_file": "resume_subsampled.csv",
    "output_file": "final_data_gpt4.csv",

    # Prompt template
    "prompt_template": PROMPT_TEMPLATE,
}


def load_data(filepath: str, occupation: str = None, sample_size: int = None) -> pd.DataFrame:
    """Load and optionally filter/sample the resume data."""
    df = pd.read_csv(filepath)

    if occupation:
        df = df[df["Occupation"] == occupation]

    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)

    return df


def process_occupation(df: pd.DataFrame, agent: GPTAgent, config: dict) -> pd.DataFrame:
    """Process a single occupation's data through the model."""
    parameters = {
        "temperature": config["temperature"],
        "max_tokens": config["max_tokens"]
    }

    processed_df = process_scores_multiple(
        df=df,
        num_run=config["num_run"],
        parameters=parameters,
        privilege_label=config["privilege_label"],
        protect_label=config["protect_label"],
        agent=agent,
        group_name=config["group_name"],
        occupation=df["Occupation"].iloc[0] if len(df) > 0 else "Unknown",
        template=config["prompt_template"]
    )

    return processed_df


def main():
    """Main function to generate final_data.csv"""
    print("=" * 60)
    print("Resume Scoring Generation Script - GPT-4 (2023-11-06)")
    print("=" * 60)

    # Validate API key
    if not CONFIG["api_key"]:
        raise ValueError(
            "API key not found. Set OPENAI_API_KEY environment variable or update CONFIG."
        )

    # Initialize the GPT-4 agent
    print(f"\nInitializing GPT-4 agent...")
    print(f"  Model: {CONFIG['deployment_name']}")
    print(f"  Endpoint: {CONFIG['endpoint_url']}")

    agent = GPTAgent(
        api_key=CONFIG["api_key"],
        azure_endpoint=CONFIG["endpoint_url"],
        deployment_name=CONFIG["deployment_name"],
        api_version=CONFIG["api_version"]
    )

    # Load data
    print(f"\nLoading data from: {CONFIG['input_file']}")
    df = load_data(
        CONFIG["input_file"],
        occupation=CONFIG["occupation"],
        sample_size=CONFIG["sample_size"]
    )
    print(f"  Loaded {len(df)} entries")

    if CONFIG["occupation"]:
        print(f"  Filtered to occupation: {CONFIG['occupation']}")

    # Get unique occupations
    occupations = df["Occupation"].unique()
    print(f"  Occupations: {list(occupations)}")

    # Process data
    print(f"\nProcessing settings:")
    print(f"  Group: {CONFIG['group_name']}")
    print(f"  Privilege label: {CONFIG['privilege_label']}")
    print(f"  Protect label: {CONFIG['protect_label']}")
    print(f"  Number of runs: {CONFIG['num_run']}")
    print(f"  Temperature: {CONFIG['temperature']}")
    print(f"  Max tokens: {CONFIG['max_tokens']}")

    all_results = []

    for occupation in occupations:
        print(f"\n{'='*40}")
        print(f"Processing occupation: {occupation}")
        print(f"{'='*40}")

        occupation_df = df[df["Occupation"] == occupation].copy()
        print(f"  Entries: {len(occupation_df)}")

        processed_df = process_occupation(occupation_df, agent, CONFIG)
        all_results.append(processed_df)

    # Combine all results
    final_df = pd.concat(all_results, ignore_index=True)

    # Save results
    print(f"\nSaving results to: {CONFIG['output_file']}")
    final_df.to_csv(CONFIG["output_file"], index=False)
    print(f"  Saved {len(final_df)} entries")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    for occupation in final_df["Occupation"].unique():
        occ_df = final_df[final_df["Occupation"] == occupation]
        print(f"\n{occupation}:")
        print(f"  Privilege Avg Score: {occ_df['Privilege_Avg_Score'].mean():.2f}")
        print(f"  Protect Avg Score: {occ_df['Protect_Avg_Score'].mean():.2f}")
        print(f"  Neutral Avg Score: {occ_df['Neutral_Avg_Score'].mean():.2f}")

    print("\n" + "=" * 60)
    print("Generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
