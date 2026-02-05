MODELS = {
    "mistral-small-24b": "mistralai/Mistral-Small-24B-Instruct-2501",
    "meta-llama-3.1-8b-instruct-turbo": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "meta-llama-3.3-70b-instruct-turbo": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "mistral-7b-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
}

MODELS_GPT = {
    # "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4o-2024-11-20": "gpt-4o-2024-11-20",
    "gpt-5-nano-2025-08-07": "gpt-5-nano-2025-08-07",
}

# NAMES = ['armstrong']
# JOBS = ['armstrong']


NAMES = ['armstrong', 'rozado', 'wen', 'wang', 'seshadri', 'karvonen', 'zollo', 'yin', 'gaeb', 'lippens']
JOBS = ['armstrong', 'rozado', 'wen', 'wang', 'karvonen', 'zollo', 'yin']

colors = ["purple", "darkblue", "green", "orange", "brown", "pink", "gray", "olive", "cyan", "magenta"]
linestyles = ['--', '-.', ':']
metric_to_plot = {'selection_rate': 'Selection Rate', 'score_difference': 'Score Difference', 'disparate_impact_ratio': 'Disparate Impact Ratio', 'regression_coefficients': 'Regression Coefficient'}