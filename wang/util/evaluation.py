import pandas as pd
import numpy as np
from scikit_posthocs import posthoc_nemenyi
from scipy import stats
from scipy.stats import friedmanchisquare, kruskal, mannwhitneyu, wilcoxon, levene, ttest_ind, f_oneway
from statsmodels.stats.multicomp import MultiComparison
from scipy.stats import spearmanr, pearsonr, kendalltau, entropy
from scipy.spatial.distance import jensenshannon
from scipy.stats import ttest_ind, friedmanchisquare, rankdata, ttest_rel
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import ttest_1samp


def test_statistic_variance_ratio(x, y):
    return np.var(x, ddof=1) / np.var(y, ddof=1)


def test_statistic_mean_difference(x, y):
    return np.mean(x) - np.mean(y)


def permutation_test_variance(x, y, num_permutations=100000):
    T_obs = test_statistic_variance_ratio(x, y)
    pooled_data = np.concatenate([x, y])
    n_A = len(x)

    perm_test_stats = [T_obs]
    for _ in range(num_permutations):
        np.random.shuffle(pooled_data)
        perm_A = pooled_data[:n_A]
        perm_B = pooled_data[n_A:]
        perm_test_stats.append(test_statistic_variance_ratio(perm_A, perm_B))

    perm_test_stats = np.array(perm_test_stats)
    p_value = np.mean(np.abs(perm_test_stats) >= np.abs(T_obs))

    return T_obs, p_value


def permutation_test_mean(x, y, num_permutations=100000):
    T_obs = test_statistic_mean_difference(x, y)
    pooled_data = np.concatenate([x, y])
    n_A = len(x)

    perm_test_stats = [T_obs]
    for _ in range(num_permutations):
        np.random.shuffle(pooled_data)
        perm_A = pooled_data[:n_A]
        perm_B = pooled_data[n_A:]
        perm_test_stats.append(test_statistic_mean_difference(perm_A, perm_B))

    perm_test_stats = np.array(perm_test_stats)
    p_value = np.mean(np.abs(perm_test_stats) >= np.abs(T_obs))

    return T_obs, p_value

def calculate_impact_ratio(selection_rates):
    """Calculate the impact ratio for each category."""
    most_selected_rate = max(selection_rates.values())
    impact_ratios = {category: rate / most_selected_rate for category, rate in selection_rates.items()}
    return impact_ratios

def statistical_parity_difference(y_true, y_pred=None, reference_group='Privilege'):
    selection_rates = y_pred if y_pred is not None else y_true
    reference_rate = selection_rates[reference_group]
    spd = {category: rate - reference_rate for category, rate in selection_rates.items()}
    return spd



def statistical_parity_difference(selection_rates):
    """Calculate statistical parity difference."""
    most_selected_rate = max(selection_rates.values())
    spd = {category: rate - most_selected_rate for category, rate in selection_rates.items()}
    return spd

def calculate_four_fifths_rule(impact_ratios):
    """Calculate whether each category meets the four-fifths rule."""
    adverse_impact = {category: (ratio < 0.8) for category, ratio in impact_ratios.items()}
    return adverse_impact

def statistical_tests(data):
    # Add ranks for each score within each row
    # ranks = data[['Privilege_Avg_Score', 'Protect_Avg_Score', 'Neutral_Avg_Score']].rank(axis=1, ascending=True)
    #
    # data['Privilege_Rank'] = ranks['Privilege_Avg_Score']
    # data['Protect_Rank'] = ranks['Protect_Avg_Score']
    # data['Neutral_Rank'] = ranks['Neutral_Avg_Score']

    """Perform various statistical tests to evaluate potential biases."""
    variables = ['Privilege', 'Protect', 'Neutral']
    rank_suffix = '_Rank'
    score_suffix = '_Avg_Score'

    # Calculate average ranks and scores
    rank_columns = [v + rank_suffix for v in variables]
    average_ranks = data[rank_columns].mean()
    average_scores = data[[v + score_suffix for v in variables]].mean()

    # Statistical tests setup
    rank_data = [data[col] for col in rank_columns]
    pairs = [('Privilege', 'Protect'), ('Protect', 'Neutral'), ('Privilege', 'Neutral')]

    pairwise_results = {'Wilcoxon Test': {}}

    # Pairwise Wilcoxon Signed-Rank Test
    for var1, var2 in pairs:

        pair_rank_score = f'{var1}{rank_suffix} vs {var2}{rank_suffix}'
        pair_score_score = f'{var1}{score_suffix} vs {var2}{score_suffix}'

        if len(data) > 20:
            wilcoxon_stat_rank, wilcoxon_p_rank = wilcoxon(data[f'{var1}{rank_suffix}'], data[f'{var2}{rank_suffix}'])
            wilcoxon_stat_score, wilcoxon_p_score = wilcoxon(data[f'{var1}{score_suffix}'], data[f'{var2}{score_suffix}'])
        else:
            wilcoxon_stat_rank, wilcoxon_p_rank = np.nan, "Sample size too small for Wilcoxon test."
            wilcoxon_stat_score, wilcoxon_p_score = np.nan, "Sample size too small for Wilcoxon test."
        pairwise_results['Wilcoxon Test'][pair_rank_score] = {"Statistic": wilcoxon_stat_rank, "p-value": wilcoxon_p_rank}
        pairwise_results['Wilcoxon Test'][pair_score_score] = {"Statistic": wilcoxon_stat_score, "p-value": wilcoxon_p_score}

    # Calculate variances for ranks
    variances = {col: data[col].var() for col in rank_columns}
    pairwise_variances = {
        'Privilege_Rank vs Protect_Rank': variances['Privilege_Rank'] > variances['Protect_Rank'],
        'Privilege_Rank vs Neutral_Rank': variances['Privilege_Rank'] > variances['Neutral_Rank'],
        'Protect_Rank vs Neutral_Rank': variances['Protect_Rank'] > variances['Neutral_Rank']
    }

    # Bias metrics calculations
    selection_rates_Avg_Score = {v: data[f'{v}{score_suffix}'].mean() for v in variables}
    selection_rates_rank = {v: data[f'{v}{rank_suffix}'].mean() for v in variables}

    impact_ratios_Avg_Score = calculate_impact_ratio(selection_rates_Avg_Score)
    spd_result_Avg_Score = statistical_parity_difference(selection_rates_Avg_Score)
    adverse_impact_Avg_Score = calculate_four_fifths_rule(impact_ratios_Avg_Score)

    impact_ratios_rank = calculate_impact_ratio(selection_rates_rank)
    spd_result_rank = statistical_parity_difference(selection_rates_rank)
    adverse_impact_rank = calculate_four_fifths_rule(impact_ratios_rank)

    # Friedman test
    friedman_stat, friedman_p = friedmanchisquare(*rank_data)
    rank_matrix_transposed = np.transpose(data[rank_columns].values)
    posthoc_results = posthoc_nemenyi(rank_matrix_transposed)

    # Perform permutation tests for variances
    T_priv_prot_var_rank, p_priv_prot_var_rank = permutation_test_variance(data['Privilege_Rank'], data['Protect_Rank'])
    T_neut_prot_var_rank, p_neut_prot_var_rank = permutation_test_variance(data['Neutral_Rank'], data['Protect_Rank'])
    T_neut_priv_var_rank, p_neut_priv_var_rank = permutation_test_variance(data['Neutral_Rank'], data['Privilege_Rank'])

    # Perform permutation tests for variances by using rank data
    T_priv_prot_var_score, p_priv_prot_var_score = permutation_test_variance(data['Privilege_Avg_Score'], data['Protect_Avg_Score'])
    T_neut_prot_var_score, p_neut_prot_var_score = permutation_test_variance(data['Neutral_Avg_Score'], data['Protect_Avg_Score'])
    T_neut_priv_var_score, p_neut_priv_var_score = permutation_test_variance(data['Neutral_Avg_Score'], data['Privilege_Avg_Score'])

    # Perform permutation tests for means
    T_priv_prot_mean_rank, p_priv_prot_mean_rank = permutation_test_mean(data['Privilege_Rank'], data['Protect_Rank'])
    T_neut_prot_mean_rank, p_neut_prot_mean_rank = permutation_test_mean(data['Neutral_Rank'], data['Protect_Rank'])
    T_neut_priv_mean_rank, p_neut_priv_mean_rank = permutation_test_mean(data['Neutral_Rank'], data['Privilege_Rank'])

    # Perform permutation tests for means by using rank data
    T_priv_prot_mean_score, p_priv_prot_mean_score = permutation_test_mean(data['Privilege_Avg_Score'], data['Protect_Avg_Score'])
    T_neut_prot_mean_score, p_neut_prot_mean_score = permutation_test_mean(data['Neutral_Avg_Score'], data['Protect_Avg_Score'])
    T_neut_priv_mean_score, p_neut_priv_mean_score = permutation_test_mean(data['Neutral_Avg_Score'], data['Privilege_Avg_Score'])

    permutation_results = {
        "Permutation Tests for Variances (score)": {
            "Privilege vs. Protect": {"Statistic": T_priv_prot_var_score, "p-value": p_priv_prot_var_score},
            "Neutral vs. Protect": {"Statistic": T_neut_prot_var_score, "p-value": p_neut_prot_var_score},
            "Neutral vs. Privilege": {"Statistic": T_neut_priv_var_score, "p-value": p_neut_priv_var_score}
        },
        "Permutation Tests for Means (score)": {
            "Privilege vs. Protect": {"Statistic": T_priv_prot_mean_score, "p-value": p_priv_prot_mean_score},
            "Neutral vs. Protect": {"Statistic": T_neut_prot_mean_score, "p-value": p_neut_prot_mean_score},
            "Neutral vs. Privilege": {"Statistic": T_neut_priv_mean_score, "p-value": p_neut_priv_mean_score}
        },
        "Permutation Tests for Variances (rank)": {
            "Privilege vs. Protect": {"Statistic": T_priv_prot_var_rank, "p-value": p_priv_prot_var_rank},
            "Neutral vs. Protect": {"Statistic": T_neut_prot_var_rank, "p-value": p_neut_prot_var_rank},
            "Neutral vs. Privilege": {"Statistic": T_neut_priv_var_rank, "p-value": p_neut_priv_var_rank}
        },
        "Permutation Tests for Means (rank)": {
            "Privilege vs. Protect": {"Statistic": T_priv_prot_mean_rank, "p-value": p_priv_prot_mean_rank},
            "Neutral vs. Protect": {"Statistic": T_neut_prot_mean_rank, "p-value": p_neut_prot_mean_rank},
            "Neutral vs. Privilege": {"Statistic": T_neut_priv_mean_rank, "p-value": p_neut_priv_mean_rank}
        }
    }

    results = {
        "Average Ranks": average_ranks.to_dict(),
        "Average Scores": average_scores.to_dict(),
        "Friedman Test": {
            "Statistic": friedman_stat,
            "p-value": friedman_p,
            "Post-hoc": posthoc_results
        },
        **pairwise_results,
        #"Levene's Test for Equality of Variances": levene_results,
        "Pairwise Comparisons of Variances": pairwise_variances,
        "Statistical Parity Difference": {
            "Avg_Score": spd_result_Avg_Score,
            "Rank": spd_result_rank
        },
        "Disparate Impact Ratios": {
            "Avg_Score": impact_ratios_Avg_Score,
            "Rank": impact_ratios_rank
        },
        "Four-Fifths Rule": {
            "Avg_Score": adverse_impact_Avg_Score,
            "Rank": adverse_impact_rank
        },
        **permutation_results
    }

    return results


#
# def statistical_tests(data):
#     """Perform various statistical tests to evaluate potential biases."""
#     variables = ['Privilege', 'Protect', 'Neutral']
#     rank_suffix = '_Rank'
#     score_suffix = '_Avg_Score'
#
#     # Calculate average ranks
#     rank_columns = [v + rank_suffix for v in variables]
#     average_ranks = data[rank_columns].mean()
#     average_scores = data[[v + score_suffix for v in variables]].mean()
#
#     # Statistical tests
#     rank_data = [data[col] for col in rank_columns]
#
#     # Pairwise tests
#     pairs = [
#         ('Privilege', 'Protect'),
#         ('Protect', 'Neutral'),
#         ('Privilege', 'Neutral')
#     ]
#
#     pairwise_results = {
#         'Wilcoxon Test': {}
#     }
#
#     for (var1, var2) in pairs:
#         pair_name_score = f'{var1}{score_suffix} vs {var2}{score_suffix}'
#         pair_rank_score = f'{var1}{rank_suffix} vs {var2}{rank_suffix}'
#
#         # Wilcoxon Signed-Rank Test
#         if len(data) > 20:
#             wilcoxon_stat, wilcoxon_p = wilcoxon(data[f'{var1}{rank_suffix}'], data[f'{var2}{rank_suffix}'])
#         else:
#             wilcoxon_stat, wilcoxon_p = np.nan, "Sample size too small for Wilcoxon test."
#         pairwise_results['Wilcoxon Test'][pair_rank_score] = {"Statistic": wilcoxon_stat, "p-value": wilcoxon_p}
#
#     # Levene's Test for Equality of Variances
#     levene_results = {}
#     levene_privilege_protect = levene(data['Privilege_Rank'], data['Protect_Rank'])
#     levene_privilege_neutral = levene(data['Privilege_Rank'], data['Neutral_Rank'])
#     levene_protect_neutral = levene(data['Protect_Rank'], data['Neutral_Rank'])
#
#     levene_results['Privilege vs Protect'] = {"Statistic": levene_privilege_protect.statistic,
#                                               "p-value": levene_privilege_protect.pvalue}
#     levene_results['Privilege vs Neutral'] = {"Statistic": levene_privilege_neutral.statistic,
#                                               "p-value": levene_privilege_neutral.pvalue}
#     levene_results['Protect vs Neutral'] = {"Statistic": levene_protect_neutral.statistic,
#                                             "p-value": levene_protect_neutral.pvalue}
#
#     # Calculate variances for ranks
#     variances = {col: data[col].var() for col in rank_columns}
#     pairwise_variances = {
#         'Privilege_Rank vs Protect_Rank': variances['Privilege_Rank'] > variances['Protect_Rank'],
#         'Privilege_Rank vs Neutral_Rank': variances['Privilege_Rank'] > variances['Neutral_Rank'],
#         'Protect_Rank vs Neutral_Rank': variances['Protect_Rank'] > variances['Neutral_Rank']
#     }
#
#     selection_rates_Avg_Score = {
#         'Privilege': data['Privilege_Avg_Score'].mean(),
#         'Protect': data['Protect_Avg_Score'].mean(),
#         'Neutral': data['Neutral_Avg_Score'].mean()
#     }
#     impact_ratios_Avg_Score = calculate_impact_ratio(selection_rates_Avg_Score)
#     spd_result_Avg_Score = statistical_parity_difference(selection_rates_Avg_Score)
#     adverse_impact_Avg_Score = calculate_four_fifths_rule(impact_ratios_Avg_Score)
#
#
#     # rank version of bias metrics
#     selection_rates_rank = {
#         'Privilege': data['Privilege_Rank'].mean(),
#         'Protect': data['Protect_Rank'].mean(),
#         'Neutral': data['Neutral_Rank'].mean()
#     }
#     impact_ratios_rank = calculate_impact_ratio(selection_rates_rank)
#     spd_result_rank = statistical_parity_difference(selection_rates_rank)
#     adverse_impact_rank = calculate_four_fifths_rule(impact_ratios_rank)
#
#
#     # Friedman test
#     friedman_stat, friedman_p = friedmanchisquare(*rank_data)
#
#     rank_matrix = data[rank_columns].values
#     rank_matrix_transposed = np.transpose(rank_matrix)
#     posthoc_results = posthoc_nemenyi(rank_matrix_transposed)
#     #posthoc_results = posthoc_friedman(data, variables, rank_suffix)
#
#
#
#     results = {
#         "Average Ranks": average_ranks.to_dict(),
#         "Average Scores": average_scores.to_dict(),
#         "Friedman Test": {
#             "Statistic": friedman_stat,
#             "p-value": friedman_p,
#             "Post-hoc": posthoc_results
#         },
#         **pairwise_results,
#         "Levene's Test for Equality of Variances": levene_results,
#         "Pairwise Comparisons of Variances": pairwise_variances,
#         "Statistical Parity Difference": {
#             "Avg_Score": spd_result_Avg_Score,
#             "Rank": spd_result_rank
#         },
#         "Disparate Impact Ratios": {
#             "Avg_Score": impact_ratios_Avg_Score,
#             "Rank": impact_ratios_rank
#         },
#         "Four-Fifths Rule": {
#             "Avg_Score": adverse_impact_Avg_Score,
#             "Rank": adverse_impact_rank
#         }
#     }
#
#     return results


# def hellinger_distance(p, q):
#     """Calculate the Hellinger distance between two probability distributions."""
#     return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))
#
#
# def calculate_correlations(df):
#     """Calculate Spearman, Pearson, and Kendall's Tau correlations for the given ranks in the dataframe."""
#     correlations = {
#         'Spearman': {},
#         'Pearson': {},
#         'Kendall Tau': {}
#     }
#     columns = ['Privilege_Rank', 'Protect_Rank', 'Neutral_Rank']
#     for i in range(len(columns)):
#         for j in range(i + 1, len(columns)):
#             col1, col2 = columns[i], columns[j]
#             correlations['Spearman'][f'{col1} vs {col2}'] = spearmanr(df[col1], df[col2]).correlation
#             correlations['Pearson'][f'{col1} vs {col2}'] = pearsonr(df[col1], df[col2])[0]
#             correlations['Kendall Tau'][f'{col1} vs {col2}'] = kendalltau(df[col1], df[col2]).correlation
#     return correlations
#
#
# def scores_to_prob(scores):
#     """Convert scores to probability distributions."""
#     value_counts = scores.value_counts()
#     probabilities = value_counts / value_counts.sum()
#     full_prob = np.zeros(int(scores.max()) + 1)
#     full_prob[value_counts.index.astype(int)] = probabilities
#     return full_prob


# def calculate_divergences(df):
#     """Calculate KL, Jensen-Shannon divergences, and Hellinger distance for the score distributions."""
#     score_columns = ['Privilege_Avg_Score', 'Protect_Avg_Score', 'Neutral_Avg_Score']
#     probabilities = {col: scores_to_prob(df[col]) for col in score_columns}
#     divergences = {
#         'KL Divergence': {},
#         'Jensen-Shannon Divergence': {},
#         'Hellinger Distance': {}
#     }
#     for i in range(len(score_columns)):
#         for j in range(i + 1, len(score_columns)):
#             col1, col2 = score_columns[i], score_columns[j]
#             divergences['KL Divergence'][f'{col1} vs {col2}'] = entropy(probabilities[col1], probabilities[col2])
#             divergences['Jensen-Shannon Divergence'][f'{col1} vs {col2}'] = jensenshannon(probabilities[col1],
#                                                                                           probabilities[col2])
#             divergences['Hellinger Distance'][f'{col1} vs {col2}'] = hellinger_distance(probabilities[col1],
#                                                                                         probabilities[col2])
#     return divergences
