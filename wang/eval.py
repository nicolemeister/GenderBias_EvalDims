import pandas as pd
import plotly.express as px
import numpy as np
import os
from scikit_posthocs import posthoc_nemenyi
from scipy import stats
from scipy.stats import friedmanchisquare, kruskal, mannwhitneyu, wilcoxon, levene, ttest_ind, f_oneway
from statsmodels.stats.multicomp import MultiComparison
from scipy.stats import spearmanr, pearsonr, kendalltau, entropy
from scipy.spatial.distance import jensenshannon
from scipy.stats import ttest_ind, friedmanchisquare, rankdata, ttest_rel
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import ttest_1samp
from util.plot import create_3d_plot, create_score_plot, create_rank_plots, create_correlation_heatmaps, calculate_distances



# Ensure a directory exists to save the plots
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)


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
    # friedman_stat, friedman_p = friedmanchisquare(*rank_data)
    # rank_matrix_transposed = np.transpose(data[rank_columns].values)
    # posthoc_results = posthoc_nemenyi(rank_matrix_transposed)

    # # Perform permutation tests for variances
    # T_priv_prot_var_rank, p_priv_prot_var_rank = permutation_test_variance(data['Privilege_Rank'], data['Protect_Rank'])
    # T_neut_prot_var_rank, p_neut_prot_var_rank = permutation_test_variance(data['Neutral_Rank'], data['Protect_Rank'])
    # T_neut_priv_var_rank, p_neut_priv_var_rank = permutation_test_variance(data['Neutral_Rank'], data['Privilege_Rank'])

    # # Perform permutation tests for variances by using rank data
    # T_priv_prot_var_score, p_priv_prot_var_score = permutation_test_variance(data['Privilege_Avg_Score'], data['Protect_Avg_Score'])
    # T_neut_prot_var_score, p_neut_prot_var_score = permutation_test_variance(data['Neutral_Avg_Score'], data['Protect_Avg_Score'])
    # T_neut_priv_var_score, p_neut_priv_var_score = permutation_test_variance(data['Neutral_Avg_Score'], data['Privilege_Avg_Score'])

    # # Perform permutation tests for means
    # T_priv_prot_mean_rank, p_priv_prot_mean_rank = permutation_test_mean(data['Privilege_Rank'], data['Protect_Rank'])
    # T_neut_prot_mean_rank, p_neut_prot_mean_rank = permutation_test_mean(data['Neutral_Rank'], data['Protect_Rank'])
    # T_neut_priv_mean_rank, p_neut_priv_mean_rank = permutation_test_mean(data['Neutral_Rank'], data['Privilege_Rank'])

    # # Perform permutation tests for means by using rank data
    # T_priv_prot_mean_score, p_priv_prot_mean_score = permutation_test_mean(data['Privilege_Avg_Score'], data['Protect_Avg_Score'])
    # T_neut_prot_mean_score, p_neut_prot_mean_score = permutation_test_mean(data['Neutral_Avg_Score'], data['Protect_Avg_Score'])
    # T_neut_priv_mean_score, p_neut_priv_mean_score = permutation_test_mean(data['Neutral_Avg_Score'], data['Privilege_Avg_Score'])

    # permutation_results = {
    #     "Permutation Tests for Variances (score)": {
    #         "Privilege vs. Protect": {"Statistic": T_priv_prot_var_score, "p-value": p_priv_prot_var_score},
    #         "Neutral vs. Protect": {"Statistic": T_neut_prot_var_score, "p-value": p_neut_prot_var_score},
    #         "Neutral vs. Privilege": {"Statistic": T_neut_priv_var_score, "p-value": p_neut_priv_var_score}
    #     },
    #     "Permutation Tests for Means (score)": {
    #         "Privilege vs. Protect": {"Statistic": T_priv_prot_mean_score, "p-value": p_priv_prot_mean_score},
    #         "Neutral vs. Protect": {"Statistic": T_neut_prot_mean_score, "p-value": p_neut_prot_mean_score},
    #         "Neutral vs. Privilege": {"Statistic": T_neut_priv_mean_score, "p-value": p_neut_priv_mean_score}
    #     },
    #     "Permutation Tests for Variances (rank)": {
    #         "Privilege vs. Protect": {"Statistic": T_priv_prot_var_rank, "p-value": p_priv_prot_var_rank},
    #         "Neutral vs. Protect": {"Statistic": T_neut_prot_var_rank, "p-value": p_neut_prot_var_rank},
    #         "Neutral vs. Privilege": {"Statistic": T_neut_priv_var_rank, "p-value": p_neut_priv_var_rank}
    #     },
    #     "Permutation Tests for Means (rank)": {
    #         "Privilege vs. Protect": {"Statistic": T_priv_prot_mean_rank, "p-value": p_priv_prot_mean_rank},
    #         "Neutral vs. Protect": {"Statistic": T_neut_prot_mean_rank, "p-value": p_neut_prot_mean_rank},
    #         "Neutral vs. Privilege": {"Statistic": T_neut_priv_mean_rank, "p-value": p_neut_priv_mean_rank}
    #     }
    # }

    results = {
        # "Average Ranks": average_ranks.to_dict(),
        # "Average Scores": average_scores.to_dict(),
        # "Friedman Test": {
        #     "Statistic": friedman_stat,
        #     "p-value": friedman_p,
        #     "Post-hoc": posthoc_results
        # },
        # **pairwise_results,
        # #"Levene's Test for Equality of Variances": levene_results,
        # "Pairwise Comparisons of Variances": pairwise_variances,
        # "Statistical Parity Difference": {
        #     "Avg_Score": spd_result_Avg_Score,
        #     "Rank": spd_result_rank
        # },
        "Disparate Impact Ratios": {
            "Avg_Score": impact_ratios_Avg_Score,
            "Rank": impact_ratios_rank
        },
        # "Four-Fifths Rule": {
        #     "Avg_Score": adverse_impact_Avg_Score,
        #     "Rank": adverse_impact_rank
        # },
        # **permutation_results
    }

    return results


def run_analysis():
    data_filepath = 'final_data.csv'
    
    # Check if data exists before crashing
    if not os.path.exists(data_filepath):
        print(f"Error: {data_filepath} not found.")
        return

    df = pd.read_csv(data_filepath) 

    # --- 1. Statistical Processing ---
    print("Running statistical tests...")
    statistical_results = statistical_tests(df)
    
    # Flatten results
    flat_statistical_results = {f"{key1}": value1 for key1, value1 in statistical_results.items()}
    
    # Combine results 
    results_combined = {**flat_statistical_results} 

    # Create DataFrame
    results_df = pd.DataFrame(list(results_combined.items()), columns=['Metric', 'Value'])

    # OUTPUT: Print to console and save to CSV
    print('\n--- Test Results ---')
    print(results_df)
    results_df.to_csv(f'{output_dir}/statistical_results.csv', index=False)
    print(f"Saved results to {output_dir}/statistical_results.csv")


    # # --- 2. 3D Plot ---
    # # Note: PNGs of 3D plots are static; you can't rotate them. 
    # # You might want to fix the camera angle in create_3d_plot if the default view isn't good.
    # fig_3d = create_3d_plot(df)
    # fig_3d.write_image(f"{output_dir}/3d_plot.png")
    # print("Saved 3d_plot.png")


    # # --- 3. Distance Calculations ---
    # point_A = np.array([0, 0, 0])
    # point_B = np.array([10, 10, 10])
    # distances = calculate_distances(df, point_A, point_B)
    # average_distance = distances.mean()
    
    # print(f'Average distance to the ideal line: {average_distance}')


    # --- 4. Score and Rank Plots ---
    score_fig = create_score_plot(df)
    score_fig.write_image(f"{output_dir}/score_plot.png")
    print("Saved score_plot.png")

    rank_fig = create_rank_plots(df)
    rank_fig.write_image(f"{output_dir}/rank_plot.png")
    print("Saved rank_plot.png")


    # --- 5. Histograms ---
    hist_fig = px.histogram(df.melt(id_vars=['Role'],
                                    value_vars=['Privilege_Avg_Score', 'Protect_Avg_Score',
                                                'Neutral_Avg_Score']),
                            x='value', color='variable', facet_col='variable',
                            title='Distribution of Scores')
    hist_fig.write_image(f"{output_dir}/dist_scores.png")
    print("Saved dist_scores.png")

    hist_rank_fig = px.histogram(
        df.melt(id_vars=['Role'], value_vars=['Privilege_Rank', 'Protect_Rank', 'Neutral_Rank']),
        x='value', color='variable', facet_col='variable', title='Distribution of Ranks')
    hist_rank_fig.write_image(f"{output_dir}/dist_ranks.png")
    print("Saved dist_ranks.png")


    # --- 6. Box Plots ---
    box_fig = px.box(df.melt(id_vars=['Role'], value_vars=['Privilege_Avg_Score', 'Protect_Avg_Score',
                                                            'Neutral_Avg_Score']),
                        x='variable', y='value', color='variable', title='Spread of Scores')
    box_fig.write_image(f"{output_dir}/box_scores.png")
    print("Saved box_scores.png")

    box_rank_fig = px.box(
        df.melt(id_vars=['Role'], value_vars=['Privilege_Rank', 'Protect_Rank', 'Neutral_Rank']),
        x='variable', y='value', color='variable', title='Spread of Ranks')
    box_rank_fig.write_image(f"{output_dir}/box_ranks.png")
    print("Saved box_ranks.png")


    # --- 7. Heatmaps ---
    heatmaps = create_correlation_heatmaps(df)
    for title, fig in heatmaps.items():
        # Clean title for filename (remove spaces/special chars)
        safe_title = "".join([c for c in title if c.isalnum() or c in (' ', '_')]).rstrip().replace(" ", "_")
        fig.write_image(f"{output_dir}/heatmap_{safe_title}.png")
        print(f"Saved heatmap_{safe_title}.png")

if __name__ == "__main__":
    run_analysis()