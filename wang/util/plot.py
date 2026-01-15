import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

def create_score_plot(df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index, y=df['Privilege_Avg_Score'],
        mode='lines+markers', name='Privilege',
        text=df['Role'], hoverinfo='text+y'
    ))

    fig.add_trace(go.Scatter(
        x=df.index, y=df['Protect_Avg_Score'],
        mode='lines+markers', name='Protection',
        text=df['Role'], hoverinfo='text+y'
    ))

    fig.add_trace(go.Scatter(
        x=df.index, y=df['Neutral_Avg_Score'],
        mode='lines+markers', name='Neutral',
        text=df['Role'], hoverinfo='text+y'
    ))

    fig.update_layout(
        title=f'Scores of Resumes',
        xaxis_title='Resume Index',
        yaxis_title='Score',
        legend_title='Score Type',
        hovermode='closest'
    )

    return fig


def create_rank_plots(df):
    fig = go.Figure()

    # Add traces for ranks
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Privilege_Rank'],
        mode='lines+markers', name='Privilege',
        text=df['Role'], hoverinfo='text+y'
    ))

    fig.add_trace(go.Scatter(
        x=df.index, y=df['Protect_Rank'],
        mode='lines+markers', name='Protection',
        text=df['Role'], hoverinfo='text+y'
    ))

    fig.add_trace(go.Scatter(
        x=df.index, y=df['Neutral_Rank'],
        mode='lines+markers', name='Neutral',
        text=df['Role'], hoverinfo='text+y'
    ))

    # Update layout
    fig.update_layout(
        title='Ranks of Scores',
        xaxis_title='Resume Index',
        yaxis_title='Rank',
        legend_title='Rank Type',
        hovermode='closest'
    )

    return fig


def create_correlation_heatmaps(df):
    scores_df = df[['Privilege_Avg_Score', 'Protect_Avg_Score', 'Neutral_Avg_Score']]
    ranks_df = df[['Privilege_Rank', 'Protect_Rank', 'Neutral_Rank']]

    # Pearson correlation
    scores_corr_pearson = scores_df.corr(method='pearson')
    ranks_corr_pearson = ranks_df.corr(method='pearson')

    # Spearman correlation
    scores_corr_spearman = scores_df.corr(method='spearman')
    ranks_corr_spearman = ranks_df.corr(method='spearman')

    # Kendall Tau correlation
    scores_corr_kendall = scores_df.corr(method='kendall')
    ranks_corr_kendall = ranks_df.corr(method='kendall')

    # Plotting the heatmaps separately
    heatmaps = {
        'Scores Pearson Correlation': scores_corr_pearson,
        'Ranks Pearson Correlation': ranks_corr_pearson,
        'Scores Spearman Correlation': scores_corr_spearman,
        'Ranks Spearman Correlation': ranks_corr_spearman,
        'Scores Kendall Correlation': scores_corr_kendall,
        'Ranks Kendall Correlation': ranks_corr_kendall
    }

    figs = {}
    for title, corr_matrix in heatmaps.items():
        fig = px.imshow(corr_matrix, text_auto=True, title=title)
        figs[title] = fig

    return figs


def point_to_line_distance(point, A, B):
    """Calculate the distance from a point to a line defined by two points A and B."""
    line_vec = B - A
    point_vec = point - A
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    point_vec_scaled = point_vec / line_len
    t = np.dot(line_unitvec, point_vec_scaled)
    nearest = line_vec * t
    dist = np.linalg.norm(nearest - point_vec)
    return dist


def calculate_distances(data, point_A, point_B):
    distances = data.apply(lambda row: point_to_line_distance(
        np.array([row['Privilege_Avg_Score'], row['Protect_Avg_Score'], row['Neutral_Avg_Score']]),
        point_A, point_B), axis=1)
    return distances


def create_3d_plot(data):
    # Define the ideal line (from point A to point B)
    point_A = np.array([0, 0, 0])
    point_B = np.array([10, 10, 10])

    # Calculate distances
    distances = calculate_distances(data, point_A, point_B)
    data['Distance_to_Ideal'] = distances

    # Label points that perfectly match the ideal line (distance close to 0)
    tolerance = 1e-6
    data['Perfect_Match'] = data['Distance_to_Ideal'].apply(lambda x: 'Yes' if x < tolerance else 'No')

    # Create a 3D scatter plot of the scores
    fig_3d = px.scatter_3d(data, x='Privilege_Avg_Score', y='Protect_Avg_Score', z='Neutral_Avg_Score',
                           color='Distance_to_Ideal', symbol='Perfect_Match',
                           hover_data={
                               'Occupation': True,
                               'Role': True,
                               'Privilege_Avg_Score': True,
                               'Protect_Avg_Score': True,
                               'Neutral_Avg_Score': True,
                               'Distance_to_Ideal': True,
                               'Perfect_Match': True
                           },
                           title='Occupation and Role Clusters based on Scores with Distance to Ideal Line')

    # Add ideal line where Neutral = Protect = Privilege
    ideal_line = go.Scatter3d(x=[0, 10], y=[0, 10], z=[0, 10], mode='lines', name='Ideal Line',
                              line=dict(color='green', dash='dash'))
    fig_3d.add_trace(ideal_line)

    return fig_3d