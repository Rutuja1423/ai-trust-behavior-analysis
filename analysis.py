import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def run_correlation(df, col1, col2):
    """
    Perform Pearson correlation between two variables.
    """
    r, p = stats.pearsonr(df[col1], df[col2])
    return r, p

def run_t_test(df, group_col, group1, group2, val_col):
    """
    Perform an independent samples t-test between two groups on a given column.
    Uses Welch's t-test (equal_var=False) by default.
    """
    g1_data = df.loc[df[group_col] == group1, val_col]
    g2_data = df.loc[df[group_col] == group2, val_col]
    
    t_stat, p_val = stats.ttest_ind(g1_data, g2_data, equal_var=False)
    
    means = {
        group1: g1_data.mean(),
        group2: g2_data.mean()
    }
    return t_stat, p_val, means

def run_anova(df, group_col, val_col):
    """
    Perform a One-Way ANOVA across all categories in group_col for val_col.
    """
    categories = df[group_col].dropna().unique()
    groups_data = [df.loc[df[group_col] == cat, val_col] for cat in categories]
    
    f_stat, p_val = stats.f_oneway(*groups_data)
    
    means = df.groupby(group_col)[val_col].mean().to_dict()
    return f_stat, p_val, means

def perform_kmeans(df, features, n_clusters=2, random_state=42):
    """
    Scale features and segment users using K-Means clustering.
    Returns:
    - df: DataFrame with 'cluster' labels appended.
    - profiles: Mean values of features for each cluster.
    - sil_score: Silhouette score of the clustering.
    - scaler: The fit StandardScaler.
    - kmeans: The fit KMeans model.
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data)
    
    df_copy = df.copy()
    df_copy["cluster"] = cluster_labels
    
    # Calculate profiles and silhouette score
    profiles = df_copy.groupby("cluster")[features].mean()
    sil_score = silhouette_score(scaled_data, cluster_labels)
    
    return df_copy, profiles, sil_score, scaler, kmeans
