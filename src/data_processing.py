import pandas as pd
import numpy as np

def load_data(file_path):
    """
    Load dataset from the specified file path.
    """
    return pd.read_csv(file_path)

def clean_data(df):
    """
    Perform data cleaning:
    - Log duplicate records and remove if any.
    - Log missing values.
    """
    duplicates = df.duplicated().sum()
    missing_summary = df.isnull().sum()
    
    # We clean duplicate records if any exist
    if duplicates > 0:
        df = df.drop_duplicates()
        
    return df, duplicates, missing_summary

def calculate_composite_scores(df):
    """
    Ensure composite scores are correctly calculated.
    In case they are missing, compute them from item responses.
    """
    ei_cols = [f"EI{i}" for i in range(1, 11)]
    trust_cols = [f"TRUST{i}" for i in range(1, 7)]
    empathy_cols = [f"EMPATHY{i}" for i in range(1, 6)]
    reliance_cols = [f"RELIANCE{i}" for i in range(1, 5)]
    comfort_cols = [f"COMFORT{i}" for i in range(1, 4)]
    
    # Calculate means along rows (excluding NaN values)
    df["ei_score"] = df[ei_cols].mean(axis=1)
    df["trust_score"] = df[trust_cols].mean(axis=1)
    df["perceived_empathy_score"] = df[empathy_cols].mean(axis=1)
    df["emotional_reliance_score"] = df[reliance_cols].mean(axis=1)
    df["comfort_score"] = df[comfort_cols].mean(axis=1)
    
    return df

def calculate_cronbach_alpha(items_df):
    """
    Calculate Cronbach's Alpha coefficient for internal consistency.
    Formula: alpha = (k / (k - 1)) * (1 - sum(s_i^2) / s_t^2)
    where k is the number of items, s_i^2 is the variance of item i,
    and s_t^2 is the variance of the total score.
    """
    # Drop rows with any missing item values for reliability calculation
    cleaned_df = items_df.dropna()
    k = cleaned_df.shape[1]
    if k <= 1:
        return 0.0
    
    item_variances = cleaned_df.var(axis=0, ddof=1)
    total_score = cleaned_df.sum(axis=1)
    total_variance = total_score.var(ddof=1)
    
    if total_variance == 0:
        return 0.0
        
    alpha = (k / (k - 1)) * (1 - item_variances.sum() / total_variance)
    return alpha
