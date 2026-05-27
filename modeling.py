import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

def fit_regression(df, formula):
    """
    Fit an OLS linear regression model using statsmodels formula api.
    """
    model = smf.ols(formula, data=df).fit()
    return model

def run_mediation(df, x_col, m_col, y_col, covariates=None, n_bootstrap=1000, alpha=0.05, random_seed=42):
    """
    Perform a formal mediation analysis using Baron & Kenny steps,
    Sobel test, and Bootstrap resampling for the indirect effect.
    
    Model:
    X -> M -> Y (controlling for covariates)
    
    Returns:
    A dictionary containing:
    - OLS model fits for Model Total, Model M, and Model Both.
    - Estimates and p-values for total, direct, and indirect effects.
    - Sobel test Z-statistic and p-value.
    - Bootstrapped 95% Confidence Interval for the indirect effect.
    """
    np.random.seed(random_seed)
    
    # Format formula strings
    cov_str = " + " + " + ".join(covariates) if covariates else ""
    formula_total = f"{y_col} ~ {x_col}{cov_str}"
    formula_m = f"{m_col} ~ {x_col}{cov_str}"
    formula_both = f"{y_col} ~ {x_col} + {m_col}{cov_str}"
    
    # 1. Fit OLS models on original data
    model_total = smf.ols(formula_total, data=df).fit()
    model_m = smf.ols(formula_m, data=df).fit()
    model_both = smf.ols(formula_both, data=df).fit()
    
    # 2. Extract path coefficients
    # Total effect c (X -> Y without mediator)
    c = model_total.params[x_col]
    total_p = model_total.pvalues[x_col]
    
    # Path a (X -> M)
    a = model_m.params[x_col]
    path_a_p = model_m.pvalues[x_col]
    
    # Path b (M -> Y controlling for X)
    b = model_both.params[m_col]
    path_b_p = model_both.pvalues[m_col]
    
    # Direct effect c_prime (X -> Y controlling for M)
    c_prime = model_both.params[x_col]
    direct_p = model_both.pvalues[x_col]
    
    # Indirect effect ab
    indirect_ab = a * b
    
    # 3. Sobel Test (Delta Method)
    # Standard errors of a and b
    se_a = model_m.bse[x_col]
    se_b = model_both.bse[m_col]
    
    # Sobel standard error: sqrt(a^2 * se_b^2 + b^2 * se_a^2)
    se_ab = np.sqrt((a ** 2) * (se_b ** 2) + (b ** 2) * (se_a ** 2))
    sobel_z = indirect_ab / se_ab
    # Two-tailed p-value
    sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z)))
    
    # 4. Bootstrap Resampling for Indirect Effect
    indirect_boot = []
    n = len(df)
    for _ in range(n_bootstrap):
        boot_df = df.sample(n=n, replace=True)
        # Re-fit models
        boot_model_m = smf.ols(formula_m, data=boot_df).fit()
        boot_model_both = smf.ols(formula_both, data=boot_df).fit()
        
        boot_a = boot_model_m.params[x_col]
        boot_b = boot_model_both.params[m_col]
        indirect_boot.append(boot_a * boot_b)
        
    indirect_boot = np.array(indirect_boot)
    
    # Calculate confidence interval bounds
    lower_bound = (alpha / 2) * 100
    upper_bound = (1 - alpha / 2) * 100
    ci_lower = np.percentile(indirect_boot, lower_bound)
    ci_upper = np.percentile(indirect_boot, upper_bound)
    
    results = {
        "models": {
            "total_model": model_total,
            "m_model": model_m,
            "both_model": model_both
        },
        "total_effect_c": c,
        "total_p": total_p,
        "path_a": a,
        "path_a_p": path_a_p,
        "path_b": b,
        "path_b_p": path_b_p,
        "direct_effect_c_prime": c_prime,
        "direct_p": direct_p,
        "indirect_effect_ab": indirect_ab,
        "sobel_z": sobel_z,
        "sobel_p": sobel_p,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "bootstrap_samples": indirect_boot
    }
    return results
