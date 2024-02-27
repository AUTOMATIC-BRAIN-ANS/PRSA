import numpy as np
from scipy.stats import skew, kurtosis

def statistical_analysis_prsa(prsa_output: np.ndarray) -> dict[str, float]:
    """
    Quantifies the PRSA output using various metrics.

    Parameters:
    prsa_output (np.ndarray): The PRSA-averaged signal.

    Returns:
    dict: A dictionary with various quantification metrics, all values adjusted to float dtype.
    """
    metrics = {
        'mean': float(np.mean(prsa_output)),
        'variance': float(np.var(prsa_output)),
        'skewness': float(skew(prsa_output)),
        'kurtosis': float(kurtosis(prsa_output)),
        'max': float(np.max(prsa_output)),
        'min': float(np.min(prsa_output)),
        'range': float(np.ptp(prsa_output)),  # Peak-to-peak range
        'rms': float(np.sqrt(np.mean(np.square(prsa_output))))
    }

    return metrics

