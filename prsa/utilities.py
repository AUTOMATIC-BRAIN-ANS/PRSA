import numpy as np
import pandas as pd

def convert_to_time_intervals(anchor_indices: np.ndarray) -> np.ndarray:
    """
    Converts anchor point indices to time intervals.

    Parameters:
    anchor_indices (np.ndarray): Indices of anchor points in the signal.

    Returns:
    np.ndarray: Time intervals of each anchor point in milliseconds, with dtype adjusted.
    """
    return (anchor_indices * 5).astype(np.float64)

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values in the DataFrame."""
    df_copy = df.copy()
    df_copy['Values'].ffill(inplace=True)
    return df_copy

def timedomain(rr: np.ndarray) -> dict[str, float]:
    """
    Calculate various time domain metrics from RR interval data.

    Parameters:
    rr (numpy.ndarray): Array of RR intervals (ms)

    Returns:
    Dict[str, float]: A dictionary containing the calculated time domain metrics.
    """
    results = {}

    # Calculate Heart Rate (HR) from RR intervals
    hr = 60000 / rr  # Conversion from RR intervals to heart rate in beats/min

    # Calculate and store various time domain metrics
    results['Mean RR (ms)'] = np.mean(rr)  # Average of RR intervals
    results['STD RR/SDNN (ms)'] = np.std(rr)  # Standard deviation of RR intervals (SDNN)
    results['Mean HR (Kubios\' style) (beats/min)'] = 60000 / np.mean(rr)  # Average HR, reciprocal of mean RR
    results['Mean HR (beats/min)'] = np.mean(hr)  # Average of calculated HR values
    results['STD HR (beats/min)'] = np.std(hr)  # Standard deviation of HR values
    results['Min HR (beats/min)'] = np.min(hr)  # Minimum HR value
    results['Max HR (beats/min)'] = np.max(hr)  # Maximum HR value
    results['RMSSD (ms)'] = np.sqrt(np.mean(np.square(np.diff(rr))))  # Root mean square of successive RR interval differences
    results['NN50'] = np.sum(np.abs(np.diff(rr)) > 50)  # Count of successive RR interval differences greater than 50 ms
    results['pNN50 (%)'] = 100 * results['NN50'] / len(rr)  # Percentage of NN50 divided by total number of RR intervals

    return results