PATH = 'D:/prsa-sonata' # Where the repository was cloned
import sys
sys.path.append(PATH)

import numpy as np

from prsa.segmentation import segment_signal
from prsa.signal_preprocessing import normalize_time_series

def perform_prsa(time_series: np.ndarray, anchors: np.ndarray, segment_length: int, normalization: bool = False) -> list[np.ndarray, np.ndarray]:
    """
    Performs Phase-Rectified Signal Averaging (PRSA) on a given time series.

    Parameters:
    time_series (np.ndarray): The input time series.
    anchors (np.ndarray): Indices of anchor points in the time series.
    segment_length (int): Length of the segments to be averaged around each anchor point.
    normalization (bool): Whether to normalize the time series before PRSA.

    Returns:
    Tuple[np.ndarray, np.ndarray]: Tuple containing the segments array and the PRSA-averaged time series.
    """
    if normalization:
        time_series = normalize_time_series(time_series)
    
    if segment_length <= 0 or segment_length > len(time_series):
        raise ValueError("Segment length must be a positive integer and less than the length of the time series.")
    
    segments = segment_signal(time_series, anchors, segment_length // 2)
    
    segments_array = np.vstack(segments)
    prsa_output = np.nanmean(segments_array, axis=0)

    return segments_array, prsa_output

def perform_rr_based_prsa(time_series: np.ndarray, anchors: np.ndarray, T: int, normalization: bool = False) -> list[np.ndarray, np.ndarray]:
    """
    Performs Phase-Rectified Signal Averaging (PRSA) on time series based on given anchor points.

    Parameters:
    time_series (np.ndarray): Array of time series data.
    anchors (np.ndarray): Indices of anchor points in the time series.
    T (int): Half the length of the segments to be averaged around each anchor point.
    normalization (bool): Whether to normalize the time series before PRSA.

    Returns:
    Tuple[np.ndarray, np.ndarray]: Tuple containing the segments array and the PRSA-averaged time series.
    """
    if normalization:
        time_series = normalize_time_series(time_series)
    
    if 2*T > len(time_series):
        raise ValueError("Segment length (2L) must be less than the length of the time series.")

    segments = []
    for anchor in anchors:
        start = max(anchor - T, 0)
        end = min(anchor + T + 1, len(time_series))
        segment = time_series[start:end]
        if len(segment) == 2*T + 1:
            segments.append(segment)

    if not segments:
        return np.array([]), np.array([])

    segments_array = np.vstack(segments)
    prsa_output = np.nanmean(segments_array, axis=0)

    return segments_array, prsa_output

def perform_rr_based_prsa_2(time_series: np.ndarray, anchors: np.ndarray, T: int, normalization: bool = False) -> np.ndarray:
    """
    Performs Phase-Rectified Signal Averaging (PRSA) on time series based on given anchor points.

    Parameters:
    time_series (np.ndarray): Array of time series data.
    anchors (np.ndarray): Indices of anchor points in the time series.
    T (int): Half the length of the segments to be averaged around each anchor point.
    normalization (bool): Whether to normalize the time series before PRSA.

    Returns:
    np.ndarray: The PRSA-averaged time series.
    """
    if normalization:
        time_series = normalize_time_series(time_series)
    
    if 2*T > len(time_series):
        raise ValueError("Segment length (2T) must be less than the length of the time series.")

    X_k = np.zeros(2 * T + 1)

    for k in range(-T, T + 1):
        values_at_k = []
        for anchor in anchors:
            if 0 <= anchor + k < len(time_series):
                values_at_k.append(time_series[anchor + k])
        
        if values_at_k:
            X_k[k + T] = np.mean(values_at_k)

    return np.array(X_k)
