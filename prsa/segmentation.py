import numpy as np

def segment_signal(time_series: np.ndarray, idx_anchors: np.ndarray, L: int) -> np.ndarray:
    """
    Segments a time_series into equal-length windows based on detected anchors
    and a length of surroundings.
    
    Parameters:
    time_series (np.ndarray): The input time series.
    idx_anchors (np.ndarray): Detected anchors.
    L (int): Length of surroundings, L must be two times less than the length 
    of the time_series.

    Returns:
    np.ndarray: Array of segmented parts of the time series.
    """
    if L <= 0:
        raise ValueError("Segment duration must be a positive number.")
    if 2 * L > len(time_series):
        raise ValueError("Double segment duration must be shorter than a length of the time_series.")

    max_padding = L

    padded_time_series = np.pad(time_series, (max_padding, max_padding), 'constant', constant_values=np.nan)

    windows = []

    for idx in idx_anchors:
        adjusted_idx = idx + max_padding

        start_idx = adjusted_idx - L
        end_idx = adjusted_idx + L
        segment = padded_time_series[start_idx:end_idx]

        windows.append(segment)
    
    return np.array(windows)

def segment_intervals(time_intervals: np.ndarray, idx_anchors: np.ndarray, L: int) -> np.ndarray:
    """
    Segments a series of time intervals into windows based on detected anchors
    and a length of surroundings.
    
    Parameters:
    time_intervals (np.ndarray): Series of time intervals.
    idx_anchors (np.ndarray): Detected anchors.
    L (int): Length of surroundings.

    Returns:
    np.ndarray: Array of segmented parts of the time intervals.
    """
    max_padding = L
    time_intervals_float = time_intervals.astype(float)
    padded_intervals = np.pad(time_intervals_float, (max_padding, max_padding), 'constant', constant_values=np.nan)

    windows = []

    for idx in idx_anchors:
        adjusted_idx = idx + max_padding
        start_idx = max(0, adjusted_idx - L)
        end_idx = min(len(padded_intervals), adjusted_idx + L)
        segment = padded_intervals[start_idx:end_idx]

        if len(segment) < 2 * L:
            segment = np.pad(segment, (0, 2 * L - len(segment)), 'constant', constant_values=np.nan)

        windows.append(segment)

    return np.array(windows)

def calculate_anchor_neighbours(time_series: np.ndarray, anchor_points: np.ndarray, T: int) -> list[np.ndarray]:
    """
    Extracts segments of the time_series around each anchor point. Each segment consists of
    the portion of the time_series from 'T' indices before to 'T' indices after the anchor point.

    This function is useful for analyzing specific parts of a time series surrounding points
    of interest (anchor points), which could be peaks, inflection points, or other significant
    features in the time_series.

    Parameters:
    time_series (np.ndarray): The input time series, a one-dimensional array of time series values.
    anchor_points (np.ndarray): An array of indices in 'time_series' that represent the anchor points.
    T (int): The number of indices before and after each anchor point to include in the segment.

    Returns:
    list of np.ndarray: A list where each element is a segment of the time series surrounding an anchor point.
                            Each segment is an array of length '2*T + 1', capturing the time series from 'T' indices
                            before to 'T' indices after the anchor point.
    """
    anchor_neighbours = []
    for i in anchor_points:
        start_idx = max(i - T, 0)
        end_idx = min(i + T + 1, len(time_series))

        segment = time_series[start_idx:end_idx]
        anchor_neighbours.append(segment)

    return anchor_neighbours