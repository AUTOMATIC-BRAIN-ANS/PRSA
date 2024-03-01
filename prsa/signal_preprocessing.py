import numpy as np
from scipy.ndimage import label
from scipy.signal import find_peaks
from scipy.stats import zscore

def standardize_time_series(time_series: np.ndarray) -> np.ndarray:
    """
    Normalize the time series to a range between 0 and 1.

    This function handles NaN values by ignoring them in the normalization process.
    The normalization is done according to the formula:
    normalized_time_series = (time_series - time_series_min) / (time_series_max - time_series_min)

    Parameters:
    time_series (np.ndarray): The input time series, potentially with NaN values.

    Returns:
    np.ndarray: The normalized time series with values between 0 and 1.
    """
    time_series_min = np.nanmin(time_series)
    time_series_max = np.nanmax(time_series)
    normalized_time_series = (time_series - time_series_min) / (time_series_max - time_series_min)
    return normalized_time_series.astype(float)

def detect_peaks(time_series: np.ndarray, filter: np.ndarray, threshold: float = 0.3) -> list[np.ndarray, np.ndarray]:
    """
    Detect peaks in a time series using template matching and thresholding.

    Parameters:
    time_series (np.ndarray): The input time series, potentially with NaN values.
    filter (np.ndarray): The template used for matching within the time series.
    threshold (float): The threshold value for peak detection after normalization.

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple containing an array of detected peak indices and the normalized similarity array.
    """
    time_series = (time_series - np.nanmean(time_series)) / np.nanstd(time_series)
    similarity = np.correlate(time_series, filter, mode="same")
    similarity = similarity / np.max(similarity)

    peaks = np.where(similarity > threshold)[0]
    return peaks, similarity.astype(float)

def group_peaks(peaks: np.ndarray, threshold: int = 5) -> np.ndarray:
    """
    Group nearby peaks within a specified threshold and return the median index of each group.

    Parameters:
    peaks (np.ndarray): An array of peak indices.
    threshold (int): The distance threshold for grouping peaks.

    Returns:
    np.ndarray: An array of the median index for each group of peaks.
    """
    output = np.empty(0)

    peak_groups, num_groups = label(np.diff(peaks) < threshold)

    for i in np.unique(peak_groups)[1:]:
        peak_group = peaks[np.where(peak_groups == i)]
        output = np.append(output, np.median(peak_group))
    return output.astype(int)

def extract_rr_intervals_template_matching(time_series: np.ndarray, sampling_rate: int = 200, threshold: float = 0.45, correction_value: int = 2, correction: bool = False) -> list[np.ndarray, np.ndarray]:
    """
    Calculate RR intervals from an ECG time series using template matching.

    Description.

    Parameters:
    time_series (np.ndarray): The input time series, potentially with NaN values.
    sampling_rate (int): The sampling rate of the ECG signal.
    threshold (float): The threshold value for peak detection in template matching.
    correction_value (int): The Z-score value beyond which RR intervals are considered outliers.
    correction (bool): Flag to apply correction for outlier RR intervals.

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple containing an array of RR intervals and the array of indices of the detected peaks.
    """
    # ABP filter
    t = np.linspace(-1, 1, 12)
    peaks_filter = np.cosh(t)

    peaks, similarity = detect_peaks(time_series, filter=peaks_filter, threshold=threshold)

    grouped_peaks = group_peaks(peaks)
    rr_intervals = np.diff(grouped_peaks)
    rr_intervals = (rr_intervals * 1000) / sampling_rate

    # Error correction
    if correction:
        rr_corrected = rr_intervals.copy()
        rr_corrected[np.abs(zscore(rr_intervals)) > correction_value] = np.median(rr_intervals)
        return rr_corrected.astype(float), grouped_peaks.astype(int)
    else:
        return rr_intervals.astype(float), grouped_peaks.astype(int)

def calculate_rr_intervals(time_series_values: np.ndarray, time_series_times: np.ndarray, height_threshold: float = None, distance_threshold: int = None) -> list[np.ndarray, np.ndarray]:
    """
    Calculate RR intervals from an ECG time series.

    The function identifies R-peaks in the ECG time series using the height and distance thresholds.
    It then calculates the RR intervals as the difference in time between successive R-peaks.

    Parameters:
    time_series_values (np.ndarray): The input time series, potentially with NaN values.
    time_series_times (np.ndarray): Time series indicating the time at which each value in the time series was measured.
    height_threshold (float): The required height of peaks to be considered as R-peaks.
    distance_threshold (int): The required minimum horizontal distance (in number of samples) between successive R-peaks.

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple containing the array of indices of the detected peaks and an array of RR intervals in the same time units as provided in the time series.
    """
    peaks, _ = find_peaks(time_series_values, height=height_threshold, distance=distance_threshold)

    rr_intervals = np.diff(time_series_times[peaks])

    original_time_step_ms = 5
    rr_intervals_ms = rr_intervals * (1 / original_time_step_ms)

    return peaks.astype(int), rr_intervals_ms.astype(float)

def clean_rr_intervals_by_derivative(rr_intervals: np.ndarray, sd_factor: float=2.5) -> np.ndarray:
    """
    Cleans RR intervals by removing points where the derivative is considered an outlier, 
    defined as exceeding a multiple of the standard deviation.
    
    Parameters:
    rr_intervals (np.ndarray): Array of RR intervals.
    sd_factor (float): Number of standard deviations to use for outlier detection.
    
    Returns:
    (np.ndarray): Cleaned array of RR intervals with outliers set to NaN.
    """
    rr_derivative = np.diff(rr_intervals)
    
    mean_derivative = np.mean(rr_derivative)
    sd_derivative = np.std(rr_derivative)
    
    threshold = mean_derivative + sd_factor * sd_derivative
    
    artifacts_indices = np.where(np.abs(rr_derivative) > threshold)[0]
    
    cleaned_rr_intervals = rr_intervals.copy()
    for idx in artifacts_indices:
        cleaned_rr_intervals[idx:idx+2] = np.nan

    valid_indices = ~np.isnan(cleaned_rr_intervals)
    interpolated_rr_intervals = np.interp(np.arange(len(cleaned_rr_intervals)),
                                          np.arange(len(cleaned_rr_intervals))[valid_indices],
                                          cleaned_rr_intervals[valid_indices])
    
    return interpolated_rr_intervals

