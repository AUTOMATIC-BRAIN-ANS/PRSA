import numpy as np

"""
Following formulas based on:
Campana LM, Owens RL, Clifford GD, Pittman SD, Malhotra A.
Phase-rectified signal averaging as a sensitive index of autonomic changes with aging. 
J Appl Physiol (1985). 
2010;108(6):1668-1673. doi:10.1152/japplphysiol.00013.2010
"""

def detect_anchors_from_rr_acc(time_series: np.ndarray, T: int, threshold: float = None) -> np.ndarray:
    """
    Identify anchor points in the time series where there is an acceleration.

    Parameters:
    - time_series (np.ndarray): The RR interval data.
    - T (int): The number of points to exclude from the start and end of the array.
    - threshold (float, optional): Threshold for change relative to the previous interval.

    Returns:
    - np.ndarray: The indices of the detected anchor points.
    """
    idxs = np.arange(T, len(time_series) - T)

    if threshold is not None:
        anchor_points = [idx for idx in idxs if time_series[idx] > time_series[idx - 1] and 
                         abs(time_series[idx] - time_series[idx - 1]) / time_series[idx - 1] < threshold]
    else:
        anchor_points = [idx for idx in idxs if time_series[idx] > time_series[idx - 1]]
    
    return np.array(anchor_points)

def detect_anchors_from_rr_dc(time_series: np.ndarray, T: int, threshold: float = None) -> np.ndarray:
    """
    Identify points in the time series where there is a deceleration.

    Parameters:
    - time_series (np.ndarray): The RR interval data.
    - T (int): The number of points to exclude from the start and end of the array.
    - threshold (float, optional): Threshold for change relative to the previous interval.

    Returns:
    - np.ndarray: The indices of the detected anchor points.
    """
    idxs = np.arange(T, len(time_series) - T)

    if threshold is not None:
        anchor_points = [idx for idx in idxs if time_series[idx] < time_series[idx - 1] and 
                         abs(time_series[idx] - time_series[idx - 1]) / time_series[idx - 1] < threshold]
    else:
        anchor_points = [idx for idx in idxs if time_series[idx] < time_series[idx - 1]]
    
    return np.array(anchor_points)

def detect_anchors_acc(time_series: np.ndarray) -> np.ndarray:
    """
    Identify points in the time series where there is an acceleration.
    The anchor points correspond to increases in the time series.

    xi > xi-1, where xi (time_series), i = 1,..., N

    Parameters:
    - time_series (np.ndarray): The time series.

    Returns:
    - np.ndarray: The indices of the detected anchor points.
    """
    anchor_points = [idx for idx in range(len(time_series)) if time_series[idx] > time_series[idx - 1]]
    
    return np.array(anchor_points)

def detect_anchors_dc(time_series: np.ndarray) -> np.ndarray:
    """
    Identify points in the time series where there is a deceleration.
    The anchor points correspond to decreases in the time series.

    xi < xi-1, where xi (time_series), i = 1,..., N

    Parameters:
    - time_series (np.ndarray): The time series.

    Returns:
    - np.ndarray: The indices of the detected anchor points.
    """
    anchor_points = [idx for idx in range(len(time_series)) if time_series[idx] < time_series[idx - 1]]

    return np.array(anchor_points)

"""
Following formulas based on:
Lobmaier, Silvia & Huhn, Evelyn & Steinburg, S. & Müller, Alexander & Schuster, Tibor & Ortiz,
Javier & Schmidt, Georg & Schneider, Karl. (2012). 
Phase-rectified signal averaging as a new method for surveillance of growth restricted fetuses. 
The journal of maternal-fetal & neonatal medicine : the official journal of the European Association of Perinatal Medicine, 
the Federation of Asia and Oceania Perinatal Societies, the International Society of Perinatal Obstetricians. 25. 10.3109/14767058.2012.696163.

Kantelhardt JW, Bauer A, Schumann AY, et al. 
Phase-rectified signal averaging for the detection of quasi-periodicities and the 
prediction of cardiovascular risk. Chaos. 
2007;17(1):015112. doi:10.1063/1.2430636
"""

def detect_anchors_threshold_acc(time_series: np.ndarray, threshold: float) -> np.ndarray:
    """
    Identify points in the time series where there is an acceleration, checking against a threshold.
    The anchor points correspond to increases in the time series that are not too large.

    Parameters:
    - time_series (np.ndarray): The time series.
    - threshold (float): The maximum percentage of change between two values acceptable as an acceleration.

    Returns:
    - np.ndarray: The indices of the detected anchor points.
    """
    anchor_points = [idx for idx in range(len(time_series)) if time_series[idx] > time_series[idx - 1]
                     and abs(time_series[idx] - time_series[idx - 1]) / time_series[idx - 1] < threshold]
    
    return np.array(anchor_points)

def detect_anchors_threshold_dc(time_series: np.ndarray, threshold: float) -> np.ndarray:
    """
    Identify points in the time series where there is a deceleration, checking against a threshold.
    The anchor points correspond to decreases in the time series that are not too large.

    Parameters:
    - time_series (np.ndarray): The time series.
    - threshold (float): The maximum percentage of change between two values acceptable as a deceleration.

    Returns:
    - np.ndarray: The indices of the detected anchor points.
    """
    anchor_points = [idx for idx in range(len(time_series)) if time_series[idx] < time_series[idx - 1]
                     and abs(time_series[idx] - time_series[idx - 1]) / time_series[idx - 1] < threshold]
    
    return np.array(anchor_points)

def detect_anchors_mean_acc(time_series: np.ndarray, T: int) -> np.ndarray:
    """
    Identify acceleration anchor points by comparing averages of T values before and after each point.

    Parameters:
    - time_series (np.ndarray): The time series.
    - T (int): The number of values to consider for the average before and after each point.

    Returns:
    - np.ndarray: The indices of the detected anchor points.
    """
    anchor_points = []
    for idx in range(T, len(time_series) - T):
        mean_before = np.mean(time_series[idx - T:idx])
        mean_after = np.mean(time_series[idx:idx + T])
        if mean_after > mean_before:
            anchor_points.append(idx)
    return np.array(anchor_points)

def detect_anchors_mean_dc(time_series: np.ndarray, T: int) -> np.ndarray:
    """
    Identify deceleration anchor points by comparing averages of T values before and after each point.

    Parameters:
    - time_series (np.ndarray): The time series.
    - T (int): The number of values to consider for the average before and after each point.

    Returns:
    - np.ndarray: The indices of the detected anchor points.
    """
    anchor_points = []
    for idx in range(T, len(time_series) - T):
        mean_before = np.mean(time_series[idx - T:idx])
        mean_after = np.mean(time_series[idx:idx + T])
        if mean_after < mean_before:
            anchor_points.append(idx)
    return np.array(anchor_points)
