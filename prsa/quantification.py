import numpy as np

"""
Source for the following formula:
Chen T, Feng G, Heiselman C, Quirk JG, Djurić PM. IMPROVING PHASE-RECTIFIED SIGNAL 
AVERAGING FOR FETAL HEART RATE ANALYSIS. Proc IEEE Int Conf Acoust Speech Signal Process. 
2022;2022:10.1109/icassp43922.2022.9747860. doi:10.1109/icassp43922.2022.9747860
"""


def calculate_capacity_baumana(prsa_output: np.ndarray, L: int, s: int) -> float:
    """
    Calculate the Acceleration Capacity (AC) of a signal using phase-rectified signal averaging.

    The formula for AC is given by:
    AC = (1 / (2s)) * Σ(x_AC[i]) from i = L+1 to L+s - (1 / (2s)) * Σ(x_AC[i]) from i = L-s+1 to L
    where:
    - x_AC[i] is the phase-rectified signal for acceleration capacity at point i,
    - L is the anchor point around which the window is considered,
    - s is the parameter for summarizing the phase-rectified curves (assumed to be even).

    Parameters:
    prsa_output (np.ndarray): The phase-rectified signal.
    L (int): The anchor point around which the capacity is calculated.
    s (int): The summarizing parameter for the phase-rectified curves.

    Returns:
    float: The calculated DC or AC value.
    """
    L = L // 2
    minuend = np.sum(prsa_output[L + 1:L + s])
    subtrahend = np.sum(prsa_output[L - s + 1:L])
    capacity = (1 / (2 * s)) * (minuend - subtrahend)

    return capacity


"""
The formula for calculating the AAC is based on the following article:
Lobmaier, Silvia & Huhn, Evelyn & Steinburg, S. & Müller, Alexander & Schuster, Tibor & Ortiz,
Javier & Schmidt, Georg & Schneider, Karl. (2012). 
Phase-rectified signal averaging as a new method for surveillance of growth restricted fetuses. 
The journal of maternal-fetal & neonatal medicine : the official journal of the European Association of Perinatal Medicine, 
the Federation of Asia and Oceania Perinatal Societies, the International Society of Perinatal Obstetricians. 25. 10.3109/14767058.2012.696163. 

However, the ADC is calculated based on the similar mathematical logic. 
"""


def calculate_aac(signal: np.ndarray, X: int) -> float:
    """
    Calculate the Acceleration Capacity (AAC) of a signal.

    The formula for AAC is given by:
    AAC = (1/X) * Σ(signal[i]) from i=0 to X-1 - (1/X) * Σ(signal[i]) from i=-X to -1

    This calculates the difference between the mean of the signal values
    immediately after the anchor point (up to X-1) and the mean of the signal
    values immediately before the anchor point (up to X back from the anchor point).

    Parameters:
    signal (np.ndarray): The input signal, where the anchor point is at index X.
    X (int): The anchor point index in the signal around which AAC is calculated.

    Returns:
    float: The calculated AAC value.
    """
    mean_after = np.mean(signal[X:X + X]) if X + X <= len(signal) else np.nan
    mean_before = np.mean(signal[0:X])

    aac = mean_after - mean_before

    return aac


def calculate_adc(signal: np.ndarray, X: int) -> float:
    """
    Calculate the Deceleration Capacity (ADC) of a signal, assuming a symmetrical definition to AAC.

    The formula for ADC is assumed to be the inverse of AAC:
    ADC = (1/X) * Σ(signal[i]) from i=-X to -1 - (1/X) * Σ(signal[i]) from i=0 to X-1

    This calculates the difference between the mean of the signal values
    immediately before the anchor point (up to X back from the anchor point)
    and the mean of the signal values immediately after the anchor point (up to X-1).

    Parameters:
    signal (np.ndarray): The input signal, where the anchor point is at index X.
    X (int): The anchor point index in the signal around which ADC is calculated.

    Returns:
    float: The calculated ADC value.
    """
    mean_before = np.mean(signal[0:X])
    mean_after = np.mean(signal[X:X + X]) if X + X <= len(signal) else np.nan

    adc = mean_before - mean_after

    return adc


"""
The next function is based on:
Campana LM, Owens RL, Clifford GD, Pittman SD, Malhotra A.
Phase-rectified signal averaging as a sensitive index of autonomic changes with aging. 
J Appl Physiol (1985). 
2010;108(6):1668-1673. doi:10.1152/japplphysiol.00013.2010

"""


def calculate_capacity_campana(prsa_output: np.ndarray) -> float:
    """
    Calculate the Deceleration or Acceleration Capacity (DC or AC) at a given anchor point.
    (Haar wavelet analysis at a specific scale of 2)

    The formula for calculating the capacity is:
    DC or AC = [RR(0) + RR(1) - RR(-1) - RR(-2)] / 4

    where:
    - RR(0) is the RR interval at the anchor point (current interval).
    - RR(1) is the next RR interval.
    - RR(-1) and RR(-2) are the two RR intervals immediately preceding the anchor point.

    Parameters:
    prsa_output (np.ndarray): The phase-rectified signal.

    Returns:
    float: The calculated DC or AC value.
    """
    if len(prsa_output) < 4:
        return np.nan

    capacity = (prsa_output[0] + prsa_output[1] - prsa_output[-1] - prsa_output[-2]) / 4

    return capacity
