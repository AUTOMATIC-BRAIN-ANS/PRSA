"""
Check if your backend works without indicating TkAgg as a backend for matplotlib.
If it doesnt uncomment first two comments.
"""
# import matplotlib
# matplotlib.use('TkAgg')
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['axes.edgecolor'] = 'black'  # Set the color of the axes here
plt.rcParams['axes.labelcolor'] = 'black'  # Set the color of the label here
plt.rcParams['xtick.color'] = 'black'  # Set the color of the x-axis ticks here
plt.rcParams['ytick.color'] = 'black'  # Set the color of the y-axis ticks here
plt.rcParams['grid.color'] = 'white'  # Set the grid color here

# Color palette
signal_color = 'black'
anchor_color = 'crimson'
prsa_output_color = 'yellow'
segment_color = 'black'
ac_anchor_color = 'crimson'
dc_anchor_color = 'forestgreen'
neighbour_color = 'black'

def plot_signal_with_anchors(time: np.ndarray, time_series: np.ndarray, idx_anchors: np.ndarray, start_time: float, end_time: float):
    """
    Plots the signal and highlights anchor points within a specified time period.

    Parameters:
    time (np.ndarray): Array of time points corresponding to the signal values, with adjusted dtype.
    time_series (np.ndarray): The signal values, with adjusted dtype.
    idx_anchors (np.ndarray): Indices of the anchor points in the signal, with adjusted dtype.
    start_time (float): The starting time for plotting.
    end_time (float): The ending time for plotting.
    """
    plt.figure(figsize=(15, 5))
    plot_mask = (time >= start_time) & (time <= end_time)
    time_plot = time[plot_mask]
    signal_plot = time_series[plot_mask]
    idx_anchors_plot = idx_anchors[(idx_anchors >= np.searchsorted(time, start_time)) & (idx_anchors <= np.searchsorted(time, end_time))] - np.searchsorted(time, start_time)

    plt.plot(time_plot, signal_plot, label='Signal')
    plt.scatter(time_plot[idx_anchors_plot], signal_plot[idx_anchors_plot], marker='o', label='Anchor Points')
    plt.title('Signal with Anchor Points in Specified Period')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_subset_signal_with_anchors(time: np.ndarray, time_series: np.ndarray, idx_anchors: np.ndarray, start_time: float, end_time: float, subset_size: int = 10):
    """
    Plots a subset of the signal's anchor points within a specified time range. This function
    is useful for focusing on a smaller number of anchor points in a large dataset.

    Parameters:
    time (np.ndarray): Array of time points corresponding to the signal values, with adjusted dtype.
    time_series (np.ndarray): The signal values, with adjusted dtype.
    idx_anchors (np.ndarray): Indices of the anchor points in the signal, with adjusted dtype.
    start_time (float): The starting time for plotting.
    end_time (float): The ending time for plotting.
    subset_size (int, optional): The number of anchor points to display. Defaults to 10.
    """
    plt.figure(figsize=(15, 5))
    plot_mask = (time >= start_time) & (time <= end_time)
    time_plot = time[plot_mask]
    signal_plot = time_series[plot_mask]
    idx_anchors_plot = idx_anchors[(idx_anchors >= np.searchsorted(time, start_time)) & (idx_anchors <= np.searchsorted(time, end_time))] - np.searchsorted(time, start_time)

    if len(idx_anchors_plot) > subset_size:
        selected_indices = np.random.choice(idx_anchors_plot, subset_size, replace=False)
    else:
        selected_indices = idx_anchors_plot

    plt.plot(time_plot, signal_plot, label='Signal')
    plt.scatter(time_plot[selected_indices], signal_plot[selected_indices], marker='o', label='Anchor Points')
    plt.title(f'Subset of Signal with Anchor Points Between {start_time} and {end_time}')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_segments_with_prsa_output(segments_array: np.ndarray, prsa_output: np.ndarray, idx_anchors: np.ndarray, prsa: bool = True) -> None:
    """
    Plots segments and overlays the PRSA output for visual comparison. This function
    is useful for analyzing how the PRSA output corresponds to specific segments in the data.

    Parameters:
    segments_array (np.ndarray): An array of segments extracted from the signal.
    prsa_output (np.ndarray): The output of the PRSA analysis.
    idx_anchors (np.ndarray): Indices of the segments to be plotted.

    """
    plt.figure(figsize=(15, 5), dpi=600)
    plt.style.use('default')

    segment_length = segments_array.shape[1]
    x_axis = np.linspace(-segment_length / 2, segment_length / 2, segment_length)

    for idx in idx_anchors:
        if idx < len(segments_array):
            plt.plot(x_axis, segments_array[idx], alpha=0.1, color='black')  
    if prsa:
        plt.plot(x_axis, prsa_output, color='red', linewidth=4, label='PRSA Output')
        plt.legend(fontsize=16, frameon=True, edgecolor='black')

    plt.xlabel('Index (relative to anchor point)', fontsize=16, fontweight='bold', color='black')
    plt.ylabel('RR interval (miliseconds)', fontsize=16, fontweight='bold', color='black')
    plt.xticks(fontsize=12, fontweight='bold', color='black')
    plt.yticks(fontsize=12, fontweight='bold', color='black')
    plt.grid(True, color='gray', linewidth=1)

    for axis in ['top', 'bottom', 'left', 'right']:
        plt.gca().spines[axis].set_linewidth(2)

    plt.show()

def plot_subset_segments_with_prsa_output(segments_array: np.ndarray, prsa_output: np.ndarray, idx_anchors: np.ndarray, subset_size: int = 10, prsa: bool = True) -> None:
    """
    Plots a subset of segments and overlays the PRSA output for clarity. This function
    helps in focusing on a smaller subset of segments for detailed analysis.

    Parameters:
    segments_array (np.ndarray): An array of segments extracted from the signal.
    prsa_output (np.ndarray): The output of the PRSA analysis.
    idx_anchors (np.ndarray): Indices of the segments to be plotted.
    subset_size (int): The number of segments to display. Defaults to 10.

    """
    plt.figure(figsize=(15, 5), dpi=600)
    plt.style.use('default')

    segment_length = segments_array.shape[1]
    x_axis = np.linspace(-segment_length / 2, segment_length / 2, segment_length)

    if len(idx_anchors) > subset_size:
        selected_indices = np.random.choice(idx_anchors, subset_size, replace=False)
    else:
        selected_indices = idx_anchors

    for idx in selected_indices:
        if idx < len(segments_array):
            plt.plot(x_axis, segments_array[idx], alpha=0.1, color='black')

    if prsa:
        plt.plot(x_axis, prsa_output, color='red', linewidth=4, label='PRSA Output')
        plt.legend(fontsize=16, frameon=True, edgecolor='black')

    plt.xlabel('Index (relative to anchor point)', fontsize=16, fontweight='bold', color='black')
    plt.ylabel('RR interval (miliseconds)', fontsize=16, fontweight='bold', color='black')
    plt.xticks(fontsize=12, fontweight='bold', color='black')
    plt.yticks(fontsize=12, fontweight='bold', color='black')
    plt.grid(True, color='gray', linewidth=1)

    for axis in ['top', 'bottom', 'left', 'right']:
        plt.gca().spines[axis].set_linewidth(2)

    plt.show()

def plot_signal_with_peaks(signal: np.ndarray, grouped_peaks: np.ndarray, start_time: Optional[float] = None, end_time: Optional[float] = None):
    """
    Plots the signal over time, with an option to focus on a specific time range.

    Parameters:
    signal (np.ndarray): Signal values.
    grouped_peaks (np.ndarray): Indices of the peaks in the signal data.
    start_time (Optional[float]): Start time for plotting a specific range.
    end_time (Optional[float]): End time for plotting a specific range.
    """
    plt.figure(figsize=(12, 8))
    plt.title("Signal with detected peaks", fontsize=24)
    
    plt.plot(np.arange(len(signal)), signal, label="Signal", color="black", linewidth=2)
    
    for i, group in enumerate(grouped_peaks):
        rounded_group = int(round(group))
        if rounded_group < len(signal):
            if i == 0:  # Only label the first peak
                plt.plot(rounded_group, signal[rounded_group], 'v', label="Peak", markersize=12, color="k", linestyle="None")
            else:
                plt.plot(rounded_group, signal[rounded_group], 'v', markersize=8, color="k", linestyle="None")

    if start_time is not None and end_time is not None:
        plt.xlim(start_time, end_time)

    plt.legend(loc="upper right", fontsize=20)
    plt.xlabel("Time (ms)", fontsize=16)
    plt.ylabel("Amplitude (arbitrary unit)", fontsize=16)
    plt.grid(True)
    plt.show()

def plot_signal(signal: np.ndarray, start_time: Optional[float] = None, end_time: Optional[float] = None, top: Optional[int] = None):
    """
    Plots the signal over time, with an option to focus on a specific time range.

    Parameters:
    signal (np.ndarray): Signal values.
    start_time (Optional[float]): Start time for plotting a specific range.
    end_time (Optional[float]): End time for plotting a specific range.
    top (Optional[int]): The maximum y-limit for the plot.
    """
    sampling_interval_ms = 5 
    sampling_interval_s = sampling_interval_ms / 1000  

    time_in_seconds = np.arange(len(signal)) * sampling_interval_s

    plt.figure(figsize=(12, 8), dpi=600)
    plt.style.use('default')
    plt.plot(time_in_seconds, signal, label="Signal", color="black", linewidth=2)

    if start_time is not None and end_time is not None:
        plt.xlim(start_time, end_time)
    plt.legend(loc="upper right", fontsize=20, frameon=True, edgecolor='black')
    plt.xlabel("Time (seconds)", fontsize=16, fontweight='bold', color='black')
    plt.ylabel("ABP (mmHg)", fontsize=16, fontweight='bold', color='black')
    plt.xticks(fontsize=12, fontweight='bold', color='black')
    plt.yticks(fontsize=12, fontweight='bold', color='black')
    plt.ylim(top=top)
    plt.tight_layout()
    plt.grid(True, color='gray', linewidth=1)

    for axis in ['top','bottom','left','right']:
        plt.gca().spines[axis].set_linewidth(2)

    plt.show()

def plot_rr_intervals_with_anchors(time: np.ndarray, rr_intervals: np.ndarray, idx_anchors_acc: np.ndarray, idx_anchors_dc: np.ndarray, start_time: float = 0, end_time: float = None):
    """
    Plots RR intervals and highlights AC and DC anchor points within a specified time period.

    Parameters:
    time (numpy.ndarray): Array of time points.
    rr_intervals (numpy.ndarray): RR interval values.
    idx_anchors_acc (numpy.ndarray): Indices of AC anchor points within the time period.
    idx_anchors_dc (numpy.ndarray): Indices of DC anchor points within the time period.
    start_time (float, optional): Start time for plotting a specific range. Defaults to 0.
    end_time (float, optional): End time for plotting a specific range. Defaults to the last time point.
    """
    rr_intervals_seconds = rr_intervals.astype(float)
    time_seconds = np.cumsum(rr_intervals_seconds / 1000)

    plt.figure(figsize=(12, 8), dpi=600)
    plt.style.use('default')

    # Determine the range of data to plot based on start and end times
    if start_time is None:
        start_time = time_seconds[0]
    if end_time is None:
        end_time = time_seconds[-1]

    mask = (time_seconds >= start_time) & (time_seconds <= end_time)
    time_filtered = time_seconds[mask]
    rr_intervals_filtered = rr_intervals_seconds[mask]

    # Adjust the mask for AC and DC anchors to use time in seconds
    mask_acc = (time_seconds[idx_anchors_acc] >= start_time) & (time_seconds[idx_anchors_acc] <= end_time)
    mask_dc = (time_seconds[idx_anchors_dc] >= start_time) & (time_seconds[idx_anchors_dc] <= end_time)
    acc_anchors_filtered = idx_anchors_acc[mask_acc]
    dc_anchors_filtered = idx_anchors_dc[mask_dc]

    # Plotting
    plt.plot(time_filtered, rr_intervals_filtered, '-', color='black', label='RR Intervals', linewidth=2)
    plt.scatter(time_seconds[acc_anchors_filtered], rr_intervals_seconds[acc_anchors_filtered], s=28, color='black', marker='x', label='AC Anchors')
    plt.scatter(time_seconds[dc_anchors_filtered], rr_intervals_seconds[dc_anchors_filtered], s=28, color='black', marker='o', label='DC Anchors')

    # Customize the legend, labels, ticks, and grid as per the provided structure
    plt.legend(loc="best", fontsize=20, frameon=True, edgecolor='black')
    plt.xlabel("Time (seconds)", fontsize=16, fontweight='bold', color='black')
    plt.ylabel("RR interval (miliseconds)", fontsize=16, fontweight='bold', color='black')
    plt.xticks(fontsize=12, fontweight='bold', color='black')
    plt.yticks(fontsize=12, fontweight='bold', color='black')
    plt.grid(True, color='gray', linewidth=1)

    # Set the axis spines to be bold
    for axis in ['top','bottom','left','right']:
        plt.gca().spines[axis].set_linewidth(2)

    plt.show()

def plot_rr_intervals_with_anchors_clean(time: np.ndarray, rr_intervals: np.ndarray, idx_anchors_acc: np.ndarray, idx_anchors_dc: np.ndarray, start_time: float = 0, end_time: float = None):
    """
    Plots RR intervals and highlights AC and DC anchor points within a specified time period.

    Parameters:
    time (numpy.ndarray): Array of time points.
    rr_intervals (numpy.ndarray): RR interval values.
    idx_anchors_acc (numpy.ndarray): Indices of AC anchor points within the time period.
    idx_anchors_dc (numpy.ndarray): Indices of DC anchor points within the time period.
    start_time (float, optional): Start time for plotting a specific range. Defaults to 0.
    end_time (float, optional): End time for plotting a specific range. Defaults to the last time point.
    """
    rr_intervals_seconds = rr_intervals.astype(float)
    time_seconds = np.cumsum(rr_intervals_seconds / 1000)

    plt.figure(figsize=(12, 8), dpi=800)
    #plt.style.use('default')

    # Determine the range of data to plot based on start and end times
    if start_time is None:
        start_time = time_seconds[0]
    if end_time is None:
        end_time = time_seconds[-1]

    mask = (time_seconds >= start_time) & (time_seconds <= end_time)
    time_filtered = time_seconds[mask]
    rr_intervals_filtered = rr_intervals_seconds[mask]

    # Adjust the mask for AC and DC anchors to use time in seconds
    mask_acc = (time_seconds[idx_anchors_acc] >= start_time) & (time_seconds[idx_anchors_acc] <= end_time)
    mask_dc = (time_seconds[idx_anchors_dc] >= start_time) & (time_seconds[idx_anchors_dc] <= end_time)
    acc_anchors_filtered = idx_anchors_acc[mask_acc]
    dc_anchors_filtered = idx_anchors_dc[mask_dc]

    # Plotting
    plt.plot(time_filtered, rr_intervals_filtered)
    plt.scatter(time_seconds[acc_anchors_filtered], rr_intervals_seconds[acc_anchors_filtered], s=28, color='red', marker='x', label='AC Anchors')
    plt.scatter(time_seconds[dc_anchors_filtered], rr_intervals_seconds[dc_anchors_filtered], s=28, color='green', marker='o', label='DC Anchors')

    # Customize the legend, labels, ticks, and grid as per the provided structure
    plt.legend(loc="best", fontsize=16, frameon=True, edgecolor='black')
    plt.xlabel("Time (seconds)", fontsize=16, fontweight='bold', color='black')
    plt.ylabel("RR interval (miliseconds)", fontsize=16, fontweight='bold', color='black')
    plt.xticks(fontsize=12, fontweight='bold', color='black')
    plt.yticks(fontsize=12, fontweight='bold', color='black')

    # # Set the axis spines to be bold
    # for axis in ['top','bottom','left','right']:
    #     plt.gca().spines[axis].set_linewidth(2)

    plt.show()

def plot_rr_intervals(rr_intervals: np.ndarray, start_time: float = None, end_time: float = None):
    """
    Plots the RR intervals against the cumulative sum of the RR intervals as the time axis.

    Parameters:
    rr_intervals (numpy.ndarray): Array of RR interval values in milliseconds.
    """
    # Ensure rr_intervals is a one-dimensional array
    if not rr_intervals.ndim == 1:
        raise ValueError("rr_intervals must be a one-dimensional array.")

    rr_intervals_seconds = rr_intervals

    # Convert cumulative sum of rr_intervals from milliseconds to seconds for the time axis
    time_axis_seconds = np.cumsum(rr_intervals_seconds / 1000)

    plt.figure(figsize=(12, 8), dpi=600)
    plt.style.use('default')

    # Plot using the converted time axis and RR intervals in seconds
    plt.plot(time_axis_seconds, rr_intervals_seconds, label="RR interval", color="black", linewidth=2)

    # Set x-axis limits if start and end times are provided
    if start_time is not None and end_time is not None:
        plt.xlim(start_time, end_time)

    # Set the properties of the legend, axis labels, ticks, and grid
    plt.legend(loc="upper right", fontsize=20, frameon=True, edgecolor='black')
    plt.xlabel("Time (seconds)", fontsize=16, fontweight='bold', color='black')
    plt.ylabel("RR interval (miliseconds)", fontsize=16, fontweight='bold', color='black')
    plt.xticks(fontsize=12, fontweight='bold', color='black')
    plt.yticks(fontsize=12, fontweight='bold', color='black')
    plt.grid(True, color='gray', linewidth=1)

    # Set the axis spines to be bold
    for axis in ['top','bottom','left','right']:
        plt.gca().spines[axis].set_linewidth(2)

    plt.show()

def plot_all_anchors_with_neighbours(anchor_neighbours: np.ndarray, prsa_output: np.ndarray, T: int, prsa: bool = True):
    """
    Plots all anchor points with their neighboring values and overlays the PRSA output. 
    This function is useful for visualizing how anchor points relate to the PRSA output across the entire dataset.

    Parameters:
    anchor_neighbours (numpy.ndarray): Array containing the neighboring values of each anchor point.
    prsa_output (numpy.ndarray): The PRSA output for comparison.
    T (int): The number of neighboring points to consider on each side of the anchor point.
    """
    plt.figure(figsize=(12, 6))
    window_size = 2 * T + 1
    x_range = np.arange(-T, T + 1)

    prsa_output = prsa_output[:window_size] if len(prsa_output) > window_size else np.pad(prsa_output, (0, window_size - len(prsa_output)), 'constant', constant_values=np.nan)

    for i in range(len(anchor_neighbours)):
        plt.plot(x_range, anchor_neighbours[i], color=neighbour_color, alpha=0.5)

    if prsa:
        plt.plot(x_range, prsa_output, color=prsa_output_color, linewidth=4, label='PRSA Output')
        plt.legend()
    plt.xlabel("Index Number (Relative to Anchor)")
    plt.ylabel("RR Interval (ms)")
    plt.title('All Anchor Points with Neighbours and PRSA Output')
    plt.grid(True)
    plt.show()

def plot_all_anchors_with_neighbours_clean(anchor_neighbours: np.ndarray, prsa_output: np.ndarray, T: int, prsa: bool = True):
    """
    Plots all anchor points with their neighboring values and overlays the PRSA output.
    This function is useful for visualizing how anchor points relate to the PRSA output across the entire dataset.

    Parameters:
    anchor_neighbours (numpy.ndarray): Array containing the neighboring values of each anchor point.
    prsa_output (numpy.ndarray): The PRSA output for comparison.
    T (int): The number of neighboring points to consider on each side of the anchor point.
    prsa (bool, optional): Whether to plot the PRSA output. Defaults to True.
    """
    plt.figure(figsize=(12, 6), dpi=800)
    window_size = 2 * T + 1
    x_range = np.arange(-T, T + 1)

    # Adjust the PRSA output length to match the window size
    prsa_output_adjusted = prsa_output[:window_size] if len(prsa_output) > window_size else np.pad(prsa_output, (0, window_size - len(prsa_output)), 'constant', constant_values=np.nan)

    # Plot each anchor neighbour
    for i in range(len(anchor_neighbours)):
        plt.plot(x_range, anchor_neighbours[i], color='#1f77b4', alpha=0.5)  # Adjusted color for better visibility

    # Plot PRSA output if enabled
    if prsa:
        plt.plot(x_range, prsa_output_adjusted, color='red', linewidth=4, label='PRSA Output')
        plt.legend(fontsize=12)

    # Customize plot labels and ticks
    plt.xlabel("Index Number (Relative to Anchor)", fontsize=16, fontweight='bold', color='black')
    plt.ylabel("RR Interval (ms)", fontsize=16, fontweight='bold', color='black')
    plt.xticks(fontsize=12, fontweight='bold', color='black')
    plt.yticks(fontsize=12, fontweight='bold', color='black')
    plt.show()

def plot_subset_of_anchors_with_neighbours_clean(anchor_neighbours: np.ndarray, prsa_output: np.ndarray, T: int, subset_size: int=10, prsa: bool=True):
    """
    Plots a subset of anchor points with their neighboring values and overlays the PRSA output.
    This function is useful for focusing on a smaller subset of anchor points for detailed analysis.

    Parameters:
    anchor_neighbours (numpy.ndarray): Array containing the neighboring values of each anchor point.
    prsa_output (numpy.ndarray): The PRSA output for comparison.
    T (int): The number of neighboring points to consider on each side of the anchor point.
    subset_size (int, optional): The number of anchor points to display. Defaults to 10.
    prsa (bool, optional): Whether to plot the PRSA output. Defaults to True.
    """
    plt.figure(figsize=(12, 6), dpi=800)
    window_size = 2 * T + 1
    x_range = np.arange(-T, T + 1)

    # Adjust the PRSA output length to match the window size
    prsa_output_adjusted = prsa_output[:window_size] if len(prsa_output) > window_size else np.pad(prsa_output, (0, window_size - len(prsa_output)), 'constant', constant_values=np.nan)

    # Select a subset of anchor neighbours if necessary
    selected_indices = np.random.choice(len(anchor_neighbours), subset_size, replace=False) if len(anchor_neighbours) > subset_size else np.arange(len(anchor_neighbours))
    for i in selected_indices:
        plt.plot(x_range, anchor_neighbours[i], color='#1f77b4', alpha=0.5)  # Adjusted color for better visibility

    # Plot PRSA output if enabled
    if prsa:
        plt.plot(x_range, prsa_output_adjusted, color='red', linewidth=4, label='PRSA Output')
        plt.legend(fontsize=12)

    # Customize plot labels and ticks
    plt.xlabel("Index Number (Relative to Anchor)", fontsize=16, fontweight='bold', color='black')
    plt.ylabel("RR Interval (ms)", fontsize=16, fontweight='bold', color='black')
    plt.xticks(fontsize=12, fontweight='bold', color='black')
    plt.yticks(fontsize=12, fontweight='bold', color='black')
    plt.show()

def plot_subset_of_anchors_with_neighbours(anchor_neighbours: np.ndarray, prsa_output: np.ndarray, T: int, subset_size: int=10, prsa: bool=True):
    """
    Plots a subset of anchor points with their neighboring values and overlays the PRSA output. 
    This function is useful for focusing on a smaller subset of anchor points for detailed analysis.

    Parameters:
    anchor_neighbours (numpy.ndarray): Array containing the neighboring values of each anchor point.
    prsa_output (numpy.ndarray): The PRSA output for comparison.
    T (int): The number of neighboring points to consider on each side of the anchor point.
    subset_size (int, optional): The number of anchor points to display. Defaults to 10.
    """
    plt.figure(figsize=(12, 6))
    window_size = 2 * T + 1
    x_range = np.arange(-T, T + 1)

    prsa_output = prsa_output[:window_size] if len(prsa_output) > window_size else np.pad(prsa_output, (0, window_size - len(prsa_output)), 'constant', constant_values=np.nan)

    if len(anchor_neighbours) > subset_size:
        selected_indices = np.random.choice(len(anchor_neighbours), subset_size, replace=False)
        selected_anchor_neighbours = [anchor_neighbours[i] for i in selected_indices]
    else:
        selected_anchor_neighbours = anchor_neighbours

    for anchor in selected_anchor_neighbours:
        plt.plot(x_range, anchor, color=neighbour_color, alpha=0.5)

    if prsa:
        plt.plot(x_range, prsa_output, color=prsa_output_color, linewidth=4, label='PRSA Output')
        plt.legend()
    plt.xlabel("Index Number (Relative to Anchor)")
    plt.ylabel("RR Interval (ms)")
    plt.title('Subset of Anchor Points with Neighbours and PRSA Output')
    plt.grid(True)
    plt.show()