#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# This file contains the main functions used in the bachelor thesis of Francesco Andreetto (2020)
# updated to be used on the new version of the pipeline for functional verification of LSPE-STRIP (2023)
# October 29th 2022, Brescia (Italy)

# Libraries & Modules
import csv
import json
import logging
import scipy.signal

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn  # This should be added to requirements.txt
import scipy.stats as scs
import scipy.ndimage as scn

from astropy.time import Time, TimeDelta
from datetime import datetime
from numba import njit
from pathlib import Path
from rich.logging import RichHandler
from striptease import DataFile
from typing import Dict, Any


def tab_cap_time(pol_name: str, file_name: str, output_dir: str) -> str:
    """
    Create a new file .txt and write the caption of a tabular\n
    Parameters:\n
    - **pol_name** (``str``): Name of the polarimeter
    - **file_name** (``str``): Name of the file to create and in which insert the caption\n
    - **output_dir** (``str``): Name of the dir where the csv file must be saved
    This specific function creates a tabular that collects the jumps in the dataset (JT).
    """
    new_file_name = f"JT_{pol_name}_{file_name}.csv"
    cap = [["# Jump", "tDelta_t [JHD]", "Delta_t [s]", "Gregorian Date", "JHD Date"]]

    path = f'../plot/{output_dir}/Time_Jump/'
    Path(path).mkdir(parents=True, exist_ok=True)
    with open(f"{path}/{new_file_name}", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(cap)

    return cap


def pol_list(path_dataset: Path) -> list:
    """
    Create a list of the polarimeters present in the datafile\n
    Parameters:\n
    - **path_dataset** (Path comprehensive of the name of the dataset file)
    """
    d = DataFile(path_dataset)
    d.read_file_metadata()
    pols = []
    for cur_pol in sorted(d.polarimeters):
        pols.append(f"{cur_pol}")
    return pols


@njit  # optimize calculations
def mean_cons(v):
    """
    Calculate consecutive means of an array.\n
    Parameters:\n
    - **v** is an array-like object-like object\n
    The mean on each couple of samples of even-odd index is computed.
    """
    n = (len(v) // 2) * 2
    mean = (v[0:n:2] + v[1:n + 1:2]) / 2
    return mean


@njit
def diff_cons(v):
    """
    Calculate consecutive difference of an array.\n
    Parameters:\n
    - **v** is an array-like object\n
    The difference between each sample of even-odd index is computed.
    """
    n = (len(v) // 2) * 2
    diff = (v[0:n:2] - v[1:n + 1:2])
    return diff


@njit
def mob_mean(v, smooth_len: int):
    """
    Calculate a mobile mean on a number of elements given by smooth_len, used to smooth plots.\n
    Parameters:\n
    - **v** is an array-like object
    - **smooth_len** (int): number of elements on which the mobile mean is calculated
    """
    m = np.zeros(len(v) - smooth_len + 1)
    for i in np.arange(len(m)):
        m[i] = np.mean(v[i:i + smooth_len])
    return m


def demodulation(dataset: dict, timestamps: list, type: str, exit: str, begin=0, end=-1) -> Dict[str, Any]:
    """
    Demodulation\n
    Calculate the double demodulation at 50Hz of the dataset provided\n
    Timestamps are chosen as mean of the two consecutive times of the DEM/PWR data\n
    Parameters:\n
    - **dataset** (``dict``): dictionary ({}) containing the dataset with the output of a polarimeter
    - **timestamps** (``list``): list ([]) containing the Timestamps of the output of a polarimeter
    - **exit** (``str``) *"Q1"*, *"Q2"*, *"U1"*, *"U2"*\n
    - **type** (``str``) of data *"DEM"* or *"PWR"*
    - **begin**, **end** (``int``): interval of dataset that has to be considered
    """
    times = mean_cons(timestamps)
    data = {}
    if type == "PWR":
        data[exit] = mean_cons(dataset[type][exit][begin:end])
    if type == "DEM":
        data[exit] = diff_cons(dataset[type][exit][begin:end])

    sci_data = {"sci_data": data, "times": times}
    return sci_data


@njit
def rolling_window(v, window: int):
    """
    Rolling Window Function\n
    Parameters:\n
    -  **v** is an array-like object
    - **window** (int)
    Return a matrix with:\n
    - A number of element per row fixed by the parameter window
    - The first element of the row j is the j element of the vector
    """
    shape = v.shape[:-1] + (v.shape[-1] - window + 1, window)
    strides = v.strides + (v.strides[-1],)
    return np.lib.stride_tricks.as_strided(v, shape=shape, strides=strides)


def RMS(data: dict, window: int, exit: str, eoa: int, begin=0, end=-1):
    """
    Calculate the RMS of a vector using the rolling window
    Parameters:\n
    - **data** is a dictionary with four keys (exits) of a particular type *"DEM"* or *"PWR"*
    - **window**: number of elements on which the RMS is calculated
    - **exit** (``str``) *"Q1"*, *"Q2"*, *"U1"*, *"U2"*
    - **eoa** (``int``): flag in order to calculate RMS for\n
        all samples (*eoa=0*), can be used for Demodulated and Total Power scientific data (50Hz)\n
        odd samples (*eoa=1*)\n
        even samples (*eoa=2*)\n
    - **begin**, **end** (``int``): interval of dataset that has to be considered
    """
    if eoa == 0:
        rms = np.std(rolling_window(data[exit][begin:end], window), axis=1)
    elif eoa == 1:
        rms = np.std(rolling_window(data[exit][begin + 1:end:2], window), axis=1)
    elif eoa == 2:
        rms = np.std(rolling_window(data[exit][begin:end - 1:2], window), axis=1)
    else:
        logging.error("Wrong EOA value: it must be 0,1 or 2.")
        raise SystemExit(1)
    return rms


def EOA(even: int, odd: int, all: int) -> str:
    """
    Parameters:\n
    - **even**, **odd**, **all** (``int``)
    If the variables are different from zero, this returns a string that contains the corresponding letters:\n
    "E" for even (``int``)\n
    "O" for odd (``int``)\n
    "A" for all (``int``)\n
    """
    eoa = ""
    if even != 0:
        eoa += "E"
    if odd != 0:
        eoa += "O"
    if all != 0:
        eoa += "A"
    return eoa


def eoa_values(eoa_str: str) -> []:
    """
    Parameters:\n
    - **eoa_str** (``str``): string of 0,1,2 or 3 letters from a combination of the letters e, o and a
    Return a string of combinations of 3 values (e, o, a) taken from a dictionary that contains 3 keys
    and the values 0 or 0 and 1 for each keys
    """
    # Initialize a dictionary with 0 values for e,o,a keys
    eoa_dict = {"e": [0], "o": [0], "a": [0]}
    eoa_list = [char for char in eoa_str]

    # If a letter appears also the value 1 is included in the dictionary
    for key in eoa_dict.keys():
        if key in eoa_list:
            eoa_dict[key].append(1)

    # Store the combinations of 0 and 1 depending on which letters were provided
    eoa_combinations = [(val1, val2, val3)
                        for val1 in eoa_dict["e"]
                        for val2 in eoa_dict["o"]
                        for val3 in eoa_dict["a"]]

    return eoa_combinations


def find_spike(v, threshold=8.5, n_chunk=5) -> []:
    """
        Look up for 'spikes' in a given array.\n
        Calculate the median and the mad and uses those to discern spikes.
        Parameters:\n
        - **v** is an array-like object
        - **threshold** (int): value used to discern what a spike is
        - **n_chunk** (int): n of blocks in which v is divided. On every block the median is computed to find spikes.
        """
    l = len(v)
    n_chunk = n_chunk
    len_chunk = l // n_chunk
    spike_idx = []

    for n_rip in range(5):
        _v_ = v[n_rip * len_chunk:(n_rip + 1) * len_chunk - 1]
        med_v = scn.median(_v_)  # type:float
        mad_v = scs.median_abs_deviation(_v_)

        dif = diff_cons(_v_)
        med_dif = scn.median(dif)
        mad_dif = scs.median_abs_deviation(dif)

        v_threshold = (threshold / 2.7) * mad_v

        logging.basicConfig(level="WARNING", format='%(message)s',
                            datefmt="[%X]", handlers=[RichHandler()])  # <3

        for idx, item in enumerate(dif):
            # spike up in differences
            if item > med_dif + threshold * mad_dif:
                if _v_[idx * 2] > med_v + v_threshold:
                    # logging.info(f"found spike up: {idx}")
                    spike_idx.append(n_rip * len_chunk + idx * 2)
                if _v_[idx * 2 + 1] < med_v - v_threshold:
                    # logging.info(f"found spike down: {idx}")
                    spike_idx.append(n_rip * len_chunk + idx * 2 + 1)
            # spike down in differences
            if item < med_dif - threshold * mad_dif:
                if _v_[idx * 2] < med_v - v_threshold:
                    # logging.info(f"found spike down: {idx}")
                    spike_idx.append(n_rip * len_chunk + idx * 2)
                if _v_[idx * 2 + 1] > med_v + v_threshold:
                    # logging.info(f"found spike up: {idx}")
                    spike_idx.append(n_rip * len_chunk + idx * 2 + 1)

    if len(spike_idx) > 0:
        logging.warning("Found Spike!\n")

    return spike_idx


def replace_spike(v, N=10, threshold=8.5, gauss=True):
    """
    Find the 'spikes' in a given array and replace them as follows.
    Parameters:\n
    - **v** (``list``): is an array-like object\n
    - **N** (int): number of elements used to calculate the mean and the std_dev to substitute the spike (see below).
    - **threshold** (int): value used to discern what a spike is (see find_spike above).
    - **gauss** (bool):\n

        *True* -> The element of the array is substituted with a number extracted with a Gaussian distribution
        around the mean of some elements of the array chosen as follows:\n
        \t i)   The 2N following elements if the spike is in position 0;\n
        \t ii)  An equal number of elements taken before and after the spike if the spike itself is in a position i<N
        \t iii) An equal number of elements N taken before and after the spike if the spike itself is in a position i>N
        \t iv)  The 2N previous elements if the spike is in a position i bigger than the length of the array minus N\n

        *False* -> The element of the array is substituted with the median of the array.
    """
    logging.basicConfig(level="INFO", format='%(message)s',
                        datefmt="[%X]", handlers=[RichHandler()])  # <3

    s = find_spike(v=v, threshold=threshold)
    a, b, c, d = 0, 0, 0, 0
    if len(s) == 0:
        return "No spikes detected in the dataset.\n"
    for i in s:
        logging.info("\nStarting new removal run")
        if gauss:
            """
            Gaussian substitution
            """
            if i == 0 or 0 < i < N:
                logging.info(f"Spike in low position ({i}): using the following {2 * N} samples")
                a = 0
                c = 1
                d = 1 + 2 * i
            if N < i < (len(v) - N):
                logging.info(f"Spike in position {i}: using {2 * N} samples, {N} before e {N} after")
                a = -N
                c = 1
                d = 1 + N
            if i > (len(v) - N - 1):
                logging.info(f"Spike in high position ({i}): using the previous {2 * N} samples")
                a = -2 * N

            new_d = np.concatenate((v[i + a:i + b], v[i + c: i + d]))
            new_m = scn.median(new_d)
            new_s = scs.median_abs_deviation(new_d)
            new_a = np.random.normal(new_m, new_s)

        else:
            """
            Non-Gaussian substitution
            """
            new_a = scn.median(v)

        v[i] = new_a

        # s = find_spike(v=v, threshold=threshold)


def find_jump(v, exp_med: float, tolerance: float) -> {}:
    """
        Find the 'jumps' in a given Time astropy object: the samples should be consequential with a fixed growth rate.
        Hence, their consecutive differences should have an expected median within a certain tolerance.
        Parameters:\n
        - **v** is a Time object from astropy => i.e. Polarimeter.times\n
        - **exp_med** (``float``) is the expected median (in seconds) of the TimeDelta
        between two consecutive values of v
        - **tolerance** (``float``) is the threshold # of seconds over which a TimeDelta is considered as an error\n
        Return:\n
        - **jumps** a dictionary containing three keys:
            - **n** (``int``) is the number of jumps found
            - **idx** (``int``) index of the jump in the array
            - **value** (``float``) is the value of the jump in JHD
            - **s_value** (``float``) is the value of the jump in seconds
            - **median_ok** (``bool``) True if there is no jump in the vector, False otherwise
    """
    dt = v[1:] - v[:-1]  # type: TimeDelta
    exp_med = exp_med / 86400  # Conversion in days
    med_dt = np.median(dt.value)  # If ".value" is not used the time needed is 1.40min vs 340ms... Same results.
    median_ok = True
    if np.abs(np.abs(med_dt) - np.abs(exp_med)) > tolerance / 86400:  # Over the tolerance
        msg = f"Median is out of range: {med_dt}, expected {exp_med}."
        logging.warning(msg)
        median_ok = False

    err_t = dt.value - med_dt

    idx = []
    value = []
    s_value = []
    n = 0
    jumps = {"n": n, "idx": idx, "value": value, "s_value": s_value, "median_ok": median_ok}
    for i, item in enumerate(err_t):
        if np.abs(item) > tolerance / 86400:
            jumps["n"] += 1
            jumps["idx"].append(i)
            jumps["value"].append(dt.value[i])
            jumps["s_value"].append(dt.value[i] * 86400)
    return jumps


def dir_format(old_string: str) -> str:
    """
    Take a string a return a new string changing white spaces into underscores, ":" into "-" and removing ".000"
    Parameters:\n
    old_string (``str``)
    """
    new_string = old_string.replace(" ", "_")
    new_string = new_string.replace(".000", "")
    new_string = new_string.replace(":", "-")
    return new_string


def csv_to_json(csv_file_path: str, json_file_path):
    """
    Convert a csv file into a json file
    Parameters:\n
    - csv_file_path (``str``): path of the csv file that have to be converted
    - json_file_path (``str``): path of the json file converted
    """
    json_array = []

    # read csv file
    with open(csv_file_path, encoding='utf-8') as csv_file:
        # load csv file data using csv library's dictionary reader
        csvReader = csv.DictReader(csv_file)

        # convert each csv row into python dict
        for row in csvReader:
            # add this python dict to json array
            json_array.append(row)

    # convert python json_array to JSON String and write to file
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json_string = json.dumps(json_array, indent=4)
        json_file.write(json_string)


def name_check(names: list) -> bool:
    """
        Check if the names of the polarimeters in the list are wrong: not the same as the polarimeters of Strip.
        Parameters:\n
        - names (``list``): list of the names of the polarimeters
    """
    for n in names:
        # Check if the letter corresponds to one of the tiles of Strip
        if n[0] not in (["B", "G", "I", "O", "R", "V", "W", "Y"]):
            return False
        # Check if the number is correct
        if n[1] not in (["0", "1", "2", "3", "4", "5", "6", "7"]):
            return False
        # The only exception is W7
        if n == "W7":
            return False
    return True


def datetime_check(date_str: str) -> bool:
    """
        Check if the string is in datatime format "YYYY-MM-DD hh:mm:ss" or not.
        Parameters:\n
        - date (``str``): string with the datetime
    """
    date_format = "%Y-%m-%d %H:%M:%S"
    try:
        datetime.strptime(date_str, date_format)
        return True
    except ValueError:
        return False


def date_update(start_datetime: str, n_samples: int, sampling_frequency: int) -> Time:
    """
    Calculates and returns the new Gregorian date in which the analysis begins, given a number of samples that
    must be skipped from the beginning of the dataset.
    Parameters:\n
    - **start_datetime** (``str``): start time of the dataset
    - **n_samples** (``int``) number of samples that must be skipped\n
    - **sampling_freq** (``int``): sampling frequency of the dataset
    """
    # Convert the str in a Time object: Julian Date MJD
    jdate = Time(start_datetime).mjd
    # A second expressed in days unit
    s = 1 / 86_400
    # Julian Date increased
    jdate += s * (n_samples / sampling_frequency)
    # New Gregorian Date
    new_date = Time(jdate, format="mjd").to_datetime().strftime("%Y-%m-%d %H:%M:%S")
    return new_date


def same_length(array1, array2) -> []:
    """
        Check if the two array are of the same length. If not, the longer becomes as long as the smaller
        Parameters:\n
        - array1, array2 (``array``): data arrays.
    """
    l1 = len(array1)
    l2 = len(array2)
    array1 = array1[:min(l1, l2)]
    array2 = array2[:min(l1, l2)]
    return [array1, array2]


# warn_threshold => used for anomalies & warnings
def correlation_mat(dict1: {}, dict2: {}, data_name: str,
                    start_datetime: str, end_datetime: str, show=False, corr_t=0.4) -> {}:
    """
       Plot a 4x4 Correlation Matrix of two generic dictionaries (also of one with itself).\n

       Parameters:\n
       - **dict1**, **dict2** (``array``): dataset
       - **data_name** (``str``): name of the dataset. Used for the title of the figure and to save the png.
       - **start_datetime** (``str``): begin date of dataset. Used for the title of the figure and to save the png.
       - **end_datetime** (``str``): end date of dataset. Used for the title of the figure and to save the png.
       - **show** (``bool``):\n
            *True* -> show the plot and save the figure\n
            *False* -> save the figure only
       - **corr_t** (``int``): if it is overcome by one of the values of the matrix a warning is produced.\n
    """
    self_correlation = False
    # If the second dictionary is not provided we are in a Self correlation case
    if dict2 == {}:
        self_correlation = True
        dict2 = dict1

    # Convert dictionaries to DataFrames
    df1 = pd.DataFrame(dict1)
    df2 = pd.DataFrame(dict2)

    # Initialize an empty DataFrame for correlations
    correlation_matrix = pd.DataFrame(index=df1.columns, columns=df2.columns)

    # Calculate correlations
    for key1 in df1.columns:
        for key2 in df2.columns:
            correlation_matrix.loc[key1, key2] = df1[key1].corr(df2[key2])

    # Self correlation case
    if self_correlation:
        for i in correlation_matrix.keys():
            # Put at Nan the values on the diagonal of the matrix (self correlations)
            correlation_matrix[i][i] = np.nan

    # Convert correlation matrix values to float
    correlation_matrix = correlation_matrix.astype(float)

    # Create a figure to plot the correlation matrices
    fig, axs = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(10, 5))
    # Set the title of the figure
    fig.suptitle(f'Correlation Matrix {data_name} - Date: {start_datetime}', fontsize=12)

    pl_m1 = sn.heatmap(correlation_matrix, annot=True, ax=axs[0], cmap='coolwarm')
    pl_m1.set_title(f"Correlation {data_name}", fontsize=14)
    pl_m2 = sn.heatmap(correlation_matrix, annot=True, ax=axs[1], cmap='coolwarm', vmin=-0.4, vmax=0.4)
    pl_m2.set_title(f"Correlation {data_name} - Fixed Scale", fontsize=14)

    # Procedure to save the png of the plot in the correct dir
    # Gregorian Date [in string format]
    gdate = [Time(start_datetime), Time(end_datetime)]
    # Directory where to save all the plots of a given analysis
    date_dir = dir_format(f"{gdate[0]}__{gdate[1]}")
    path = f'../plot/{date_dir}/Correlation_Matrix/'
    # Check if the dir exists. If not, it will be created.
    Path(path).mkdir(parents=True, exist_ok=True)
    fig.savefig(f'{path}{data_name}_CorrMat.png')

    # If show is True the plot is visible on video
    if show:
        plt.show()
    plt.close(fig)

    return {"There will be a dict of anomalies"}


def data_plot(pol_name: str,
              dataset: dict,
              timestamps: list,
              start_datetime: str, end_datetime: str,
              begin: int, end: int,
              type: str,
              even: str, odd: str, all: str,
              demodulated: bool, rms: bool, fft: bool,
              window: int, smooth_len: int, nperseg: int,
              show: bool):
    """
    Generic function that create a Plot of the dataset provided.\n
    Parameters:
        -**pol_name** (``str``): name of the polarimeter we want to analyze
        - **dataset** (``dict``): dictionary ({}) containing the dataset with the output of a polarimeter
        - **timestamps** (``list``): list ([]) containing the Timestamps of the output of a polarimeter
        - **start_datetime** (``str``): start time
        - **end_datetime** (``str``): end time
        - **begin**, **end** (``int``): interval of dataset that has to be considered\n

        - **type** (``str``) of data *"DEM"* or *"PWR"*\n
        - **even**, **odd**, **all** (int): used to set the transparency of the dataset (0=transparent, 1=visible)\n

        - **demodulated** (``bool``): if true, demodulated data are computed, if false even-odd-all output are plotted
        - **rms** (``bool``) if true, the rms are computed
        - **fft** (``bool``) if true, the fft are computed

        - **window** (``int``): number of elements on which the RMS is calculated
        - **smooth_len** (``int``): number of elements on which the mobile mean is calculated
        - **nperseg** (``int``): number of elements of the array of scientific data on which the fft is calculated
        - **show** (``bool``): *True* -> show the plot and save the figure, *False* -> save the figure only
    """
    # Initialize the plot directory from start_datetime and end_datetime
    path_dir = dir_format(f"{Time(start_datetime)}__{Time(end_datetime)}")

    # Initialize the name of the plot
    name_plot = f"{pol_name} "

    # ------------------------------------------------------------------------------------------------------------------
    # Step 1: define the operations: FFT, RMS, OUTPUT

    # Calculate fft
    if fft:
        # Update the name of the plot
        name_plot += " FFT"
        # Update the name of the plot directory
        path_dir += "/FFT"
    else:
        pass

    # Calculate rms
    if rms:
        # Update the name of the plot
        name_plot += " RMS"
        # Update the name of the plot directory
        path_dir += "/RMS"
    else:
        pass

    if not fft and not rms:
        # Update the name of the plot directory
        path_dir += "/SCIDATA" if demodulated else "/OUTPUT"

    # ------------------------------------------------------------------------------------------------------------------
    # Step 2: define type of data
    # Demodulated Scientific Data vs Scientific Output
    if type == "DEM":
        # Update the name of the plot
        name_plot += " DEMODULATED" if demodulated else f" {type} {EOA(even, odd, all)}"
        # Update the name of the plot directory
        path_dir += "/DEMODULATED" if demodulated else f"/{type}"
    elif type == "PWR":
        # Update the name of the plot
        name_plot += " TOTPOWER" if demodulated else f"{type} {EOA(even, odd, all)}"
        # Update the name of the plot directory
        path_dir += "/TOTPOWER" if demodulated else f"/{type}"
    else:
        logging.error("Wrong type! Choose between DEM or PWR!")
        raise SystemExit(1)

    # ------------------------------------------------------------------------------------------------------------------
    # Step 3: Creating the plot

    # Updating the start datatime for the plot title
    begin_date = date_update(n_samples=begin, start_datetime=start_datetime, sampling_frequency=100)

    # Creating the figure with the subplots
    fig, axs = plt.subplots(nrows=2, ncols=4, constrained_layout=True, figsize=(20, 12))
    # Title of the figure
    fig.suptitle(f'POL {name_plot}\nDate: {begin_date}', fontsize=14)

    logging.info(f"Plot of POL {name_plot}")
    # The 4 plots are repeated on two rows (uniform Y-scale below)
    for row in range(2):
        col = 0  # type: int
        for exit in ["Q1", "Q2", "U1", "U2"]:
            # Setting the Y-scale uniform on the 2nd row
            if row == 1:
                axs[row, col].sharey(axs[1, 0])

            # ------------------------------------------------------------------------------------------------------
            # Demodulation: Scientific Data
            if demodulated:
                # Creating a dict with the Scientific Data of an exit of a specific type and their new timestamps
                sci_data = demodulation(dataset=dataset, timestamps=timestamps,
                                        type=type, exit=exit, begin=begin, end=end)
                if rms:
                    # Calculate the RMS of the Scientific Data
                    rms_sd = RMS(sci_data["sci_data"], window=window, exit=exit, eoa=0, begin=begin, end=end)

                    # Plot of FFT of the RMS of the SciData DEMODULATED/TOTPOWER -----------------------------------
                    if fft:
                        f, s = scipy.signal.welch(rms_sd, fs=50, nperseg=min(len(rms_sd), nperseg), scaling="spectrum")
                        axs[row, col].plot(f[f < 25.], s[f < 25.],
                                           linewidth=0.2, marker=".", color="mediumvioletred",
                                           label=f"{name_plot[3:]}")

                    # Plot of RMS of the SciData DEMODULATED/TOTPOWER ----------------------------------------------
                    else:
                        # Smoothing of the rms of the SciData. Smooth_len=1 -> No smoothing
                        rms_sd = mob_mean(rms_sd, smooth_len=smooth_len)

                        axs[row, col].plot(sci_data["times"][begin:len(rms_sd) + begin], rms_sd,
                                           color="mediumvioletred", label=f"{name_plot[3:]}")

                else:
                    # Plot of the FFT of the SciData DEMODULATED/TOTPOWER ------------------------------------------
                    if fft:
                        f, s = scipy.signal.welch(sci_data["sci_data"][exit][begin:end], fs=50,
                                                  nperseg=min(len(sci_data["sci_data"][exit][begin:end]), nperseg),
                                                  scaling="spectrum")
                        axs[row, col].plot(f[f < 25.], s[f < 25.],
                                           linewidth=0.2, marker=".", color="mediumpurple",
                                           label=f"{name_plot[3:]}")

                    # Plot of the SciData DEMODULATED/TOTPOWER -----------------------------------------------------
                    else:
                        # Smoothing of the SciData  Smooth_len=1 -> No smoothing
                        y = mob_mean(sci_data["sci_data"][exit][begin:end], smooth_len=smooth_len)
                        axs[row, col].plot(sci_data["times"][begin:len(y) + begin], y,
                                           color="mediumpurple", label=f"{name_plot[3:]}")

            # ------------------------------------------------------------------------------------------------------
            # Output
            else:
                # If even, odd, all are equal to 0
                if not (even or odd or all):
                    # Do not plot anything
                    logging.error("No plot can be printed if even, odd, all values are all 0.")
                    raise SystemExit(1)
                else:
                    if rms:
                        rms_even = []
                        rms_odd = []
                        rms_all = []
                        # Calculate the RMS of the Scientific Output: Even, Odd, All
                        if even:
                            rms_even = RMS(dataset[type], window=window, exit=exit, eoa=2, begin=begin, end=end)
                        if odd:
                            rms_odd = RMS(dataset[type], window=window, exit=exit, eoa=1, begin=begin, end=end)
                        if all:
                            rms_all = RMS(dataset[type], window=window, exit=exit, eoa=0, begin=begin, end=end)

                        # Plot of FFT of the RMS of the Output DEM/PWR ---------------------------------------------
                        if fft:
                            if even:
                                f, s = scipy.signal.welch(rms_even, fs=50, nperseg=min(len(rms_even), nperseg),
                                                          scaling="spectrum")
                                axs[row, col].plot(f[f < 25.], s[f < 25.], color="royalblue",
                                                   linewidth=0.2, marker=".", alpha=even, label=f"Even samples")
                            if odd:
                                f, s = scipy.signal.welch(rms_odd, fs=50, nperseg=min(len(rms_odd), nperseg),
                                                          scaling="spectrum")
                                axs[row, col].plot(f[f < 25.], s[f < 25.], color="crimson",
                                                   linewidth=0.2, marker=".", alpha=odd, label=f"Odd samples")
                            if all:
                                f, s = scipy.signal.welch(rms_all, fs=100, nperseg=min(len(rms_all), nperseg),
                                                          scaling="spectrum")
                                axs[row, col].plot(f[f < 25.], s[f < 25.], color="forestgreen",
                                                   linewidth=0.2, marker=".", alpha=all, label="All samples")

                        # Plot of RMS of the Output DEM/PWR --------------------------------------------------------
                        else:
                            if even:
                                axs[row, col].plot(timestamps[begin:end - 1:2][:-window - smooth_len + 1],
                                                   mob_mean(rms_even, smooth_len=smooth_len)[:-1],
                                                   color="royalblue", alpha=even, label="Even Output")
                            if odd:
                                axs[row, col].plot(timestamps[begin + 1:end:2][:-window - smooth_len + 1],
                                                   mob_mean(rms_odd, smooth_len=smooth_len)[:-1],
                                                   color="crimson", alpha=odd, label="Odd Output")
                            if all != 0:
                                axs[row, col].plot(timestamps[begin:end][:-window - smooth_len + 1],
                                                   mob_mean(rms_all, smooth_len=smooth_len)[:-1],
                                                   color="forestgreen", alpha=all, label="All Output")

                    else:
                        # Plot of the FFT of the Output DEM/PWR ----------------------------------------------------
                        if fft:
                            if even:
                                f, s = scipy.signal.welch(dataset[type][exit][begin:end - 1:2], fs=50,
                                                          nperseg=min(len(dataset[type][exit][begin:end - 1:2]),
                                                                      nperseg),
                                                          scaling="spectrum")
                                axs[row, col].plot(f[f < 25.], s[f < 25.], color="royalblue",
                                                   linewidth=0.2, marker=".", alpha=even, label="Even samples")
                            if odd:
                                f, s = scipy.signal.welch(dataset[type][exit][begin + 1:end:2], fs=50,
                                                          nperseg=min(len(dataset[type][exit][begin + 1:end:2]),
                                                                      nperseg),
                                                          scaling="spectrum")
                                axs[row, col].plot(f[f < 25.], s[f < 25.], color="crimson",
                                                   linewidth=0.2, marker=".", alpha=odd, label="Odd samples")
                            if all:
                                f, s = scipy.signal.welch(dataset[type][exit][begin:end], fs=100,
                                                          nperseg=min(len(dataset[type][exit][begin:end]), nperseg),
                                                          scaling="spectrum")
                                axs[row, col].plot(f[f < 25.], s[f < 25.], color="forestgreen",
                                                   linewidth=0.2, marker=".", alpha=all, label="All samples")

                        # Plot of the Output DEM/PWR ---------------------------------------------------------------
                        else:
                            if even != 0:
                                axs[row, col].plot(timestamps[begin:end - 1:2][:- smooth_len],
                                                   mob_mean(dataset[type][exit][begin:end - 1:2],
                                                            smooth_len=smooth_len)[:-1],
                                                   color="royalblue", alpha=even, label="Even Output")
                            if odd != 0:
                                axs[row, col].plot(timestamps[begin + 1:end:2][:- smooth_len],
                                                   mob_mean(dataset[type][exit][begin + 1:end:2],
                                                            smooth_len=smooth_len)[:-1],
                                                   color="crimson", alpha=odd, label="Odd Output")
                            if all != 0:
                                axs[row, col].plot(timestamps[begin:end][:- smooth_len],
                                                   mob_mean(dataset[type][exit][begin:end],
                                                            smooth_len=smooth_len)[:-1],
                                                   color="forestgreen", alpha=all, label="All Output")

            # Subplots properties ----------------------------------------------------------------------------------

            # Title subplot
            axs[row, col].set_title(f'{exit}', size=15)

            # X-axis
            x_label = "Time [s]"
            if fft:
                x_label = "Frequency [Hz]"
                axs[row, col].set_xscale('log')
            axs[row, col].set_xlabel(f"{x_label}", size=10)

            # Y-axis
            y_label = "Output [ADU]"
            if fft:
                y_label = "Power Spectral Density [ADU**2/Hz]"
                axs[row, col].set_yscale('log')
            else:
                if rms:
                    y_label = "RMS [ADU]"

            axs[row, col].set_ylabel(f"{y_label}", size=10)

            # Legend
            axs[row, col].legend(prop={'size': 10}, loc=7)

            # Skipping to the following column of the subplot grid
            col += 1
    # ------------------------------------------------------------------------------------------------------------------
    # Step 4: producing the file in the correct dir
    # Creating the name of the png file: introducing _ in place of white spaces
    name_file = dir_format(name_plot)

    logging.debug(f"Title plot: {name_plot}, name file: {name_file}, name dir: {path_dir}")

    path = f'../plot/{path_dir}/'
    Path(path).mkdir(parents=True, exist_ok=True)
    fig.savefig(f'{path}{name_file}.png')

    # If true, show the plot on video
    if show:
        plt.show()
    plt.close(fig)

    return 88


def correlation_plot(array1: [], array2: [], dict1: dict, dict2: dict, time1: [], time2: [],
                     data_name1: str, data_name2: str, start_datetime: str, end_datetime: str, show=False,
                     corr_t=0.4):
    """
        Create a Correlation Plot of two dataset: two array, two dictionaries or one array and one dictionary.\n
        Parameters:\n
        - **array1**, **array2** (``array``): arrays ([]) of n1 and n2 elements
        - **dict1**, **dict2** (``dict``): dictionaries ({}) with N1, N2 keys
        - **time1**, **time2** (``array``): arrays ([]) of timestamps: not necessary if the dataset have same length.
        - **data_name1**, **data_name2** (``str``): names of the dataset. Used for titles, labels and to save the png.
        - **start_datetime** (``str``): begin date of dataset. Used for the title of the figure and to save the png.
        - **end_datetime** (``str``): end date of dataset. Used for the title of the figure and to save the png.
        - **show** (``bool``):\n
            *True* -> show the plot and save the figure\n
            *False* -> save the figure only
        - **corr_t** (``int``): if it is overcome by the correlation value of a plot, a warning is produced.\n

    """
    # Data comprehension -----------------------------------------------------------------------------------------------
    # Type check
    # array and timestamps array must be list
    if not (all(isinstance(l, list) for l in (array1, array2, time1, time2))):
        logging.error("Wrong type: note that array1, array2, time1, time2 are list.")
    # dict1 and dict2 must be dictionaries
    if not (all(isinstance(d, dict) for d in (dict1, dict2))):
        logging.error("Wrong type: note that dict1 and dict2 are dictionaries.")

    # Case 1 : two 1D arrays
    # Single plot: scatter correlation between two array
    if array1 != [] and array2 != [] and dict1 == {} and dict2 == {}:
        n_rows = 1
        n_col = 1
        fig_size = (4, 4)

    # Case 2: one 1D array and one dictionary with N keys
    # Plot 1xN: scatter correlation between an array and the exits of a dictionary
    elif array1 != [] and array2 == [] and dict1 != {} and dict2 == {}:
        logging.debug(f"I should be here!")
        # If the object are different, the sampling frequency may be different
        # hence the timestamps array must be provided to interpolate
        if time1 == [] or time2 == []:
            logging.error("Different sampling frequency: provide timestamps array.")
            raise SystemExit(1)
        else:
            n_rows = 1
            n_col = len(dict1.keys())
            fig_size = (4 * n_col, 4 * n_rows)

    # Case 3: two dictionaries with N keys
    # Plot NxN: scatter correlation between each dictionary exit
    elif array1 == [] and array2 == [] and dict1 != {} and dict2 != {}:
        n_rows = len(dict1.keys())
        n_col = len(dict2.keys())
        fig_size = (4 * n_col, 4 * n_rows)

    else:
        msg = ("Wrong data. Please insert only two of the four dataset: array1, array2, dict1 or dict2. "
               "Please do not insert array2 if there is not an array1. "
               "Please do not insert dict2 if there is not a dict1.")
        logging.error(f"{msg}")
        raise SystemExit(1)
    # ------------------------------------------------------------------------------------------------------------------

    logging.debug(f"Number of col:{n_col}, of row:{n_rows}, fig size: {fig_size}")
    data_name = f"{data_name1}-{data_name2}"
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_col, constrained_layout=True, figsize=fig_size)
    fig.suptitle(f'Correlation {data_name} \n Date: {start_datetime}', fontsize=10)

    # array1 vs array2 -------------------------------------------------------------------------------------------------
    if n_col == 1:

        # Check if the dataset is a TS and store the label for the plots
        label1 = f"{data_name1} Temperature [K]" if data_name1[0] in ("T", "E") else f"{data_name1} Output [ADU]"
        label2 = f"{data_name2} Temperature [K]" if data_name2[0] in ("T", "E") else f"{data_name2} Output [ADU]"

        # Arrays with different length must be interpolated
        if len(array1) != len(array2):
            if time1 == [] or time2 == []:
                logging.error("Different sampling frequency: provide timestamps array.")
                raise SystemExit(1)
            else:
                # Find the longest array (x) and the shortest to be interpolated
                x, short_array, label_x, label_y = (array1, array2, label1, label2) if len(array1) > len(array2) \
                    else (array2, array1, label2, label1)
                x_t, short_t = (time1, time2) if x is array1 else (time2, time1)

                # Interpolation of the shortest array
                y = np.interp(x_t, short_t, short_array)

        # Arrays with same length
        else:
            x = array1
            y = array2
            label_x = label1
            label_y = label2

        axs.plot(x, y, "*", color="firebrick", label="Corr Data")
        # XY-axis
        axs.set_xlabel(f"{label_x}")
        axs.set_ylabel(f"{label_y}")
        # Legend
        axs.legend(prop={'size': 9}, loc=4)
    # ------------------------------------------------------------------------------------------------------------------

    elif n_col > 1:
        # dict1 vs dict2 -----------------------------------------------------------------------------------------------
        if n_rows > 1:
            for r, r_exit in enumerate(dict1.keys()):
                for c, c_exit in enumerate(dict2.keys()):
                    if r == 0:
                        axs[r, c].sharey(axs[1, 0])
                        axs[r, c].sharex(axs[1, 0])

                    x = dict1[r_exit]
                    y = dict2[c_exit]
                    axs[r, c].plot(x, y, "*", color="teal", label="Corr Data")

                    # Subplot title
                    axs[r, c].set_title(f'Corr {c_exit} - {r_exit}')
                    # XY-axis
                    axs[r, c].set_xlabel(f"{data_name1} {r_exit} Output [ADU]")
                    axs[r, c].set_ylabel(f"{data_name2} {c_exit} Output [ADU]")
                    # Legend
                    axs[r, c].legend(prop={'size': 9}, loc=4)

        # array1 vs dict1 ----------------------------------------------------------------------------------------------
        else:
            for c, exit in enumerate(dict1.keys()):
                x = dict1[exit]
                y = np.interp(time2, time1, array1)
                axs[c].plot(x, y, "*", color="lawngreen", label="Corr Data")

                label_x = f"{data_name2} Output [ADU]"
                label_y = f"{data_name1} Temperature [K]"

                # Subplot title
                axs[c].set_title(f'Corr {exit}')
                # XY-axis
                axs[c].set_xlabel(f"{label_x} ")
                axs[c].set_ylabel(f"{label_y}")
                # Legend
                axs[c].legend(prop={'size': 9}, loc=4)
    else:
        return

    # Procedure to save the png of the plot in the correct dir
    # Gregorian Date [in string format]
    gdate = [Time(start_datetime), Time(end_datetime)]
    # Directory where to save all the plots of a given analysis
    date_dir = dir_format(f"{gdate[0]}__{gdate[1]}")
    path = f'../plot/{date_dir}/Correlation_Plot/'
    # Check if the dir exists. If not, it will be created.
    Path(path).mkdir(parents=True, exist_ok=True)
    fig.savefig(f'{path}{data_name}_CorrPlot.png')

    # If show is True the plot is visible on video
    if show:
        plt.show()
    plt.close(fig)
