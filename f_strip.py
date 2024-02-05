# -*- encoding: utf-8 -*-

# This file contains the main functions used in the bachelor thesis of Francesco Andreetto (2020)
# updated to be used on the new version of the pipeline for functional verification of LSPE-STRIP (2024)

# October 29th 2022, Brescia (Italy) - January 31st 2024, Bologna (Italy)

# Libraries & Modules
import csv
import json
import logging
import scipy.signal
import warnings

from astropy.time import Time, TimeDelta
from datetime import datetime
from matplotlib import pyplot as plt
from numba import njit
from pathlib import Path
from striptease import DataFile
from typing import Dict, Any

import numpy as np
import scipy.stats as scs
import scipy.ndimage as scn


def tab_cap_time(pol_name: str, file_name: str, output_dir: str) -> str:
    """
        Create a new file .csv and write the caption of a tabular\n
            Parameters:\n
        - **pol_name** (``str``): Name of the polarimeter
        - **file_name** (``str``): Name of the file to create and in which insert the caption\n
        - **output_dir** (``str``): Name of the dir where the csv file must be saved
        This specific function creates a tabular that collects the jumps in the dataset (JT).
    """
    new_file_name = f"JT_{pol_name}_{file_name}.csv"
    cap = [["# Jump", "Jump value [JHD]", "Jump value [s]", "Gregorian Date", "JHD Date"]]

    path = f'../Data_Results/{output_dir}/Time_Jump/'
    Path(path).mkdir(parents=True, exist_ok=True)
    # Open the file to append the heading
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
    # Read the Datafile
    d.read_file_metadata()
    # Initialize a list to collect pol names
    pols = []
    for cur_pol in sorted(d.polarimeters):
        # Append pol names to the list
        pols.append(f"{cur_pol}")
    return pols


@njit  # optimize calculations
def mean_cons(v):
    """
        Calculate consecutive means between the elements of an array.\n
            Parameters:\n
        - **v** is an array-like object\n
        The mean on each couple of samples of even-odd index is computed.
    """
    n = (len(v) // 2) * 2
    mean = (v[0:n:2] + v[1:n + 1:2]) / 2
    return mean


@njit
def diff_cons(v):
    """
        Calculate consecutive difference between the elements of an array.\n
            Parameters:\n
        - **v** is an array-like object\n
        The difference between each couple of samples of even-odd index is computed.
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
        # Compute the mean on a number smooth_len of elements, than move forward the window by 1 element in the array
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
    # Calculate consecutive mean of the Timestamps
    times = mean_cons(timestamps)
    data = {}

    # Calculate consecutive mean of PWR Outputs -> Get TOTAL POWER Scientific Data
    if type == "PWR":
        data[exit] = mean_cons(dataset[type][exit][begin:end])
    # Calculate consecutive differences of DEM Outputs -> Get DEMODULATED Scientific Data
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
        Accepts a vector and return a matrix with:\n
        - A number of element per row fixed by the parameter window
        - The first element of the row j is the j element of the vector
    """
    shape = v.shape[:-1] + (v.shape[-1] - window + 1, window)
    strides = v.strides + (v.strides[-1],)
    return np.lib.stride_tricks.as_strided(v, shape=shape, strides=strides)


def RMS(data: dict, window: int, exit: str, eoa: int, begin=0, end=-1) -> []:
    """
        Calculate the RMS of a vector using the rolling window\n
            Parameters:\n
        - **data** is a dictionary with four keys (exits) of a particular type *"DEM"* or *"PWR"*
        - **window**: number of elements on which the RMS is calculated
        - **exit** (``str``) *"Q1"*, *"Q2"*, *"U1"*, *"U2"*
        - **eoa** (``int``): flag used to calculate RMS for:\n
            - all samples (*eoa=0*), can be used for Demodulated and Total Power scientific data (50Hz)\n
            - odd samples (*eoa=1*)\n
            - even samples (*eoa=2*)\n
        - **begin**, **end** (``int``): interval of dataset that has to be considered
    """
    rms = []
    if eoa == 0:
        try:
            rms = np.std(rolling_window(data[exit][begin:end], window), axis=1)
        except ValueError as e:
            logging.warning(f"{e}. "
                            f"Impossible to compute RMS.\n\n")
    elif eoa == 1:
        try:
            rms = np.std(rolling_window(data[exit][begin + 1:end:2], window), axis=1)
        except ValueError as e:
            logging.warning(f"{e}. "
                            f"Impossible to compute RMS.\n\n")
    elif eoa == 2:
        try:
            rms = np.std(rolling_window(data[exit][begin:end - 1:2], window), axis=1)
        except ValueError as e:
            logging.warning(f"{e}. "
                            f"Impossible to compute RMS.\n\n")
    else:
        logging.error("Wrong EOA value: it must be 0,1 or 2.")
        raise SystemExit(1)
    return rms


def EOA(even: int, odd: int, all: int) -> str:
    """
        Parameters:\n
        - **even**, **odd**, **all** (``int``)
        If these variables are different from zero, this function returns a string with the corresponding letters:\n
        - "E" for even (``int``)\n
        - "O" for odd (``int``)\n
        - "A" for all (``int``)\n
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
        Return a list in which each element is a tuple of 3 values.
        Those values can be 0 or 1 depending on the letters (e, o, a) provided.
        Note: if a letter is present in the eoa_str, then that letter will assume both value 0 and 1. Only 0 otherwise.

            Parameters:\n
        - **eoa_str** (``str``): string of 0,1,2 or 3 letters from a combination of the letters e, o and a
    """
    # Initialize a dictionary with 0 values for e,o,a keys
    eoa_dict = {"E": [0], "O": [0], "A": [0]}
    eoa_list = [char for char in eoa_str]

    # If a letter appears also the value 1 is included in the dictionary
    for key in eoa_dict.keys():
        if key in eoa_list:
            eoa_dict[key].append(1)

    # Store the combinations of 0 and 1 depending on which letters were provided
    eoa_combinations = [(val1, val2, val3)
                        for val1 in eoa_dict["E"]
                        for val2 in eoa_dict["O"]
                        for val3 in eoa_dict["A"]]

    return eoa_combinations


def letter_combo(in_str: str) -> []:
    """
        Return a list in which each element is a combination of E,O,A letters.

        Parameters:\n
        - **in_str** (``str``): generic string of max 3 letters
    """
    result = []

    for length in range(1, 4):
        for i in range(len(in_str) - length + 1):
            result.append(in_str[i:i + length])

    return result


def find_spike(v, data_type: str, threshold=4.4, n_chunk=10) -> []:
    """
        Look up for 'spikes' in a given array.\n
        Calculate the median and the mad and uses those to discern spikes.

            Parameters:\n
        - **v** is an array-like object
        - **type** (str): if "DEM" look for spikes in two sub arrays (even and odd output) if "FFT" select only spike up
        - **threshold** (int): value used to discern what a spike is
        - **n_chunk** (int): n of blocks in which v is divided. On every block the median is computed to find spikes.
    """
    # Initialize a spike list to collect the indexes of the problematic samples
    spike_idx = []
    # Number of steps of the algorithm
    steps = 1
    # If DEM output look for spike on even and odd: two steps needed
    if data_type == "DEM":
        steps = 2

    # Start spike algorithm
    for i in range(steps):
        # logging.debug(f"Step: {i+1}/{steps}.")
        new_v = v[i:-1 - i:steps]
        # Length of the new vector
        l = len(new_v)
        # Calculate the length of a chunk used to divide the array
        len_chunk = l // n_chunk
        # Repeat the research of the spikes on every chunk of data
        for n_rip in range(n_chunk):
            # Creating a sub array dividing the new_v in n_chunk
            _v_ = new_v[n_rip * len_chunk:(n_rip + 1) * len_chunk - 1]
            # Calculate the Median
            med_v = scn.median(_v_)  # type:float
            # Calculate the Mean Absolute Deviation
            mad_v = scs.median_abs_deviation(_v_)

            for idx, item in enumerate(_v_):
                if item > med_v + threshold * mad_v or item < med_v - threshold * mad_v:
                    s_idx = n_rip * len_chunk + idx
                    if data_type == "DEM":
                        s_idx = s_idx * 2 + i
                    spike_idx.append(s_idx)

            # If the data_type is an FFT
            if data_type == "FFT":
                # Selecting local spikes UP: avoiding contour spikes
                spike_idx = [i for i in spike_idx if v[i] > v[i - 1] and v[i] > v[i + 1]]

    if len(spike_idx) > 0:
        logging.warning(f"Found Spike in {data_type}!\n")

    return spike_idx


def select_spike(spike_idx: list, s: list, freq: list) -> []:
    """
        Select the most relevant spikes in an array of FFT data

            Parameters:\n
        - **spike_idx** (``list``): is an array-like object containing the indexes of the spikes present in the s array
        - **s** (``list``): is an array-like object that contains spikes
        - **freq** (``list``): is an array-like object that contains the frequency corresponding to the s values
    """
    # Select only the most "significant" spikes
    idx_sel = []
    # Divide the array in sub-arrays on the base of the frequency
    for a in range(-16, 8):
        s_spike = [s[i] for i in spike_idx if (10 ** (a / 4) < freq[i] < 10 ** ((a + 1) / 4))]
        # Keep only the idx of the maxes
        idx_sel += [i for i in spike_idx if
                    (10 ** (a / 4) < freq[i] < 10 ** ((a + 1) / 4)) and s[i] == max(s_spike)]
    return idx_sel


def find_jump(v, exp_med: float, tolerance: float) -> {}:
    """
        Find the 'jumps' in a given Time astropy object: the samples should be consequential with a fixed growth rate.
        Hence, their consecutive differences should have an expected median within a certain tolerance.

            Parameters:\n
        - **v** is a Time object from astropy => i.e. Polarimeter.times\n
        - **exp_med** (``float``) expected median (in seconds) of the TimeDelta between two consecutive values of v
        - **tolerance** (``float``) threshold number of seconds over which a TimeDelta is considered as an error\n

            Return:\n
        - **jumps** a dictionary containing five keys:
            - **n** (``int``) is the number of jumps found
            - **idx** (``int``) index of the jump in the array
            - **value** (``float``) is the value of the jump in JHD
            - **s_value** (``float``) is the value of the jump in seconds
            - **median_ok** (``bool``) True if there is no jump in the vector, False otherwise
    """
    # Create a TimeDelta object from the Time object given in input
    dt = (v[1:] - v[:-1]).sec  # type: TimeDelta

    # Calculate the median of the TimeDelta
    med_dt = np.median(dt)
    median_ok = True

    # If the tolerance is overcome -> a warning message is produced
    if np.abs(np.abs(med_dt) - np.abs(exp_med)) > tolerance:
        msg = f"Median is out of range: {med_dt}, expected {exp_med}."
        logging.warning(msg)
        median_ok = False

    # Discrepancy between dt and their median
    err_t = dt - med_dt

    # Initializing the lists with the information about time jumps
    idx = []
    value = []
    s_value = []
    n = 0

    # Initializing the dict with the information about time jumps
    jumps = {"n": n, "idx": idx, "value": value, "s_value": s_value,
             "median": med_dt, "exp_med": exp_med, "tolerance": tolerance, "median_ok": median_ok,
             "5per": np.percentile(dt, 5), "95per": np.percentile(dt, 95)}

    # Store the info
    for i, item in enumerate(err_t):
        # logging.debug(f"Discrepancy value: {item}")
        if np.abs(item) > tolerance:
            jumps["n"] += 1
            jumps["idx"].append(i)
            # Convert the value in days
            jumps["value"].append(dt[i] / 86400)
            jumps["s_value"].append(dt[i])

    return jumps


def dir_format(old_string: str) -> str:
    """
        Take a string and return a new string changing white spaces into underscores, ":" into "-" and removing ".000"

            Parameters:\n
        - **old_string** (``str``)
    """
    new_string = old_string.replace(" ", "_")
    new_string = new_string.replace(".000", "")
    new_string = new_string.replace(":", "-")
    return new_string


def csv_to_json(csv_file_path: str, json_file_path):
    """
        Convert a csv file into a json file.

            Parameters:\n
        - **csv_file_path** (``str``): path of the csv file that have to be converted
        - **json_file_path** (``str``): path of the json file converted
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

    # Check if the Json Path exists, if not it is created
    # Note: 61 is the number of characters of the path of the date_dir + the json dir
    json_dir = json_file_path[:61]
    logging.info(json_dir)
    Path(json_dir).mkdir(parents=True, exist_ok=True)

    # convert python json_array to JSON String and write to file
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json_string = json.dumps(json_array, indent=4)
        json_file.write(json_string)


def merge_report(md_reports_path: str, total_report_path: str):
    """
        Merge together all the md report files into a single md report file.

            Parameters:\n
        - **md_reports_path** (``str``): path of the md files that have to be merged
        - **total_report_path** (``str``): path of the md file merged
    """
    # Ensure the output directory exists, create it if not
    output_directory = Path(total_report_path).parent
    output_directory.mkdir(parents=True, exist_ok=True)

    # List all files .md that start with a number in the directory
    files = [f for f in Path(md_reports_path).iterdir() if f.suffix == '.md' and f.name[0].isdigit()]

    # Sort files based on the number at the beginning of their names
    files_sorted = sorted(files, key=lambda x: (int(x.name.split('_')[0]), x.name))

    # Create or overwrite the destination file
    with open(total_report_path, 'w', encoding='utf-8') as outfile:
        for file_path in files_sorted:
            with file_path.open('r', encoding='utf-8') as infile:
                outfile.write(infile.read() + '\n\n')


def name_check(names: list) -> bool:
    """
        Check if the names of the polarimeters in the list are wrong: not the same as the polarimeters of Strip.

            Parameters:\n
        - **names** (``list``): list of the names of the polarimeters
    """
    for n in names:
        # Check if the letter corresponds to one of the tiles of Strip
        if n[0] in (["B", "G", "I", "O", "R", "V", "W", "Y"]):
            pass
        else:
            return False
        # Check if the number is correct
        if n[1] in (["0", "1", "2", "3", "4", "5", "6", "7"]):
            pass
        else:
            return False
        # Check the white space after every polarimeter
        try:
            if n[2] != "":
                return False
        except IndexError:
            pass

        # The only exception is W7
        if n == "W7":
            return False
    return True


def datetime_check(date_str: str) -> bool:
    """
        Check if the string is in datatime format "YYYY-MM-DD hh:mm:ss" or not.

            Parameters:\n
        - **date** (``str``): string with the datetime
    """
    date_format = "%Y-%m-%d %H:%M:%S"
    try:
        datetime.strptime(date_str, date_format)
        return True
    except ValueError:
        return False


def date_update(start_datetime: str, n_samples: int, sampling_frequency: int, ms=False) -> Time:
    """
    Calculates and returns the new Gregorian date in which the analysis begins, given a number of samples that
    must be skipped from the beginning of the dataset.

        Parameters:\n
    - **start_datetime** (``str``): start time of the dataset
    - **n_samples** (``int``): number of samples that must be skipped\n
    - **sampling_freq** (``int``): number of data collected per second
    - **ms** (``bool``): if True the new Gregorian date has also milliseconds
    """
    # Convert the str in a Time object: Julian Date MJD
    jdate = Time(start_datetime).mjd
    # A second expressed in days unit
    s = 1 / 86_400
    # Julian Date increased
    jdate += s * (n_samples / sampling_frequency)
    # New Gregorian Date
    if not ms:
        new_date = Time(jdate, format="mjd").to_datetime().strftime("%Y-%m-%d %H:%M:%S")
    else:
        new_date = Time(jdate, format="mjd").to_datetime().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    return new_date


def same_length(array1, array2) -> []:
    """
        Check if the two array are of the same length. If not, the longer becomes as long as the smaller
            Parameters:\n
        - **array1**, **array2** (``array``): data arrays.
    """
    l1 = len(array1)
    l2 = len(array2)
    array1 = array1[:min(l1, l2)]
    array2 = array2[:min(l1, l2)]
    return [array1, array2]


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
        - **pol_name** (``str``): name of the polarimeter we want to analyze
        - **dataset** (``dict``): dictionary containing the dataset with the output of a polarimeter
        - **timestamps** (``list``): list containing the Timestamps of the output of a polarimeter
        - **start_datetime** (``str``): start time
        - **end_datetime** (``str``): end time
        - **begin**, **end** (``int``): interval of dataset that has to be considered\n

        - **type** (``str``): defines the scientific output, *"DEM"* or *"PWR"*\n
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
        name_plot += " TOTPOWER" if demodulated else f" {type} {EOA(even, odd, all)}"
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
                # Avoid UserWarning on y-axis log-scale
                try:
                    # Set the y-axis of the current plot as the first of the raw
                    axs[row, col].sharey(axs[1, 0])
                except ValueError as e:
                    logging.warning(f"{e} "
                                    f"Negative data found in Spectral Analysis (FFT): impossible to use log scale.\n\n")
                    continue
                except Exception as e:
                    logging.warning(f"{e} "
                                    f"Negative data found in Spectral Analysis (FFT): impossible to use log scale.\n\n")
                    continue

            # ------------------------------------------------------------------------------------------------------
            # Demodulation: Scientific Data
            if demodulated:
                try:
                    # Creating a dict with the Scientific Data of an exit of a specific type and their new timestamps
                    sci_data = demodulation(dataset=dataset, timestamps=timestamps,
                                            type=type, exit=exit, begin=begin, end=end)
                    if rms:
                        # Calculate the RMS of the Scientific Data
                        rms_sd = RMS(sci_data["sci_data"], window=window, exit=exit, eoa=0, begin=begin, end=end)

                        # Plot of FFT of the RMS of the SciData DEMODULATED/TOTPOWER -----------------------------------
                        if fft:
                            f, s = scipy.signal.welch(rms_sd, fs=50, nperseg=min(len(rms_sd), nperseg),
                                                      scaling="spectrum")
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

                except ValueError as e:
                    logging.warning(f"{e}. Impossible to process {name_plot}.\n\n")
                    pass
            # ------------------------------------------------------------------------------------------------------
            # Output
            else:
                # If even, odd, all are equal to 0
                if not (even or odd or all):
                    # Do not plot anything
                    logging.error("No plot can be printed if even, odd, all values are all 0.")
                    raise SystemExit(1)
                else:
                    try:
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

                    except ValueError as e:
                        logging.warning(f"{e}. Impossible to process {name_plot}. \n\n")
                        pass

                    else:
                        try:
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

                        except ValueError as e:
                            logging.warning(f"{e}. Impossible to process {name_plot}.\n\n")
                            pass

            # Subplots properties ----------------------------------------------------------------------------------

            # Title subplot
            axs[row, col].set_title(f'{exit}', size=15)

            # Treat UserWarning as errors to catch them
            warnings.simplefilter("ignore", UserWarning)

            # X-axis
            x_label = "Time [s]"
            if fft:
                x_label = "Frequency [Hz]"

                try:
                    axs[row, col].set_xscale('log')
                except ValueError as e:
                    logging.warning(f"{e} "
                                    f"Negative data found in Spectral Analysis (FFT): impossible to use log scale.\n\n")
                    continue
                except Exception as e:
                    logging.warning(f"{e} "
                                    f"Negative data found in Spectral Analysis (FFT): impossible to use log scale.\n\n")
                    continue

            axs[row, col].set_xlabel(f"{x_label}", size=10)

            # Y-axis
            y_label = "Output [ADU]"
            if fft:
                y_label = "Power Spectral Density [ADU**2/Hz]"

                try:
                    axs[row, col].set_yscale('log')
                except ValueError as e:
                    logging.warning(f"{e} "
                                    f"Negative data found in Spectral Analysis (FFT): impossible to use log scale.\n\n")
                    continue
                except Exception as e:
                    logging.warning(f"{e} "
                                    f"Negative data found in Spectral Analysis (FFT): impossible to use log scale.\n\n")
                    continue

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

    path = f'../Data_Results/{path_dir}/'
    Path(path).mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(f'{path}{name_file}.png')
    except ValueError as e:
        logging.warning(f"{e}. Impossible to save the pictures.\n\n")
        pass

    # If true, show the plot on video
    if show:
        plt.show()
    plt.close(fig)

    return 88
