#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# This file contains the main functions used in the bachelor thesis of Francesco Andreetto (2020)
# updated to be used on the new version of the software of LSPE-STRIP
# October 29th 2022, Brescia (Italy)

import csv
import json
import logging
import numpy as np
import scipy.stats as scs
import scipy.ndimage as scn

from astropy.time import TimeDelta
from numba import njit
from pathlib import Path
from rich.logging import RichHandler
from striptease import DataFile


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
    len_chunk = l//n_chunk
    spike_idx = []

    for n_rip in range(5):
        _v_ = v[n_rip * len_chunk:(n_rip+1) * len_chunk - 1]
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
            jumps["s_value"].append(dt.value[i]*86400)
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

