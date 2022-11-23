#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# This file contains the main functions used in the bachelor thesis of Francesco Andreetto (2020)
# updated to be used on the new version of the software of LSPE-STRIP
# October 29th 2022, Brescia (Italy)

import numpy as np
import scipy.stats as scs
import scipy.ndimage as scn
from striptease import DataFile
from pathlib import Path
from numba import njit


def tab_cap_time(path_dataset: Path):
    """
    Create a new file txt and write the caption of a tabular\n
    Parameters:\n
    - **path_dataset** (Path: without the name of the file?)\n
    This specific function creates a tabular that collects the jumps in the dataset (JT)
    """
    # Found bug here. Need to be fixed
    new_file_name = f"JT_{path_dataset.stem}.txt"
    cap = "Name_Polarimeter\tIndex\tDelta_t\tJHD\n"

    file = open(new_file_name, "w")
    file.write(cap)
    file.close()


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
        pols.append(f"POL_{cur_pol}")
    return pols


@njit  # optimize calculations
def mean_cons(v):
    """
    Calculate consecutive means of an array.\n
    Parameters:\n
    - **v** is an array\n
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
    - **v** is an array\n
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
    -  **v** is an array
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
    - **v** is an array
    - **smooth_len** (int): number of elements on which the mobile mean is calculated
    """
    m = np.zeros(len(v) - smooth_len + 1)
    for i in np.arange(len(m)):
        m[i] = np.mean(v[i:i + smooth_len])
    return m


@njit
def find_spike(v, threshold=8.5) -> []:
    """
        Look up for 'spikes' in a given array.\n
        Calculate the mean of the max and the min of the array
        Parameters:\n
        - **v** is an array
        - **threshold** (int): value used to discern what a spike is
        """
    med_v = scn.median(v)  # type:float
    mad_v = scs.median_abs_deviation(v)

    dif = diff_cons(v)
    med_dif = scn.median(dif)
    mad_dif = scs.median_abs_deviation(dif)

    spike_idx = []

    for idx, item in enumerate(dif):
        # spike up in differences
        if item > med_dif + threshold * mad_dif:
            if v[idx * 2] > med_v + (threshold / 2.7) * mad_v:
                print(f"found spike up: {idx}")
                spike_idx.append(idx * 2)
            if v[idx * 2 + 1] < med_v - (threshold / 2.7) * mad_v:
                print(f"found spike down: {idx}")
                spike_idx.append(idx * 2 + 1)
        # spike down in differences
        if item < med_dif - threshold * mad_dif:
            if v[idx * 2] < med_v - (threshold / 2.7) * mad_v:
                print(f"found spike down: {idx}")
                spike_idx.append(idx * 2)
            if v[idx * 2 + 1] > med_v + (threshold / 2.7) * mad_v:
                print(f"found spike up: {idx}")
                spike_idx.append(idx * 2 + 1)

    return spike_idx


def remove_spike(v, N=10, threshold=8.5, gauss=True):
    """
    Find the 'spikes' in a given array and remove them.
    Parameters:\n
    - **v** is an array\n
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
    s = find_spike(v=v, threshold=threshold)
    a, b, c, d = 0, 0, 0, 0
    if len(s) == 0:
        return "No spikes detected in the dataset.\n"
    for i in s:
        print("\nStarting new removal run")
        if gauss:
            """
            Gaussian substitution
            """
            if i == 0:
                print(f"Spike in position zero: using the following {2 * N} samples")
                c = 1
                d = 1 + 2 * N
            if 0 < i < N:
                print(f"Spike in low position ({i}): using 2N samples with N={i}")
                a = -i  # N = i
                c = 1
                d = 1 + i
            if N < i < (len(v) - N):
                print(f"Spike in position {i}: using {2 * N} samples, {N} before e {N} after")
                a = -N
                c = 1
                d = 1 + N
            if i > (len(v) - N - 1):
                print(f"Spike in high position ({i}): using the previous {2 * N} samples")
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

        s = find_spike(v=v, threshold=threshold)
