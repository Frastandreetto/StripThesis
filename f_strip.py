#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# This file contains the main functions used in the bachelor thesis of Francesco Andreetto (2020)
# updated to be used on the new version of the software of LSPE-STRIP
# October 29th 2022, Brescia (Italy)

import numpy as np
from striptease import DataFile
from pathlib import Path
from numba import njit


def tab_cap_time(path_dataset: Path):
    """
    Create a new file txt and write the caption of a tabular\n
    Parameter: Path (without the name of the file?)\n
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
    Parameter: Path (comprehensive of the name of the dataset file)
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
    Parameter: v is an array\n
    The mean on each couple of samples of even-odd index is computed.
    """
    n = (len(v) // 2) * 2
    mean = (v[0:n:2] + v[1:n + 1:2]) / 2
    return mean


@njit
def diff_cons(v):
    """
    Calculate consecutive difference of an array.\n
    Parameter: v is an array\n
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
    -  v is an array
    - window (int)
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
    - v is an array
    - smooth_len (int): number of elements on which the mobile mean is calculated
    """
    m = np.zeros(len(v) - smooth_len + 1)
    for i in np.arange(len(m)):
        m[i] = np.mean(v[i:i + smooth_len])
    return m
