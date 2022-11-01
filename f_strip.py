#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# This file contains the main functions used in the bachelor thesis of Francesco Andreetto (2020)
# updated to be used on the new version of the software of LSPE-STRIP
# October 29th 2022, Brescia (Italy)

import numpy as np
from striptease import DataFile, DataStorage, Tag
from pathlib import Path
from numba import njit


# -------------------------------------------------------------------------------------------------------------------
# Create a new file txt and write the caption of a tabular
# Parameter: Path
# This specific function creates a tabular that collects the jumps in the dataset (JT)
# -------------------------------------------------------------------------------------------------------------------
def tab_cap_time(path_dataset: Path):
    new_file_name = f"JT_{path_dataset.stem}.txt"
    cap = "Name_Polarimeter\tIndex\tDelta_t\tJHD\n"

    file = open(new_file_name, "w")
    file.write(cap)
    file.close()


# --------------------------------------------------------------------------------------------------------------------
# Create a list of the polarimeters present in the datafile
# Parameter: str
# --------------------------------------------------------------------------------------------------------------------
def pol_list(path_dataset: Path) -> list:
    d = DataFile(path_dataset)
    d.read_file_metadata()
    pols = []
    for cur_pol in sorted(d.polarimeters):
        pols.append(f"POL_{cur_pol}")
    return pols


# -------------------------------------------------------------------------------------------------------------------
# Calculate consecutive means of an array
# Parameter: array
# the mean on each couple of samples of even-odd index is computed
# -------------------------------------------------------------------------------------------------------------------
@njit  # optimize calculations
def mean_cons(v):
    n = (len(v) // 2) * 2
    mean = (v[0:n:2] + v[1:n + 1:2]) / 2
    return mean


# -------------------------------------------------------------------------------------------------------------------
# Calculate consecutive difference of an array
# Parameter: array
# The difference between each sample of even-odd index is computed
# -------------------------------------------------------------------------------------------------------------------
@njit
def diff_cons(v):
    n = (len(v) // 2) * 2
    diff = (v[0:n:2] - v[1:n + 1:2])
    return diff


# -----------------------------------------------------------------------------------------------------
# Rolling Window Function
# Parameters: array, window
# Return a matrix with:
# - a number of element per line fixed by the parameter window
# - the first element of the row j is the j element of the vector
# -----------------------------------------------------------------------------------------------------
@njit
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


# ------------------------------------------------------------------------------------------------------
# Calculate a mobile mean on a number of elements given by smooth_len, used to smooth plots
# Parameters: array, smooth length
# ------------------------------------------------------------------------------------------------------
@njit
def mob_mean(a, smooth_len):
    m = np.zeros(len(a) - smooth_len + 1)
    for i in np.arange(len(m)):
        m[i] = np.mean(a[i:i + smooth_len])
    return m
