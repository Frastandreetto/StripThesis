#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# This file contains part of the code used in the bachelor thesis of Francesco Andreetto (2020)
# updated to be used on the new version of the software of LSPE-STRIP
# November 1st 2022, Brescia (Italy)

from pathlib import Path
from typing import List

import numpy as np

from striptease import DataStorage


########################################################################################################
# Class for a Polarimeter
########################################################################################################
class Polarimeter:

    def __init__(self, name_pol: str, path_file: Path):

        self.name = name_pol
        self.path_file = path_file

        self.ds = DataStorage(self.path_file)

        # find a better way to set the start time
        # Esiste un modo per dedurre dal nome del file hdf5 il range temporale dei dati? NOTA: gli passo un path
        tag = self.ds.get_tags(mjd_range=("2021-01-01 00:00:00", "2023-01-01 00:00:00"))
        self.tag = [x for x in tag if f"pol{name_pol}" in x.name][0]

        self.start_time = 0
        self.STRIP_SAMPLING_FREQ = 0
        self.norm_mode = 0

        self.times = []  # type: List[float]

        power = {}
        dem = {}
        self.data = {"DEM": dem, "PWR": power}

    # ------------------------------------------------------------------------------------------------------------------
    # Load all dataset in the polarimeter
    # ------------------------------------------------------------------------------------------------------------------
    def Load_Pol(self):

        for type in ["DEM", "PWR"]:
            for exit in ["Q1", "Q2", "U1", "U2"]:
                self.times, self.data[type][exit] = self.ds.load_sci(mjd_range=self.tag, polarimeter=self.name,
                                                                     data_type=type, detector=exit)
        self.start_time = self.times[0].unix

    # ------------------------------------------------------------------------------------------------------------------
    # Load only a specific typer of dataset "PWR" or "DEM" in the polarimeter
    # Parameter: str
    # ------------------------------------------------------------------------------------------------------------------
    def Load_X(self, type):

        for exit in ["Q1", "Q2", "U1", "U2"]:
            self.times, self.data[type][exit] = self.ds.load_sci(mjd_range=self.tag, polarimeter=self.name,
                                                                 data_type=type, detector=exit)
        self.start_time = self.times[0].unix

    # ------------------------------------------------------------------------------------------------------------------
    # Data cleansing: scientific data with value zero at the beginning and at the end are removed from the dataset
    # Control that a channel doesn't turn on before the others (maybe unuseful)
    # ------------------------------------------------------------------------------------------------------------------
    def Clip_Values(self):

        start_idx = np.inf
        end_idx = 0
        dem_idx = {}
        pwr_idx = {}
        nonzero_idx = {"DEM": dem_idx, "PWR": pwr_idx}
        for type in [x for x in ["DEM", "PWR"] if not self.data[x] == {}]:
            for exit in ["Q1", "Q2", "U1", "U2"]:
                # This array contains the indexes of all nonzero values
                nonzero_idx[type][exit] = np.arange(len(self.data[type][exit]))[self.data[type][exit] != 0]
                if start_idx > np.min(nonzero_idx[type][exit]):  # start_idx is the position of the first nonzero value
                    start_idx = np.min(nonzero_idx[type][exit])
                if end_idx < np.max(nonzero_idx[type][exit]):  # end_idx is the position of the last nonzero value
                    end_idx = np.max(nonzero_idx[type][exit])

        # Cleaning operations
        self.times = self.times[start_idx:end_idx + 1]
        for type in [x for x in ["DEM", "PWR"] if not self.data[x] == {}]:
            for exit in ["Q1", "Q2", "U1", "U2"]:
                self.data[type][exit] = self.data[type][exit][start_idx:end_idx + 1]

    # ------------------------------------------------------------------------------------------------------------------
    # Strip Sampling Frequency
    # It depends on the electronics hence it's the same for all polarimeters
    # Note: it must be defined before time normalization
    # ------------------------------------------------------------------------------------------------------------------
    def STRIP_SAMPLING_FREQUENCY_HZ(self):
        type = "DEM"
        if self.data[type] == {}:
            type = "PWR"

        self.STRIP_SAMPLING_FREQ = int(
            len(self.data[type]["Q1"]) / (self.times[-1].datetime - self.times[0].datetime).total_seconds())

    # ------------------------------------------------------------------------------------------------------------------
    # Timestamp nomalization
    # Parameter: norm_mode (int) can be set in two ways:
    # 0) the output is expressed in function of the number of samples
    # 1) the output is expressed in function of the time in s from the beginning of the experience
    # -------------------------------------------------------------------------------------------------------------------
    def Norm(self, norm_mode):
        # Number of samples
        if norm_mode == 0:
            self.times = np.arange(len(self.times))
        # Seconds
        if norm_mode == 1:
            self.times = np.arange(len(self.times)) / self.STRIP_SAMPLING_FREQ

    # ----------------------------------------------------------------------------------------------------------
    # Prepare the polarimeter in three steps:
    # 1. Clean dataset with Clip_Values()
    # 2. Calculate Strip Sampling Frequency
    # 3. Normalize timestamps
    # ------------------------------------------------------------------------------------------------------------
    def Prepare(self, norm_mode):
        self.norm_mode = norm_mode

        self.Clip_Values()
        if self.STRIP_SAMPLING_FREQ > 0:
            print(f"Warning: the dataset has already been normalized. "
                  f"Strip Sampling Frequency = {self.STRIP_SAMPLING_FREQ}.")
            return 0
        self.STRIP_SAMPLING_FREQUENCY_HZ()
        self.Norm(norm_mode)

        print("The dataset is now normalized.")
        ###################################################################
        # NOTA: Migliorare usando la libreria di logging
        ###################################################################
