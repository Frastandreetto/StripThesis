#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# This file contains part of the code used in the bachelor thesis of Francesco Andreetto (2020)
# updated to be used on the new version of the software of LSPE-STRIP
# November 1st 2022, Brescia (Italy)

from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from matplotlib import pyplot as plt

import f_strip
from striptease import DataStorage

import f_strip as fz


########################################################################################################
# Class for a Polarimeter
########################################################################################################
class Polarimeter:

    # ------------------------------------------------------------------------------------------------------------------
    # Constructor
    # Parameters:
    # - name_pol (str): name of the polarimeter
    # - path_file (Path): location of the data file (without the name of the file)
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, name_pol: str, path_file: Path):

        self.name = name_pol
        self.path_file = path_file

        self.ds = DataStorage(self.path_file)

        # Find a better way to set the start time
        # How to get it from the name of the file? Is more useful to pass a time interval?
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
    # Parameter: type (str) "DEM" or "PWR"
    # ------------------------------------------------------------------------------------------------------------------
    def Load_X(self, type: str):

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
    # Timestamp Normalization
    # Parameter: norm_mode (int) can be set in two ways:
    # 0) the output is expressed in function of the number of samples
    # 1) the output is expressed in function of the time in s from the beginning of the experience
    # ------------------------------------------------------------------------------------------------------------------
    def Norm(self, norm_mode: int):
        # Number of samples
        if norm_mode == 0:
            self.times = np.arange(len(self.times))
        # Seconds
        if norm_mode == 1:
            self.times = np.arange(len(self.times)) / self.STRIP_SAMPLING_FREQ

    # ------------------------------------------------------------------------------------------------------------------
    # Prepare the polarimeter in three steps:
    # 1. Clean dataset with Clip_Values()
    # 2. Calculate Strip Sampling Frequency
    # 3. Normalize timestamps
    # ------------------------------------------------------------------------------------------------------------------
    def Prepare(self, norm_mode: int):
        self.norm_mode = norm_mode

        self.Clip_Values()
        if self.STRIP_SAMPLING_FREQ > 0:
            print(f"Warning: the dataset has already been normalized. "
                  f"Strip Sampling Frequency = {self.STRIP_SAMPLING_FREQ}.")
            return 0
        self.STRIP_SAMPLING_FREQUENCY_HZ()
        self.Norm(norm_mode)

        print("The dataset is now normalized.")
        if norm_mode == 0:
            print("Dataset in function of sample number [#]")
        if norm_mode == 1:
            print("Dataset in function of time [s].")
        ###################################################################
        # NOTE: logging library still needed
        ###################################################################

    # ---------------------------------------------------------------------------------------------------------------------
    # Demodulation
    # Calculate the Scientific data DEMODULATED or TOTAL POWER at 50Hz
    # Timestamps are chosen as mean of the two consecutive times of the DEM/PWR data
    # Parameters:
    # - exit (str) "Q1", "Q2", "U1", "U2"
    # - type (str) "DEM" or "PWR"
    # ---------------------------------------------------------------------------------------------------------------------
    def Demodulation(self, exit: str, type: str) -> Dict[str, Any]:
        times = fz.mob_mean(self.times)
        data = {}
        if type == "PWR":
            data = fz.mean_cons(self.data[type][exit])
        if type == "DEM":
            data = fz.diff_cons(self.data[type][exit])

        sci_data = {"sci_data": data, "times": times}
        return sci_data

    # ------------------------------------------------------------------------------------------------------------------
    # Plot Functions
    # ------------------------------------------------------------------------------------------------------------------
    # Plot the 4 exits PWR or DEM of the Polarimeter
    # Parameters:
    # type (str) "PWR" or "DEM"
    # begin, end (int): interval of dataset that has to be considered
    # ------------------------------------------------------------------------------------------------------------------
    def Plot_Output(self, type: str, begin: int, end: int):
        fig = plt.figure(figsize=(20, 4))
        o = 0
        for exit in ["Q1", "Q2", "U1", "U2"]:
            o = o + 1
            ax = fig.add_subplot(1, 4, o)
            ax.plot(self.times[begin:end], self.data[type][exit][begin:end], "*")
            ax.set_title(f"{exit}")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(f"Output {type}")

        fig.savefig(
            f'/home/francesco/Scrivania/Tesi/plot/{self.name}_{type}.png')
        plt.close(fig)

    # ------------------------------------------------------------------------------------------------------------------
    # Plot of Raw data DEM or PWR (Even, Odd or All)
    # Parameters:
    # - type (str) of data DEM or PWR
    # - even, odd, all (int): used for the transparency of the datas (0=transparent, 1=visible)
    # - begin, end (int): interval of the samples I want to plot
    # ------------------------------------------------------------------------------------------------------------------
    def Plot_EvenOddAll(self, type: str, even: int, odd: int, all: int, begin=100, end=-100):
        # Put a double line, uno con true yaxis e uno con false
        y_scale_limits = [np.inf, -np.inf]
        fig, axs = plt.subplots(nrows=2, ncols=4, constrained_layout=True, figsize=(17, 12))
        n = 0  # type: int
        for i in range(2):
            for exit in ["Q1", "Q2", "U1", "U2"]:
                n = n + 1
                # Output Plot
                axs = fig.add_subplot(2, 4, n)

                axs.plot(self.times[begin:end - 1:2],
                         self.data[f"{type}"][exit][begin:end - 1:2], ".b", alpha=even, label="Even Datas")
                axs.plot(self.times[begin + 1:end:2],
                         self.data[f"{type}"][exit][begin + 1:end:2], ".r", alpha=odd, label="Odd Datas")
                axs.plot(self.times,
                         self.data[f"{type}"][exit], ".g", alpha=all, label="All datas")
                if i == 0:
                    y_scale_limits[0] = np.min([y_scale_limits[0], np.min(self.data[f"{type}"][exit])])
                    y_scale_limits[1] = np.max([y_scale_limits[1], np.max(self.data[f"{type}"][exit])])
                # Title
                axs.set_title(f'{type} {exit}')
                # X-axis
                if self.norm_mode == 0:
                    axs.set_xlabel("# Samples")
                if self.norm_mode == 1:
                    axs.set_xlabel("Time [s]")
                # Y-axis
                axs.set_ylabel(f"Output [{type}]")
                if i == 1:
                    axs.set_ylim(y_scale_limits[0], y_scale_limits[1])
                # Legend
                axs.legend(prop={'size': 9})

        fig.savefig(f'/home/francesco/Scrivania/Tesi/plot/{self.name}_{type}_EOA_{all}.png')
        plt.close(fig)

    # ------------------------------------------------------------------------------------------------------------------
    # Plot of Raw data DEM or PWR (Even, Odd or All)
    # Parameters:
    # - type (str) of data DEM or PWR
    # - window: number of elements on which the RMS is calculated
    # - even, odd, all (int): used for the transparency of the datas (0=transparent, 1=visible)
    # - begin, end (int): interval of the samples I want to plot
    # - smooth_len (int): used for the mobile mean
    # ------------------------------------------------------------------------------------------------------------------
    def Plot_RMS_EOA(self, type: str, window: int, even: int, odd: int, all: int, begin=100, end=-100, smooth_len=1):
        # Put a double line, uno con true yaxis e uno con false
        y_scale_limits = [np.inf, -np.inf]
        fig, axs = plt.subplots(nrows=2, ncols=4, constrained_layout=True, figsize=(17, 12))
        n = 0  # type: int
        for i in range(2):
            for exit in ["Q1", "Q2", "U1", "U2"]:
                n = n + 1
                # Output Plot
                axs = fig.add_subplot(2, 4, n)

                rms_all = fz.mob_mean(RMS(self.data, window=window, type=type, exit=exit, eoa=0, begin=begin, end=end),
                                      smooth_len=smooth_len)
                axs.plot(self.times[begin:end - 1:2][:-window - smooth_len + 2],
                         fz.mob_mean(RMS(self.data, window=window, type=type, exit=exit, eoa=2, begin=begin, end=end),
                                     smooth_len=smooth_len),
                         "b", alpha=even, label="Even Datas")
                axs.plot(self.times[begin + 1:end:2][:-window - smooth_len + 2],
                         fz.mob_mean(RMS(self.data, window=window, type=type, exit=exit, eoa=1, begin=begin, end=end),
                                     smooth_len=smooth_len),
                         "r", alpha=odd, label="Odd Datas")
                axs.plot(self.times[begin:end][:-window - smooth_len + 2],
                         rms_all,
                         "g", alpha=all, label="All datas")
                if i == 0:
                    y_scale_limits[0] = np.min([y_scale_limits[0], np.min(rms_all)])
                    y_scale_limits[1] = np.max([y_scale_limits[1], np.max(rms_all)])
                # Title
                axs.set_title(f'RMS {type} {exit}')
                # X-axis
                if self.norm_mode == 0:
                    axs.set_xlabel("# Samples")
                if self.norm_mode == 1:
                    axs.set_xlabel("Time [s]")
                # Y-axis
                axs.set_ylabel(f"RMS [{type}]")
                if i == 1:
                    axs.set_ylim(y_scale_limits[0], y_scale_limits[1])
                # Legend
                axs.legend(prop={'size': 9})

        fig.savefig(f'/home/francesco/Scrivania/Tesi/plot/{self.name}_{type}_RMS_EOA={all}_smooth={smooth_len}.png')
        plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------------
# Calculate the RMS of a vector using the rolling window
# Parameters:
# - window: number of elements on which the RMS is calculated
# - type (str) "DEM" or "PWR"
# - exit (str) "Q1", "Q2", "U1", "U2"
# - eoa (int): flag in order to calculate RMS for
#           all samples (eoa=0), can be used for Demodulated and Total Power scientific data (50Hz)
#           odd samples (eoa=1)
#           even samples (eoa=2)
# begin, end (int): interval of dataset that has to be considered
# ---------------------------------------------------------------------------------------------------------------------
def RMS(data, window: int, type: str, exit: str, eoa: int, begin=100, end=-100):
    if eoa == 0:
        rms = np.std(fz.rolling_window(data[type][exit][begin:end], window), axis=1)
    if eoa == 1:
        rms = np.std(fz.rolling_window(data[type][exit][begin + 1:end:2], window), axis=1)
    if eoa == 2:
        rms = np.std(fz.rolling_window(data[type][exit][begin:end - 1:2], window), axis=1)
    return rms

