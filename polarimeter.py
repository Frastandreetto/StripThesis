#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# This file contains part of the code used in the bachelor thesis of Francesco Andreetto (2020)
# updated to be used on the new version of the software of LSPE-STRIP
# November 1st 2022, Brescia (Italy)

from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import scipy as scipy
from matplotlib import pyplot as plt

import f_strip
from striptease import DataStorage

import f_strip as fz


########################################################################################################
# Class for a Polarimeter
########################################################################################################
class Polarimeter:

    def __init__(self, name_pol: str, path_file: Path):
        """
        Constructor
        Parameters:
        - name_pol (str): name of the polarimeter
        - path_file (Path): location of the data file (without the name of the file)
        """
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

    def Load_Pol(self):
        """
        Load all dataset in the polarimeter
        All type "DEM" and "PWR"
        All the exit "Q1", "Q2", "U1", "U2"
        """
        for type in ["DEM", "PWR"]:
            for exit in ["Q1", "Q2", "U1", "U2"]:
                self.times, self.data[type][exit] = self.ds.load_sci(mjd_range=self.tag, polarimeter=self.name,
                                                                     data_type=type, detector=exit)
        self.start_time = self.times[0].unix

    def Load_X(self, type: str):
        """
        Load only a specific type of dataset "PWR" or "DEM" in the polarimeter
        Parameter: type (str) "DEM" or "PWR"
        """
        for exit in ["Q1", "Q2", "U1", "U2"]:
            self.times, self.data[type][exit] = self.ds.load_sci(mjd_range=self.tag, polarimeter=self.name,
                                                                 data_type=type, detector=exit)
        self.start_time = self.times[0].unix

    def Clip_Values(self):
        """
        Data cleansing: scientific data with value zero at the beginning and at the end are removed from the dataset
        Control that a channel doesn't turn on before the others (maybe unuseful)
        """

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

    def STRIP_SAMPLING_FREQUENCY_HZ(self):
        """
        Strip Sampling Frequency
        It depends on the electronics hence it's the same for all polarimeters
        Note: it must be defined before time normalization
        """
        type = "DEM"
        if self.data[type] == {}:
            type = "PWR"

        self.STRIP_SAMPLING_FREQ = int(
            len(self.data[type]["Q1"]) / (self.times[-1].datetime - self.times[0].datetime).total_seconds())

    def Norm(self, norm_mode: int):
        """
        Timestamp Normalization
        Parameter: norm_mode (int) can be set in two ways:
        0) the output is expressed in function of the number of samples
        1) the output is expressed in function of the time in s from the beginning of the experience
        """
        if norm_mode == 0:
            self.times = np.arange(len(self.times))  # Number of samples
        if norm_mode == 1:
            self.times = np.arange(len(self.times)) / self.STRIP_SAMPLING_FREQ  # Seconds

    def Prepare(self, norm_mode: int):
        """
        Prepare the polarimeter in three steps:\n
            1. Clean dataset with Clip_Values()
            2. Calculate Strip Sampling Frequency
            3. Normalize timestamps
        """
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

    def Demodulation(self, type: str, exit: str) -> Dict[str, Any]:
        """
        Demodulation\n
        Calculate the Scientific data DEMODULATED or TOTAL POWER at 50Hz\n
        Timestamps are chosen as mean of the two consecutive times of the DEM/PWR data\n
        Parameters:\n
        - exit (str) "Q1", "Q2", "U1", "U2"\n
        - type (str) of data "DEM" or "PWR"
        """
        times = fz.mean_cons(self.times)
        data = {}
        if type == "PWR":
            data[exit] = fz.mean_cons(self.data[type][exit])
        if type == "DEM":
            data[exit] = fz.diff_cons(self.data[type][exit])

        sci_data = {"sci_data": data, "times": times}
        return sci_data

    # ------------------------------------------------------------------------------------------------------------------
    # PLOT FUNCTIONS
    # ------------------------------------------------------------------------------------------------------------------

    def Plot_Output(self, type: str, begin: int, end: int):
        """
        Plot the 4 exits PWR or DEM of the Polarimeter\n
        Parameters:\n
        - type (str) of data "DEM" or "PWR"\n
        - begin, end (int): interval of dataset that has to be considered\n
        """
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

    def Plot_EvenOddAll(self, type: str, even: int, odd: int, all: int, begin=100, end=-100, smooth_len=1):
        """
        Plot of Raw data DEM or PWR (Even, Odd or All)\n
        Parameters:\n
        - type (str) of data "DEM" or "PWR"\n
        - even, odd, all (int): used to set the transparency of the dataset (0=transparent, 1=visible)\n
        - begin, end (int): interval of dataset that has to be considered\n
        - smooth_len (int): number of elements on which the mobile mean is calculated
        Note: the 4 plots are repeated on two rows (uniform Y-scale below)\n
        """
        y_scale_limits = [np.inf, -np.inf]
        fig, axs = plt.subplots(nrows=2, ncols=4, constrained_layout=True, figsize=(17, 12))

        for i in range(2):
            n = 0  # type: int
            for exit in ["Q1", "Q2", "U1", "U2"]:

                axs[i, n].plot(self.times[begin:end - 1:2][:- smooth_len],
                               fz.mob_mean(self.data[type][exit][begin:end - 1:2], smooth_len=smooth_len)[:-1],
                               "b", alpha=even, label="Even Data")
                axs[i, n].plot(self.times[begin + 1:end:2][:- smooth_len],
                               fz.mob_mean(self.data[type][exit][begin + 1:end:2], smooth_len=smooth_len)[:-1],
                               "r", alpha=odd, label="Odd Data")
                axs[i, n].plot(self.times[begin:end][:- smooth_len],
                               fz.mob_mean(self.data[type][exit][begin:end], smooth_len=smooth_len)[:-1],
                               "g", alpha=all, label="All data")
                if i == 0:
                    y_scale_limits[0] = np.min([y_scale_limits[0], np.min(self.data[type][exit])])
                    y_scale_limits[1] = np.max([y_scale_limits[1], np.max(self.data[type][exit])])
                # Title
                axs[i, n].set_title(f'{type} {exit}')
                # X-axis
                if self.norm_mode == 0:
                    axs[i, n].set_xlabel("# Samples")
                if self.norm_mode == 1:
                    axs[i, n].set_xlabel("Time [s]")
                # Y-axis
                axs[i, n].set_ylabel(f"Output [{type}]")
                if i == 1:
                    axs[i, n].set_ylim(y_scale_limits[0], y_scale_limits[1])
                # Legend
                axs[i, n].legend(prop={'size': 9})

                n += 1

        eoa = EOA(even=even, odd=odd, all=all)
        fig.savefig(
            f'/home/francesco/Scrivania/Tesi/plot/{self.name}_{type}_{eoa}_smooth={smooth_len}.png')
        plt.close(fig)

    def Plot_RMS_EOA(self, type: str, window: int, even: int, odd: int, all: int, begin=100, end=-100, smooth_len=1):
        """
        Plot of Raw data DEM or PWR (Even, Odd or All)\n
        Parameters:\n
        - type (str) of data "DEM" or "PWR"
        - window (int): number of elements on which the RMS is calculated
        - even, odd, all (int): used for the transparency of the data (0=transparent, 1=visible)
        - begin, end (int): interval of dataset that has to be considered
        - smooth_len (int): number of elements on which the mobile mean is calculated
        Note: the 4 plots are repeated on two rows (uniform Y-scale below)\n
        """
        y_scale_limits = [np.inf, -np.inf]
        fig, axs = plt.subplots(nrows=2, ncols=4, constrained_layout=True, figsize=(17, 12))

        for i in range(2):
            n = 0  # type: int
            for exit in ["Q1", "Q2", "U1", "U2"]:
                rms_all = fz.mob_mean(RMS(self.data[type], window=window, exit=exit, eoa=0, begin=begin, end=end),
                                      smooth_len=smooth_len)

                axs[i, n].plot(self.times[begin:end - 1:2][:-window - smooth_len + 2],
                               fz.mob_mean(RMS(self.data[type], window=window, exit=exit, eoa=2, begin=begin, end=end),
                                           smooth_len=smooth_len),
                               "b", alpha=even, label="Even Data")
                axs[i, n].plot(self.times[begin + 1:end:2][:-window - smooth_len + 2],
                               fz.mob_mean(RMS(self.data[type], window=window, exit=exit, eoa=1, begin=begin, end=end),
                                           smooth_len=smooth_len),
                               "r", alpha=odd, label="Odd Data")
                axs[i, n].plot(self.times[begin:end][:-window - smooth_len + 2],
                               rms_all,
                               "g", alpha=all, label="All data")
                if i == 0:
                    y_scale_limits[0] = np.min([y_scale_limits[0], np.min(rms_all)])
                    y_scale_limits[1] = np.max([y_scale_limits[1], np.max(rms_all)])
                # Title
                axs[i, n].set_title(f'RMS {type} {exit}')
                # X-axis
                if self.norm_mode == 0:
                    axs[i, n].set_xlabel("# Samples")
                if self.norm_mode == 1:
                    axs[i, n].set_xlabel("Time [s]")
                # Y-axis
                axs[i, n].set_ylabel(f"RMS [{type}]")
                if i == 1:
                    axs[i, n].set_ylim(y_scale_limits[0], y_scale_limits[1])
                # Legend
                axs[i, n].legend(prop={'size': 9})

                n += 1

        eoa = EOA(even=even, odd=odd, all=all)
        fig.savefig(
            f'/home/francesco/Scrivania/Tesi/plot/{self.name}_{type}_RMS_{eoa}_smooth={smooth_len}.png')
        plt.close(fig)

    def Plot_Correlation_EvenOdd(self, type: str, begin=100, end=-100, smooth_len=1):
        """
        Plot of Raw data DEM or PWR: Even vs Odd to see the correlation\n
        Parameters:\n
        - type (str) of data "DEM" or "PWR"\n
        - begin, end (int): interval of dataset that has to be considered\n
        - smooth_len (int): number of elements on which the mobile mean is calculated
        Note: the 4 plots are repeated on two rows (uniform Y-scale below)\n
        """
        y_scale_limits = [np.inf, -np.inf]
        x_scale_limits = [np.inf, -np.inf]
        fig, axs = plt.subplots(nrows=2, ncols=4, constrained_layout=True, figsize=(17, 12))

        for i in range(2):
            n = 0  # type: int
            for exit in ["Q1", "Q2", "U1", "U2"]:
                x = fz.mob_mean(self.data[type][exit][begin:end - 1:2], smooth_len=smooth_len)
                y = fz.mob_mean(self.data[type][exit][begin + 1:end:2], smooth_len=smooth_len)
                axs[i, n].plot(x, y, "*", color="orange", label="Corr Data")
                if i == 0:
                    x_scale_limits[0] = np.min([x_scale_limits[0], np.min(x) - 0.2 * np.abs(np.mean(x))])
                    x_scale_limits[1] = np.max([x_scale_limits[1], np.max(x) + 0.2 * np.abs(np.mean(x))])
                    y_scale_limits[0] = np.min([y_scale_limits[0], np.min(y) - 0.2 * np.abs(np.mean(y))])
                    y_scale_limits[1] = np.max([y_scale_limits[1], np.max(y) + 0.2 * np.abs(np.mean(y))])
                # Title
                axs[i, n].set_title(f'Corr {type} {exit}')
                # XY-axis
                axs[i, n].set_xlabel("Even Samples")
                axs[i, n].set_ylabel(f"Odd Samples")
                if i == 1:
                    axs[i, n].set_ylim(y_scale_limits[0], y_scale_limits[1])
                    axs[i, n].set_xlim(x_scale_limits[0], x_scale_limits[1])
                # Legend
                axs[i, n].legend(prop={'size': 9})

                n += 1

        fig.savefig(
            f'/home/francesco/Scrivania/Tesi/plot/{self.name}_{type}_Correlation_EO_smooth={smooth_len}.png')
        plt.close(fig)

    def Plot_Correlation_RMS_EO(self, type: str, window: int, begin=100, end=-100, smooth_len=1):
        """
        Plot of Raw data DEM or PWR: Even vs Odd to see the correlation\n
        Parameters:\n
        - type (str) of data "DEM" or "PWR"\n
        - window (int): number of elements on which the RMS is calculated
        - begin, end (int): interval of dataset that has to be considered\n
        - smooth_len (int): number of elements on which the mobile mean is calculated
        Note: the 4 plots are repeated on two rows (uniform Y-scale below)\n
        """
        y_scale_limits = [np.inf, -np.inf, 0]  # type: []
        x_scale_limits = [np.inf, -np.inf, 0]  # type: []
        fig, axs = plt.subplots(nrows=2, ncols=4, constrained_layout=True, figsize=(17, 12))

        for i in range(2):
            n = 0  # type: int
            for exit in ["Q1", "Q2", "U1", "U2"]:
                x = fz.mob_mean(RMS(self.data[type], window=window, exit=exit, eoa=2, begin=begin, end=end),
                                smooth_len=smooth_len)
                y = fz.mob_mean(RMS(self.data[type], window=window, exit=exit, eoa=1, begin=begin, end=end),
                                smooth_len=smooth_len)
                axs[i, n].plot(x, y, "*", color="teal", label="Corr RMS")

                if i == 0:
                    x_scale_limits[0] = np.min([x_scale_limits[0], np.min(x) - 0.2 * np.abs(np.mean(x))])
                    x_scale_limits[1] = np.max([x_scale_limits[1], np.max(x) + 0.2 * np.abs(np.mean(x))])
                    y_scale_limits[0] = np.min([y_scale_limits[0], np.min(y) - 0.2 * np.abs(np.mean(y))])
                    y_scale_limits[1] = np.max([y_scale_limits[1], np.max(y) + 0.2 * np.abs(np.mean(y))])
                # Title
                axs[i, n].set_title(f'RMS Corr {type} {exit}')
                # XY-axis
                axs[i, n].set_xlabel("RMS Even Samples")
                axs[i, n].set_ylabel(f"RMS Odd Samples")
                if i == 1:
                    axs[i, n].set_ylim(y_scale_limits[0],
                                       y_scale_limits[1])
                    axs[i, n].set_xlim(x_scale_limits[0],
                                       x_scale_limits[1])
                # Legend
                axs[i, n].legend(prop={'size': 9})

                n += 1

        fig.savefig(
            f'/home/francesco/Scrivania/Tesi/plot/{self.name}_{type}_Correlation_RMS_EO_smooth={smooth_len}.png')
        plt.close(fig)

    def Plot_SciData(self, type: str, begin=100, end=-100, smooth_len=1):
        """
        Plot of Scientific data DEMODULATED or TOTAL POWER\n
        Parameters:\n
        - type (str) of data "DEM" or "PWR"\n
        - begin, end (int): interval of dataset that has to be considered\n
        - smooth_len (int): number of elements on which the mobile mean is calculated
        Note: the 4 plots are repeated on two rows (uniform Y-scale below)\n
        """
        y_scale_limits = [np.inf, -np.inf]
        if type == "DEM":
            data_name = "DEMODULATED"
        if type == "PWR":
            data_name = "TOT_POWER"

        fig, axs = plt.subplots(nrows=2, ncols=4, constrained_layout=True, figsize=(17, 12))
        for i in range(2):
            n = 0  # type: int
            for exit in ["Q1", "Q2", "U1", "U2"]:
                sci_data = self.Demodulation(type=type, exit=exit)

                axs[i, n].plot(sci_data["times"][begin:end - 1][:-smooth_len],
                               fz.mob_mean(sci_data["sci_data"][exit][begin:end - 2], smooth_len=smooth_len),
                               label=f"{data_name}")
                if i == 0:
                    y_scale_limits[0] = np.min([y_scale_limits[0], np.min(sci_data["sci_data"][exit][begin:end - 2])])
                    y_scale_limits[1] = np.max([y_scale_limits[1], np.max(sci_data["sci_data"][exit][begin:end - 2])])
                # Title
                axs[i, n].set_title(f'{data_name} {exit}')
                # X-axis
                if self.norm_mode == 0:
                    axs[i, n].set_xlabel("# Samples")
                if self.norm_mode == 1:
                    axs[i, n].set_xlabel("Time [s]")
                # Y-axis
                axs[i, n].set_ylabel(f"{data_name}")
                if i == 1:
                    axs[i, n].set_ylim(y_scale_limits[0], y_scale_limits[1])
                # Legend
                axs[i, n].legend(prop={'size': 9})

                n += 1

        fig.savefig(f'/home/francesco/Scrivania/Tesi/plot/{self.name}_{data_name}_smooth={smooth_len}.png')
        plt.close(fig)

    def Plot_RMS_SciData(self, type: str, window: int, begin=100, end=-100, smooth_len=1):
        """
        Plot of Raw data DEM or PWR (Even, Odd or All)\n
        Parameters:\n
        - type (str) of data "DEM" or "PWR"
        - window (int): number of elements on which the RMS is calculated
        - begin, end (int): interval of dataset that has to be considered
        - smooth_len (int): number of elements on which the mobile mean is calculated
        Note: the 4 plots are repeated on two rows (uniform Y-scale below)
        """
        y_scale_limits = [np.inf, -np.inf]
        if type == "DEM":
            data_name = "DEMODULATED"
        if type == "PWR":
            data_name = "TOT_POWER"

        fig, axs = plt.subplots(nrows=2, ncols=4, constrained_layout=True, figsize=(17, 12))

        for i in range(2):
            n = 0  # type: int
            for exit in ["Q1", "Q2", "U1", "U2"]:
                sci_data = self.Demodulation(type=type, exit=exit)
                rms_all = fz.mob_mean(RMS(sci_data["sci_data"], window=window, exit=exit, eoa=0, begin=begin, end=end),
                                      smooth_len=smooth_len)

                axs[i, n].plot(sci_data["times"][begin:end][:-window - smooth_len + 2], rms_all,
                               "b", label=f"RMS {data_name}")

                if i == 0:
                    y_scale_limits[0] = np.min([y_scale_limits[0], np.min(rms_all)])
                    y_scale_limits[1] = np.max([y_scale_limits[1], np.max(rms_all)])
                # Title
                axs[i, n].set_title(f'RMS {data_name} {exit}')
                # X-axis
                if self.norm_mode == 0:
                    axs[i, n].set_xlabel("# Samples")
                if self.norm_mode == 1:
                    axs[i, n].set_xlabel("Time [s]")
                # Y-axis
                axs[i, n].set_ylabel(f"RMS {data_name}")
                if i == 1:
                    axs[i, n].set_ylim(y_scale_limits[0], y_scale_limits[1])
                # Legend
                axs[i, n].legend(prop={'size': 9})

                n += 1

        fig.savefig(f'/home/francesco/Scrivania/Tesi/plot/{self.name}_{data_name}_RMS_smooth={smooth_len}.png')
        plt.close(fig)

    # ------------------------------------------------------------------------------------------------------------------
    # TIMESTAMPS JUMP ANALYSIS
    # ------------------------------------------------------------------------------------------------------------------

    def Jump_Plot(self, norm_mode: int):
        """
        Load and Prepare the polarimeter.
        Then plot the timestamps and of the Delta time between two consecutive Timestamps\n
        Parameter:
            norm_mode (int) can be set in two ways:
                0) the output is expressed in function of the number of samples
                1) the output is expressed in function of the time in s from the beginning of the experience
        """
        self.Load_Pol()
        self.Prepare(norm_mode=norm_mode)

        fig, axs = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(13, 3))
        # Times, we should see dot-shape jumps
        axs[0].plot(np.arange(len(self.times)), self.times, '*')
        axs[0].set_title(f"{self.name} Timestamps")
        axs[0].set_xlabel("# Sample")
        axs[0].set_ylabel("Time [s]")

        # Delta t
        deltat = fz.diff_cons(self.times)
        axs[1].plot(deltat, "*g")
        axs[1].set_title(f"Delta t {self.name}")
        axs[1].set_xlabel("# Sample")
        axs[1].set_ylabel("Delta t [s]")
        axs[1].set_ylim(-1.0, 1.0)

        fig.savefig(f'/home/francesco/Scrivania/Tesi/plot/{self.name}_Timestamps.png')
        plt.close(fig)


def RMS(data, window: int, exit: str, eoa: int, begin=100, end=-100):
    """
    Calculate the RMS of a vector using the rolling window
    Parameters:
    - data is a dictionary with four keys (exits) of a particular type "DEM" or "PWR"
    - window: number of elements on which the RMS is calculated
    - exit (str) "Q1", "Q2", "U1", "U2"
    - eoa (int): flag in order to calculate RMS for
        all samples (eoa=0), can be used for Demodulated and Total Power scientific data (50Hz)
        odd samples (eoa=1)
        even samples (eoa=2)
    - begin, end (int): interval of dataset that has to be considered
    """
    if eoa == 0:
        rms = np.std(fz.rolling_window(data[exit][begin:end], window), axis=1)
    if eoa == 1:
        rms = np.std(fz.rolling_window(data[exit][begin + 1:end:2], window), axis=1)
    if eoa == 2:
        rms = np.std(fz.rolling_window(data[exit][begin:end - 1:2], window), axis=1)
    return rms


def EOA(even: int, odd: int, all: int) -> str:
    """
    Return a string that contains the letters of the samples plotted:
    "E" for even (int)
    "O" for odd (int)
    "A" for all (int)
    """
    eoa = " "
    if even != 0:
        eoa += "E"
    if odd != 0:
        eoa += "O"
    if all != 0:
        eoa += "A"
    return eoa
