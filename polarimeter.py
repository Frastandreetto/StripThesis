#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# This file contains part of the code used in the bachelor thesis of Francesco Andreetto (2020)
# updated to be used on the new version of the software of LSPE-STRIP
# November 1st 2022, Brescia (Italy)

from pathlib import Path
from typing import List

import numpy as np
from matplotlib import pyplot as plt

from striptease import DataStorage

import f_strip as fz


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
    # Timestamp nomalization
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
        # NOTA: Migliorare usando la libreria di logging
        ###################################################################

    # ------------------------------------------------------------------------------------------------------------------
    # Plot Functions
    # ------------------------------------------------------------------------------------------------------------------
    # Plot the 4 exits DEM or PWR of the Polarimeter
    # ------------------------------------------------------------------------------------------------------------------
    def Plot(self, type: str, begin: int, end: int):
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
    # Plot of:
    # 1) Raw data DEM or PWR
    # 2) RMS using Rolling Window method
    # Parameters:
    # - kind (str) of data DEM or PWR
    # - n_samples (int): number of samples I want to plot
    # - smooth_len (int): used for the mobile mean
    # - even, odd, every (int): used for the transparency of the datas (0=transparent, 1=visible)
    # - y_scale (bool) True -> Homogeneous scale on y-axis. Default: False
    # ------------------------------------------------------------------------------------------------------------------
    def Analyze(self, type: str, n_samples: int, smooth_len: int, even: int, odd: int, every: int, y_scale=False):

        chunk_length = 5 * self.STRIP_SAMPLING_FREQ
        up = [-np.inf, -np.inf]
        down = [np.inf, np.inf]

        print(f"{type} data are now plotted.")
        # True: Homogeneous scale on Y-axis
        if y_scale:
            print("Homogeneous scale on Y-axis.")
            # Look up for max & min -------------------------------------------
            for exit in ["Q1", "Q2", "U1", "U2"]:
                # Output
                up[0] = np.max([up[0], np.max(self.data[type][exit][:n_samples - 1:2])])
                down[0] = np.min([down[0], np.min(self.data[type][exit][1:n_samples:2])])
                # RMS
                up[1] = np.max([up[1], np.max(np.std(self.data[type][exit][:n_samples - 1:2]))])
                down[1] = np.min([down[1], np.min(np.std(self.data[type][exit][1:n_samples:2]))])

            # Sth more clever needed
            up[0] = np.max(up[0]) + np.max(up[0]) / 3
            down[0] = np.min(down[0]) - np.abs(np.min(down[0]) / 3)
            up[1] = np.max(up[1]) + np.max(up[1]) / 3
            down[1] = np.min(down[1]) - np.abs(np.min(down[1]) / 3)
        # -----------------------------------------------------------------

        f1 = plt.figure(figsize=(15, 4))
        f2 = plt.figure(figsize=(15, 4))

        n = 0  # type: int
        for exit in ["Q1", "Q2", "U1", "U2"]:
            n = n + 1
            # Output Plot
            ax1 = f1.add_subplot(1, 4, n)
            ax1.plot(self.times[0:-1:2], self.data[f"{type}"][exit][0:-1:2], ".b", alpha=even, label="Even Datas")
            ax1.plot(self.times[1::2], self.data[f"{type}"][exit][1::2], ".r", alpha=odd, label="Odd Datas")
            ax1.plot(self.times, self.data[f"{type}"][exit], ".g", alpha=every, label="All datas")

            # RMS Plot
            ax2 = f2.add_subplot(1, 4, n)
            ax2.plot((self.times[:n_samples - 1:2])[:-smooth_len - chunk_length + 2], fz.mob_mean(
                np.std(fz.rolling_window(self.data[type][exit][:n_samples - 1:2], chunk_length), axis=1), smooth_len),
                     "b", alpha=even, label="Even Datas")
            ax2.plot((self.times[1:n_samples:2])[:-smooth_len - chunk_length + 2], fz.mob_mean(
                np.std(fz.rolling_window(self.data[type][exit][1:n_samples:2], chunk_length), axis=1), smooth_len),
                     "r", alpha=odd, label="Odd Datas")
            ax2.plot((self.times[:n_samples])[:-smooth_len - chunk_length + 2], fz.mob_mean(
                np.std(fz.rolling_window(self.data[type][exit][:n_samples], chunk_length), axis=1), smooth_len), "g",
                     alpha=every, label="All Datas")
            # Title
            ax1.set_title(f'{type} {exit}')
            ax2.set_title(f'{type} {exit}')
            # X-axis
            if self.norm_mode == 0:
                ax1.set_xlabel("# Samples")
                ax2.set_xlabel("# Samples")
            if self.norm_mode == 1:
                ax1.set_xlabel("Time [s]")
                ax2.set_xlabel("Time [s]")
            # Y-axis
            ax1.set_ylabel(f"Output [{type}]")
            ax2.set_ylabel(f"RMS Rolling [{type}] {exit}")
            # True: Homogeneous scale on y-axis
            if y_scale:
                ax1.set_ylim(down[0], up[0])
                ax2.set_ylim(down[1], up[1])
            # Legend
            ax1.legend(prop={'size': 9})
            ax2.legend(prop={'size': 9})

        f1.savefig(f'/home/francesco/Scrivania/Tesi/plot/{self.name}_{type}_EOA.png')
        plt.close(f1)
        f2.savefig(f'/home/francesco/Scrivania/Tesi/plot/{self.name}_{type}_EOA_RMS.png')
        plt.close(f2)
