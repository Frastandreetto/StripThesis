#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# This file contains the class Thermal_Sensors
# Use this class with the new version of the pipeline for functional verification of LSPE-STRIP (2023).

# Creation: August 15th 2023, Brescia (Italy)

# Libraries & Modules
import logging
import scipy

import numpy as np

from astropy.time import Time
from matplotlib import pyplot as plt
from striptease import DataStorage
from scipy import signal
from pathlib import Path
from rich.logging import RichHandler

# MyLibraries & MyModules
import f_strip as fz

# Use the module logging to produce nice messages on the shell
logging.basicConfig(level="INFO", format='%(message)s',
                    datefmt="[%X]", handlers=[RichHandler()])


########################################################################################################
# Class for the Thermal_Sensors
########################################################################################################
class Thermal_Sensors:

    def __init__(self, path_file: str, start_datetime: str, end_datetime: str, status: int, nperseg_thermal: int):
        """
        Constructor

            Parameters:
                - **path_file** (``str``): location of the data file (without the name of the file)
                - **start_datetime** (``str``): start time
                - **end_datetime** (``str``): end time
                - **status** (``int``): defines the status of the multiplexer of the thermal sensors to analyze: 0 or 1.
                - **nperseg_thermal** (``int``): number of elements of thermal measures on which the fft is calculated.
                Then the average of all periodograms is computed to produce the spectrogram.
                Changing this parameter allow to reach lower frequencies in the FFT plot:
                in particular, the limInf of the x-axis is fs/nperseg.

        """
        # Member DataStorage: used to load and store thermal measures
        self.ds = DataStorage(path_file)

        # Julian Date MJD
        # Used by load_cryo to store the thermal measures
        self.date = [Time(start_datetime).mjd, Time(end_datetime).mjd]
        # Gregorian Date [in string format]
        self.gdate = [Time(start_datetime), Time(end_datetime)]
        # Directory where to save all plot for a given analysis
        self.date_dir = fz.dir_format(f"{self.gdate[0]}__{self.gdate[1]}")

        self.status = status
        # List containing the names of the TS divided in the two states: "0" and "1".
        # The TS in each state are divided in groups that suggest their positions on Strip or their functions.
        if self.status == 0:
            self.ts_names = {
                "TILES": ["TS-CX4-Module-G", "TS-CX6-Module-O", "TS-CX2-Module-V"],
                "FRAME": ["TS-CX10-Frame-120", "TS-DT6-Frame-South"],
                "POLAR": ["TS-CX12-Pol-W", "TS-CX14-Pol-Qy"],
                "100-200K": ["TS-CX16-Filter", "TS-DT3-Shield-Base"],  # , "TS-SP2-L-Support" # Excluded for now
                # "VERIFY": ["EX-CX18-SpareCx"],  # Excluded for now
                "COLD_HEAD": ["TS-SP1-SpareDT"]}
        elif self.status == 1:
            self.ts_names = {
                "TILES": ["TS-CX3-Module-B", "TS-CX7-Module-I", "TS-CX1-Module-R", "TS-CX5-Module-Y"],
                "FRAME": ["TS-CX8-Frame-0", "TS-CX9-Frame-60", "TS-CX11-Frame-North", "TS-CX15-IF-Frame-0"],
                "POLAR": ["TS-CX13-Pol-Qx"],
                "100-200K": ["TS-DT5-Shield-Side"],  # , "TS-CX17-Wheel-Center" # Excluded for now
                "VERIFY": ["EX-DT2-SpareDT"],
            }
        # The following TS was also excluded during the system level tests in March 2023.
        # Pay attention if it must be included for your purpose now!
        # TS-SP2-SpareCx

        # Thermal Measures
        # List of the timestamps of every measure
        thermal_times = []
        # Dictionary for the raw/calibrated dataset
        raw = {}
        calibrated = {}
        thermal_data = {"raw": raw, "calibrated": calibrated}
        # Dictionary of all the measures
        self.ts = {"thermal_times": thermal_times, "thermal_data": thermal_data}

        # nperseg: number of elements on which the periodogram is calculated.
        self.nperseg_thermal = nperseg_thermal

        # --------------------------------------------------------------------------------------------------------------
        # Warnings
        # Which warnings are we expecting?
        time_warning = []
        corr_warning = []
        # spike_warning = []
        self.warnings = {"time_warning": time_warning,
                         "corr_warning": corr_warning,
                         # "spike_warning": spike_warning
                         }

    # ------------------------------------------------------------------------------------------------------------------
    # THERMIC METHODS
    # ------------------------------------------------------------------------------------------------------------------
    def Load_TS(self):
        """
        Load all Thermal Sensor's data taking the names from the list in the constructor.
        """
        # Use a for with the function "zip" to tell the function load_cryo below to store raw or calibrated data.
        for calib, bol in zip(["raw", "calibrated"], [True, False]):
            for group in self.ts_names.keys():
                for sensor_name in self.ts_names[group]:
                    self.ts["thermal_times"], self.ts["thermal_data"][calib][sensor_name] \
                        = self.ds.load_cryo(self.date, sensor_name, get_raw=bol)
                    # Conversion to list to better handle the data array
                    self.ts["thermal_data"][calib][sensor_name] = list(self.ts["thermal_data"][calib][sensor_name])

    def Norm_TS(self) -> []:
        """
        Check if the TIME array and the CALIBRATED DATA array have the same length.
        Normalize all Thermal Sensor's timestamps putting one every 30 seconds from the beginning of the dataset.
        """
        # Check if the TIME array and the CALIBRATED DATA array have the same length
        # Return a list of problematic TS
        problematic_ts = []
        for group in self.ts_names.keys():
            for sensor_name in self.ts_names[group]:
                len_times = len(self.ts["thermal_times"])
                len_data = len(self.ts["thermal_data"]["calibrated"][sensor_name])
                if len_times != len_data:
                    if len_times == len_data + 1 and self.status == 1:
                        self.ts["thermal_times"] = self.ts["thermal_times"][1:]
                    else:
                        # Save a warning message that will be print on video and on the report
                        msg = f"The Thermal sensor: {sensor_name} has a sampling problem.\n"
                        logging.error(msg)
                        self.warnings["time_warning"].append(msg + "<br />")
                        problematic_ts.append(sensor_name)

        # Set the starting point for the new timestamps: 0s for status 0 and 10s for status 1
        if self.status == 0:
            start = 0.
        elif self.status == 1:
            start = 10.
        else:
            start = np.nan
            logging.error("Invalid status value. Please choose between the values 0 and 1 for a single analysis.")
            SystemExit(1)

        # Assign new timestamps equally spaced every 30s
        self.ts["thermal_times"] = start + np.arange(start=0, stop=len(self.ts["thermal_times"]) * 30, step=30)
        # Conversion to list to better handle the data array
        self.ts["thermal_times"] = list(self.ts["thermal_times"])

        return problematic_ts

    def Analyse_TS(self) -> {}:
        """
        Analise all Thermal Sensors' output: calculate the mean the std deviation for both raw and calibrated samples.
        """
        # Dictionary for raw and calibrated mean
        raw_m = {}
        cal_m = {}
        # Dictionary for raw and calibrated std dev
        raw_std = {}
        cal_std = {}
        # Dictionary for raw and calibrated nan percentage
        raw_nan = {}
        cal_nan = {}
        # Dictionary for raw and calibrated max
        raw_max = {}
        cal_max = {}
        # Dictionary for raw and calibrated min
        raw_min = {}
        cal_min = {}

        results = {
            "raw": {"max": raw_max, "min": raw_min, "mean": raw_m, "dev_std": raw_std, "nan_percent": raw_nan},
            "calibrated": {"max": cal_max, "min": cal_min, "mean": cal_m, "dev_std": cal_std, "nan_percent": cal_nan}
        }

        for calib in ["raw", "calibrated"]:
            for group in self.ts_names.keys():
                for sensor_name in self.ts_names[group]:
                    # Initialize the Nan percentage to zero
                    results[calib]["nan_percent"][sensor_name] = 0.

                    # Collect the TS data in the variable data to better handle them
                    data = self.ts["thermal_data"][calib][sensor_name]
                    # Calculate the mean of the TS data
                    m = np.mean(data)

                    # If the mean is Nan, at least one of the measures is Nan
                    if np.isnan(m):
                        # If there is no TS data, the Nan percentage is 100%
                        if len(data) == 0:
                            logging.warning(f"No measures found for {sensor_name}.")
                            results[calib]["nan_percent"][sensor_name] = 100.
                        # Otherwise, compute the Nan percentage
                        else:
                            # Find the number of nan measures in data
                            n_nan = len([t for t in np.isnan(data) if t == True])
                            results[calib]["nan_percent"][sensor_name] = round((n_nan / len(data)), 4) * 100.

                        # If the Nan percentage is smaller than 5%, the dataset is valid
                        if results[calib]["nan_percent"][sensor_name] < 5:
                            # The Nan values are found and removed
                            data = np.delete(data, np.argwhere(np.isnan(data)))
                            # The mean is calculated again
                            m = np.mean(data)

                    # Check if the std dev is zero
                    if np.std(data) == 0:
                        msg = f"Std Dev for {sensor_name} is 0 "
                        # Check if it is due to the fact that there is only one datum
                        if len(self.ts["thermal_data"][calib][sensor_name]) == 1:
                            logging.warning(msg + "because there is only one measure in the dataset.")
                        else:
                            logging.warning(msg)

                    # If possible collect the other variables of interests: max, min, mean and std deviation
                    try:
                        results[calib]["max"][sensor_name] = max(data)
                    except ValueError:
                        results[calib]["max"][sensor_name] = -np.inf
                        logging.error("No measure found: impossible to find the max of the dataset.")

                    try:
                        results[calib]["min"][sensor_name] = min(data)
                    except ValueError:
                        results[calib]["min"][sensor_name] = np.inf
                        logging.error("No measure found: impossible to find the min of the dataset.")

                    results[calib]["mean"][sensor_name] = m
                    results[calib]["dev_std"][sensor_name] = np.std(data)

        return results

    ####################################################################################################################
    # Check this to fill properly in the Report
    ####################################################################################################################
    def Thermal_table(self, results) -> str:
        """
        Create a string with the md code for a table of thermal results.
        In the table there are the following info:  the sensor name, the status of acquisition (0 or 1),
        the group of the sensor, the max value, the min value, the mean, the standard deviation
        and the NaN percentage found for both RAW and calibrated (CAL) Temperatures.
        """
        md_table = " "
        for calib in ['raw', 'calibrated']:
            md_table += (f"\n\n"
                         f"{calib} data"
                         f"\n\n"
                         "| TS Name | Status | Group | Max value [K] | Min value [K] | Mean [K] | Std_Dev [K] | NaN % |"
                         "\n"
                         "|:-------:|:------:|:-----:|:-------------:|:-------------:|:--------:|:-----------:|:-----:|"
                         "\n"
                         )
            for group in self.ts_names.keys():
                for sensor_name in self.ts_names[group]:
                    md_table += (f"|{sensor_name}|{self.status}|{group}|"
                                 f"{round(results[calib]['max'][sensor_name], 4)}|"
                                 f"{round(results[calib]['min'][sensor_name], 4)}|"
                                 f"{round(results[calib]['mean'][sensor_name], 4)}|"
                                 f"{round(results[calib]['dev_std'][sensor_name], 4)}|"
                                 f"{round(results[calib]['nan_percent'][sensor_name], 4)}"
                                 f"\n")
        return md_table

    def Plot_TS(self, show=False):
        """
        Plot all the calibrated acquisitions of Thermal Sensors.\n

            Parameter:\n
            - **show** (``bool``): *True* -> show the plot and save the figure, *False* -> save the figure only
        """

        col = ["cornflowerblue", "indianred", "limegreen", "gold"]

        # Prepare the shape of the fig: in each row there is a subplot with the data of the TS of a same group.
        n_rows = len(self.ts_names.keys())
        fig, axs = plt.subplots(nrows=n_rows, ncols=1, constrained_layout=True, figsize=(13, 15))

        # Set the title of the figure
        fig.suptitle(f'Plot Thermal Sensors status {self.status} - Date: {self.gdate[0]}', fontsize=10)

        # Plot the dataset
        for i, group in enumerate(self.ts_names.keys()):
            for j, sensor_name in enumerate(self.ts_names[group]):
                # Make sure that the time array and the data array are of the same length
                values = fz.same_length(self.ts["thermal_times"], self.ts["thermal_data"]["calibrated"][sensor_name])
                # Plot the TS data vs time
                axs[i].scatter(values[0], values[1], marker=".", color=col[j], label=sensor_name)

                axs[i].set_xlabel("Time [s]")
                axs[i].set_ylabel("Temperature [K]")
                axs[i].set_title(f"TS GROUP {group} - Status {self.status}")
                axs[i].legend(prop={'size': 9}, loc=7)

        # Procedure to save the png of the plot in the correct dir
        path = f"../plot/{self.date_dir}/Thermal_Output/"
        Path(path).mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{path}Thermal_status_{self.status}.png')

        # If show is True the plot is visible on video
        if show:
            plt.show()
        plt.close(fig)

    def Plot_FFT_TS(self, show=False):
        """
        Plot the FFT of the calibrated acquisitions of Thermal Sensors of the polarimeter.\n
            Parameters:\n
         - **show** (``bool``): *True* -> show the plot and save the figure, *False* -> save the figure only
        """

        # Prepare the shape of the fig: in each row there is a subplot with the data of the TS of a same group.
        n_rows = len(self.ts_names.keys())
        fig, axs = plt.subplots(nrows=n_rows, ncols=1, constrained_layout=True, figsize=(15, 10))

        # Set the title of the figure
        fig.suptitle(f'Plot Thermal Sensors FFT status {self.status}- Date: {self.gdate[0]}', fontsize=14)

        # Note: the steps used by the periodogram are 1/20.
        fs = 1 / 20.
        for i, group in enumerate(self.ts_names.keys()):
            for j, sensor_name in enumerate(self.ts_names[group]):
                # The COLD HEAD TS has a specific color: cyan
                if sensor_name == "TS-SP1-SpareDT":
                    color = "cyan"
                # The other TS will have another color: teal
                else:
                    color = "teal"
                # Calculate the periodogram
                # Choose the length of the data segment (between 10**4 and nperseg provided) on which calculate the fft
                # Changing this parameter allow to reach lower freq in the plot: the limInf of the x-axis is fs/nperseg.
                f, s = scipy.signal.welch(self.ts["thermal_data"]["calibrated"][sensor_name],
                                          fs=fs, nperseg=min(int(fs * 10 ** 4), self.nperseg_thermal))
                # Plot the periodogram (fft)
                axs[i].plot(f[f < 25.], s[f < 25.],
                            linewidth=0.2, label=f"{sensor_name}", marker=".", markerfacecolor=color, markersize=4)

                # Title
                axs[i].set_title(f"FFT TS GROUP {group}")
                # XY-axis
                axs[i].set_yscale("log")
                axs[i].set_xscale("log")
                axs[i].set_xlabel(f"$Frequency$ $[Hz]$")
                axs[i].set_ylabel(f"PSD [K**2/Hz]")
                # Legend
                axs[i].legend(prop={'size': 9}, loc=7)

        # Procedure to save the png of the plot in the correct dir
        path = f"../plot/{self.date_dir}/Thermal_Output/FFT/"
        Path(path).mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{path}FFT_Thermal_status_{self.status}.png', dpi=600)

        # If show is True the plot is visible on video
        if show:
            plt.show()
        plt.close(fig)
