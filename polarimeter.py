#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# This file contains part of the code used in the bachelor thesis of Francesco Andreetto (2020)
# updated to be used on the new version of the software of LSPE-STRIP
# November 1st 2022, Brescia (Italy)

# Libraries & Modules
import logging
import sys

import numpy as np
import pandas as pd
import scipy as scipy
import scipy.stats as scs
import seaborn as sn  # This should be added to requirements.txt

from astropy.time import Time
from matplotlib import pyplot as plt
from pathlib import Path
from rich.logging import RichHandler

import f_strip
from striptease import DataStorage
from typing import List, Dict, Any

# MyLibraries & MyModules
import f_strip as fz


########################################################################################################
# Class for a Polarimeter
########################################################################################################
class Polarimeter:

    def __init__(self, name_pol: str, path_file: str, start_datetime: str, end_datetime: str):
        """
        Constructor
        Parameters:
        - **name_pol** (``str``): name of the polarimeter
        - **path_file** (``str``): location of the data file (without the name of the file)
        - **start_datetime** (``str``): start time
        - **end_datetime** (``str``): end time
        """
        self.name = name_pol
        self.ds = DataStorage(path_file)

        # tag = self.ds.get_tags(mjd_range=(Time(start_datetime), Time(end_datetime)))

        self.STRIP_SAMPLING_FREQ = 0
        self.norm_mode = 0

        # Julian Date MJD
        self.date = [Time(start_datetime).mjd, Time(end_datetime).mjd]  # self.tag.mjd_start
        # Gregorian Date [in string format]
        self.gdate = [Time(start_datetime), Time(end_datetime)]
        # Time(self.date, format="mjd").to_datetime().strftime("%Y-%m-%d %H:%M:%S")

        # Dictionary for scientific Analysis
        self.times = []  # type: List[float]

        power = {}
        dem = {}
        self.data = {"DEM": dem, "PWR": power}

        # Dictionary for thermal Analysis
        self.thermal_list = {
            "0": ["TS-CX12-Pol-W", "TS-CX14-Pol-Qy", "TS-CX16-Filter", "TS-CX6-Module-O", "TS-SP1-SpareDT",
                  "TS-CX10-Frame-120", "TS-SP2-L-Support", "EX-CX18-SpareCx", "TS-DT3-Shield-Base",
                  "TS-DT6-Frame-South", "TS-CX2-Module-V", "TS-CX4-Module-G"],
            "1": ["TS-CX9-Frame-60", "TS-CX11-Frame-North", "TS-CX15-IF-Frame-0", "TS-CX5-Module-Y",
                  "TS-CX7-Module-I", "TS-CX8-Frame-0", "TS-CX13-Pol-Qx", "TS-CX17-Wheel-Center",
                  "EX-DT2-SpareDT", "TS-DT5-Shield-Side", "TS-CX1-Module-R", "TS-CX3-Module-B"]}

        # TS-SP2-SpareCx
        thermal_times = {"0": [], "1": []}
        raw = {}
        calibrated = {}
        thermal_data = {"raw": raw, "calibrated": calibrated}
        self.thermal_sensors = {"thermal_times": thermal_times, "thermal_data": thermal_data}

        # Dictionary for Housekeeping Analysis
        self.hk_list = {"V": ["VG0_HK", "VG1_HK", "VG2_HK", "VG3_HK", "VG4_HK", "VG5_HK",
                              "VD0_HK", "VD1_HK", "VD2_HK", "VD3_HK", "VD4_HK", "VD5_HK"],
                        "I": ["IG0_HK", "IG1_HK", "IG2_HK", "IG3_HK", "IG4_HK", "IG5_HK",
                              "ID0_HK", "ID1_HK", "ID2_HK", "ID3_HK", "ID4_HK", "ID5_HK"],
                        "O": ["DET0_OFFS", "DET1_OFFS", "DET2_OFFS", "DET3_OFFS"]
                        }
        tensions = {}
        currents = {}
        offset = {}
        t_tensions = {}
        t_currents = {}
        t_offset = {}
        self.hk = {"V": tensions, "I": currents, "O": offset}
        self.hk_t = {"V": t_tensions, "I": t_currents, "O": t_offset}

        # Warnings
        time_warning = []
        corr_warning = []
        eo_warning = []
        spike_warning = []
        self.warnings = {"time_warning": time_warning,
                         "corr_warning": corr_warning,
                         "eo_warning": eo_warning,
                         "spike_warning": spike_warning}

    def Load_Pol(self):
        """
        Load all dataset in the polarimeter
        All type "DEM" and "PWR"
        All the exit "Q1", "Q2", "U1", "U2"
        """
        for type in self.data.keys():
            for exit in ["Q1", "Q2", "U1", "U2"]:
                self.times, self.data[type][exit] = self.ds.load_sci(mjd_range=self.date, polarimeter=self.name,
                                                                     data_type=type, detector=exit)
        self.STRIP_SAMPLING_FREQ = 0

    def Load_X(self, type: str):
        """
        Load only a specific type of dataset "PWR" or "DEM" in the polarimeter
        Parameters:\n **type** (``str``) *"DEM"* or *"PWR"*
        """
        for exit in ["Q1", "Q2", "U1", "U2"]:
            self.times, self.data[type][exit] = self.ds.load_sci(mjd_range=self.date, polarimeter=self.name,
                                                                 data_type=type, detector=exit)
        self.STRIP_SAMPLING_FREQ = 0

    def Load_Times(self, range: []):
        """
        Load the times in the polarimeter and put to 0 the STRIP Sampling Frequency.
        Useful to calculate quickly the STRIP Sampling Frequency in the further steps without loading the whole Pol.
        Parameters:\n **range** (``Time``) is an array-like object containing the Time objects: start_date and end_date.
        """
        self.times, _ = self.ds.load_sci(mjd_range=range, polarimeter=self.name, data_type="DEM", detector="Q1")
        self.STRIP_SAMPLING_FREQ = 0

    def Date_Update(self, n_samples: int, modify=True) -> Time:
        """
        Calculates and returns the new Gregorian date in which the experience begins, given a number of samples that
        must be skipped from the beginning of the dataset.
        Parameters:\n
        **n_samples** (``int``) number of samples that must be skipped\n
        **modify** (``bool``)\n
        \t*"True"* -> The beginning date is definitely modified and provided.\n
        \t*"False"* -> A copy of the beginning date is modified and provided.\n
        """
        s = 1 / 86_400
        if modify:
            self.date[0] += s * (n_samples / 100)  # Julian Date increased
            self.gdate[0] = Time(self.date[0], format="mjd").to_datetime().strftime("%Y-%m-%d %H:%M:%S")  # Gregorian
            return self.gdate[0]
        else:
            new_jdate = self.date[0]
            new_jdate += s * (n_samples / 100)  # Julian Date increased
            new_date = Time(new_jdate, format="mjd").to_datetime().strftime("%Y-%m-%d %H:%M:%S")  # Gregorian Date
            return new_date

    def Clip_Values(self):
        """
        Data cleansing: scientific data with value zero at the beginning and at the end are removed from the dataset
        Control that a channel doesn't turn on before the others (maybe unuseful)
        """
        begin_zerovalues_idx = 0
        end_zerovalues_idx = 10_000_000

        for type in [x for x in self.data.keys() if not self.data[x] == {}]:
            for exit in ["Q1", "Q2", "U1", "U2"]:
                for count, item in reversed(list(enumerate(self.data[type][exit]))):
                    if item != 0:
                        end_zerovalues_idx = np.min([end_zerovalues_idx, count + 1])
                        break
                for count, item in enumerate(self.data[type][exit]):
                    if item != 0:
                        begin_zerovalues_idx = np.max([begin_zerovalues_idx, count])
                        break

        # Cleansing operations
        self.times = self.times[begin_zerovalues_idx:end_zerovalues_idx + 1]
        for type in [x for x in self.data.keys() if not self.data[x] == {}]:
            for exit in ["Q1", "Q2", "U1", "U2"]:
                self.data[type][exit] = self.data[type][exit][begin_zerovalues_idx:end_zerovalues_idx + 1]

        # Updating the new beginning time of the dataset
        _ = self.Date_Update(n_samples=begin_zerovalues_idx, modify=True)

    def STRIP_SAMPLING_FREQUENCY_HZ(self, warning=True):
        """
        Strip Sampling Frequency
        It depends on the electronics hence it's the same for all polarimeters
        Note: it must be defined before time normalization
        """
        self.STRIP_SAMPLING_FREQ = int(
            len(self.times) / (self.times[-1].datetime - self.times[0].datetime).total_seconds())

        if warning:
            if self.STRIP_SAMPLING_FREQ != 100:
                msg = f"Sampling frequency is {self.STRIP_SAMPLING_FREQ} different from the std value of 100.\n " \
                      f"This can cause inversions in even-odd sampling. \n" \
                      f"Some changes in the offset might have occurred: Some channel turned off?\n" \
                      f"There is at least a hole in the sampling: after the normalization, seconds are not significant."
                logging.error(msg)
                self.warnings["eo_warning"].append(msg)

    def Norm(self, norm_mode: int):
        """
        Timestamp Normalization\n
        Parameters:\n **norm_mode** (``int``) can be set in two ways:
        0) the output is expressed in function of the number of samples
        1) the output is expressed in function of the time in s from the beginning of the experience
        2) the output is expressed in function of the number of the Julian Date JHD
        """
        if norm_mode == 0:
            self.times = np.arange(len(self.times))  # Number of samples
        if norm_mode == 1:
            self.times = np.arange(len(self.times)) / self.STRIP_SAMPLING_FREQ  # Seconds
        if norm_mode == 2:
            self.times = self.times.value  # JHD

    def Prepare(self, norm_mode: int):
        """
        Prepare the polarimeter in three steps:\n
            1. Clean dataset with Clip_Values()
            2. Calculate Strip Sampling Frequency
            3. Normalize timestamps
        """
        logging.basicConfig(level="INFO", format='%(message)s',
                            datefmt="[%X]", handlers=[RichHandler()])  # <3

        self.norm_mode = norm_mode

        self.Clip_Values()
        if self.STRIP_SAMPLING_FREQ > 0:
            logging.warning(f"The dataset has already been normalized. "
                            f"Strip Sampling Frequency = {self.STRIP_SAMPLING_FREQ}.")
            return 0
        self.STRIP_SAMPLING_FREQUENCY_HZ()
        self.Norm(norm_mode)

        logging.info(f"Pol {self.name}: the dataset is now normalized.")
        if norm_mode == 0:
            logging.info("Dataset in function of sample number [#]")
        if norm_mode == 1:
            logging.info("Dataset in function of time [s].")

    def Demodulation(self, type: str, exit: str, begin=0, end=-1) -> Dict[str, Any]:
        """
        Demodulation\n
        Calculate the Scientific data DEMODULATED or TOTAL POWER at 50Hz\n
        Timestamps are chosen as mean of the two consecutive times of the DEM/PWR data\n
        Parameters:\n
        - **exit** (``str``) *"Q1"*, *"Q2"*, *"U1"*, *"U2"*\n
        - **type** (``str``) of data *"DEM"* or *"PWR"*
        - **begin**, **end** (``int``): interval of dataset that has to be considered
        """
        times = fz.mean_cons(self.times)
        data = {}
        if type == "PWR":
            data[exit] = fz.mean_cons(self.data[type][exit][begin:end])
        if type == "DEM":
            data[exit] = fz.diff_cons(self.data[type][exit][begin:end])

        sci_data = {"sci_data": data, "times": times}
        return sci_data

    # ------------------------------------------------------------------------------------------------------------------
    # THERMIC ANALYSIS
    # ------------------------------------------------------------------------------------------------------------------
    def Load_Thermal_Sensors(self):
        """
        Load all Thermal Sensor's data taking the names from the list in the constructor.
        """
        for status in range(2):
            for calib, bol in zip(["raw", "calibrated"], [True, False]):
                for sensor_name in self.thermal_list[f"{status}"]:
                    self.thermal_sensors["thermal_times"][f"{status}"], \
                        self.thermal_sensors["thermal_data"][calib][sensor_name] \
                        = self.ds.load_cryo(self.date, sensor_name, get_raw=bol)

    def Norm_Thermal(self):
        """
        Normalize all Thermal Sensor's timestamps putting one every 10 seconds from the beginning of the dataset.
        """
        for status, start in zip(["0", "1"], [0., 3.]):
            sensor_name = self.thermal_list[status]
            for i in range(12):
                l1 = len(self.thermal_sensors["thermal_times"][status])
                l2 = len(self.thermal_sensors["thermal_data"]["calibrated"][sensor_name[i]])
                if l1 != l2:
                    msg = f"The Thermal sensor: {sensor_name[i]} has a sampling problem.\n"
                    logging.error(msg)
                    self.warnings["time_warning"].append(msg + "<br />")

                self.thermal_sensors["thermal_times"][status] = \
                    start + np.arange(0, len(self.thermal_sensors["thermal_times"][status]) * 10, 10)

    def Analyse_Thermal(self) -> {}:
        """
        Analise all Thermal Sensors' output: calculate the mean the std deviation for both raw and calibrated samples.
        """
        raw_m = {}
        cal_m = {}
        mean = {"raw": raw_m, "calibrated": cal_m}

        raw_std = {}
        cal_std = {}
        dev_std = {"raw": raw_std, "calibrated": cal_std}

        raw_nan = {}
        cal_nan = {}
        nan_percent = {"raw": raw_nan, "calibrated": cal_nan}

        raw_max = {}
        cal_max = {}
        max_value = {"raw": raw_max, "calibrated": cal_max}

        raw_min = {}
        cal_min = {}
        min_value = {"raw": raw_min, "calibrated": cal_min}

        results = {"max": max_value, "min": min_value, "mean": mean, "dev_std": dev_std, "nan_percent": nan_percent}

        for status in range(2):
            for calib in ["raw", "calibrated"]:
                for sensor_name in self.thermal_list[f"{status}"]:
                    results["nan_percent"][calib][sensor_name] = 0.

                    data = self.thermal_sensors["thermal_data"][calib][sensor_name]
                    m = np.mean(data)
                    if np.isnan(m):
                        n_nan = len([t for t in np.isnan(data) if t == True])

                        if len(data) == 0:
                            results["nan_percent"][calib][sensor_name] = 100.
                        else:
                            results["nan_percent"][calib][sensor_name] = round((n_nan / len(data)), 4) * 100.

                        if results["nan_percent"][calib][sensor_name] < 5:
                            data = np.delete(data, np.argwhere(np.isnan(data)))
                            m = np.mean(data)

                    results["max"][calib][sensor_name] = max(data)
                    results["min"][calib][sensor_name] = min(data)
                    results["mean"][calib][sensor_name] = m
                    results["dev_std"][calib][sensor_name] = np.std(data)

        return results

    def Thermal_table(self, results) -> str:
        """
        Create a string with the html code for a table of thermal results.
        Now are listed in the table: the status of acquisition (0 or 1), the sensor name, the max value, the min value,
        the mean, the standard deviation and the NaN percentage found for both RAW and calibrated (CAL) Temperatures.
        """
        html_table = ""
        for status in range(2):
            for sensor_name in self.thermal_list[f"{status}"]:
                html_table += f"<tr><td align=center>{status}</td><td align=center>{sensor_name}</td>"
                for calib in ['raw', 'calibrated']:
                    html_table += f"<td align=center>{round(results['max'][calib][sensor_name], 4)}</td>" \
                                  f"<td align=center>{round(results['min'][calib][sensor_name], 4)}</td>" \
                                  f"<td align=center>{round(results['mean'][calib][sensor_name], 4)}</td>" \
                                  f"<td align=center>{round(results['dev_std'][calib][sensor_name], 4)}</td>" \
                                  f"<td align=center>{round(results['nan_percent'][calib][sensor_name], 4)}</td>"

                html_table += f"</tr>"

        return html_table

    def Plot_Thermal(self, status: int, show=False):
        """
        Plot all the calibrated acquisitions of Thermal Sensors of the polarimeter.\n
            Parameters:\n
         - **show** (``bool``): *True* -> show the plot and save the figure, *False* -> save the figure only
         - **status** (``int``): *0* or *1* -> It refers to the status of the multiplexer that acquire the TS
        """
        thermals = {
            "0": {
                "TILES": ["TS-CX4-Module-G", "TS-CX6-Module-O", "TS-CX2-Module-V"],
                "FRAME": ["TS-CX10-Frame-120", "TS-DT6-Frame-South"],
                "POLAR": ["TS-CX12-Pol-W", "TS-CX14-Pol-Qy"],
                "100-200": ["TS-CX16-Filter", "TS-DT3-Shield-Base"],  # , "TS-SP2-L-Support" # Excluded for the moment
                # "VERIFY": ["EX-CX18-SpareCx"],  # Excluded for the moment
                "COL_HEAD": ["TS-SP1-SpareDT"]
            },
            "1": {
                "TILES": ["TS-CX3-Module-B", "TS-CX7-Module-I", "TS-CX1-Module-R", "TS-CX5-Module-Y"],
                "FRAME": ["TS-CX8-Frame-0", "TS-CX9-Frame-60", "TS-CX11-Frame-North", "TS-CX15-IF-Frame-0"],
                "POLAR": ["TS-CX13-Pol-Qx"],
                "100-200K": ["TS-DT5-Shield-Side"],  # , "TS-CX17-Wheel-Center" # Excluded for the moment
                "VERIFY": ["EX-DT2-SpareDT"],
            }}
        col = ["cornflowerblue", "indianred", "limegreen", "gold"]

        if status == 0 or status == 1:
            n_rows = len(thermals[f"{status}"].keys())
            fig, axs = plt.subplots(nrows=n_rows, ncols=1, constrained_layout=True, figsize=(13, 15))
        else:
            sys.exit("Invalid Status. It must be 0 or 1.")

        fig.suptitle(f'Plot Thermal Sensors status {status}- Date: {self.gdate[0]}', fontsize=14)

        for i, group in enumerate(thermals[f"{status}"].keys()):
            for j, sensor_name in enumerate(thermals[f"{status}"][group]):
                l1 = len(self.thermal_sensors["thermal_times"][f"{status}"])
                l2 = len(self.thermal_sensors["thermal_data"]["calibrated"][sensor_name])
                axs[i].scatter(
                    self.thermal_sensors["thermal_times"][f"{status}"][:min(l1, l2)],
                    self.thermal_sensors["thermal_data"]["calibrated"][sensor_name][:min(l1, l2)],
                    marker=".", color=col[j], label=sensor_name)

                axs[i].set_xlabel("Time [s]")
                axs[i].set_ylabel("Temperature [K]")
                axs[i].set_title(f"TS GROUP {group} - Status {status}")
                axs[i].legend(prop={'size': 9}, loc=7)

        date_dir = fz.dir_format(f"{self.gdate[0]}__{self.gdate[1]}")
        path = f"../plot/{date_dir}/Thermal_Output/"
        Path(path).mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{path}{self.name}_thermal_status_{status}.png')
        if show:
            plt.show()
        plt.close(fig)

    def Plot_Correlation_TS(self, type: str, begin=0, end=-1, show=False):
        """
        Plot of Correlation between Raw data DEM or PWR & Thermal Sensor Outputs.\n
        Parameters:\n
        - **type** (``str``) of data "DEM" or "PWR"\n
        - **show** (``bool``): *True* -> show the plot and save the figure, *False* -> save the figure only
        """
        fig, axs = plt.subplots(nrows=4, ncols=4, constrained_layout=True, figsize=(17, 15))

        begin_date = self.Date_Update(n_samples=begin, modify=False)
        fig.suptitle(f'Correlation Thermal Sensors vs {type} Output - Date: {begin_date}', fontsize=14)

        # Note: all Thermal Sensors chosen are in configuration "0"
        ts_list = ["TS-CX6-Module-O", "TS-CX2-Module-V", "TS-SP1-SpareDT", "TS-CX16-Filter"]

        for idx, ts in enumerate(ts_list):
            n = 0  # type: int
            for exit in self.data[f"{type}"].keys():
                if idx == 0:
                    axs[idx, n].sharey(axs[1, 0])
                    axs[idx, n].sharex(axs[1, 0])

                x = self.data[type][exit][begin:end]
                y = np.interp(self.times[begin:end], self.thermal_sensors["thermal_times"]["0"],
                              self.thermal_sensors["thermal_data"]["calibrated"][f"{ts}"])
                axs[idx, n].plot(x, y, "*", color="steelblue", label="Corr Data")

                # Title
                axs[idx, n].set_title(f'Corr {ts} vs {type} {exit}')
                # XY-axis
                # axs[idx, n].set_aspect('equal')
                axs[idx, n].set_xlabel(f"{type} Output")
                axs[idx, n].set_ylabel(f"Temperature [K]")
                # Legend
                axs[idx, n].legend(prop={'size': 9}, loc=7)

                n += 1

        date_dir = fz.dir_format(f"{self.gdate[0]}__{self.gdate[1]}")
        path = f"../plot/{date_dir}/Thermal_Correlation/{self.name}/"
        Path(path).mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{path}{self.name}_{type}_Correlation_TS.png')
        if show:
            plt.show()
        plt.close(fig)

    # ------------------------------------------------------------------------------------------------------------------
    # HOUSE-KEEPING ANALYSIS
    # ------------------------------------------------------------------------------------------------------------------
    def Load_HouseKeeping(self):
        """
        Load all House-Keeping parameters taking the names from the list in the constructor.
        """
        group = "BIAS"
        for item in self.hk_list.keys():
            if item == "O":
                group = "DAQ"
            for hk_name in self.hk_list[item]:
                self.hk_t[item][hk_name], self.hk[item][hk_name] = self.ds.load_hk(mjd_range=self.date,
                                                                                   group=group,
                                                                                   subgroup=f"POL_{self.name}",
                                                                                   par=hk_name
                                                                                   )

    def Norm_HouseKeeping(self):
        """
        Normalize all House-Keeping's timestamps putting one every 1.4 seconds from the beginning of the dataset.
        """
        for item in self.hk_list.keys():
            for hk_name in self.hk_list[item]:
                l1 = len(self.hk_t[item][hk_name])
                l2 = len(self.hk[item][hk_name])
                if l1 != l2:
                    msg = f"The House-Keeping: {hk_name} has a sampling problem.\n"
                    logging.error(msg)
                    self.warnings["time_warning"].append(msg + "<br />")

                if item == "O":
                    self.hk_t[item][hk_name] = np.arange(0, len(self.hk[item][hk_name]) * 30, 30)
                    l1 = len(self.hk_t[item][hk_name])
                    self.hk_t[item][hk_name] = self.hk_t[item][hk_name][:min(l1, l2)]
                else:
                    self.hk_t[item][hk_name] = np.arange(0, len(self.hk[item][hk_name]) * 1.4, 1.4)
                    l1 = len(self.hk_t[item][hk_name])
                    self.hk_t[item][hk_name] = self.hk_t[item][hk_name][:min(l1, l2)]

    def Analyse_HouseKeeping(self) -> {}:
        """
        Analise the following HouseKeeping parameters: I Drain, I Gate, V Drain, V Gate, Offset.\n
        See self.hk_list in the constructor.\n
        Calculate the mean the std deviation.
        """
        I_m = {}
        V_m = {}
        O_m = {}
        mean = {"I": I_m, "V": V_m, "O": O_m}

        I_std = {}
        V_std = {}
        O_std = {}
        dev_std = {"I": I_std, "V": V_std, "O": O_std}

        I_nan = {}
        V_nan = {}
        O_nan = {}
        nan_percent = {"I": I_nan, "V": V_nan, "O": O_nan}

        I_max = {}
        V_max = {}
        O_max = {}
        hk_max = {"I": I_max, "V": V_max, "O": O_max}

        I_min = {}
        V_min = {}
        O_min = {}
        hk_min = {"I": I_min, "V": V_min, "O": O_min}

        results = {"max": hk_max, "min": hk_min, "mean": mean, "dev_std": dev_std, "nan_percent": nan_percent}

        for item in self.hk_list.keys():
            for hk_name in self.hk_list[item]:
                results["nan_percent"][item][hk_name] = 0.

                data = self.hk[item][hk_name]
                m = np.mean(data)
                if np.isnan(m):
                    n_nan = len([t for t in np.isnan(data) if t == True])

                    if len(data) == 0:
                        results["nan_percent"][item][hk_name] = 100.
                    else:
                        results["nan_percent"][item][hk_name] = round((n_nan / len(data)), 4) * 100.

                    if results["nan_percent"][item][hk_name] < 5:
                        data = np.delete(data, np.argwhere(np.isnan(data)))
                        m = np.mean(data)

                results["max"][item][hk_name] = max(data)
                results["min"][item][hk_name] = min(data)
                results["mean"][item][hk_name] = m
                results["dev_std"][item][hk_name] = np.std(data)

        return results

    def HK_table(self, results) -> str:
        """
        Create a string with the html code for a table of thermal results.
        Now are listed in the table: the HK-Parameter name, the max value, the min value, the mean,
        the standard deviation and the NaN percentage.
        The HouseKeeping parameters included are: I Drain, I Gate, V Drain, V Gate, Offset.
        """
        html_table = ""
        for item in self.hk_list.keys():
            if item == "V":
                unit = "[&mu;V]"
            elif item == "I":
                unit = "[&mu;A]"
            else:
                unit = "[ADU]"

            html_table += "<tr>" \
                          "<th>Parameter</th>" \
                          f"<th>Max Value {unit}</th>" \
                          f"<th>Min Value {unit}</th>" \
                          f"<th>Mean {unit}</th>" \
                          f"<th>Std_Dev {unit}</th>" \
                          "<th>NaN %</th></tr> "
            for hk_name in self.hk_list[item]:
                html_table += f"<tr>" \
                              f"<td align=center>{hk_name}</td>" \
                              f"<td align=center>{round(results['max'][item][hk_name], 4)}</td>" \
                              f"<td align=center>{round(results['min'][item][hk_name], 4)}</td>" \
                              f"<td align=center>{round(results['mean'][item][hk_name], 4)}</td>" \
                              f"<td align=center>{round(results['dev_std'][item][hk_name], 4)}</td>" \
                              f"<td align=center>{round(results['nan_percent'][item][hk_name], 4)}</td>" \
                              f"</tr>"
        return html_table

    def Plot_HouseKeeping_VI(self, show=False):
        """
        Plot all the acquisitions of HouseKeeping parameters of the polarimeter: Drain Voltage, Gate Voltage,
        Drain Current and Gate Current. 1 every 1.4 seconds\n
        Parameters:\n
         - **show** (``bool``): *True* -> show the plot and save the figure, *False* -> save the figure only
        """
        col = ["plum", "gold"]
        label = ["Voltage [$\mu$V]", "Current [mA]"]
        for idx, item in enumerate(["V", "I"]):
            hk_name = self.hk_list[item]

            fig, axs = plt.subplots(nrows=4, ncols=3, constrained_layout=True, figsize=(15, 12), sharey='row')
            fig.suptitle(f'Plot Housekeeping parameters: {item} - Date: {self.gdate[0]}', fontsize=14)
            for i in range(4):
                for j in range(3):

                    l1 = len(self.hk_t[item][hk_name[3 * i + j]])
                    l2 = len(self.hk[item][hk_name[3 * i + j]])

                    if l1 != l2:
                        msg = f"The House-Keeping: {hk_name[3 * i + j]} has a sampling problem.\n"
                        logging.error(msg)
                        self.warnings["time_warning"].append(msg + "<br />")

                    axs[i, j].scatter(self.hk_t[item][hk_name[3 * i + j]][:min(l1, l2)],
                                      self.hk[item][hk_name[3 * i + j]][:min(l1, l2)], marker=".", color=col[idx])

                    axs[i, j].set_xlabel("Time [s]")
                    axs[i, j].set_ylabel(f"{label[idx]}")
                    axs[i, j].set_title(f"{hk_name[3 * i + j]}")

            date_dir = fz.dir_format(f"{self.gdate[0]}__{self.gdate[1]}")
            path = f"../plot/{date_dir}/HouseKeeping/{self.name}/"
            Path(path).mkdir(parents=True, exist_ok=True)
            fig.savefig(f'{path}{self.name}_HK_{item}.png')
            if show:
                plt.show()
            plt.close(fig)

    def Plot_HouseKeeping_OFF(self, show=False):
        """
        Plot all the acquisitions of HouseKeeping parameters of Offset: 1 every 30 sec.\n
        Parameters:\n
         - **show** (``bool``): *True* -> show the plot and save the figure, *False* -> save the figure only
        """
        col = ["teal"]
        label = ["Offset [ADU]"]
        for idx, item in enumerate(["O"]):
            hk_name = self.hk_list[item]

            fig, axs = plt.subplots(nrows=1, ncols=4, constrained_layout=True, figsize=(15, 4), sharey='row')
            fig.suptitle(f'Plot Housekeeping parameters: {item} - Date: {self.gdate[0]}', fontsize=14)
            for j in range(4):
                axs[j].scatter(self.hk_t[item][hk_name[j]],
                               self.hk[item][hk_name[j]],
                               marker=".", color=col[idx])
                axs[j].set_xlabel("Time [s]")

                axs[j].set_ylabel(f"{label[idx]}")
                axs[j].set_title(f"{hk_name[j]}")

            date_dir = fz.dir_format(f"{self.gdate[0]}__{self.gdate[1]}")
            path = f"../plot/{date_dir}/HouseKeeping/{self.name}/"
            Path(path).mkdir(parents=True, exist_ok=True)
            fig.savefig(f'{path}{self.name}_HK_{item}.png')
            if show:
                plt.show()
            plt.close(fig)

    # ------------------------------------------------------------------------------------------------------------------
    # PLOT FUNCTIONS
    # ------------------------------------------------------------------------------------------------------------------

    def Plot_Output(self, type: str, begin: int, end: int, show=True):
        """
        Plot the 4 exits PWR or DEM of the Polarimeter\n
        Parameters:\n
        - **type** (``str``) of data *"DEM"* or *"PWR"*\n
        - **begin**, **end** (``int``): interval of dataset that has to be considered\n
        - **show** (``bool``): *True* -> show the plot and save the figure, *False* -> save the figure only
        """
        fig = plt.figure(figsize=(17, 5))

        begin_date = self.Date_Update(n_samples=begin, modify=False)
        fig.suptitle(f'{self.name} Output {type} - Date: {begin_date}', fontsize=14)
        o = 0
        for exit in ["Q1", "Q2", "U1", "U2"]:
            o = o + 1
            ax = fig.add_subplot(1, 4, o)
            ax.plot(self.times[begin:end], self.data[type][exit][begin:end], "*")
            ax.set_title(f"{exit}")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(f"Output {type}")
        plt.tight_layout()

        date_dir = fz.dir_format(f"{self.gdate[0]}__{self.gdate[1]}")
        path = f"../plot/{date_dir}/Output/"
        Path(path).mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{path}{self.name}_{type}.png')
        if show:
            plt.show()
        plt.close(fig)

    def Plot_EvenOddAll(self, type: str, even: int, odd: int, all: int, begin=0, end=-1, smooth_len=1, show=True):
        """
        Plot of Raw data DEM or PWR (Even, Odd or All)\n
        Parameters:\n
        - **type** (``str``) of data *"DEM"* or *"PWR"*\n
        - **even**, **odd**, **all** (int): used to set the transparency of the dataset (0=transparent, 1=visible)\n
        - **begin**, **end** (``int``): interval of dataset that has to be considered\n
        - **smooth_len** (``int``): number of elements on which the mobile mean is calculated
        - **show** (``bool``): *True* -> show the plot and save the figure, *False* -> save the figure only
        Note: the 4 plots are repeated on two rows (uniform Y-scale below)\n
        """
        fig, axs = plt.subplots(nrows=2, ncols=4, constrained_layout=True, figsize=(17, 9))

        eoa = EOA(even=even, odd=odd, all=all)

        begin_date = self.Date_Update(n_samples=begin, modify=False)
        fig.suptitle(f'Plot {eoa} {type} - Date: {begin_date}', fontsize=14)

        for i in range(2):
            n = 0  # type: int
            for exit in ["Q1", "Q2", "U1", "U2"]:
                if i == 1:
                    axs[i, n].sharey(axs[1, 0])

                if even != 0:
                    axs[i, n].plot(self.times[begin:end - 1:2][:- smooth_len],
                                   fz.mob_mean(self.data[type][exit][begin:end - 1:2], smooth_len=smooth_len)[:-1],
                                   color="royalblue", alpha=even, label="Even Output")

                if odd != 0:
                    axs[i, n].plot(self.times[begin + 1:end:2][:- smooth_len],
                                   fz.mob_mean(self.data[type][exit][begin + 1:end:2], smooth_len=smooth_len)[:-1],
                                   color="crimson", alpha=odd, label="Odd Output")

                if all != 0:
                    axs[i, n].plot(self.times[begin:end][:- smooth_len],
                                   fz.mob_mean(self.data[type][exit][begin:end], smooth_len=smooth_len)[:-1],
                                   color="forestgreen", alpha=all, label="All Output")
                # Title
                axs[i, n].set_title(f'{type} {exit}')
                # X-axis
                # axs[i, n].set_xticklabels(rotation=45, ha="right")
                if self.norm_mode == 0:
                    axs[i, n].set_xlabel("# Samples")
                if self.norm_mode == 1:
                    axs[i, n].set_xlabel("Time [s]")
                # Y-axis
                axs[i, n].set_ylabel(f"Output [{type}]")
                # Legend
                axs[i, n].legend(prop={'size': 9}, loc=7)

                n += 1

        date_dir = fz.dir_format(f"{self.gdate[0]}__{self.gdate[1]}")
        path = f"../plot/{date_dir}/EvenOddAll_Analysis/EOA_Output/{self.name}/"
        Path(path).mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{path}{self.name}_{type}_{eoa}_smooth={smooth_len}.png')
        if show:
            plt.show()
        plt.close(fig)

    def Plot_RMS_EOA(self, type: str, window: int, even: int, odd: int, all: int, begin=0, end=-1, smooth_len=1,
                     show=True):
        """
        Plot of Raw data DEM or PWR (Even, Odd or All)\n
        Parameters:\n
        - **type** (``str``) of data *"DEM"* or *"PWR"*
        - **window** (``int``): number of elements on which the RMS is calculated
        - **even**, **odd**, **all** (``int``): used for the transparency of the data (0=transparent, 1=visible)
        - **begin**, **end** (``int``): interval of dataset that has to be considered
        - **smooth_len** (``int``): number of elements on which the mobile mean is calculated
        - **show** (``bool``): *True* -> show the plot and save the figure, *False* -> save the figure only
        Note: the 4 plots are repeated on two rows (uniform Y-scale below)\n
        """
        fig, axs = plt.subplots(nrows=2, ncols=4, constrained_layout=True, figsize=(17, 9))

        eoa = EOA(even=even, odd=odd, all=all)

        begin_date = self.Date_Update(n_samples=begin, modify=False)
        fig.suptitle(f'RMS {eoa} {type} - Date: {begin_date}', fontsize=14)

        for i in range(2):
            n = 0  # type: int
            for exit in ["Q1", "Q2", "U1", "U2"]:

                if i == 1:
                    axs[i, n].sharey(axs[1, 0])

                if even != 0:
                    axs[i, n].plot(self.times[begin:end - 1:2][:-window - smooth_len + 2],
                                   fz.mob_mean(
                                       RMS(self.data[type], window=window, exit=exit, eoa=2, begin=begin, end=end),
                                       smooth_len=smooth_len),
                                   color="royalblue", alpha=even, label="Even Output")
                if odd != 0:
                    axs[i, n].plot(self.times[begin + 1:end:2][:-window - smooth_len + 2],
                                   fz.mob_mean(
                                       RMS(self.data[type], window=window, exit=exit, eoa=1, begin=begin, end=end),
                                       smooth_len=smooth_len),
                                   color="crimson", alpha=odd, label="Odd Output")
                if all != 0:
                    axs[i, n].plot(self.times[begin:end][:-window - smooth_len + 2],
                                   fz.mob_mean(
                                       RMS(self.data[type], window=window, exit=exit, eoa=0, begin=begin, end=end),
                                       smooth_len=smooth_len),
                                   color="forestgreen", alpha=all, label="All Output")

                # Title
                axs[i, n].set_title(f'RMS {type} {exit}')
                # X-axis
                if self.norm_mode == 0:
                    axs[i, n].set_xlabel("# Samples")
                if self.norm_mode == 1:
                    axs[i, n].set_xlabel("Time [s]")
                # Y-axis
                axs[i, n].set_ylabel(f"RMS [{type}]")
                # Legend
                axs[i, n].legend(prop={'size': 9}, loc=7)

                n += 1

        date_dir = fz.dir_format(f"{self.gdate[0]}__{self.gdate[1]}")
        path = f'../plot/{date_dir}/EvenOddAll_Analysis/EOA_RMS/{self.name}/'
        Path(path).mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{path}{self.name}_{type}_RMS_{eoa}_smooth={smooth_len}.png')
        if show:
            plt.show()
        plt.close(fig)

    def Plot_Correlation_EvenOdd(self, type: str, begin=0, end=-1, show=True):
        """
        Plot of Raw data DEM or PWR: Even vs Odd to see the correlation\n
        Parameters:\n
        - **type** (``str``) of data "DEM" or "PWR"\n
        - **begin**, **end** (``int``): interval of dataset that has to be considered\n
        - **show** (``bool``): *True* -> show the plot and save the figure, *False* -> save the figure only
        Note: the 4 plots are repeated on two rows (uniform Y-scale below)\n
        """
        fig, axs = plt.subplots(nrows=2, ncols=4, constrained_layout=True, figsize=(17, 9))

        begin_date = self.Date_Update(n_samples=begin, modify=False)
        fig.suptitle(f'Correlation {type} - Date: {begin_date}', fontsize=14)

        for i in range(2):
            n = 0  # type: int
            for exit in ["Q1", "Q2", "U1", "U2"]:

                if i == 1:
                    axs[i, n].sharey(axs[1, 0])
                    axs[i, n].sharex(axs[1, 0])

                x = self.data[type][exit][begin:end - 1:2]
                y = self.data[type][exit][begin + 1:end:2]
                axs[i, n].plot(x, y, "*", color="orange", label="Corr Data")

                # Title
                axs[i, n].set_title(f'Corr {type} {exit}')
                # XY-axis
                axs[i, n].set_aspect('equal')
                axs[i, n].set_xlabel("Even Samples")
                axs[i, n].set_ylabel(f"Odd Samples")
                # Legend
                axs[i, n].legend(prop={'size': 9}, loc=7)

                n += 1

        date_dir = fz.dir_format(f"{self.gdate[0]}__{self.gdate[1]}")
        path = f"../plot/{date_dir}/EvenOddAll_Analysis/Correlation/EO_Output/{self.name}/"
        Path(path).mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{path}{self.name}_{type}_Correlation_EO.png')
        if show:
            plt.show()
        plt.close(fig)

    def Plot_Correlation_RMS_EO(self, type: str, window: int, begin=0, end=-1, show=True):
        """
        Plot of Raw data DEM or PWR: Even vs Odd to see the correlation\n
        Parameters:\n
        - **type** (``str``) of data "DEM" or "PWR"\n
        - **window** (``int``): number of elements on which the RMS is calculated
        - **begin**, **end** (``int``): interval of dataset that has to be considered\n
        - **show** (``bool``): *True* -> show the plot and save the figure, *False* -> save the figure only
        Note: the 4 plots are repeated on two rows (uniform Y-scale below)\n
        """
        fig, axs = plt.subplots(nrows=2, ncols=4, constrained_layout=True, figsize=(17, 9))

        begin_date = self.Date_Update(n_samples=begin, modify=False)
        fig.suptitle(f'Correlation RMS {type} - Date: {begin_date}', fontsize=14)

        for i in range(2):
            n = 0  # type: int
            for exit in ["Q1", "Q2", "U1", "U2"]:
                if i == 1:
                    axs[i, n].sharey(axs[1, 0])
                    axs[i, n].sharex(axs[1, 0])

                x = RMS(self.data[type], window=window, exit=exit, eoa=2, begin=begin, end=end)
                y = RMS(self.data[type], window=window, exit=exit, eoa=1, begin=begin, end=end)
                axs[i, n].plot(x, y, "*", color="teal", label="Corr RMS")

                # Title
                axs[i, n].set_title(f'RMS Corr {type} {exit}')
                # XY-axis
                axs[i, n].set_aspect('equal')
                axs[i, n].set_xlabel("RMS Even Samples")
                axs[i, n].set_ylabel(f"RMS Odd Samples")
                # Legend
                axs[i, n].legend(prop={'size': 9}, loc=7)

                n += 1

        date_dir = fz.dir_format(f"{self.gdate[0]}__{self.gdate[1]}")
        path = f'../plot/{date_dir}/EvenOddAll_Analysis/Correlation/EO_RMS/{self.name}/'
        Path(path).mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{path}{self.name}_{type}_Correlation_RMS_EO.png')
        if show:
            plt.show()
        plt.close(fig)

    def Plot_SciData(self, type: str, begin=0, end=-1, smooth_len=1, show=True):
        """
        Plot of Scientific data DEMODULATED or TOTAL POWER\n
        Parameters:\n
        - **type** (``str``) of data *"DEM"* or *"PWR"*\n
        - **begin**, **end** (``int``): interval of dataset that has to be considered\n
        - **smooth_len** (``int``): number of elements on which the mobile mean is calculated
        - **show** (``bool``): *True* -> show the plot and save the figure, *False* -> save the figure only
        Note: the 4 plots are repeated on two rows (uniform Y-scale below)\n
        """
        assert (type == "DEM" or type == "PWR"), "Typo: type must be the string 'DEM' or 'PWR'"
        if type == "DEM":
            data_name = "DEMODULATED"
        elif type == "PWR":
            data_name = "TOT_POWER"
        else:
            data_name = "WRONG NAME"

        fig, axs = plt.subplots(nrows=2, ncols=4, constrained_layout=True, figsize=(17, 9))

        begin_date = self.Date_Update(n_samples=begin, modify=False)
        fig.suptitle(f'Plot Scientific data {data_name} - Date: {begin_date}', fontsize=14)

        for i in range(2):
            n = 0  # type: int
            for exit in ["Q1", "Q2", "U1", "U2"]:
                if i == 1:
                    axs[i, n].sharey(axs[1, 0])

                sci_data = self.Demodulation(type=type, exit=exit)
                y = fz.mob_mean(sci_data["sci_data"][exit][begin:end - 2], smooth_len=smooth_len)

                axs[i, n].plot(sci_data["times"][begin:len(y) + begin], y,
                               color="mediumpurple", label=f"{data_name}")

                # Title
                axs[i, n].set_title(f'{data_name} {exit}')
                # X-axis
                if self.norm_mode == 0:
                    axs[i, n].set_xlabel("# Samples")
                if self.norm_mode == 1:
                    axs[i, n].set_xlabel("Time [s]")
                # Y-axis
                axs[i, n].set_ylabel(f"{data_name}")
                # Legend
                axs[i, n].legend(prop={'size': 9}, loc=7)

                n += 1

        date_dir = fz.dir_format(f"{self.gdate[0]}__{self.gdate[1]}")
        path = f'../plot/{date_dir}/SciData_Analysis/SciData_Output/{self.name}/'
        Path(path).mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{path}{self.name}_{data_name}_smooth={smooth_len}.png')
        if show:
            plt.show()
        plt.close(fig)

    def Plot_RMS_SciData(self, type: str, window: int, begin=0, end=-1, smooth_len=1, show=True):
        """
        Plot of Raw data DEM or PWR (Even, Odd or All)\n
        Parameters:\n
        - **type** (``str``) of data *"DEM"* or *"PWR"*
        - **window** (``int``): number of elements on which the RMS is calculated
        - **begin**, **end** (``int``): interval of dataset that has to be considered
        - **smooth_len** (``int``): number of elements on which the mobile mean is calculated
        - **show** (``bool``): *True* -> show the plot and save the figure, *False* -> save the figure only
        Note: the 4 plots are repeated on two rows (uniform Y-scale below)
        """
        assert (type == "DEM" or type == "PWR"), "Typo: type must be the string 'DEM' or 'PWR'"
        if type == "DEM":
            data_name = "DEMODULATED"
        elif type == "PWR":
            data_name = "TOT_POWER"
        else:
            data_name = "WRONG NAME"

        fig, axs = plt.subplots(nrows=2, ncols=4, constrained_layout=True, figsize=(17, 9))

        begin_date = self.Date_Update(n_samples=begin, modify=False)
        fig.suptitle(f'Plot RMS Scientific data {data_name} - Date: {begin_date}', fontsize=14)

        for i in range(2):
            n = 0  # type: int
            for exit in ["Q1", "Q2", "U1", "U2"]:
                if i == 1:
                    axs[i, n].sharey(axs[1, 0])

                sci_data = self.Demodulation(type=type, exit=exit)
                rms_all = fz.mob_mean(RMS(sci_data["sci_data"], window=window, exit=exit, eoa=0, begin=begin, end=end),
                                      smooth_len=smooth_len)

                axs[i, n].plot(sci_data["times"][begin:len(rms_all) + begin], rms_all,
                               color="mediumvioletred", label=f"RMS {data_name}")

                # Title
                axs[i, n].set_title(f'RMS {data_name} {exit}')
                # X-axis
                if self.norm_mode == 0:
                    axs[i, n].set_xlabel("# Samples")
                if self.norm_mode == 1:
                    axs[i, n].set_xlabel("Time [s]")
                # Y-axis
                axs[i, n].set_ylabel(f"RMS {data_name}")
                # Legend
                axs[i, n].legend(prop={'size': 9}, loc=7)

                n += 1

        date_dir = fz.dir_format(f"{self.gdate[0]}__{self.gdate[1]}")
        path = f'../plot/{date_dir}/SciData_Analysis/SciData_RMS/{self.name}/'
        Path(path).mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{path}{self.name}_{data_name}_RMS_smooth={smooth_len}.png')
        if show:
            plt.show()
        plt.close(fig)

    # ------------------------------------------------------------------------------------------------------------------
    # TIMESTAMPS JUMP ANALYSIS
    # ------------------------------------------------------------------------------------------------------------------

    def Jump_Plot(self, show=True):
        """
        Then plot the timestamps and of the Delta time between two consecutive Timestamps.\n
        Note: the Polarimeter must be Loaded but not Prepared aka do not normalize the Timestamps.
        Parameters:\n
        - **show** (bool):\n
            *True* -> show the plot and save the figure\n
            *False* -> save the figure only
        """

        fig, axs = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(13, 3))
        # Times, we should see dot-shape jumps
        axs[0].plot(np.arange(len(self.times)), self.times.value, '*')
        axs[0].set_title(f"{self.name} Timestamps")
        axs[0].set_xlabel("# Sample")
        axs[0].set_ylabel("Time [s]")

        # Delta t
        deltat = self.times.value[:-1] - self.times.value[1:]  # t_n - t_(n+1)
        axs[1].plot(deltat, "*", color="forestgreen")
        axs[1].set_title(f"$\Delta$t {self.name}")
        axs[1].set_xlabel("# Sample")
        axs[1].set_ylabel("$\Delta$ t [s]")
        axs[1].set_ylim(-1.0, 1.0)

        date_dir = fz.dir_format(f"{self.gdate[0]}__{self.gdate[1]}")
        path = f'../plot/{date_dir}/Timestamps_Jump_Analysis/'
        Path(path).mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{path}{self.name}_Timestamps.png')
        if show:
            plt.show()
        plt.close(fig)

    # ------------------------------------------------------------------------------------------------------------------
    # FOURIER SPECTRA ANALYSIS
    # ------------------------------------------------------------------------------------------------------------------

    def Plot_FFT_EvenOdd(self, type: str, even: int, odd: int, all: int, begin=0, end=-1, nseg=np.inf, show=True,
                         spike_check=False):
        """
        Plot of Fourier Spectra of Even Odd data\n
        Parameters:\n
        - **type** (``str``) of data *"DEM"* or *"PWR"*
        - **even**, **odd**, **all** (``int``): used for the transparency of the datas (*0*=transparent, *1*=visible)
        - **begin**, **end** (``int``): interval of dataset that has to be considered
        - **nseg**: number of elements on which the periodogram is calculated. Then the average of all periodograms is
         computed to produce the spectrogram. Changing this parameter allow to reach lower frequencies in the FFT plot:
         in particular, the limInf of the x-axis is fs/nseg.
        - **show** (bool):\n
            *True* -> show the plot and save the figure\n
            *False* -> save the figure only
        Note: plots on two rows (uniform Y-scale below)
        - **spike_check** (bool):\n
            *True* -> look for spikes in the fft\n
            *False* -> do nothing
        """
        # Note: The Sampling Frequency for the Even-Odd Data is 50Hz the half of STRIP one
        fs = self.STRIP_SAMPLING_FREQ
        scaling = "spectrum"
        fig, axs = plt.subplots(nrows=2, ncols=4, constrained_layout=True, figsize=(17, 9))

        eoa = EOA(even=even, odd=odd, all=all)
        begin_date = self.Date_Update(n_samples=begin, modify=False)
        fig.suptitle(f'FFT Output {eoa} {type} - Date: {begin_date}', fontsize=14)

        for i in range(2):
            n = 0  # type: int
            for exit in ["Q1", "Q2", "U1", "U2"]:

                # Spike check
                msg = ""
                first_msg = True
                threshold = 2

                if i == 1:
                    axs[i, n].sharey(axs[1, 0])

                if all != 0:
                    f, s = scipy.signal.welch(self.data[type][exit][begin:end], fs=fs,
                                              nperseg=min(len(self.data[type][exit][begin:end]), nseg),
                                              scaling=scaling)
                    axs[i, n].plot(f[f < 25.], s[f < 25.], color="forestgreen", linewidth=0.2, marker=".",
                                   alpha=all, label="All samples")

                    # Spike check
                    if spike_check and i == 0:
                        if len(f_strip.find_spike(f, threshold=threshold)) != 0:
                            msg += f"Spikes in FFT type: {type}, exit: {exit} - all"
                            first_msg = False

                if even != 0:
                    f, s = scipy.signal.welch(self.data[type][exit][begin:end - 1:2], fs=fs / 2,
                                              nperseg=min(len(self.data[type][exit][begin:end - 1:2]), nseg),
                                              scaling=scaling)
                    axs[i, n].plot(f[f < 25.], s[f < 25.], color="royalblue", linewidth=0.2, marker=".",
                                   alpha=even, label=f"Even samples")

                    # Spike check
                    if spike_check and i == 0:
                        if len(f_strip.find_spike(f, threshold=threshold)) != 0:
                            if first_msg:
                                msg += f"Spikes in FFT type:{type}, exit:{exit} - "
                                first_msg = False
                            else:
                                msg += ", "
                            msg += f"even"

                if odd != 0:
                    f, s = scipy.signal.welch(self.data[type][exit][begin + 1:end:2], fs=fs / 2,
                                              nperseg=min(len(self.data[type][exit][begin + 1:end:2]), nseg),
                                              scaling=scaling)
                    axs[i, n].plot(f[f < 25.], s[f < 25.], color="crimson", linewidth=0.2, marker=".",
                                   alpha=odd, label=f"Odd samples")

                    # Spike check
                    if spike_check and i == 0:
                        if len(f_strip.find_spike(f, threshold=threshold)) != 0:
                            if first_msg:
                                msg += f"Spikes in FFT type: {type}, exit: {exit} - "
                                first_msg = False
                            else:
                                msg += ", "
                            msg += f"odd"
                if not first_msg:
                    self.warnings["spike_warning"].append(msg + ".<br /><p></p>")

                # Title
                axs[i, n].set_title(f"FFT {type} {exit}")
                # X-axis
                axs[i, n].set_xlabel("Frequency [Hz]")
                axs[i, n].set_xscale('log')
                # Y-axis
                axs[i, n].set_ylabel(f"FFT [{type}]")
                axs[i, n].set_yscale('log')
                # Legend
                axs[i, n].legend(prop={'size': 6}, loc=7)

                n += 1

        eoa = EOA(even=even, odd=odd, all=all)
        date_dir = fz.dir_format(f"{self.gdate[0]}__{self.gdate[1]}")
        path = f'../plot/{date_dir}/EvenOddAll_Analysis/FFT_EOA_Output/{self.name}/'
        Path(path).mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{path}{self.name}_FFT_{type}_{eoa}.png')
        if show:
            plt.show()
        plt.close(fig)

    def Plot_FFT_RMS_EO(self, type: str, window: int, even: int, odd: int, all: int, begin=0, end=-1, nseg=np.inf,
                        show=True):
        """
        Plot of Fourier Spectra of the RMS of Even Odd data\n
        Parameters:\n
        - **type** (``str``) of data "DEM" or "PWR"
        - **window** (``int``): number of elements on which the RMS is calculated
        - **even**, **odd**, **all** (``int``): used for the transparency of the datas (0=transparent, 1=visible)
        - **begin**, **end** (``int``): interval of dataset that has to be considered
        - **nseg**: number of elements on which the periodogram is calculated. Then the average of all periodograms is
         computed to produce the spectrogram. Changing this parameter allow to reach lower frequencies in the FFT plot:
         in particular, the limInf of the x-axis is fs/nseg.
        - **show** (bool):\n
            *True* -> show the plot and save the figure\n
            *False* -> save the figure only
        Note: plots on two rows (uniform Y-scale below)
        """
        # The Sampling Frequency for the Scientific Data is 50Hz the half of STRIP one
        fs = self.STRIP_SAMPLING_FREQ
        scaling = "spectrum"
        fig, axs = plt.subplots(nrows=2, ncols=4, constrained_layout=True, figsize=(17, 9))

        eoa = EOA(even=even, odd=odd, all=all)

        begin_date = self.Date_Update(n_samples=begin, modify=False)
        fig.suptitle(f'FFT RMS {eoa} {type} - Date: {begin_date}', fontsize=14)

        for i in range(2):
            n = 0  # type: int
            for exit in ["Q1", "Q2", "U1", "U2"]:
                if i == 1:
                    axs[i, n].sharey(axs[1, 0])

                if all != 0:
                    rms = RMS(self.data[type], window=window, exit=exit, eoa=0, begin=begin, end=end)
                    f, s = scipy.signal.welch(rms, fs=fs, nperseg=min(len(rms), nseg), scaling=scaling)
                    axs[i, n].plot(f[f < 25.], s[f < 25.], color="forestgreen", linewidth=0.2, marker=".",
                                   alpha=all, label="All samples")

                if even != 0:
                    rms = RMS(self.data[type], window=window, exit=exit, eoa=2, begin=begin, end=end)
                    f, s = scipy.signal.welch(rms, fs=fs / 2, nperseg=min(len(rms), nseg), scaling=scaling)
                    axs[i, n].plot(f[f < 25.], s[f < 25.], color="royalblue", linewidth=0.2, marker=".",
                                   alpha=even, label=f"Even samples")

                if odd != 0:
                    rms = RMS(self.data[type], window=window, exit=exit, eoa=1, begin=begin, end=end)
                    f, s = scipy.signal.welch(rms, fs=fs / 2, nperseg=min(len(rms), nseg), scaling=scaling)
                    axs[i, n].plot(f[f < 25.], s[f < 25.], color="crimson", linewidth=0.2, marker=".",
                                   alpha=odd, label=f"Odd samples")

                # Title
                axs[i, n].set_title(f"FFT {type} {exit}")
                # X-axis
                axs[i, n].set_xlabel("Frequency [Hz]")
                axs[i, n].set_xscale('log')
                # Y-axis
                axs[i, n].set_ylabel(f"FFT [{type}]")
                axs[i, n].set_yscale('log')
                # Legend
                axs[i, n].legend(prop={'size': 6}, loc=7)

                n += 1

        eoa = EOA(even=even, odd=odd, all=all)
        date_dir = fz.dir_format(f"{self.gdate[0]}__{self.gdate[1]}")
        path = f'../plot/{date_dir}/EvenOddAll_Analysis/FFT_EOA_RMS/{self.name}/'
        Path(path).mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{path}{self.name}_FFT_RMS_{type}_{eoa}.png')
        if show:
            plt.show()
        plt.close(fig)

    def Plot_FFT_SciData(self, type: str, begin=0, end=-1, nseg=np.inf, show=True, spike_check=False):
        """
        Plot of Fourier Spectra of Scientific data\n
        Parameters:\n
        - **type** (``str``) of data *"DEM"* or *"PWR"*
        - **begin**, **end** (``int``): interval of dataset that has to be considered
        - **nseg**: number of elements on which the periodogram is calculated. Then the average of all periodograms is
         computed to produce the spectrogram. Changing this parameter allow to reach lower frequencies in the FFT plot:
         in particular, the limInf of the x-axis is fs/nseg.
        - **show** (bool):\n
            *True* -> show the plot and save the figure\n
            *False* -> save the figure only
        Note: plots on two rows (uniform Y-scale below)
        - **spike_check** (bool):\n
            *True* -> look for spikes in the fft\n
            *False* -> do nothing
        """
        # The Sampling Frequency for the Scientific Data is 50Hz the half of STRIP one
        fs = self.STRIP_SAMPLING_FREQ / 2
        scaling = "spectrum"

        assert (type == "DEM" or type == "PWR"), "Typo: type must be the string 'DEM' or 'PWR'"
        if type == "DEM":
            data_name = "DEMODULATED"
        elif type == "PWR":
            data_name = "TOT_POWER"
        else:
            data_name = "WRONG NAME"

        fig, axs = plt.subplots(nrows=2, ncols=4, constrained_layout=True, figsize=(17, 9))

        begin_date = self.Date_Update(n_samples=begin, modify=False)
        fig.suptitle(f'FFT Scientific data {data_name} - Date: {begin_date}', fontsize=14)

        # Spike check
        msg = ""
        threshold = 2
        first_msg = True

        for i in range(2):

            if first_msg == False and i == 1:
                self.warnings["spike_warning"].append(msg + ".<br /><p></p>")

            n = 0  # type: int
            for exit in ["Q1", "Q2", "U1", "U2"]:
                if i == 1:
                    axs[i, n].sharey(axs[1, 0])

                sci_data = self.Demodulation(type=type, exit=exit)

                f, s = scipy.signal.welch(sci_data["sci_data"][exit][begin:end], fs=fs,
                                          nperseg=min(len(sci_data["sci_data"][exit][begin:end]), nseg),
                                          scaling=scaling)
                axs[i, n].plot(f[f < 25.], s[f < 25.], linewidth=0.2, marker=".",
                               color="mediumpurple", label=f"{data_name}")

                # Spike check
                if spike_check and i == 0:
                    if len(f_strip.find_spike(f, threshold=threshold)) != 0:
                        if first_msg:
                            msg += f"Spikes in FFT {data_name} exit: "
                            first_msg = False
                        else:
                            msg += ","
                        msg += f" {exit}"

                # Title
                axs[i, n].set_title(f"FFT {data_name} {exit}")
                # X-axis
                axs[i, n].set_xlabel("Frequency [Hz]")
                axs[i, n].set_xscale('log')
                # Y-axis
                axs[i, n].set_ylabel(f"FFT [{data_name}]")
                axs[i, n].set_yscale('log')
                # Legend
                axs[i, n].legend(prop={'size': 6}, loc=7)

                n += 1

        date_dir = fz.dir_format(f"{self.gdate[0]}__{self.gdate[1]}")
        path = f'../plot/{date_dir}/SciData_Analysis/FFT_Output_SciData/{self.name}/'
        Path(path).mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{path}{self.name}_FFT_{data_name}.png')
        if show:
            plt.show()
        plt.close(fig)

    def Plot_FFT_RMS_SciData(self, type: str, window: int, begin=0, end=-1, nseg=np.inf, show=True):
        """
        Plot of Fourier Spectra of the RMS of Scientific data\n
        Parameters:\n
        - **type** (``str``) of data *"DEM"* or *"PWR"*
        - **window** (``int``): number of elements on which the RMS is calculated
        - **begin**, **end** (``int``): interval of dataset that has to be considered
        - **nseg**: number of elements on which the periodogram is calculated. Then the average of all periodograms is
         computed to produce the spectrogram. Changing this parameter allow to reach lower frequencies in the FFT plot:
         in particular, the limInf of the x-axis is fs/nseg.
        - **show** (bool):\n
            *True* -> show the plot and save the figure\n
            *False* -> save the figure only
        Note: plots on two rows (uniform Y-scale below)
        """
        # The Sampling Frequency for the Scientific Data is 50Hz the half of STRIP one
        fs = self.STRIP_SAMPLING_FREQ / 2
        scaling = "spectrum"

        assert (type == "DEM" or type == "PWR"), "Typo: type must be the string 'DEM' or 'PWR'"
        if type == "DEM":
            data_name = "DEMODULATED"
        elif type == "PWR":
            data_name = "TOT_POWER"
        else:
            data_name = "WRONG NAME"

        fig, axs = plt.subplots(nrows=2, ncols=4, constrained_layout=True, figsize=(17, 9))

        begin_date = self.Date_Update(n_samples=begin, modify=False)
        fig.suptitle(f'FFT RMS Scientific Data {data_name} - Date: {begin_date}', fontsize=14)

        for i in range(2):
            n = 0  # type: int
            for exit in ["Q1", "Q2", "U1", "U2"]:
                if i == 1:
                    axs[i, n].sharey(axs[1, 0])

                sci_data = self.Demodulation(type=type, exit=exit)

                rms = RMS(sci_data["sci_data"], window=window, exit=exit, eoa=0, begin=begin, end=end)
                f, s = scipy.signal.welch(rms, fs=fs, nperseg=min(len(rms), nseg), scaling=scaling)

                axs[i, n].plot(f[f < 25.], s[f < 25.], linewidth=0.2, marker=".",
                               color="mediumvioletred", label=f"RMS {data_name}")

                # Title
                axs[i, n].set_title(f"FFT RMS {data_name} {exit}")
                # X-axis
                axs[i, n].set_xlabel("Frequency [Hz]")
                axs[i, n].set_xscale('log')
                # Y-axis
                axs[i, n].set_ylabel(f"FFT RMS [{data_name}]")
                axs[i, n].set_yscale('log')
                # Legend
                axs[i, n].legend(prop={'size': 6}, loc=7)

                n += 1

        date_dir = fz.dir_format(f"{self.gdate[0]}__{self.gdate[1]}")
        path = f'../plot/{date_dir}/SciData_Analysis/FFT_RMS_SciData/{self.name}/'
        Path(path).mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{path}{self.name}_FFT_RMS_{data_name}.png')
        if show:
            plt.show()
        plt.close(fig)

    def Plot_Correlation_Mat(self, type: str, begin=0, end=-1, scientific=True,
                             even=False, odd=False, show=False, warn_threshold=0.4):
        """
       Plot the 4x4 Correlation Matrix of the outputs of the four channel Q1, Q2, U1 and U2.\n
       Choose between of the Output or the Scientific Data.\n
       Parameters:\n
       - **type** (``str``) of data *"DEM"* or *"PWR"*
       - **begin**, **end** (``int``): interval of dataset that has to be considered
       - **scientific** (``bool``):\n
            *True* -> Scientific data are processed\n
            *False* -> Outputs are processed
       - **even** (``bool``):\n
            *True* -> Even Outputs are processed\n
            *False* -> Other Outputs are processed
       - **odd** (``bool``):\n
            *True* -> Odd Outputs are processed\n
            *False* -> Other Outputs are processed
       - **show** (``bool``):\n
            *True* -> show the plot and save the figure\n
            *False* -> save the figure only
       - **warn_threshold** (``int``): if it is overcome by one of the values of the matrix a warning is produced.\n
       """
        assert (type == "DEM" or type == "PWR"), "Typo: type must be the string 'DEM' or 'PWR'"
        sci = {}
        data_name = ""  # type: str
        if scientific:
            for exit in self.data[type].keys():
                sci_data = self.Demodulation(type=type, exit=exit, begin=begin, end=end)
                sci[exit] = sci_data["sci_data"][exit]
                if type == "DEM":
                    data_name = "DEMODULATED Data"
                elif type == "PWR":
                    data_name = "TOT_POWER Data"
        else:
            if even:
                for exit in self.data[type].keys():
                    sci[exit] = self.data[type][exit][begin:end - 1:2]
                    data_name = f"{type}_OUTPUT_EVEN Data"
            elif odd:
                for exit in self.data[type].keys():
                    sci[exit] = self.data[type][exit][begin + 1:end:2]
                    data_name = f"{type}_OUTPUT_ODD Data"
            else:
                for exit in self.data[type].keys():
                    sci[exit] = self.data[type][exit][begin:end]
                    data_name = f"{type}_OUTPUT Data"

        fig, axs = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(14, 7))

        begin_date = self.Date_Update(n_samples=begin, modify=False)
        fig.suptitle(f'Correlation Matrix {data_name} - Date: {begin_date}', fontsize=14)

        sci_data = pd.DataFrame(sci)
        corr_matrix = sci_data.corr()

        keys = list(corr_matrix.keys())
        rows = ""
        need_cap = False
        for i in corr_matrix.keys():
            """
            Put at nan the values on the diagonal of the matrix (self correlations)
            """
            corr_matrix[i][i] = np.nan
            """
            Write a warning in the report if there is high correlation between the channels
            """
            keys.remove(i)
            for j in keys:
                logging.debug(f"Correlation {i} with {j}.")
                if np.abs(corr_matrix[i][j]) > warn_threshold:
                    msg = f"High correlation ({round(corr_matrix[i][j], 6)}) " \
                          f"found in {data_name} between channel {i} and {j}."
                    logging.warning(msg)

                    need_cap = True
                    rows += f"<td align=center>{data_name}</td>" \
                            f"<td align=center>{i}</td>" \
                            f"<td align=center>{j}</td>" \
                            f"<td align=center>{round(corr_matrix[i][j], 6)}</td></tr>"
        if need_cap:
            corr_table = "<p></p>" \
                         "<style>" \
                         "table, th, td {border:1px solid black;}" \
                         "</style>" \
                         "<body>" \
                         "<p></p>" \
                         "<p></p>" \
                         "<p></p>" \
                         "<table style='width:100%' align=center>" \
                         "<tr>" \
                         "<th>Data type</th><th>Channel 1</th><th>Channel 2</th><th>Correlation Value</th>" \
                         "</tr>"
            self.warnings["corr_warning"].append(corr_table)
        self.warnings["corr_warning"].append(rows + "</table></body><p></p><p></p><p></p>")

        pl_m1 = sn.heatmap(corr_matrix, annot=True, ax=axs[0], cmap='coolwarm')
        pl_m1.set_title(f"Correlation {data_name}", fontsize=18)
        pl_m2 = sn.heatmap(corr_matrix, annot=True, ax=axs[1], cmap='coolwarm', vmin=-0.4, vmax=0.4)
        pl_m2.set_title(f"Correlation {data_name} - Fixed Scale", fontsize=18)

        date_dir = fz.dir_format(f"{self.gdate[0]}__{self.gdate[1]}")
        path = f'../plot/{date_dir}/Correlation_Matrix/Data/{self.name}/'
        Path(path).mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{path}{self.name}_CorrMat_{data_name[:-5]}.png')
        if show:
            plt.show()
        plt.close(fig)

    def Plot_Correlation_Mat_RMS(self, type: str, begin=0, end=-1, scientific=True,
                                 even=False, odd=False, show=False, warn_threshold=0.4):
        """
       Plot the 4x4 Correlation Matrix of the RMS of the outputs of the four channel Q1, Q2, U1 and U2.\n
       Choose between of the Output or the Scientific Data.\n
       Parameters:\n
       - **type** (``str``) of data *"DEM"* or *"PWR"*
       - **begin**, **end** (``int``): interval of dataset that has to be considered
       - **scientific** (``bool``):\n
            *True* -> Scientific data are processed\n
            *False* -> Outputs are processed
        - **even** (``bool``):\n
            *True* -> Even Outputs are processed\n
            *False* -> Other Outputs are processed
       - **odd** (``bool``):\n
            *True* -> Odd Outputs are processed\n
            *False* -> Other Outputs are processed
       - **show** (bool):\n
            *True* -> show the plot and save the figure\n
            *False* -> save the figure only
       - **warn_threshold** (``int``): if it is overcome by one of the values of the matrix a warning is produced.\n
       """
        assert (type == "DEM" or type == "PWR"), "Typo: type must be the string 'DEM' or 'PWR'"
        sci = {}
        data_name = ""  # type: str
        if scientific:
            for exit in self.data[type].keys():
                sci_data = self.Demodulation(type=type, exit=exit)
                sci[exit] = RMS(sci_data["sci_data"], window=100, exit=exit, eoa=0, begin=begin, end=end)

                if type == "DEM":
                    data_name = "RMS_DEMODULATED"
                elif type == "PWR":
                    data_name = "RMS_TOT_POWER"
        else:
            if even:
                data_name = f"RMS_{type}_EVEN"
                eoa = 2
            elif odd:
                data_name = f"RMS_{type}_ODD"
                eoa = 1
            else:
                data_name = f"RMS_{type}"
                eoa = 0
            for exit in self.data[type].keys():
                sci[exit] = RMS(self.data[type],
                                window=100, exit=exit, eoa=eoa, begin=begin, end=end)

        fig, axs = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(14, 7))

        begin_date = self.Date_Update(n_samples=begin, modify=False)
        fig.suptitle(f'Correlation Matrix {data_name} - Date: {begin_date}', fontsize=14)

        sci_data = pd.DataFrame(sci)
        corr_matrix = sci_data.corr()

        keys = list(corr_matrix.keys())
        rows = ""
        need_cap = False
        for i in corr_matrix.keys():
            """
            Put at nan the values on the diagonal of the matrix (self correlations)
            """
            corr_matrix[i][i] = np.nan
            """
            Write a warning in the report if there is high correlation between the channels
            """
            keys.remove(i)
            for j in keys:
                logging.debug(f"Correlation {i} with {j}.")
                if np.abs(corr_matrix[i][j]) > warn_threshold:
                    msg = f"High correlation ({round(corr_matrix[i][j], 6)}) " \
                          f"found in {data_name} between channel {i} and {j}."
                    logging.warning(msg)

                    need_cap = True
                    rows += f"<td align=center>{data_name}</td>" \
                            f"<td align=center>{i}</td>" \
                            f"<td align=center>{j}</td>" \
                            f"<td align=center>{round(corr_matrix[i][j], 6)}</td></tr>"
        if need_cap:
            corr_table = "<p></p>" \
                         "<style>" \
                         "table, th, td {border:1px solid black;}" \
                         "</style>" \
                         "<body>" \
                         "<p></p>" \
                         "<p></p>" \
                         "<p></p>" \
                         "<table style='width:100%' align=center>" \
                         "<tr>" \
                         "<th>Data type</th><th>Channel 1</th><th>Channel 2</th><th>Correlation Value</th>" \
                         "</tr>"
            self.warnings["corr_warning"].append(corr_table)
        self.warnings["corr_warning"].append(rows + "</table></body><p></p><p></p><p></p>")

        pl_m1 = sn.heatmap(corr_matrix, annot=True, ax=axs[0], cmap='coolwarm')
        pl_m1.set_title(f"Correlation {data_name}", fontsize=18)
        pl_m2 = sn.heatmap(corr_matrix, annot=True, ax=axs[1], cmap='coolwarm', vmin=-0.4, vmax=0.4)
        pl_m2.set_title(f"Correlation {data_name} - Fixed Scale", fontsize=18)

        date_dir = fz.dir_format(f"{self.gdate[0]}__{self.gdate[1]}")
        path = f'../plot/{date_dir}/Correlation_Matrix/RMS/{self.name}/'
        Path(path).mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{path}{self.name}_CorrMat_{data_name}.png')
        if show:
            plt.show()
        plt.close(fig)

    def Write_Jump(self, start_datetime: str) -> {}:
        """
        Find the 'jumps' in the timestamps of a given dataset and produce a file .txt with a description for every jump,
        including: Name_Polarimeter - Jump_Index - Delta_t before - tDelta_t after - Gregorian Date - JHD.\n
        Parameters:\n
        - **start_datetime** (``str``): start time, format: "%Y-%m-%d %H:%M:%S". That must be the start_time used
        to define the polarimeter for which the jumps dictionary has been created with the function "find_jump" above.\n
        """
        logging.basicConfig(level="INFO", format='%(message)s',
                            datefmt="[%X]", handlers=[RichHandler()])  # <3

        logging.info("Looking for jumps...\n")
        jumps = fz.find_jump(v=self.times, exp_med=0.01, tolerance=0.1)
        logging.info("Done.\n")

        if jumps["n"] == 0:
            t_warn = "No Time Jumps found in the dataset."
            logging.info(t_warn)
            self.warnings["time_warning"].append(t_warn + "<p></p>")
        else:
            t_warn = f"In the dataset there are {jumps['n']} Time Jumps."
            logging.info(t_warn + "\n\n")

            # .txt file with all time jumps.
            logging.info("I'm going to produce the caption for the file.")
            _ = fz.tab_cap_time(pol_name=self.name, file_name=start_datetime)
            new_file_name = f"JT_{self.name}_{start_datetime}.txt"

            html_tab_content = "<p></p><style>table, th, td {border:1px solid black;}</style><body>" \
                               f"<h2>Time Jumps Pol {self.name}</h2>" \
                               "<p></p><table style='width:100%' align=center>" \
                               "<tr><th># Jump</th><th>Jump value [JHD]</th><th>Jump value [s]</th>" \
                               "<th>Gregorian Date</th><th>Julian Date [JHD]</th>" \
                               "</tr>"
            i = 1
            for idx, j_value, j_val_s in zip(jumps["idx"], jumps["value"], jumps["s_value"]):
                jump_instant = self.times.value[idx]
                greg_jump_instant = Time(jump_instant, format="mjd").to_datetime().strftime("%Y-%m-%d %H:%M:%S")

                html_tab_content += f"<td align=center>{i}</td>" \
                                    f"<td align=center>{j_value}</td>" \
                                    f"<td align=center>{j_val_s}</td>" \
                                    f"<td align=center>{greg_jump_instant}</td>" \
                                    f"<td align=center>{jump_instant}</td>" \
                                    f"</tr>"

                tab_content = f"{i}\t\t{j_value}\t\t{j_val_s}\t{greg_jump_instant}\t\t{jump_instant}\n"
                with open(new_file_name, "at") as new_file:
                    new_file.write(tab_content)
                i += 1

            html_tab_content += "</table></body><p></p><p>"
            self.warnings["time_warning"].append(html_tab_content)
        return jumps

    def Inversion_EO_Time(self, jumps_pos: list, threshold=3.):
        """
        Find the inversions between even and odd output during the sampling due to time jumps.\n
        It could be also used to find even-odd inversions given a generic vector of position defining the intervals.\n
        Parameters:\n
        - **jump_pos** (``list``): obtained with the function find_jump: it contains the positions of the time jumps.\n
        """
        logging.basicConfig(level="INFO", format='%(message)s',
                            datefmt="[%X]", handlers=[RichHandler()])  # <3
        l = len(jumps_pos)
        if l == 0:
            msg = f"No jumps in the timeline: hence no inversions even-odd are due to time jumps.\n"
            logging.warning(msg)
            self.warnings["eo_warning"].append(msg)
        else:
            for type in self.data.keys():
                for idx, item in enumerate(jumps_pos):
                    if idx == 0:
                        a = 0
                    else:
                        a = jumps_pos[idx - 1]

                    b = item

                    if l > idx + 1:
                        c = jumps_pos[idx + 1]
                    else:
                        c = -1

                    for exit in self.data[type].keys():
                        logging.debug(f"{exit}) a: {a}, b: {b}, c: {c}")
                        mad_even = scs.median_abs_deviation(self.data[type][exit][b:c - 1:2])
                        mad_odd = scs.median_abs_deviation(self.data[type][exit][b + 1:c:2])

                        m_even_1 = np.median(self.data[type][exit][a:b - 1:2])
                        m_even_2 = np.median(self.data[type][exit][b:c - 1:2])

                        m_odd_1 = np.median(self.data[type][exit][a + 1:b:2])
                        m_odd_2 = np.median(self.data[type][exit][b + 1:c:2])

                        if (
                                (m_even_1 > m_even_2 + threshold * mad_even and m_odd_1 < m_odd_2 - threshold * mad_odd)
                                or
                                (m_even_1 < m_even_2 - threshold * mad_even and m_odd_1 > m_odd_2 + threshold * mad_odd)
                        ):
                            inversion_jdate = self.times[item]
                            inversion_date = Time(inversion_jdate, format="mjd").to_datetime().strftime(
                                "%Y-%m-%d %H:%M:%S")

                            msg = f"Inversion Even-Odd at {inversion_date} in {type} Output in channel {exit}.<br />"

                            self.warnings["eo_warning"].append(msg)
                            logging.warning(msg)

    def spike_report(self) -> str:
        """
            Look up for 'spikes' in the DEM and PWR output of the Polarimeter.\n
            Create a table in html language (basically a str) in which the spikes found are listed.
        """
        cap = False
        spike_tab = ""
        rows = ""
        for type in self.data.keys():
            for exit in self.data[type].keys():

                spike_idxs = f_strip.find_spike(self.data[type][exit])
                if len(spike_idxs) != 0:
                    if not cap:
                        spike_tab += "<p></p>" \
                                     "<style>" \
                                     "table, th, td {border:1px solid black;}" \
                                     "</style>" \
                                     "<body>" \
                                     "<p></p>" \
                                     "<table style='width:100%' align=center>" \
                                     "<tr>" \
                                     "<th>Spike Number</th>" \
                                     "<th>Data Type</th>" \
                                     "<th>Exit</th>" \
                                     "<th>Spike Time [JHD]</th>" \
                                     "<th>Spike Value - Median [ADU]</th></tr>"
                        cap = True

                    for idx, item in enumerate(spike_idxs):
                        rows += f"<td align=center>{idx + 1}</td>" \
                                f"<td align=center>{type}</td>" \
                                f"<td align=center>{exit}</td>" \
                                f"<td align=center>{self.times[item]}</td>" \
                                f"<td align=center>{self.data[type][exit][item] - np.median(self.data[type][exit])}</td>" \
                                f"</tr>"
                        logging.info(f"Spike n.{idx} in {exit} - {type}.\n")
        if cap:
            spike_tab += rows + "</table></body><p></p><p></p><p></p>"
        else:
            spike_tab = "No spikes detected in DEM and PWR Output.<br /><p></p>"

        return spike_tab


def RMS(data, window: int, exit: str, eoa: int, begin=0, end=-1):
    """
    Calculate the RMS of a vector using the rolling window
    Parameters:\n
    - **data** is a dictionary with four keys (exits) of a particular type *"DEM"* or *"PWR"*
    - **window**: number of elements on which the RMS is calculated
    - **exit** (``str``) *"Q1"*, *"Q2"*, *"U1"*, *"U2"*
    - **eoa** (``int``): flag in order to calculate RMS for\n
        all samples (*eoa=0*), can be used for Demodulated and Total Power scientific data (50Hz)\n
        odd samples (*eoa=1*)\n
        even samples (*eoa=2*)\n
    - **begin**, **end** (``int``): interval of dataset that has to be considered
    """
    if eoa == 0:
        rms = np.std(fz.rolling_window(data[exit][begin:end], window), axis=1)
    elif eoa == 1:
        rms = np.std(fz.rolling_window(data[exit][begin + 1:end:2], window), axis=1)
    elif eoa == 2:
        rms = np.std(fz.rolling_window(data[exit][begin:end - 1:2], window), axis=1)
    else:
        rms = np.nan
    return rms


def EOA(even: int, odd: int, all: int) -> str:
    """
    Parameters:\n
    - **even**, **odd**, **all** (``int``)
    If the variables are different from zero, this returns a string that contains the letters of the samples plotted:\n
    "E" for even (``int``)\n
    "O" for odd (``int``)\n
    "A" for all (``int``)\n
    """
    eoa = ""
    if even != 0:
        eoa += "E"
    if odd != 0:
        eoa += "O"
    if all != 0:
        eoa += "A"
    return eoa
