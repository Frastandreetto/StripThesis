#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# This file contains the function "thermal_hk" that operates an analysis of the Thermal Sensors (TS) of Strip.
# This function will be used during the system level test campaign of the LSPE-Strip instrument.
# August 18th 2023, Brescia (Italy)

# Libraries & Modules
import logging

from rich.logging import RichHandler

# Use the module logging to produce nice messages on the shell
logging.basicConfig(level="INFO", format='%(message)s',
                    datefmt="[%X]", handlers=[RichHandler()])


def thermal_hk(path_file: str, start_datetime: str, end_datetime: str,
               status: str, fft: bool, nperseg_thermal: int, corr_t: float,
               output_dir: str):
    """
    Performs the analysis of one or more polarimeters producing a complete report.
    The analysis can include plots of: Even-Odd Output, Scientific Data, FFT and correlation Matrices.
    The reports produced include also info about the state of the housekeeping parameters and the thermal sensors.

        Parameters:
            - **path_file** (``str``): location of the data file, it is indeed the location of the hdf5 file's index
            - **start_datetime** (``str``): start time
            - **end_datetime** (``str``): end time
            - **status** (``int``): status of the multiplexer of the TS to analyze: 0, 1 or 2 (which stands for both).
            - **fft** (``bool``): If true, the code will compute the power spectra of the TS.
            - **nperseg_thermal** (``int``): number of elements of thermal measures on which the fft is calculated.
            - **corr_t** (``float``): lim sup for the correlation value between two dataset:
             if the value computed is higher than the threshold, a warning is produced.
             - **output_dir** (`str`): Path of the dir that will contain the reports with the results of the analysis.
    """
    logging.info('I am C, and I am working for you!')
    return
