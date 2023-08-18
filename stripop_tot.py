#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# This file contains the function "tot" that operates a complete analysis of a provided group of polarimeters.
# This function will be used during the system level test campaign of the LSPE-Strip instrument.
# August 14th 2023, Brescia (Italy)

# Libraries & Modules
import logging

from rich.logging import RichHandler

# Use the module logging to produce nice messages on the shell
logging.basicConfig(level="INFO", format='%(message)s',
                    datefmt="[%X]", handlers=[RichHandler()])


def tot(path_file: str, start_datetime: str, end_datetime: str, name_pol: str,
        eoa: str, smooth: int, window: int,
        fft: bool, nperseg: int, nperseg_thermal: int,
        spike_data: bool, spike_fft: bool,
        corr_mat: bool, corr_t: float,
        output_dir: str):
    """
    Performs the analysis of one or more polarimeters producing a complete report.
    The analysis can include plots of: Even-Odd Output, Scientific Data, FFT, correlation and  Matrices.
    The reports produced include also info about the state of the housekeeping parameters and the thermal sensors.

        Parameters:
            - **path_file** (``str``): location of the data file, it is indeed the location of the hdf5 file's index
            - **start_datetime** (``str``): start time
            - **end_datetime** (``str``): end time
            - **name_pol** (``str``): name of the polarimeter. If more than one, write them into ' ' separated by space.
            - **eoa** (``str``): states which data analyze. Even samples (e), odd samples (o), all samples (a).
            - **smooth** (``int``): Smoothing length used to flatter the data. smooth=1 equals no smooth.
            - **window** (``int``): Integer number used to convert the array of the data into a matrix
            with a number "window" of elements per row and then calculate the RMS on every row.
            window=1 equals no conversion.
            - **fft** (``bool``): If true, the code will compute the power spectra of the scientific data.
            - **nperseg** (``int``): int value that defines the number of elements of the array of scientific data o
            n which the fft is calculated.
            - **nperseg_thermal** (``int``): int value that defines the number of elements of thermal measures
            on which the fft is calculated.
            - **spike_data** (``bool``): If true, the code will look for spikes in Sci-data.
            - **spike_fft** (``bool``): If true, the code will look for spikes in FFT.
            - **corr_mat** (``bool``): If true, the code will compute the correlation matrices
            of the even-odd and scientific data.
            - **corr_t** (``float``): Floating point number used as lim sup for the correlation value
             between two dataset: if the value computed is higher than the threshold, a warning is produced.
            - **output_dir** (`str`): Path of the dir that will contain the reports with the results of the analysis.
    """
    logging.info('I am A, and I am working for you!')
    return
