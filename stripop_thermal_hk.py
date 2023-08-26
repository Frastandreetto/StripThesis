#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# This file contains the function "thermal_hk" that operates an analysis of the Thermal Sensors (TS) of Strip.
# This function will be used during the system level test campaign of the LSPE-Strip instrument.
# August 18th 2023, Brescia (Italy)

# Libraries & Modules
import logging

from rich.logging import RichHandler

# My Modules
import thermalsensors as ts
import f_strip as fz

# Use the module logging to produce nice messages on the shell
logging.basicConfig(level="INFO", format='%(message)s',
                    datefmt="[%X]", handlers=[RichHandler()])


def thermal_hk(path_file: str, start_datetime: str, end_datetime: str,
               status: str, fft: bool, nperseg_thermal: int, corr_t: float,
               output_plot_dir: str, output_report_dir: str,
               command_line: str):
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
             - **command_line** (`str`): Command line used to start the pipeline.
    """
    logging.info('Ready to analyze the Thermal Sensors.')
    TS = ts.Thermal_Sensors(path_file=path_file, start_datetime=start_datetime, end_datetime=end_datetime,
                            status=status, nperseg_thermal=nperseg_thermal)
    # Loading the TS
    logging.info('Loading TS.')
    TS.Load_TS()
    # Normalizing TS measures: flag to specify the sampling frequency? Now 30s
    logging.info('Normalizing TS.')
    # Saving a list of sampling problematic TS
    problematic_TS = TS.Norm_TS()

    # Analyzing TS and collecting the results
    logging.info('Analyzing TS.')
    ts_results = TS.Analyse_TS()

    # Preparing html table for the report
    logging.info('Producing TS table for the report.')
    th_table_html = TS.Thermal_table(results=ts_results)

    # Plots of all TS
    logging.info(f'Plotting all TS measures for status {status} of the multiplexer.')
    TS.Plot_TS()

    # Fourier's analysis if asked
    if fft:
        logging.info(f'Plotting the FFT of all the TS measures for status {status} of the multiplexer.')
        TS.Plot_FFT_TS()

    # TS Correlation plots
    # Collecting all names
    all_names = [name for groups in TS.ts_names.values() for name in groups if name not in problematic_TS]
    # Printing the plots with no repetitions
    logging.info(f'Plotting Correlation plots of the TS with each other.')
    for i, n1 in enumerate(all_names):
        for n2 in all_names[i + 1:]:
            fz.correlation_plot(array1=TS.ts["thermal_data"]["calibrated"][n1],
                                array2=TS.ts["thermal_data"]["calibrated"][n2],
                                dict1={},
                                dict2={},
                                time1=TS.ts["thermal_times"],
                                time2=TS.ts["thermal_times"],
                                data_name1=f"{status}_{n1}",
                                data_name2=f"{n2}",
                                start_datetime=start_datetime,
                                end_datetime=end_datetime,
                                corr_t=corr_t
                                )
    # Print the report
    logging.info(f"Once ready, I will put the report into: {output_report_dir}.")

    return
