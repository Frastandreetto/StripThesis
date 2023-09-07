#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# This file contains the function "thermal_hk" that operates an analysis of the Thermal Sensors (TS) of Strip.
# This function will be used during the system level test campaign of the LSPE-Strip instrument.
# August 18th 2023, Brescia (Italy)

# Libraries & Modules
import logging

from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from rich.logging import RichHandler

# My Modules
import thermalsensors as ts
import f_strip as fz

# Use the module logging to produce nice messages on the shell
logging.basicConfig(level="INFO", format='%(message)s',
                    datefmt="[%X]", handlers=[RichHandler()])


def thermal_hk(path_file: str, start_datetime: str, end_datetime: str,
               status: str, fft: bool, nperseg_thermal: int,
               ts_sam_exp_med: float, ts_sam_tolerance: float,
               corr_t: float, corr_plot: bool, corr_mat: bool,
               output_plot_dir: str, output_report_dir: str):
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
            - **ts_sam_exp_med** (``float``): the exp sampling delta between two consecutive timestamps of TS
            - **ts_sam_tolerance** (``float``): the acceptance sampling tolerances of the TS
            - **corr_t** (``float``): lim sup for the correlation value between two dataset:
             if the value computed is higher than the threshold, a warning is produced.
             - **output_dir** (`str`): Path of the dir that will contain the reports with the results of the analysis.
             - **command_line** (`str`): Command line used to start the pipeline.
    """
    logging.info('\nLoading dir and templates information...')

    # Initializing the data-dict for the report
    report_data = {"output_plot_dir": output_plot_dir}

    # root: location of the file.txt with the information to build the report
    root = "../striptease/templates"
    templates_dir = Path(root)

    # Creating the Jinja2 environment
    env = Environment(loader=FileSystemLoader(templates_dir))
    # ------------------------------------------------------------------------------------------------------------------

    logging.info('Ready to analyze the Thermal Sensors.')
    TS = ts.Thermal_Sensors(path_file=path_file, start_datetime=start_datetime, end_datetime=end_datetime,
                            status=status, nperseg_thermal=nperseg_thermal)
    # Loading the TS
    logging.info('Loading TS.')
    TS.Load_TS()

    # Analyzing TS Sampling
    sampling_warn = TS.TS_Sampling_Table(sam_exp_med=ts_sam_exp_med, sam_tolerance=ts_sam_tolerance)

    # Normalizing TS measures: flag to specify the sampling frequency? Now 30s
    logging.info('Normalizing TS.')
    # Saving a list of sampling problematic TS
    problematic_TS = TS.Norm_TS()

    # Analyzing TS and collecting the results
    logging.info('Analyzing TS.')
    ts_results = TS.Analyse_TS()

    # Preparing table in markdown for the report
    logging.info('Producing TS table for the report.')
    th_table = TS.Thermal_table(results=ts_results)

    # Plots of all TS
    logging.info(f'Plotting all TS measures for status {status} of the multiplexer.')
    TS.Plot_TS()

    # Fourier's analysis if asked
    if fft:
        logging.info(f'Plotting the FFT of all the TS measures for status {status} of the multiplexer.')
        TS.Plot_FFT_TS()

    # TS Correlation plots
    if not corr_plot:
        pass
    else:
        # Collecting all names
        all_names = [name for groups in TS.ts_names.values() for name in groups if name not in problematic_TS]
        # Printing the plots with no repetitions
        logging.info(f'Plotting Correlation plots of the TS with each other.')
        for i, n1 in enumerate(all_names):
            for n2 in all_names[i + 1:]:
                TS.warnings["corr_warning"].extend(fz.correlation_plot(array1=TS.ts["thermal_data"]["calibrated"][n1],
                                                                       array2=TS.ts["thermal_data"]["calibrated"][n2],
                                                                       dict1={},
                                                                       dict2={},
                                                                       time1=TS.ts["thermal_times"],
                                                                       time2=TS.ts["thermal_times"],
                                                                       data_name1=f"{status}_{n1}",
                                                                       data_name2=f"{n2}",
                                                                       start_datetime=start_datetime,
                                                                       corr_t=corr_t,
                                                                       plot_dir=output_plot_dir))
    # Add some other correlations (?)
    if not corr_mat:
        pass
    else:
        logging.info("I'll plot correlation matrices.\n")
        # Add Plot correlation mat - which ones (?)

    # --------------------------------------------------------------------------------------------------------------
    # REPORT TS
    # --------------------------------------------------------------------------------------------------------------
    logging.info(f"\nOnce ready, I will put the TS report for the status {status} into: {output_report_dir}.")

    # Updating the report_data dict
    report_data.update({"th_table": th_table, "status": status})

    # Getting instructions to create the head of the report
    template_ts = env.get_template('report_thermals.txt')

    # Report TS generation
    filename = Path(f"{output_report_dir}/report_ts_status_{status}.md")
    with open(filename, 'w') as outf:
        outf.write(template_ts.render(report_data))

    # --------------------------------------------------------------------------------------------------------------
    # REPORT WARNINGS
    # --------------------------------------------------------------------------------------------------------------
    # Updating the report_data dict for the warning report
    report_data.update({"t_warn": TS.warnings["time_warning"],
                        "sampling_warn": sampling_warn,
                        "corr_warn": TS.warnings["corr_warning"],
                        })

    # Getting instructions to create the head of the report
    template_ts = env.get_template('report_warnings.txt')

    # Report generation
    filename = Path(f"{output_report_dir}/report_ts_warnings_{status}.md")
    with open(filename, 'w') as outf:
        outf.write(template_ts.render(report_data))

    return
