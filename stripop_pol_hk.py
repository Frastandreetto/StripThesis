#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# This file contains the function "pol_hk" that operates an analysis of the Thermal Sensors (TS) of Strip.
# This function will be used during the system level test campaign of the LSPE-Strip instrument.
# August 18th 2023, Brescia (Italy)

# Libraries & Modules
import logging

from rich.logging import RichHandler

# MyLibraries & MyModules
import polarimeter as pol

# Use the module logging to produce nice messages on the shell
logging.basicConfig(level="INFO", format='%(message)s',
                    datefmt="[%X]", handlers=[RichHandler()])


def pol_hk(path_file: str, start_datetime: str, end_datetime: str, name_pol: str,
           output_plot_dir: str, output_report_dir: str,
           command_line: str):
    """
    Performs only the analysis of the Housekeeping parameters of the polarimeter(s) provided.
        Parameters:
            - **path_file** (``str``): location of the data file, it is indeed the location of the hdf5 file's index
            - **start_datetime** (``str``): start time
            - **end_datetime** (``str``): end time
            - **name_pol** (``str``): name of the polarimeter. If more than one, write them into ' ' separated by space.
            - **output_dir** (`str`): Path of the dir that will contain the reports with the results of the analysis.
            - **command_line** (`str`): Command line used to start the pipeline.
    """
    logging.info('Ready to analyze the HouseKeeping Parameters.')
    # Converting the string of polarimeters into a list
    name_pol = name_pol.split()
    # Repeating the analysis for all the polarimeters in the list
    for np in name_pol:
        logging.warning(f'Parsing {np}')
        # Initializing a Polarimeter
        p = pol.Polarimeter(name_pol=np, path_file=path_file,
                            start_datetime=start_datetime, end_datetime=end_datetime)
        # Loading the HK
        logging.info('Loading HK.')
        p.Load_HouseKeeping()
        # Normalizing the HK measures
        logging.info('Normalizing HK.')
        p.Norm_HouseKeeping()

        # Analyzing HK and collecting the results
        logging.info('Analyzing HK.')
        hk_results = p.Analyse_HouseKeeping()

        # Preparing html table for the report
        logging.info('Producing HK table for the report.')
        hk_table_html = p.HK_table(results=hk_results)

        # Plots of the Bias HK: Tensions and Currents
        logging.info('Plotting Bias HK.')
        p.Plot_Housekeeping(hk_kind="V", show=False)
        p.Plot_Housekeeping(hk_kind="I", show=False)
        # Plots of the Offsets
        logging.info('Plotting Offset.')
        p.Plot_Housekeeping(hk_kind="O", show=False)

        # Add some correlations (?)

        # Print the report
        logging.info(f"Once ready, I will put the report into: {output_report_dir}.")

    return
