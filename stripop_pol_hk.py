#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# This file contains the function "pol_hk" that operates an analysis of the Thermal Sensors (TS) of Strip.
# This function will be used during the system level test campaign of the LSPE-Strip instrument.
# August 18th 2023, Brescia (Italy)

# Libraries & Modules
import logging

from rich.logging import RichHandler

# Use the module logging to produce nice messages on the shell
logging.basicConfig(level="INFO", format='%(message)s',
                    datefmt="[%X]", handlers=[RichHandler()])


def pol_hk(path_file: str, start_datetime: str, end_datetime: str, name_pol: str, output_dir: str):
    """
    Performs only the analysis of the Housekeeping parameters of the polarimeter(s) provided.
        Parameters:
            - **path_file** (``str``): location of the data file, it is indeed the location of the hdf5 file's index
            - **start_datetime** (``str``): start time
            - **end_datetime** (``str``): end time
            - **name_pol** (``str``): name of the polarimeter. If more than one, write them into ' ' separated by space.
            - **output_dir** (`str`): Path of the dir that will contain the reports with the results of the analysis.

    """
    logging.info('I am B, and I am working for you!')
    return
