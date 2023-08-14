#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# This file contains the new LSPE-Strip official pipeline for functional verification.
# It includes different analysis modalities: ...
# July 23rd 2023, Brescia (Italy) - ...

import argparse
import logging

from pathlib import Path
from rich.logging import RichHandler

import f_strip


def main():
    """
        Pipeline that can be used in three different operative modalities:
        A) "tot" -> Performs the analysis of one or more polarimeters producing a complete report.
        The analysis can include plots of: Even-Odd Output, Scientific Data, FFT and correlation Matrices.
        The reports produced include also info about the state of the housekeeping parameters and the thermal sensors.
        Parameters:
            - **path_file** (``str``): location of the data file, it is indeed the location of the hdf5 file's index
            - **start_datetime** (``str``): start time
            - **end_datetime** (``str``): end time
            - **name_pol** (``str``): name of the polarimeter. If more than one write them into ' ' separated by space.

        B) "pol_hk" -> Performs only the analysis of the Housekeeping parameters of the polarimeter(s) provided.
        Parameters:
            - **path_file** (``str``): location of the data file, it is indeed the location of the hdf5 file's index
            - **start_datetime** (``str``): start time
            - **end_datetime** (``str``): end time
            - **name_pol** (``str``): name of the polarimeter. If more than one write them into ' ' separated by space.

        C) "thermal_hk" -> Performs only the analysis of the thermal sensors of Strip.
        Parameters:
            - **path_file** (``str``): location of the data file, it is indeed the location of the hdf5 file's index
            - **start_datetime** (``str``): start time
            - **end_datetime** (``str``): end time
    """
    # Use the module logging to produce nice messages on the shell
    logging.basicConfig(level="INFO", format='%(message)s',
                        datefmt="[%X]", handlers=[RichHandler()])

    # Create the top-level argument parser by instantiating ArgumentParser
    # Note: the description is optional. It will appear if the help (-h) is required from the command line
    parser = argparse.ArgumentParser(prog='PROGRAM', description="Official pipeline for the functional verification "
                                                                 "of the LSPE-Strip instrument.\n"
                                                                 "Please choose one of the modalities A, B or C. "
                                                                 "Remember to provide all the positional arguments.")
    # Create a subparser
    subparsers = parser.add_subparsers(help='Operation modalities of the pipeline.', dest="subcommand")

    ####################################################################################################################
    # MODALITY A: TOT
    # Analyzes all the polarimeters provided by the user
    ####################################################################################################################

    # Create the parser for the command A: "tot"
    parser_A = subparsers.add_parser("tot", help="A) Analyzes all the polarimeters provided.")

    # Positional Argument (mandatory)
    # path_file
    parser_A.add_argument("path_file", action='store', type=str,
                          help='- Location of the hdf5 file index (database)')
    # start_datetime
    parser_A.add_argument("start_datetime", action='store', type=str,
                          help='- Starting datetime of the analysis. Format: "YYYY-MM-DD hh:mm:ss".')
    # end_datetime
    parser_A.add_argument("end_datetime", action='store', type=str,
                          help='- Ending datetime of the analysis. Format: "YYYY-MM-DD hh:mm:ss".')
    # name_pol
    parser_A.add_argument('name_pol', type=str, help="str containing the name(s) of the polarimeter(s).")

    # Flags (optional)
    # Even Odd All
    parser_A.add_argument('--even_odd_all', '-eoa', type=str, default='eoa',
                          help='Choose which data analyze by adding a letter in the string: '
                               'even samples (e), odd samples (o) or all samples (a).')
    # Smoothing length
    parser_A.add_argument('--smooth', '-sm', type=int, default=1,
                          help='Smoothing length used to flatter the data. smooth=1 equals no smooth.')
    # Rolling Window
    parser_A.add_argument('--window', '-w', type=int, default=1,
                          help='Integer number used to convert the array of the data into a matrix '
                               'with a number "window" of elements per row and then calculate the RMS on every row. '
                               'window=1 equals no conversion.')
    # nperseg FFT data
    parser_A.add_argument('--nperseg', '-nps', type=int, default=256,
                          help='int value that defines the number of elements of the array of scientific data'
                               'on which the fft is calculated.')
    # nperseg FFT Thermal
    parser_A.add_argument('--nperseg_thermal', '-nps_th', type=int, default=256,
                          help='int value that defines the number of elements of the array of thermal measures'
                               'on which the fft is calculated.')
    # FFT
    parser_A.add_argument('--fourier', '-fft', action="store_true",
                          help='If true, the code will compute the power spectra of the scientific data.')
    # Spikes Sci-data
    parser_A.add_argument('--spike_data', '-sd', action="store_true",
                          help='If true, the code will look for spikes in Sci-data')
    # Spikes FFT
    parser_A.add_argument('--spike_fft', '-sf', action="store_true",
                          help='If true, the code will look for spikes in FFT')

    # Correlation Matrices
    parser_A.add_argument('--corr_mat', '-cm', action="store_true",
                          help='If true, the code will compute the correlation matrices '
                               'of the even-odd and scientific data.')
    # Correlation Matrices Threshold
    parser_A.add_argument('--corr_mat_t', '-cmt', type=float, default=0.4,
                          help='Floating point number used as lim sup for the correlation value between two dataset: '
                               'if the value computed is higher than the threshold, a warning is produced.')

    ####################################################################################################################
    # MODALITY B: POL_HK
    # Analyzes the housekeeping parameters of the polarimeters provided by the user.
    ####################################################################################################################

    # Create the parser for the command B: "pol_hk"
    parser_B = subparsers.add_parser('pol_hk',
                                     help="B) Analyzes the housekeeping parameters of the polarimeters provided.")

    # Positional Argument (mandatory)
    # path_file
    parser_B.add_argument("path_file", action='store', type=str,
                          help='- Location of the hdf5 file index (database)')
    # start_datetime
    parser_B.add_argument("start_datetime", action='store', type=str,
                          help='- Starting datetime of the analysis. Format: "YYYY-MM-DD hh:mm:ss".')
    # end_datetime
    parser_B.add_argument("end_datetime", action='store', type=str,
                          help='- Ending datetime of the analysis. Format: "YYYY-MM-DD hh:mm:ss".')
    # name_pol
    parser_B.add_argument('name_pol', type=str, help="str containing the name(s) of the polarimeter(s).")

    ####################################################################################################################
    # MODALITY C: THERMAL_HK
    # Analyzes the thermal sensors of LSPE-Strip.
    ####################################################################################################################

    # Create the parser for the command C: "thermal_hk"
    parser_C = subparsers.add_parser('thermal_hk', help="C) Analyzes the thermal sensors of LSPE-Strip.")

    # Positional Argument (mandatory)
    # path_file
    parser_C.add_argument("path_file", action='store', type=str,
                          help='- Location of the hdf5 file index (database)')
    # start_datetime
    parser_C.add_argument("start_datetime", action='store', type=str,
                          help='- Starting datetime of the analysis. Format: "YYYY-MM-DD hh:mm:ss".')
    # end_datetime
    parser_C.add_argument("end_datetime", action='store', type=str,
                          help='- Ending datetime of the analysis. Format: "YYYY-MM-DD hh:mm:ss".')
    # Flags (optional)
    # nperseg FFT Thermal
    parser_C.add_argument('--nperseg_thermal', '-nps_th', type=int, default=256,
                          help='int value that defines the number of elements of the array of thermal measures'
                               'on which the fft is calculated.')

    ####################################################################################################################
    # Option for all the modalities
    # Output directory of the reports
    parser.add_argument('--output_dir', '-od', type=str, default='../plot/default_reports',
                        help='Path of the dir that will contain the reports with the results of the analysis.')

    ####################################################################################################################
    # Call .parse_args() on the parser to get the Namespace object that contains all the user’s arguments.
    args = parser.parse_args()
    logging.info(args)

    ####################################################################################################################
    # CHECKS & Tests

    # Check if the dir provided by the user exists.
    try:
        Path(args.path_file)
    except AttributeError:
        logging.error("Modality not selected. Type -h for help!")
        raise SystemExit(1)

    if not Path(args.path_file).exists():
        logging.error(f"The target directory {args.path_file} does not exist. "
                      f"Please select a real location of the hdf5 file index.\n"
                      "Note: no quotation marks needed.")
        raise SystemExit(1)

    # Check on datatime objects: start_datetime & end_datetime
    if not f_strip.datetime_check(args.start_datetime):
        logging.error("start_datetime: wrong datetime format.")
        raise SystemExit(1)
    if not f_strip.datetime_check(args.end_datetime):
        logging.error("end_datetime: wrong datetime format.")
        raise SystemExit(1)

    # Check on the consequentiality of the datetime
    if args.end_datetime < args.start_datetime:
        logging.error("end_datetime is before than start_datetime: wrong datetime values.")
        raise SystemExit(1)

    # Check on the names of the polarimeters
    if args.subcommand == "tot" or args.subcommand == "pol_hk":
        name_pol = args.name_pol.split()

        if not f_strip.name_check(name_pol):
            logging.error("The names of the polarimeters provided are not valid. Please check them again."
                          "They must be written as follows: 'B0 B1'")
            raise SystemExit(1)

    ####################################################################################################################
    if args.subcommand == "tot":
        logging.info("The total analysis is beginning... Have a seat!")
        # f_strip.total()
    elif args.subcommand == "pol_hk":
        logging.info("The housekeeping analysis is beginning...")
        # f_strip.pol_hk()
    elif args.subcommand == "thermal_hk":
        logging.info("The thermal analysis is beginning...")
        # f_strip.thermal_hk()


if __name__ == "__main__":
    main()
