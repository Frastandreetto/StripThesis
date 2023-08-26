#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# This file contains the new LSPE-Strip official pipeline for functional verification.
# It includes different analysis modalities: total analysis, housekeeping analysis and thermal analysis
# July 23rd 2023, Brescia (Italy) - ...

# Libraries & Modules
import argparse
import logging
import sys

from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from rich.logging import RichHandler

# MyLibraries & MyModules
import f_strip as fz
import stripop_tot as strip_a
import stripop_pol_hk as strip_b
import stripop_thermal_hk as strip_c


def main():
    """
        Pipeline that can be used in three different operative modalities:

        A) "tot" -> Performs the analysis of one or more polarimeters producing a complete report.
        The analysis can include plots of: Even-Odd Output, Scientific Data, FFT, correlation and Matrices.
        The reports produced include also info about the state of the housekeeping parameters and the thermal sensors.

            Parameters:
                - **path_file** (``str``): location of the data file, it is indeed the location of the hdf5 file's index
                - **start_datetime** (``str``): start time
                - **end_datetime** (``str``): end time
                - **name_pol** (``str``): name of the polarimeter. If more than one write them spaced into ' '.

        B) "pol_hk" -> Performs only the analysis of the Housekeeping parameters of the polarimeter(s) provided.

            Parameters:
                - **path_file** (``str``): location of the data file, it is indeed the location of the hdf5 file's index
                - **start_datetime** (``str``): start time
                - **end_datetime** (``str``): end time
                - **name_pol** (``str``): name of the polarimeter. If more than one write them spaced into ' '.

        C) "thermal_hk" -> Performs only the analysis of the thermal sensors of Strip.

            Parameters:
            - **path_file** (``str``): location of the data file, it is indeed the location of the hdf5 file's index
            - **start_datetime** (``str``): start time
            - **end_datetime** (``str``): end time
            - **status** (``int``): status of the multiplexer of the TS to analyze: 0, 1 or 2 (which stands for both).
            - **fft** (``bool``): If true, the code will compute the power spectra of the TS.
            - **nperseg_thermal** (``int``): number of elements of thermal measures on which the fft is calculated.
            - **corr_t** (``float``): lim sup for the correlation value between two dataset:
             if the value computed is higher than the threshold, a warning is produced.

    """
    # Use the module logging to produce nice messages on the shell
    logging.basicConfig(level="INFO", format='%(message)s',
                        datefmt="[%X]", handlers=[RichHandler()])

    # Command Line used to start the pipeline
    command_line = " ".join(sys.argv)

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
    # Thermal Sensors
    parser_A.add_argument('--thermal_sensors', '-ts', action="store_true", default=False,
                          help='If true, the code will analyze the Thermal Sensors of Strip.')
    # Housekeeping Parameters
    parser_A.add_argument('--housekeeping', '-hk', action="store_true", default=False,
                          help='If true, the code will analyze the Housekeeping parameters of the Polarimeters.')
    # Even Odd All
    parser_A.add_argument('--even_odd_all', '-eoa', type=str, default='eoa',
                          help='Choose which data analyze by adding a letter in the string: '
                               'even samples (e), odd samples (o) or all samples (a).')
    # Scientific Data
    parser_A.add_argument('--scientific', '-sci', action="store_true", default=False,
                          help='If true, the code will compute the double demodulation analyze the scientific data.')
    # Smoothing length
    parser_A.add_argument('--smooth', '-sm', type=int, default=1,
                          help='Smoothing length used to flatter the data. smooth=1 equals no smooth.')
    # Rolling Window
    parser_A.add_argument('--window', '-w', type=int, default=2,
                          help='Integer number used to convert the array of the data into a matrix '
                               'with a number "window" of elements per row and then calculate the RMS on every row.')
    # FFT
    parser_A.add_argument('--fourier', '-fft', action="store_true",
                          help='If true, the code will compute the power spectra of the scientific data.')
    # nperseg FFT data
    parser_A.add_argument('--nperseg', '-nps', type=int, default=256,
                          help='int value that defines the number of elements of the array of scientific data'
                               'on which the fft is calculated.')
    # nperseg FFT Thermal
    parser_A.add_argument('--nperseg_thermal', '-nps_th', type=int, default=256,
                          help='int value that defines the number of elements of the array of thermal measures'
                               'on which the fft is calculated.')
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
    # Correlation Threshold
    parser_A.add_argument('--corr_t', '-ct', type=float, default=0.4,
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
    # FFT
    parser_C.add_argument('--fourier', '-fft', action="store_true",
                          help='If true, the code will compute the power spectra of the thermal measures.')
    # nperseg FFT Thermal
    parser_C.add_argument('--nperseg_thermal', '-nps_th', type=int, default=256,
                          help='int value that defines the number of elements of the array of thermal measures'
                               'on which the fft is calculated.')
    # Status
    parser_C.add_argument('--status', '-stat', type=int, default=2, choices=[0, 1, 2],
                          help='int value that defines the status of the multiplexer of the TS to analyze: 0 or 1. '
                               'If it is set on 2, both states will be analyzed.')
    # Correlation Threshold
    parser_C.add_argument('--corr_t', '-ct', type=float, default=0.4,
                          help='Floating point number used as lim sup for the correlation value between two dataset: '
                               'if the value computed is higher than the threshold, a warning is produced.')

    ####################################################################################################################
    # Option for all the modalities
    # Output directory of the plots
    parser.add_argument('--output_plot_dir', '-opd', type=str, default='../plot/',
                        help='Path of the dir that will contain the plots of the analysis.')
    # Output directory of the reports
    parser.add_argument('--output_report_dir', '-ord', type=str, default='../plot/Reports',
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
        logging.error('Modality not selected. Type -h for help!')
        raise SystemExit(1)

    if not Path(args.path_file).exists():
        logging.error(f'The target directory {args.path_file} does not exist. '
                      f'Please select a real location of the hdf5 file index.\n'
                      'Note: no quotation marks needed.')
        raise SystemExit(1)

    # Check on datatime objects: start_datetime & end_datetime
    if not fz.datetime_check(args.start_datetime):
        logging.error('start_datetime: wrong datetime format.')
        raise SystemExit(1)
    if not fz.datetime_check(args.end_datetime):
        logging.error('end_datetime: wrong datetime format.')
        raise SystemExit(1)

    # Check on datetime
    # Consequentiality of the datetime
    if args.end_datetime < args.start_datetime:
        logging.error('end_datetime is before than start_datetime: wrong datetime values.')
        raise SystemExit(1)
    # Same datetime
    if args.end_datetime == args.start_datetime:
        logging.error('end_datetime is equal to start_datetime: wrong datetime values.')
        raise SystemExit(1)

    # MODE A: check on EOA string
    if args.subcommand == "tot":
        if args.even_odd_all not in [' ', 'e', 'o', 'a', 'eo', 'ea', 'oa', 'eoa']:
            logging.error('Wrong data name:. Please choose between the options: e, o, a, eo, ea, oa, eoa')
            raise SystemExit(1)

    # MODE A and B: Check on the names of the polarimeters
    if args.subcommand == "tot" or args.subcommand == "pol_hk":
        name_pol = args.name_pol.split()

        if not fz.name_check(name_pol):
            logging.error('The names of the polarimeters provided are not valid. Please check them again.'
                          'They must be written as follows: \'B0 B1\'')
            raise SystemExit(1)

    # MODE C: Check on the status value
    if args.subcommand == "thermal_hk":
        if args.status not in ([0, 1, 2]):
            logging.error('Invalid status value. Please choose between the values 0 and 1 for a single analysis. '
                          'Choose the value 2 to have both.')
            raise SystemExit(1)

    ####################################################################################################################
    # Operations: A-B-C
    if args.subcommand == "tot":
        logging.info('The total analysis is beginning... Take a seat!')
        # Total Analysis Operation
        strip_a.tot(path_file=args.path_file, start_datetime=args.start_datetime, end_datetime=args.end_datetime,
                    thermal_sensors=args.thermal_sensors, housekeeping=args.housekeeping,
                    name_pol=args.name_pol, eoa=args.even_odd_all, scientific=args.scientific,
                    smooth=args.smooth, window=args.window,
                    fft=args.fourier, nperseg=args.nperseg, nperseg_thermal=args.nperseg_thermal,
                    spike_data=args.spike_data, spike_fft=args.spike_fft, corr_mat=args.corr_mat, corr_t=args.corr_t,
                    command_line=command_line,
                    output_plot_dir=args.output_plot_dir, output_report_dir=args.output_report_dir)

    elif args.subcommand == "pol_hk":
        logging.info('The housekeeping analysis is beginning...')
        # Housekeeping Analysis Operation
        strip_b.pol_hk(path_file=args.path_file, start_datetime=args.start_datetime, end_datetime=args.end_datetime,
                       name_pol=args.name_pol, command_line=command_line,
                       output_plot_dir=args.output_plot_dir, output_report_dir=args.output_report_dir)

    elif args.subcommand == "thermal_hk":
        # Thermal Sensors Analysis Operation
        logging.info('The thermal analysis is beginning...')

        # If status is not specified, the analysis is done on both the states of the multiplexer
        if args.status == 2:
            for status in [0, 1]:
                strip_c.thermal_hk(path_file=args.path_file,
                                   start_datetime=args.start_datetime, end_datetime=args.end_datetime,
                                   status=status, fft=args.fourier, nperseg_thermal=args.nperseg_thermal,
                                   corr_t=args.corr_t, command_line=command_line,
                                   output_plot_dir=args.output_plot_dir, output_report_dir=args.output_report_dir)
        else:
            strip_c.thermal_hk(path_file=args.path_file,
                               start_datetime=args.start_datetime, end_datetime=args.end_datetime,
                               status=args.status, fft=args.fourier, nperseg_thermal=args.nperseg_thermal,
                               corr_t=args.corr_t, command_line=command_line,
                               output_plot_dir=args.output_plot_dir, output_report_dir=args.output_report_dir)

    # --------------------------------------------------------------------------------------------------------------
    # REPORT
    # --------------------------------------------------------------------------------------------------------------
    logging.info(f"\nI am putting the header report into: {args.output_report_dir}.")

    header_report_data = {
        "path_file": args.path_file,
        "analysis_date": str(f"{args.start_datetime} - {args.end_datetime}"),
        "output_plot_dir": args.output_plot_dir,
        "output_report_dir": args.output_report_dir,
        "command_line": command_line
    }

    # root: location of the file.txt with the information to build the report
    root = "../striptease/templates"
    templates_dir = Path(root)

    # Creating the Jinja2 environment
    env = Environment(loader=FileSystemLoader(templates_dir))
    # Getting instructions to create the head of the report
    header_template = env.get_template('report_header.txt')

    # Check if the dir exists. If not, it will be created.
    Path(args.output_report_dir).mkdir(parents=True, exist_ok=True)
    # Report generation
    filename = Path(f"{args.output_report_dir}/report_head_{args.subcommand}.md")
    with open(filename, 'w') as outf:
        outf.write(header_template.render(header_report_data))


if __name__ == "__main__":
    main()
