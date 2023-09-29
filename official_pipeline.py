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

    ####################################################################################################################
    # Create a COMMON ARGUMENT parser for shared arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    # path_file
    common_parser.add_argument("path_file", action='store', type=str,
                               help='- Location of the hdf5 file index (database)')
    # start_datetime
    common_parser.add_argument("start_datetime", action='store', type=str,
                               help='- Starting datetime of the analysis. Format: "YYYY-MM-DD hh:mm:ss".')
    # end_datetime
    common_parser.add_argument("end_datetime", action='store', type=str,
                               help='- Ending datetime of the analysis. Format: "YYYY-MM-DD hh:mm:ss".')

    # Sampling Parameters ----------------------------------------------------------------------------------------------
    # Housekeeping Sampling Expected Median
    common_parser.add_argument('--hk_sam_exp_med', '-hk_sem',
                               type=lambda x: [float(val) for val in x.split(',')], default=[1.4, 1.4, 10.],
                               help='Contains the exp sampling delta between two consecutive timestamps of the hk. '
                                    '(default: %(default)s)', metavar='')
    # Housekeeping Sampling Tolerance
    common_parser.add_argument('--hk_sam_tolerance', '-hk_st',
                               type=lambda x: [float(val) for val in x.split(',')], default=[0.1, 0.1, 0.5],
                               help='Contains the acceptance sampling tolerances of the hk parameters: I,V,O. '
                                    '(default: %(default)s)', metavar='')
    # Thermal Sensors Sampling Expected Median
    common_parser.add_argument('--ts_sam_exp_med', '-ts_sem', type=float, default=60.,
                               help='the exp sampling delta between two consecutive timestamps of the Thermal Sensors. '
                                    '(default: %(default)s)', metavar='')
    # Thermal Sensors Sampling Tolerance
    common_parser.add_argument('--ts_sam_tolerance', '-ts_st', type=float, default=1.,
                               help='the acceptance sampling tolerances of the Thermal Sensors (default: %(default)s).',
                               metavar='')

    # Correlation Parameters -------------------------------------------------------------------------------------------
    # Correlation Plot
    common_parser.add_argument('--corr_plot', '-cp', action="store_true",
                               help='If true, the code will compute the correlation plots '
                                    'of the even-odd and scientific data.')
    # Correlation Matrices
    common_parser.add_argument('--corr_mat', '-cm', action="store_true",
                               help='If true, the code will compute the correlation matrices '
                                    'of the even-odd and scientific data.')
    # Correlation Threshold
    common_parser.add_argument('--corr_t', '-ct', type=float, default=0.4,
                               help='Floating point number used as lim sup for the corr value between two dataset: '
                                    'if the value computed is higher than the threshold, a warning is produced '
                                    '(default: %(default)s).', metavar='')
    # Cross Correlation
    common_parser.add_argument('--cross_corr', '-cc', action="store_true",
                               help='If true, compute the 55x55 corr matr between the exits of all polarimeters.')

    # Output parameters ------------------------------------------------------------------------------------------------
    # Output directory of the plots
    common_parser.add_argument('--output_plot_dir', '-opd', type=str, default='../plot',
                               help='Path of the dir that will contain the plots of the analysis '
                                    '(default: %(default)s).', metavar='')
    # Output directory of the reports
    common_parser.add_argument('--output_report_dir', '-ord', type=str, default='../plot/Reports',
                               help='Path of the dir that will contain the reports with the results of the analysis '
                                    '(default: %(default)s).', metavar='')

    ####################################################################################################################
    # Create subparsers
    subparsers = parser.add_subparsers(help='Operation modalities of the pipeline.', dest="subcommand")

    ####################################################################################################################
    # MODALITY A: TOT
    # Analyzes all the polarimeters provided by the user
    ####################################################################################################################

    # Create the parser for the command A: "tot"
    parser_A = subparsers.add_parser("tot", parents=[common_parser], help="A) Analyzes all the polarimeters provided.")

    # Positional Arguments (mandatory)
    # name_pol
    parser_A.add_argument('name_pol', type=str, help="- str containing the name(s) of the polarimeter(s). "
                                                     "Write \'all\' to perform the complete analysis")

    # Flags (optional)
    # Thermal Sensors
    parser_A.add_argument('--thermal_sensors', '-ts', action="store_true", default=False,
                          help='If true, the code will analyze the Thermal Sensors of Strip'
                               '(default: %(default)s).')
    # Housekeeping Parameters
    parser_A.add_argument('--housekeeping', '-hk', action="store_true", default=False,
                          help='If true, the code will analyze the Housekeeping parameters of the Polarimeters '
                               '(default: %(default)s).')
    # Even Odd All
    parser_A.add_argument('--even_odd_all', '-eoa', type=str, default='EOA',
                          help='Choose which data analyze by adding a letter in the string: '
                               'even samples (E), odd samples (O) or all samples (A) (default: %(default)s).',
                          metavar='')
    # Scientific Data
    parser_A.add_argument('--scientific', '-sci', action="store_true", default=False,
                          help='If true, compute the double demodulation and analyze the scientific data '
                               '(default: %(default)s).')
    # Rms
    parser_A.add_argument('--rms', '-rms', action="store_true", default=False,
                          help='If true, compute the rms on the scientific output and data '
                               '(default: %(default)s).')
    # Scientific Output Sampling Tolerance
    parser_A.add_argument('--sam_tolerance', '-st', type=float, default=0.005,
                          help='The acceptance sampling tolerances of the Scientific Output of Strip '
                               '(default: %(default)s).', metavar='')
    # Smoothing length
    parser_A.add_argument('--smooth', '-sm', type=int, default=1,
                          help='Smoothing length used to flatter the data. smooth=1 equals no smooth '
                               '(default: %(default)s).', metavar='')
    # Rolling Window
    parser_A.add_argument('--window', '-w', type=int, default=2,
                          help='Integer number used to convert the array of the data into a matrix '
                               'with a number "window" of elements per row and then calculate the RMS on every row '
                               '(default: %(default)s).', metavar='')
    # FFT
    parser_A.add_argument('--fourier', '-fft', action="store_true",
                          help='If true, the code will compute the power spectra of the scientific data.')
    # nperseg FFT data
    parser_A.add_argument('--nperseg', '-nps', type=int, default=256,
                          help='int value that defines the number of elements of the array of scientific data'
                               'on which the fft is calculated (default: %(default)s).', metavar='')
    # nperseg FFT Thermal
    parser_A.add_argument('--nperseg_thermal', '-nps_th', type=int, default=256,
                          help='int value that defines the number of elements of the array of thermal measures'
                               'on which the fft is calculated (default: %(default)s).', metavar='')
    # Spikes Sci-data
    parser_A.add_argument('--spike_data', '-sd', action="store_true",
                          help='If true, the code will look for spikes in Sci-data')
    # Spikes FFT
    parser_A.add_argument('--spike_fft', '-sf', action="store_true",
                          help='If true, the code will look for spikes in FFT')

    ####################################################################################################################
    # MODALITY B: POL_HK
    # Analyzes the housekeeping parameters of the polarimeters provided by the user.
    ####################################################################################################################

    # Create the parser for the command B: "pol_hk"
    parser_B = subparsers.add_parser('pol_hk', parents=[common_parser],
                                     help="B) Analyzes the housekeeping parameters of the polarimeters provided.")

    # Positional Argument (mandatory)
    # name_pol
    parser_B.add_argument('name_pol', type=str, help="- str containing the name(s) of the polarimeter(s). "
                                                     "Write \'all\' to perform the complete analysis")

    ####################################################################################################################
    # MODALITY C: THERMAL_HK
    # Analyzes the thermal sensors of LSPE-Strip.
    ####################################################################################################################

    # Create the parser for the command C: "thermal_hk"
    parser_C = subparsers.add_parser('thermal_hk', parents=[common_parser],
                                     help="C) Analyzes the thermal sensors of LSPE-Strip.")

    # Flags (optional)
    # FFT
    parser_C.add_argument('--fourier', '-fft', action="store_true",
                          help='If true, the code will compute the power spectra of the thermal measures.')
    # nperseg FFT Thermal
    parser_C.add_argument('--nperseg_thermal', '-nps_th', type=int, default=256,
                          help='int value that defines the number of elements of the array of thermal measures'
                               'on which the fft is calculated (default: %(default)s).', metavar='')
    # Status
    parser_C.add_argument('--status', '-stat', type=int, default=2, choices=[0, 1, 2],
                          help='int value that defines the status of the multiplexer of the TS to analyze: 0 or 1. '
                               'If it is set on 2, both states will be analyzed (default: %(default)s).')
    # Spikes TS
    parser_C.add_argument('--spike_ts', '-s_ts', action="store_true",
                          help='If true, the code will look for spikes in Thermal Sensors')
    # Spikes FFT TS
    parser_C.add_argument('--spike_fft', '-sf_ts', action="store_true",
                          help='If true, the code will look for spikes in FFT of the Thermal Sensors')

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
        # Create a set of E,O,A
        args.even_odd_all = set(str(args.even_odd_all).upper())
        if not args.even_odd_all.issubset({"E", "O", "A"}):
            logging.error('Wrong data name: Please choose between the options: E, O, A, EO, EA, OA, EOA')
            raise SystemExit(1)

    # MODE A and B: Check on the names of the polarimeters
    if args.subcommand == "tot" or args.subcommand == "pol_hk":

        # Case with all the polarimeter
        if args.name_pol == "all":
            # Assign all pol names to the corresponding arg
            args.name_pol = ("B0 B1 B2 B3 B4 B5 B6 I0 I1 I2 I3 I4 I5 I6 "
                             "G0 G1 G2 G3 G4 G5 G6 O0 O1 O2 O3 O4 O5 O6 "
                             "R0 R1 R2 R3 R4 R5 R6 V0 V1 V2 V3 V4 V5 V6 "
                             "W1 W2 W3 W4 W5 W6 Y0 Y1 Y2 Y3 Y4 Y5 Y6")

        # Create a list of polarimeters names
        name_pol = args.name_pol.split()
        # Check the names of the polarimeters provided
        if not fz.name_check(name_pol):
            logging.error('The names of the polarimeters provided are not valid. Please check the parameter name_pol. '
                          '\nThe names must be written as follows: \'B0 B1\'')
            raise SystemExit(1)

    # MODE C: Check on the status value
    if args.subcommand == "thermal_hk":
        if args.status not in ([0, 1, 2]):
            logging.error('Invalid status value. Please choose between the values 0 and 1 for a single analysis. '
                          'Choose the value 2 to have both.')
            raise SystemExit(1)

    # Common Mode: Check if hk_sam_exp_med and hk_sam_tolerance are float
    for item in args.hk_sam_exp_med:
        if not isinstance(item, (int, float)):
            logging.error(f'Invalid Sampling Median Value: {item}. Please choose a number (float or int).')
            raise SystemExit(1)
    for item in args.hk_sam_tolerance:
        if not isinstance(item, (int, float)):
            logging.error(f'Invalid Sampling Tolerance Value: {item}. Please choose a number (float or int).')
            raise SystemExit(1)

    # Store the values into a dict
    hk_sam_exp_med = {"I": args.hk_sam_exp_med[0], "V": args.hk_sam_exp_med[1], "O": args.hk_sam_exp_med[2]}
    hk_sam_tolerance = {"I": args.hk_sam_tolerance[0], "V": args.hk_sam_tolerance[1], "O": args.hk_sam_tolerance[2]}

    ####################################################################################################################
    # Reports Requirements

    logging.info('\nLoading dir and templates information...')

    # Directory where to save all the reports of a given analysis
    date_dir = fz.dir_format(f"{args.start_datetime}__{args.end_datetime}")

    # Creating the correct path for the PLOT dir: adding the date_dir
    args.output_plot_dir = f"{args.output_plot_dir}/{date_dir}"
    # Check if the dir exists. If not, it will be created.
    Path(args.output_plot_dir).mkdir(parents=True, exist_ok=True)

    # Creating the correct path for the REPORT dir: adding the date_dir
    args.output_report_dir = f"{args.output_report_dir}/{date_dir}"
    # Check if the dir exists. If not, it will be created.
    Path(args.output_report_dir).mkdir(parents=True, exist_ok=True)

    ####################################################################################################################
    # Operations: A-B-C
    if args.subcommand == "tot":
        logging.info('The total analysis is beginning... Take a seat!')
        # Total Analysis Operation
        strip_a.tot(path_file=args.path_file, start_datetime=args.start_datetime, end_datetime=args.end_datetime,
                    thermal_sensors=args.thermal_sensors, housekeeping=args.housekeeping,
                    sam_tolerance=args.sam_tolerance,
                    hk_sam_exp_med=hk_sam_exp_med, hk_sam_tolerance=hk_sam_tolerance,
                    ts_sam_exp_med=args.ts_sam_exp_med, ts_sam_tolerance=args.ts_sam_tolerance,
                    name_pol=args.name_pol, eoa=args.even_odd_all, scientific=args.scientific, rms=args.rms,
                    smooth=args.smooth, window=args.window,
                    fft=args.fourier, nperseg=args.nperseg, nperseg_thermal=args.nperseg_thermal,
                    spike_data=args.spike_data, spike_fft=args.spike_fft,
                    corr_plot=args.corr_plot, corr_mat=args.corr_mat, corr_t=args.corr_t, cross_corr=args.cross_corr,
                    output_plot_dir=args.output_plot_dir, output_report_dir=args.output_report_dir)

    elif args.subcommand == "pol_hk":
        logging.info('The housekeeping analysis is beginning...')
        # Housekeeping Analysis Operation
        strip_b.pol_hk(path_file=args.path_file, start_datetime=args.start_datetime, end_datetime=args.end_datetime,
                       name_pol=args.name_pol,
                       hk_sam_exp_med=hk_sam_exp_med, hk_sam_tolerance=hk_sam_tolerance,
                       output_plot_dir=args.output_plot_dir, output_report_dir=args.output_report_dir,
                       corr_plot=args.corr_plot, corr_mat=args.corr_mat, corr_t=args.corr_t)

    elif args.subcommand == "thermal_hk":
        # Thermal Sensors Analysis Operation
        logging.info('The thermal analysis is beginning...')

        # If status is not specified, the analysis is done on both the states of the multiplexer
        if args.status == 2:
            for status in [0, 1]:
                strip_c.thermal_hk(path_file=args.path_file,
                                   start_datetime=args.start_datetime, end_datetime=args.end_datetime,
                                   status=status, fft=args.fourier, nperseg_thermal=args.nperseg_thermal,
                                   ts_sam_exp_med=args.ts_sam_exp_med, ts_sam_tolerance=args.ts_sam_tolerance,
                                   spike_ts=args.spike_ts, spike_fft=args.spike_fft,
                                   corr_t=args.corr_t, corr_plot=args.corr_plot, corr_mat=args.corr_mat,
                                   output_plot_dir=args.output_plot_dir, output_report_dir=args.output_report_dir)
        else:
            strip_c.thermal_hk(path_file=args.path_file,
                               start_datetime=args.start_datetime, end_datetime=args.end_datetime,
                               status=args.status, fft=args.fourier, nperseg_thermal=args.nperseg_thermal,
                               ts_sam_exp_med=args.ts_sam_exp_med, ts_sam_tolerance=args.ts_sam_tolerance,
                               spike_ts=args.spike_ts, spike_fft=args.spike_fft,
                               corr_t=args.corr_t, corr_plot=args.corr_plot, corr_mat=args.corr_mat,
                               output_plot_dir=args.output_plot_dir, output_report_dir=args.output_report_dir)

    ####################################################################################################################
    # REPORT Production
    ####################################################################################################################
    logging.info(f"\nI am putting the header report into: {args.output_report_dir}.")

    # Convert the Namespace object to a dictionary
    args_dict = vars(args)

    # Dictionary with the data used to create the report
    header_report_data = {
        "command_line": command_line,
        "path_file": args.path_file,
        "analysis_date": str(f"{args.start_datetime} - {args.end_datetime}"),
        "output_plot_dir": args.output_plot_dir,
        "output_report_dir": args.output_report_dir,
        "args_dict": args_dict
    }

    # root: location of the file.txt with the information to build the report
    root = "../striptease/templates/validation_templates"
    templates_dir = Path(root)

    # Creating the Jinja2 environment
    env = Environment(loader=FileSystemLoader(templates_dir))
    # Getting instructions to create the head of the report
    header_template = env.get_template('report_header.txt')

    # Report generation: header
    filename = Path(f"{args.output_report_dir}/report_{args.subcommand}_head.md")
    with open(filename, 'w') as outf:
        outf.write(header_template.render(header_report_data))


if __name__ == "__main__":
    main()
