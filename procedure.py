#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# This file contains a first version (0.0.1) of the new LSPE-STRIP pipeline to produce a complete scan of a polarimeter.
# December 7th 2022, Brescia (Italy)

import logging
import sys
from rich.logging import RichHandler

import polarimeter as pol


def main():
    """
    Produce a scan of a single polarimeter performing a complete analysis on Even-Odd Output, Scientific Data, FFT and
    correlation Matrices.
    Parameters:
        - **name_pol** (``str``): name of the polarimeter
        - **path_file** (``str``): location of the data file, it is indeed the location of the hdf5 file's index
        - **start_datetime** (``str``): start time
        - **end_datetime** (``str``): end time
    """
    logging.basicConfig(level="INFO", format='%(message)s',
                        datefmt="[%X]", handlers=[RichHandler()])  # <3

    if len(sys.argv) != 5:
        # Note: When I run the code with "python procedure.py" I already give an argument:
        # procedure.py is sys.argv[0]
        logging.error("Wrong number of parameters.\n"
                      "Usage: python procedure.py name_pol path_file start_datetime end_datetime")
        sys.exit(1)

    name_pol = sys.argv[1]
    path_file = sys.argv[2]
    start_datetime = sys.argv[3]
    end_datetime = sys.argv[4]
    p = pol.Polarimeter(name_pol=name_pol, path_file=path_file, start_datetime=start_datetime,
                        end_datetime=end_datetime)
    p.Load_Pol()
    p.Prepare(1)

    logging.info("The procedure started.\n")
    for type in p.data.keys():
        logging.info(f"Going to Plot {type} Output.")
        p.Plot_Output(type=f"{type}", begin=0, end=-1, show=False)
        logging.info(f"Done.\n")

    logging.info("\nEven-Odd Analysis started.\n")
    i = 1
    for type in p.data.keys():
        logging.info(f"Going to Plot Even Odd {type} Output and RMS.")
        for smooth in [1, 100, 200, 400, 500, 1000]:
            p.Plot_EvenOddAll(type=type, even=1, odd=1, all=0, begin=0, end=-1, smooth_len=smooth, show=False)
            p.Plot_EvenOddAll(type=type, even=1, odd=1, all=1, begin=0, end=-1, smooth_len=smooth, show=False)
            p.Plot_EvenOddAll(type=type, even=0, odd=0, all=1, begin=0, end=-1, smooth_len=smooth, show=False)
            logging.info(f"{type}: {3 * i}/36) Output plot done.")

            p.Plot_RMS_EOA(type=type, window=100, even=1, odd=1, all=0, begin=0, end=-1, smooth_len=smooth,
                           show=False)
            p.Plot_RMS_EOA(type=type, window=100, even=1, odd=1, all=1, begin=0, end=-1, smooth_len=smooth,
                           show=False)
            p.Plot_RMS_EOA(type=type, window=100, even=0, odd=0, all=1, begin=0, end=-1, smooth_len=smooth,
                           show=False)
            logging.info(f"{type}: {3 * i}/36) RMS plot done.")
            i += 1
        logging.info(f"{type}: Done.\n")

    for type in p.data.keys():
        logging.info(f"Going to Plot {type} Even-Odd Correlation.")
        p.Plot_Correlation_EvenOdd(type, begin=0, end=-1, show=False)
        logging.info(f"Done.\n")

    i = 1
    for type in p.data.keys():
        logging.info(f"Going to Plot {type} Even-Odd FFT.")
        p.Plot_FFT_EvenOdd(type=type, even=1, odd=1, all=0, begin=0, end=-1, show=False)
        p.Plot_FFT_EvenOdd(type=type, even=1, odd=1, all=1, begin=0, end=-1, show=False)
        p.Plot_FFT_EvenOdd(type=type, even=0, odd=0, all=1, begin=0, end=-1, show=False)
        logging.info(f"FFT {type}: Done.")

        logging.info(f"Going to Plot {type} Even-Odd FFT of the RMS.")
        p.Plot_FFT_RMS_EO(type=type, window=100, even=1, odd=1, all=0, begin=0, end=-1, show=False)
        p.Plot_FFT_RMS_EO(type=type, window=100, even=1, odd=1, all=1, begin=0, end=-1, show=False)
        p.Plot_FFT_RMS_EO(type=type, window=100, even=0, odd=0, all=1, begin=0, end=-1, show=False)
        logging.info(f"FFT RMS {type}: Done.")
    i += 1

    logging.info("\nEven Odd Analysis is now completed.\nScientific Data Analysis started.")

    i = 1
    for type in p.data.keys():
        logging.info(f"Going to Plot Scientific Data {type} and RMS.")
        for smooth in [1, 100, 200, 400, 500, 1000]:
            p.Plot_SciData(type=type, smooth_len=smooth, show=False)
            logging.info(f"{type}: {i}/12) Data plot done.")

            p.Plot_RMS_SciData(type=type, window=100, begin=0, end=-1, smooth_len=smooth, show=False)
            logging.info(f"{type}: {i}/12) RMS plot done.")

            i += 1
    i = 1
    for type in p.data.keys():
        logging.info(f"Going to Plot Scientific Data {type} FFT and FFT of  RMS.")
        p.Plot_FFT_SciData(type=type, begin=0, end=-1, show=False)
        logging.info(f"{type}: {i}/2) Data FFT plot done.")

        p.Plot_FFT_RMS_SciData(type=type, window=100, begin=0, end=-1, show=False)
        logging.info(f"{type}: {i}/2) RMS FFT plot done.")

    logging.info("Scientific Data Analysis is now completed. Correlation Matrices will be now produced.\n")

    for type in p.data.keys():
        for s in [True, False]:
            logging.info(f"Going to plot {type} Correlation Matrix with scientific parameter = {s}")
            p.Plot_Correlation_Mat(type=type, scientific=s, show=False)

    logging.info("Analysis completed.\n")


if __name__ == "__main__":
    main()
