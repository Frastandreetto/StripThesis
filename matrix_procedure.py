#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# This file contains a first version (0.0.1) of the matrix_procedure to produce all Correlation Matrix Plots.
# December 26th 2022, Brescia (Italy)

# Libraries & Modules
import logging
import sys

from rich.logging import RichHandler

# MyLibraries & MyModules
import polarimeter as pol


def main():
    """
    Produce a scan of a list of polarimeter Calculating all Correlation Matrices.
    Parameters:
        - **path_file** (``str``): location of the data file, it is indeed the location of the hdf5 file's index
        - **start_datetime** (``str``): start time
        - **end_datetime** (``str``): end time
        - **name_pol** (``str``): name of the polarimeter
    """
    logging.basicConfig(level="INFO", format='%(message)s',
                        datefmt="[%X]", handlers=[RichHandler()])  # <3

    if len(sys.argv) < 5:
        # Note: When I run the code with "python procedure.py" I already give an argument:
        # procedure.py is sys.argv[0]
        logging.error("Wrong number of parameters.\n"
                      "Usage: python matrix_procedure.py path_datafile start_datetime end_datetime name_pol")
        sys.exit(1)

    path_file = sys.argv[1]  # type: str
    start_datetime = sys.argv[2]  # type: str
    end_datetime = sys.argv[3]  # type: str
    pol_list = list(sys.argv[4:])

    logging.info(f"The procedure started.\n{len(pol_list)}")
    for name_pol in pol_list:
        logging.warning(name_pol)
        p = pol.Polarimeter(name_pol=name_pol, path_file=path_file, start_datetime=start_datetime,
                            end_datetime=end_datetime)
        p.Load_Pol()
        p.Prepare(1)

        logging.info(f"Analyzing polarimeter {name_pol}.\n")

        for type in p.data.keys():
            for s in [True, False]:
                logging.info(f"Going to plot {type} Correlation Matrix with scientific parameter = {s}")
                p.Plot_Correlation_Mat(type=type, scientific=s, show=False)
                p.Plot_Correlation_Mat_RMS(type=type, scientific=s, show=False)

        logging.info("Analysis completed.\n")


if __name__ == "__main__":
    main()
