#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
import json
import logging
import sys
from pathlib import Path

from mako.template import Template

from rich.logging import RichHandler


# This file contains a program to produce a report of a polarimeter.
# December 20th 2022, Brescia (Italy)


def main():
    """
    Produce a report for a single polarimeter including plots of the analysis on Even-Odd Output, Scientific Data, FFT
    and correlation Matrices.
    Parameters:
        - **name_pol** (``str``): name of the polarimeter
        - **output_dir** (``str``): location of the report, once is done
        - **plot_dir** (``str``): location of the plots to be included in the report
        - **start_datetime** (``str``): start time
        - **end_datetime** (``str``): end time
    """
    logging.basicConfig(level="INFO", format='%(message)s',
                        datefmt="[%X]", handlers=[RichHandler()])  # <3

    if len(sys.argv) != 6:
        # report_producer.py is sys.argv[0]
        logging.error("Wrong number of parameters.\n"
                      "Usage: python report_producer.py name_pol output_dir plot_dir start_datetime end_datetime")
        sys.exit(1)

    name_pol = sys.argv[1]  # type: str
    output_dir = sys.argv[2]  # type: str
    plot_dir = sys.argv[3]  # type: str
    start_datetime = sys.argv[4]  # type: str
    end_datetime = sys.argv[5]  # type: str

    logging.info(f"The procedure started.\nGoing to produce the report")

    # Save the report using the variable "result"

    template = Template(filename="report_producer.txt")
    with open(output_dir + f'/report_{name_pol}.md', "wt") as outf:
        print(template.render(
            name_polarimeter=name_pol,
            analysis_date=str(f"{start_datetime} - {end_datetime}"),
            output_dir=output_dir,
            plot_dir=plot_dir,
            command_line=" ".join(sys.argv)
        ), file=outf)


if __name__ == "__main__":
    main()
