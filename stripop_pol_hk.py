#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# This file contains the function "pol_hk" that operates an analysis of the Thermal Sensors (TS) of Strip.
# This function will be used during the system level test campaign of the LSPE-Strip instrument.
# August 18th 2023, Brescia (Italy)

# Libraries & Modules
import logging

from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from rich.logging import RichHandler

# MyLibraries & MyModules
import polarimeter as pol
import f_correlation_strip as fz_c

# Use the module logging to produce nice messages on the shell
logging.basicConfig(level="INFO", format='%(message)s',
                    datefmt="[%X]", handlers=[RichHandler()])


def pol_hk(path_file: str, start_datetime: str, end_datetime: str, name_pol: str,
           corr_plot: bool, corr_mat: bool, corr_t: float,
           hk_sam_exp_med: dict, hk_sam_tolerance: dict,
           output_plot_dir: str, output_report_dir: str):
    """
    Performs only the analysis of the Housekeeping parameters of the polarimeter(s) provided.
        Parameters:
            - **path_file** (``str``): location of the data file, it is indeed the location of the hdf5 file's index
            - **start_datetime** (``str``): start time
            - **end_datetime** (``str``): end time
            - **name_pol** (``str``): name of the polarimeter. If more than one, write them into ' ' separated by space.
            - **hk_sam_exp_med** (``dict``): contains the exp sampling delta between two consecutive timestamps of HK
            - **hk_sam_tolerance** (``dict``): contains the acceptance sampling tolerances of the hk parameters: I,V,O
            - **output_dir** (`str`): Path of the dir that will contain the reports with the results of the analysis.
            - **command_line** (`str`): Command line used to start the pipeline.
    """
    logging.info('\nLoading dir and templates information...')

    # Initializing the data-dict for the report
    report_data = {"output_plot_dir": output_plot_dir}

    # root: location of the file.txt with the information to build the report
    root = "../striptease/templates/validation_templates"
    templates_dir = Path(root)

    # Creating the Jinja2 environment
    env = Environment(loader=FileSystemLoader(templates_dir))
    # ------------------------------------------------------------------------------------------------------------------

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

        # Analyzing HK Sampling
        sampling_warn = p.HK_Sampling_Table(sam_exp_med=hk_sam_exp_med, sam_tolerance=hk_sam_tolerance)

        # Normalizing the HK measures
        logging.info('Normalizing HK.')
        p.Norm_HouseKeeping()

        # Analyzing HK and collecting the results
        logging.info('Analyzing HK.')
        hk_results = p.Analyse_HouseKeeping()

        # Preparing html table for the report
        logging.info('Producing HK table for the report.')
        hk_table = p.HK_table(results=hk_results)

        # Plots of the Bias HK: Tensions and Currents
        logging.info('Plotting Bias HK.')
        p.Plot_Housekeeping(hk_kind="V", show=False)
        p.Plot_Housekeeping(hk_kind="I", show=False)
        # Plots of the Offsets
        logging.info('Plotting Offset.')
        p.Plot_Housekeeping(hk_kind="O", show=False)

        # Correlation plots between all HK parameters
        if not corr_plot:
            pass
        else:
            logging.info("Starting correlation plot.")
            # Get all HK names
            all_names = p.hk_list["I"] + p.hk_list["V"] + p.hk_list["O"]
            # Plot correlation plots
            for idx, hk_name1 in enumerate(all_names):
                logging.info(hk_name1)
                for hk_name2 in all_names[idx + 1:]:
                    logging.info(hk_name2)
                    # Setting the names of the items: I, V, O
                    item1 = hk_name1[0] if hk_name1[0] != "D" else "O"
                    item2 = hk_name2[0] if hk_name2[0] != "D" else "O"
                    logging.info(item1)
                    p.warnings["corr_warning"].extend(
                        fz_c.correlation_plot(array1=list(p.hk[item1][hk_name1]),
                                              array2=list(p.hk[item2][hk_name2]),
                                              dict1={},
                                              dict2={},
                                              time1=list(p.hk_t[item1][hk_name1]),
                                              time2=list(p.hk_t[item2][hk_name2]),
                                              data_name1=f"{hk_name1}",
                                              data_name2=f"{hk_name2}",
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
        # REPORT HK
        # --------------------------------------------------------------------------------------------------------------
        logging.info(f"\nOnce ready, I will put the HK report into: {output_report_dir}.")

        # Updating the report_data dict
        report_data.update({"hk_table": hk_table})

        # Getting instructions to create the HK report
        template_hk = env.get_template('report_hk.txt')

        # Report HK generation
        filename = Path(f"{output_report_dir}/report_hk.md")
        with open(filename, 'w') as outf:
            outf.write(template_hk.render(report_data))

        # --------------------------------------------------------------------------------------------------------------
        # REPORT WARNINGS
        # --------------------------------------------------------------------------------------------------------------
        # Updating the report_data dict for the warning report
        report_data.update({"t_warn": p.warnings["time_warning"],
                            "sampling_warn": sampling_warn,
                            "corr_warn": p.warnings["corr_warning"],
                            })

        # Getting instructions to create the head of the report
        template_ts = env.get_template('report_warnings.txt')

        # Report generation
        filename = Path(f"{output_report_dir}/report_hk_warnings.md")
        with open(filename, 'w') as outf:
            outf.write(template_ts.render(report_data))

        return
