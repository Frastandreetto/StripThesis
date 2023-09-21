#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# This file contains the function "tot" that operates a complete analysis of a provided group of polarimeters.
# This function will be used during the system level test campaign of the LSPE-Strip instrument.
# August 14th 2023, Brescia (Italy)

# Libraries & Modules
import logging
import os

from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from rich.logging import RichHandler

# MyLibraries & MyModules
import f_strip as fz
import f_correlation_strip as fz_c
import polarimeter as pol
import thermalsensors as ts

# Use the module logging to produce nice messages on the shell
logging.basicConfig(level="INFO", format='%(message)s',
                    datefmt="[%X]", handlers=[RichHandler()])


def tot(path_file: str, start_datetime: str, end_datetime: str, name_pol: str,
        thermal_sensors: bool, housekeeping: bool, scientific: bool,
        eoa: str, rms: bool, smooth: int, window: int,
        fft: bool, nperseg: int, nperseg_thermal: int,
        spike_data: bool, spike_fft: bool,
        sam_tolerance: float,
        hk_sam_exp_med: float, hk_sam_tolerance: float,
        ts_sam_exp_med: float, ts_sam_tolerance: float,
        corr_plot: bool, corr_mat: bool, corr_t: float, cross_corr: bool,
        output_plot_dir: str, output_report_dir: str):
    """
    Performs the analysis of one or more polarimeters producing a complete report.
    The analysis can include plots of: Even-Odd Output, Scientific Data, FFT, correlation and  Matrices.
    The reports produced include also info about the state of the housekeeping parameters and the thermal sensors.

        Parameters:
            - **path_file** (``str``): location of the data file, it is indeed the location of the hdf5 file's index
            - **start_datetime** (``str``): start time
            - **end_datetime** (``str``): end time
            - **name_pol** (``str``): name of the polarimeter. If more than one, write them into ' ' separated by space.
            - **eoa** (``str``): states which data analyze. Even samples (e), odd samples (o), all samples (a).
            - **smooth** (``int``): Smoothing length used to flatter the data. smooth=1 equals no smooth.
            - **window** (``int``): Integer number used to convert the array of the data into a matrix
            with a number "window" of elements per row and then calculate the RMS on every row.
            window=1 equals no conversion.
            - **scientific** (``bool``): If true, compute the double demodulation and analyze the scientific data.
            - **rms** (``bool``): If true, compute the rms on the scientific output and data.
            - **thermal_sensors** (``bool``): If true, the code analyzes the Thermal Sensors of Strip.
            - **housekeeping** (``bool``): If true, the code analyzes the Housekeeping parameters of the Polarimeters.
            - **fft** (``bool``): If true, the code computes the power spectra of the scientific data.
            - **nperseg** (``int``): number of elements of the array of scientific data on which the fft is calculated
            - **nperseg_thermal** (``int``): int value that defines the number of elements of thermal measures
            on which the fft is calculated.
            - **spike_data** (``bool``): If true, the code will look for spikes in Sci-data.
            - **spike_fft** (``bool``): If true, the code will look for spikes in FFT.
            - **sam_tolerance** (``float``): the acceptance sampling tolerances of the Scientific Output.
            - **ts_sam_exp_med** (``float``): the exp sampling delta between two consecutive timestamps of TS.
            - **ts_sam_tolerance** (``float``): the acceptance sampling tolerances of the TS.
            - **hk_sam_exp_med** (``dict``): contains the exp sampling delta between two consecutive timestamps of hk.
            - **hk_sam_tolerance** (``dict``): contains the acceptance sampling tolerances of the hk parameters: I,V,O.
            - **corr_plot** (``bool``): If true, compute the correlation plot of the even-odd and scientific data.
            - **corr_mat** (``bool``): If true, compute the correlation matrices of the even-odd and scientific data.
            - **corr_t** (``float``): Floating point number used as lim sup for the correlation value
            between two dataset: if the value computed is higher than the threshold, a warning is produced.
            - **cross_corr** (``bool``): If true, compute the 55x55 correlation matrices between all the polarimeters.
            - **output_dir** (`str`): Path of the dir that will contain the reports with the results of the analysis.
            - **command_line** (`str`): Command line used to start the pipeline.
    """
    logging.info('\nLoading dir and templates information...')

    # Initializing the data-dict for the report
    report_data = {"output_plot_dir": output_plot_dir}

    # Initializing warning lists
    t_warn = []
    sampling_warn = []
    corr_warn = []
    spike_warn = []

    # root: location of the file.txt with the information to build the report
    root = "../striptease/templates/validation_templates"
    templates_dir = Path(root)

    # Creating the Jinja2 environment
    env = Environment(loader=FileSystemLoader(templates_dir))

    logging.info('\nReady to analyze Strip.')

    # ------------------------------------------------------------------------------------------------------------------
    # Thermal Sensors Analysis
    # ------------------------------------------------------------------------------------------------------------------
    if not thermal_sensors:
        pass
    else:
        logging.info('\nReady to analyze the Thermal Sensors.')
        for status in [0, 1]:
            TS = ts.Thermal_Sensors(path_file=path_file, start_datetime=start_datetime, end_datetime=end_datetime,
                                    status=status, nperseg_thermal=nperseg_thermal)

            # Loading the TS
            logging.info(f'Loading TS. Status {status}')
            TS.Load_TS()

            # TS Sampling warnings -------------------------------------------------------------------------------------
            sampling_warn.extend(TS.TS_Sampling_Table(sam_exp_med=ts_sam_exp_med, sam_tolerance=ts_sam_tolerance))
            # ----------------------------------------------------------------------------------------------------------

            # Normalizing TS measures
            logging.info(f'Normalizing TS. Status {status}')
            # Saving a list of sampling problematic TS
            problematic_TS = TS.Norm_TS()

            # TS Time warnings ----------------------------------------------------------------------------------------
            t_warn.extend(TS.warnings["time_warning"])
            # ----------------------------------------------------------------------------------------------------------

            # Analyzing TS and collecting the results
            logging.info(f'Analyzing TS. Status {status}')
            ts_results = TS.Analyse_TS()

            # Preparing md table for the report
            logging.info(f'Producing TS table for the report. Status {status}')
            th_table = TS.Thermal_table(results=ts_results)

            # Plots of all TS
            logging.info(f'Plotting all TS measures for status {status} of the multiplexer.')
            TS.Plot_TS()

            # Fourier's analysis if asked
            if fft:
                logging.info(f'Plotting the FFT of all the TS measures for status {status} of the multiplexer.')
                TS.Plot_FFT_TS()

            # TS Correlation plots
            # Add Correlation Plots TS0 - TS1?
            if not corr_plot:
                pass
            else:
                if status == 0:
                    pass
                # Compute correlation plots only one time
                else:
                    # Collecting all names, excluding the problematic TS
                    all_names = [name for groups in TS.ts_names.values()
                                 for name in groups if name not in problematic_TS]
                    # Printing the plots with no repetitions
                    logging.info(f'Plotting Correlation plots of the TS with each other.')
                    for i, n1 in enumerate(all_names):
                        for n2 in all_names[i + 1:]:
                            # Correlation warnings ---------------------------------------------------------------------
                            corr_warn.extend(fz_c.correlation_plot(array1=TS.ts["thermal_data"]["calibrated"][n1],
                                                                   array2=TS.ts["thermal_data"]["calibrated"][n2],
                                                                   dict1={},
                                                                   dict2={},
                                                                   time1=TS.ts["thermal_times"],
                                                                   time2=TS.ts["thermal_times"],
                                                                   data_name1=f"{status}_{n1}",
                                                                   data_name2=f"{n2}",
                                                                   start_datetime=start_datetime,
                                                                   corr_t=corr_t,
                                                                   plot_dir=output_plot_dir
                                                                   ))
            # TS Correlation Matrices:
            # TS status 0 - Self Correlations
            # TS status 1 - Self Correlations
            # TS status 0 - TS status 1
            if not corr_mat:
                pass
            else:
                if status == 0:
                    pass
                # Compute correlation matrix only one time
                else:
                    # Define two Thermal Sensors, one per status
                    ts_0 = ts.Thermal_Sensors(path_file=path_file, start_datetime=start_datetime,
                                              end_datetime=end_datetime,
                                              status=0, nperseg_thermal=nperseg_thermal)
                    ts_1 = ts.Thermal_Sensors(path_file=path_file, start_datetime=start_datetime,
                                              end_datetime=end_datetime,
                                              status=1, nperseg_thermal=nperseg_thermal)
                    # Loading the two Thermal Sensors
                    ts_0.Load_TS()
                    ts_1.Load_TS()
                    # Assign the dict
                    ts0_d = ts_0.ts["thermal_data"]["calibrated"]
                    ts1_d = ts_1.ts["thermal_data"]["calibrated"]
                    # Plotting the 3 correlation Matrices
                    for d1, d2, name1, name2 in [(ts0_d, {}, "TS0", "SelfCorr"), (ts1_d, {}, "TS1", "SelfCorr"),
                                                 (ts0_d, ts1_d, "TS0", "TS1")]:
                        logging.info(f"Plotting correlation matrices {name1} - {name2}.\n")
                        # Correlation warnings -------------------------------------------------------------------------
                        TS.warnings["corr_warning"].extend(
                            fz_c.correlation_mat(dict1=d1, dict2=d2, data_name1=name1, data_name2=name2,
                                                 start_datetime=start_datetime,
                                                 show=False, plot_dir=output_plot_dir))

            # ----------------------------------------------------------------------------------------------------------
            # REPORT TS
            # ----------------------------------------------------------------------------------------------------------
            logging.info(f"\nOnce ready, I will put the TS report for the status {status} into: {output_report_dir}.")

            # Updating the report_data dict
            report_data.update({'th_tab': th_table, 'status': status})

            # Getting instructions to create the TS report
            template_ts = env.get_template('report_thermals.txt')

            # Report TS generation
            filename = Path(f"{output_report_dir}/report_ts_status_{status}.md")
            with open(filename, 'w') as outf:
                outf.write(template_ts.render(report_data))
    # ------------------------------------------------------------------------------------------------------------------
    # Multi Polarimeter Analysis
    # ------------------------------------------------------------------------------------------------------------------
    if cross_corr:
        logging.warning(
            f'-------------------------------------------------------------------------------------'
            f'\nCross Correlation Matrices between all the polarimeters.')
        corr_warn.extend(fz_c.cross_corr_mat(path_file=path_file,
                                             start_datetime=start_datetime, end_datetime=end_datetime,
                                             show=False, corr_t=corr_t))

    # ------------------------------------------------------------------------------------------------------------------
    # Single Polarimeter Analysis
    # ------------------------------------------------------------------------------------------------------------------

    logging.info("\nReady to analyze the Polarimeters now.")
    # Converting the string of polarimeters into a list
    name_pol = name_pol.split()
    # Repeating the analysis for all the polarimeters in the list
    for np in name_pol:
        logging.warning(f'--------------------------------------------------------------------------------------'
                        f'\nParsing {np}')
        # Initializing a Polarimeter
        p = pol.Polarimeter(name_pol=np, path_file=path_file,
                            start_datetime=start_datetime, end_datetime=end_datetime)

        # --------------------------------------------------------------------------------------------------------------
        # Housekeeping Analysis
        # --------------------------------------------------------------------------------------------------------------
        if not housekeeping:
            pass
        else:
            logging.warning('--------------------------------------------------------------------------------------'
                            '\nHousekeeping Analysis.\nLoading HK.')
            # Loading the HK
            p.Load_HouseKeeping()

            # HK Sampling warnings -------------------------------------------------------------------------------------
            sampling_warn.extend(p.HK_Sampling_Table(sam_exp_med=hk_sam_exp_med, sam_tolerance=hk_sam_tolerance))
            # ----------------------------------------------------------------------------------------------------------

            # Normalizing the HK measures
            logging.info('Normalizing HK.')
            p.Norm_HouseKeeping()

            # HK Time warnings -----------------------------------------------------------------------------------------
            t_warn.extend(p.warnings["time_warning"])
            # ----------------------------------------------------------------------------------------------------------

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
                # Correlation plots
                # Between all HK parameters I, V, O
                for idx, hk_name1 in enumerate(all_names):
                    logging.info(hk_name1)
                    for hk_name2 in all_names[idx + 1:]:
                        logging.info(hk_name2)
                        # Setting the names of the items: I, V, O
                        item1 = hk_name1[0] if hk_name1[0] != "D" else "O"
                        item2 = hk_name2[0] if hk_name2[0] != "D" else "O"
                        logging.info(item1)
                        # Correlation Warnings -------------------------------------------------------------------------
                        corr_warn.extend(
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
            if not corr_mat:
                pass
            else:
                logging.info("I'll plot correlation matrices.\n")
                if not thermal_sensors:
                    pass
                # Plotting the 4 correlation Matrices:
                # TS status 0 - V HK Parameter
                # TS status 0 - I HK Parameter
                # TS status 1 - V HK Parameter
                # TS status 1 - I HK Parameter
                else:
                    # Thermal Sensor status
                    for status, name1 in [(0, "TS0"), (1, "TS1")]:
                        # HK parameters
                        for item, name2 in [("I", "I_HK"), ("V", "V_HK")]:
                            logging.info(f"Plotting correlation matrices {name1} - {name2}.\n")
                            # Initializing a TS
                            TS = ts.Thermal_Sensors(path_file=path_file, start_datetime=start_datetime,
                                                    end_datetime=end_datetime,
                                                    status=status, nperseg_thermal=nperseg_thermal)
                            # Loading thermal measures
                            TS.Load_TS()
                            # Preparing dictionaries
                            d1 = TS.ts["thermal_data"]["calibrated"]
                            d2 = p.hk[item]
                            # Correlation warnings -------------------------------------------------------------
                            corr_warn.extend(
                                fz_c.correlation_mat(dict1=d1, dict2=d2, data_name1=name1, data_name2=name2,
                                                     start_datetime=start_datetime,
                                                     show=False, plot_dir=output_plot_dir))
            # ----------------------------------------------------------------------------------------------------------
            # REPORT HK
            # ----------------------------------------------------------------------------------------------------------
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
        # Scientific Output Analysis
        # --------------------------------------------------------------------------------------------------------------
        # Loading the Scientific Outputs
        logging.warning('--------------------------------------------------------------------------------------'
                        '\nScientific Analysis. \nLoading Scientific Outputs.')
        p.Load_Pol()

        # Holes: Analyzing Scientific Output Sampling ------------------------------------------------------------------
        _ = p.Write_Jump(sam_tolerance=sam_tolerance)
        sampling_warn.extend(p.warnings["sampling_warning"])
        # --------------------------------------------------------------------------------------------------------------

        # Looking for spikes in the dataset ----------------------------------------------------------------------------
        if spike_data:
            logging.info('Looking for spikes in the dataset.')
            spike_warn.extend(p.Spike_Report(fft=False, nperseg=0))

        if spike_fft:
            logging.info('Looking for spikes in the FFT of the dataset.')
            spike_warn.extend(p.Spike_Report(fft=True, nperseg=10 ** 6))
        # --------------------------------------------------------------------------------------------------------------

        # Preparing the Polarimeter for the analysis: normalization and data cleanse
        logging.info('Preparing the Polarimeter.')
        # Dataset in function of time [s]
        p.Prepare(norm_mode=1)

        for type in ['DEM', 'PWR']:
            # Plot the Scientific Output
            logging.info(f'Plotting {type} Outputs.')
            p.Plot_Output(type=type, begin=0, end=-1, show=False)

            # ----------------------------------------------------------------------------------------------------------
            # Even Odd All Analysis
            # ----------------------------------------------------------------------------------------------------------
            if eoa == ' ':
                pass
            else:
                logging.warning(f'-------------------------------------------------------------------------------------'
                                f'\nEven-Odd-All Analysis. Data type: {type}.')
                combos = fz.eoa_values(eoa)
                for combo in combos:
                    # If even, odd, all are equal to 0
                    if all(value == 0 for value in combo):
                        # Do nothing
                        pass
                    else:
                        # Plotting Even Odd All Outputs
                        logging.info(f'\nEven = {combo[0]}, Odd = {combo[1]}, All = {combo[2]}.')
                        logging.info(f'Plotting Even Odd All Outputs. Type {type}.')
                        fz.data_plot(pol_name=np, dataset=p.data, timestamps=p.times,
                                     start_datetime=start_datetime, end_datetime=end_datetime, begin=0, end=-1,
                                     type=type, even=combo[0], odd=combo[1], all=combo[2],
                                     demodulated=False, rms=False, fft=False,
                                     window=window, smooth_len=smooth, nperseg=nperseg,
                                     show=False)
                        if rms:
                            # Plotting Even Odd All Outputs RMS
                            logging.info(f'Plotting Even Odd All Outputs RMS. Type {type}.')
                            fz.data_plot(pol_name=np, dataset=p.data, timestamps=p.times,
                                         start_datetime=start_datetime, end_datetime=end_datetime, begin=0, end=-1,
                                         type=type, even=combo[0], odd=combo[1], all=combo[2],
                                         demodulated=False, rms=True, fft=False,
                                         window=window, smooth_len=smooth, nperseg=nperseg,
                                         show=False)

                        if fft:
                            logging.warning("--------------------------------------------------------------------------"
                                            "\nSpectral Analysis Even-Odd-All")
                            # Plotting Even Odd All FFT
                            logging.info(f'Plotting Even Odd All FFT. Type {type}.')
                            fz.data_plot(pol_name=np, dataset=p.data, timestamps=p.times,
                                         start_datetime=start_datetime, end_datetime=end_datetime, begin=0, end=-1,
                                         type=type, even=combo[0], odd=combo[1], all=combo[2],
                                         demodulated=False, rms=False, fft=True,
                                         window=window, smooth_len=smooth, nperseg=nperseg,
                                         show=False)
                            if rms:
                                # Plotting Even Odd All FFT of the RMS
                                logging.info(f'Plotting Even Odd All FFT of the RMS. Type {type}.')
                                fz.data_plot(pol_name=np, dataset=p.data, timestamps=p.times,
                                             start_datetime=start_datetime, end_datetime=end_datetime, begin=0, end=-1,
                                             type=type, even=combo[0], odd=combo[1], all=combo[2],
                                             demodulated=False, rms=True, fft=True,
                                             window=window, smooth_len=smooth, nperseg=nperseg,
                                             show=False)
                # ------------------------------------------------------------------------------------------------------
                # REPORT EOA OUTPUT
                # ------------------------------------------------------------------------------------------------------
                # Produce the report only the second time: when all the plots are ready
                if type == "PWR":
                    logging.info(f"\nOnce ready, I will put the EOA report into: {output_report_dir}.")

                    eoa_letters = fz.letter_combo(str(eoa))

                    # Updating the report_data dict
                    report_data.update({"name_pol": np, "fft": fft, "rms": rms, "eoa_letters": eoa_letters})

                    # Getting instructions to create the HK report
                    template_hk = env.get_template('report_eoa.txt')

                    # Report HK generation
                    filename = Path(f"{output_report_dir}/report_eoa.md")
                    with open(filename, 'w') as outf:
                        outf.write(template_hk.render(report_data))

            # ----------------------------------------------------------------------------------------------------------
            # Scientific Data Analysis
            # ----------------------------------------------------------------------------------------------------------
            if not scientific:
                pass
            else:
                # Plot of Scientific Data
                logging.warning("--------------------------------------------------------------------------------------"
                                "\nScientific Data Analysis.")
                logging.info(f'\nPlot of Scientific Data. Type {type}.')
                fz.data_plot(pol_name=np, dataset=p.data, timestamps=p.times,
                             start_datetime=start_datetime, end_datetime=end_datetime, begin=0, end=-1,
                             type=type, even=1, odd=1, all=1,
                             demodulated=True, rms=False, fft=False,
                             window=window, smooth_len=smooth, nperseg=nperseg,
                             show=False)
                if rms:
                    # Plot of RMS of Scientific Data
                    logging.info(f'Plot of RMS of Scientific Data. Type {type}.')
                    fz.data_plot(pol_name=np, dataset=p.data, timestamps=p.times,
                                 start_datetime=start_datetime, end_datetime=end_datetime, begin=0, end=-1,
                                 type=type, even=1, odd=1, all=1,
                                 demodulated=True, rms=True, fft=False,
                                 window=window, smooth_len=smooth, nperseg=nperseg,
                                 show=False)

                # Plot of FFT of Scientific Data
                if fft:
                    logging.warning("----------------------------------------------------------------------------------"
                                    "\nSpectral Analysis Scientific Data.")
                    logging.info(f'Plot of FFT of Scientific Data. Type {type}.')
                    fz.data_plot(pol_name=np, dataset=p.data, timestamps=p.times,
                                 start_datetime=start_datetime, end_datetime=end_datetime, begin=0, end=-1,
                                 type=type, even=1, odd=1, all=1,
                                 demodulated=True, rms=False, fft=True,
                                 window=window, smooth_len=smooth, nperseg=nperseg,
                                 show=False)
                    if rms:
                        # Plot of FFT of the RMS of Scientific Data
                        logging.info(f'Plot of FFT of the RMS of Scientific Data. Type {type}.')
                        fz.data_plot(pol_name=np, dataset=p.data, timestamps=p.times,
                                     start_datetime=start_datetime, end_datetime=end_datetime, begin=0, end=-1,
                                     type=type, even=1, odd=1, all=1,
                                     demodulated=True, rms=True, fft=True,
                                     window=window, smooth_len=smooth, nperseg=nperseg,
                                     show=False)

                # ------------------------------------------------------------------------------------------------------
                # REPORT SCIENTIFIC DATA
                # ------------------------------------------------------------------------------------------------------
                # Produce the report only the second time: when all the plots are ready
                if type == "PWR":
                    logging.info(f"\nOnce ready, I will put the SCIENTIFIC DATA report into: {output_report_dir}.")

                    report_data.update({"name_pol": np, "fft": fft})

                    # Getting instructions to create the SCIDATA report
                    template_hk = env.get_template('report_sci.txt')

                    # Report SCIDATA generation
                    filename = Path(f"{output_report_dir}/report_sci.md")
                    with open(filename, 'w') as outf:
                        outf.write(template_hk.render(report_data))

            # ----------------------------------------------------------------------------------------------------------
            # Correlation plots
            # ----------------------------------------------------------------------------------------------------------
            if corr_plot:
                logging.warning(f'-------------------------------------------------------------------------------------'
                                f'\nCorrelation plots. Type {type}.')
                # Correlation Plot Example
                corr_warn.extend(fz_c.correlation_plot(array1=[], array2=[], dict1=p.data["DEM"], dict2=p.data["DEM"],
                                                       time1=p.times, time2=p.times,
                                                       data_name1=f"{np}_DEM", data_name2=f"{np}_PWR",
                                                       start_datetime=start_datetime, show=False, corr_t=0.4,
                                                       plot_dir=output_plot_dir))
                # ------------------------------------------------------------------------------------------------------
                # REPORT CORRELATION PLOT
                # ------------------------------------------------------------------------------------------------------
                # Produce the report only the second time: when all the plots are ready
                if type == "PWR":
                    logging.info(f"\nOnce ready, I will put the CORR PLOT report into: {output_report_dir}.")

                    # Get a list of PNG files in the directory
                    # Excluding TS correlation plots
                    excluded_prefixes = ['0', '1']
                    png_files = [file for file in os.listdir(f"{output_plot_dir}/Correlation_Plot/")
                                 if file.lower().endswith('.png')
                                 and not any(file.startswith(prefix) for prefix in excluded_prefixes)]

                    report_data.update({"name_pol": np, "png_files": png_files})

                    # Getting instructions to create the CORR MAT report
                    template_hk = env.get_template('report_corr_plot.txt')

                    # Report CORR MAT generation
                    filename = Path(f"{output_report_dir}/report_corr_plot.md")
                    with open(filename, 'w') as outf:
                        outf.write(template_hk.render(report_data))

            # ----------------------------------------------------------------------------------------------------------
            # Correlation matrices (?)
            # ----------------------------------------------------------------------------------------------------------
            if corr_mat:
                logging.warning(f'---------------------------------------------------------------------------------'
                                f'\nCorrelation matrices with threshold {corr_t}(?). Type {type}.')
                # Correlation Mat Example
                # Note: if there are the same data in the corr plot there will be repetitions in the warnings report
                corr_warn.extend(fz_c.correlation_mat(dict1=p.data["DEM"], dict2=p.data["DEM"],
                                                      data_name1=f"{np}_DEM", data_name2=f"{np}_PWR",
                                                      start_datetime=start_datetime, show=False, corr_t=0.4,
                                                      plot_dir=output_plot_dir))
                # ------------------------------------------------------------------------------------------------------
                # REPORT CORRELATION MATRIX
                # ------------------------------------------------------------------------------------------------------
                # Produce the report only the second time: when all the plots are ready
                if type == "PWR":
                    logging.info(f"\nOnce ready, I will put the CORR MATRIX report into: {output_report_dir}.")

                    # Get a list of PNG files in the directory
                    png_files = [file for file in os.listdir(f"{output_plot_dir}/Correlation_Matrix/")
                                 if file.lower().endswith('.png')]

                    report_data.update({"name_pol": np, "png_files": png_files})

                    # Getting instructions to create the CORR MAT report
                    template_hk = env.get_template('report_corr_mat.txt')

                    # Report CORR MAT generation
                    filename = Path(f"{output_report_dir}/report_corr_mat.md")
                    with open(filename, 'w') as outf:
                        outf.write(template_hk.render(report_data))

        # ------------------------------------------------------------------------------------------------------
        # REPORT WARNINGS
        # ------------------------------------------------------------------------------------------------------
        # Updating the report_data dict for the warning report
        report_data.update({"t_warn": t_warn,
                            "sampling_warn": sampling_warn,
                            "corr_warn": corr_warn,
                            "spike_warn": spike_warn
                            })

        # Getting instructions to create the head of the report
        template_ts = env.get_template('report_warnings.txt')

        # Report generation
        filename = Path(f"{output_report_dir}/report_tot_warnings.md")
        with open(filename, 'w') as outf:
            outf.write(template_ts.render(report_data))
    return
