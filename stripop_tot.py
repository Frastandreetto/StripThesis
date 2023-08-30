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
import polarimeter as pol
import thermalsensors as ts

# Use the module logging to produce nice messages on the shell
logging.basicConfig(level="INFO", format='%(message)s',
                    datefmt="[%X]", handlers=[RichHandler()])


def tot(path_file: str, start_datetime: str, end_datetime: str, name_pol: str,
        thermal_sensors: bool, housekeeping: bool, scientific: bool,
        eoa: str, smooth: int, window: int,
        fft: bool, nperseg: int, nperseg_thermal: int,
        spike_data: bool, spike_fft: bool,
        corr_plot: bool, corr_mat: bool, corr_t: float,
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
            - **scientific** (``bool``): If true, the code computes the double demodulation analyze the scientific data.
            - **thermal_sensors** (``bool``): If true, the code analyzes the Thermal Sensors of Strip.
            - **housekeeping** (``bool``): If true, the code analyzes the Housekeeping parameters of the Polarimeters.
            - **fft** (``bool``): If true, the code computes the power spectra of the scientific data.
            - **nperseg** (``int``): number of elements of the array of scientific data on which the fft is calculated
            - **nperseg_thermal** (``int``): int value that defines the number of elements of thermal measures
            on which the fft is calculated.
            - **spike_data** (``bool``): If true, the code will look for spikes in Sci-data.
            - **spike_fft** (``bool``): If true, the code will look for spikes in FFT.
            - **corr_plot** (``bool``): If true, compute the correlation plot of the even-odd and scientific data.
            - **corr_mat** (``bool``): If true, compute the correlation matrices of the even-odd and scientific data.
            - **corr_t** (``float``): Floating point number used as lim sup for the correlation value
             between two dataset: if the value computed is higher than the threshold, a warning is produced.
            - **output_dir** (`str`): Path of the dir that will contain the reports with the results of the analysis
            - **command_line** (`str`): Command line used to start the pipeline.
    """
    logging.info('\nLoading dir and templates information...')

    # root: location of the file.txt with the information to build the report
    root = "../striptease/templates"
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
            # Normalizing TS measures: flag to specify the sampling frequency? Now 30s
            logging.info(f'Normalizing TS. Status {status}')
            # Saving a list of sampling problematic TS
            problematic_TS = TS.Norm_TS()

            # Analyzing TS and collecting the results
            logging.info(f'Analyzing TS. Status {status}')
            ts_results = TS.Analyse_TS()

            # Preparing html table for the report
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
            # Collecting all names
            all_names = [name for groups in TS.ts_names.values() for name in groups if name not in problematic_TS]
            # Printing the plots with no repetitions
            logging.info(f'Plotting Correlation plots of the TS with each other.')
            for i, n1 in enumerate(all_names):
                for n2 in all_names[i + 1:]:
                    fz.correlation_plot(array1=TS.ts["thermal_data"]["calibrated"][n1],
                                        array2=TS.ts["thermal_data"]["calibrated"][n2],
                                        dict1={},
                                        dict2={},
                                        time1=TS.ts["thermal_times"],
                                        time2=TS.ts["thermal_times"],
                                        data_name1=f"{status}_{n1}",
                                        data_name2=f"{n2}",
                                        start_datetime=start_datetime,
                                        corr_t=corr_t
                                        )
            # ----------------------------------------------------------------------------------------------------------
            # REPORT TS
            # ----------------------------------------------------------------------------------------------------------
            logging.info(f"\nOnce ready, I will put the TS report for the status {status} into: {output_report_dir}.")

            report_data = {
                "output_plot_dir": output_plot_dir,
                "th_tab": th_table,
                "status": status
                # Waiting for Warnings
                # "t_warnings": 0,
                # "corr_warnings": corr_warner,
            }

            # Getting instructions to create the TS report
            template_ts = env.get_template('report_thermals.txt')

            # Report TS generation
            filename = Path(f"{output_report_dir}/report_ts_status_{status}.md")
            with open(filename, 'w') as outf:
                outf.write(template_ts.render(report_data))

    # ------------------------------------------------------------------------------------------------------------------
    # Polarimeters Analysis
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

            # Add some correlations (?)

            # ----------------------------------------------------------------------------------------------------------
            # REPORT HK
            # ----------------------------------------------------------------------------------------------------------
            logging.info(f"\nOnce ready, I will put the HK report into: {output_report_dir}.")

            report_data = {
                "output_plot_dir": output_plot_dir,
                "hk_table": hk_table,
                # Waiting for Warnings
                # "t_warnings": 0,
                # "corr_warnings": corr_warner,
            }

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
                logging.info(f"\nOnce ready, I will put the EOA report into: {output_report_dir}.")

                report_data = {
                    "output_plot_dir": output_plot_dir,
                    "name_pol": np,
                    "fft": fft,
                    # Waiting for Warnings
                    # "t_warnings": 0,
                    # "corr_warnings": corr_warner,
                    # "spikes_warnings": spikes_warner
                }
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
                logging.info(f"\nOnce ready, I will put the SCIENTIFIC DATA report into: {output_report_dir}.")

                report_data = {
                    "output_plot_dir": output_plot_dir,
                    "name_pol": np,
                    "fft": fft,
                    # Waiting for Warnings
                    # "t_warnings": 0,
                    # "corr_warnings": corr_warner,
                    # "spikes_warnings": spikes_warner
                }
                # Getting instructions to create the SCIDATA report
                template_hk = env.get_template('report_sci.txt')

                # Report SCIDATA generation
                filename = Path(f"{output_report_dir}/report_sci.md")
                with open(filename, 'w') as outf:
                    outf.write(template_hk.render(report_data))

            # ----------------------------------------------------------------------------------------------------------
            # Correlation plots (?)
            # ----------------------------------------------------------------------------------------------------------
            if corr_plot:
                logging.warning(f'-------------------------------------------------------------------------------------'
                                f'\nCorrelation plots (?). Type {type}.')
                # Correlation Plot Example
                fz.correlation_plot(array1=[], array2=[], dict1=p.data["DEM"], dict2=p.data["DEM"],
                                    time1=p.times, time2=p.times,
                                    data_name1=f"{np}_DEM", data_name2=f"{np}_PWR",
                                    start_datetime=start_datetime, show=False, corr_t=0.4, plot_dir=output_plot_dir)
                # ------------------------------------------------------------------------------------------------------
                # REPORT CORRELATION PLOT
                # ------------------------------------------------------------------------------------------------------
                logging.info(f"\nOnce ready, I will put the CORR PLOT report into: {output_report_dir}.")

                # Get a list of PNG files in the directory
                # Excluding TS correlation plots
                excluded_prefixes = ['0', '1']
                png_files = [file for file in os.listdir(f"{output_plot_dir}/Correlation_Plot/")
                             if file.lower().endswith('.png')
                             and not any(file.startswith(prefix) for prefix in excluded_prefixes)]

                logging.info(png_files)

                report_data = {
                    "output_plot_dir": output_plot_dir,
                    "name_pol": np,
                    "png_files": png_files
                    # Waiting for Warnings
                    # "corr_warnings": corr_warner
                }
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
                fz.correlation_mat(dict1=p.data["DEM"], dict2=p.data["DEM"], data_name=f"{np}_DEM_PWR",
                                   start_datetime=start_datetime, show=False, corr_t=0.4,
                                   plot_dir=output_plot_dir)
                # ------------------------------------------------------------------------------------------------------
                # REPORT CORRELATION MATRIX
                # ------------------------------------------------------------------------------------------------------
                logging.info(f"\nOnce ready, I will put the CORR MATRIX report into: {output_report_dir}.")

                # Get a list of PNG files in the directory
                png_files = [file for file in os.listdir(f"{output_plot_dir}/Correlation_Matrix/")
                             if file.lower().endswith('.png')]

                logging.info(png_files)

                report_data = {
                    "output_plot_dir": output_plot_dir,
                    "name_pol": np,
                    "png_files": png_files
                    # Waiting for Warnings
                    # "corr_warnings": corr_warner
                }
                # Getting instructions to create the CORR MAT report
                template_hk = env.get_template('report_corr_mat.txt')

                # Report CORR MAT generation
                filename = Path(f"{output_report_dir}/report_corr_mat.md")
                with open(filename, 'w') as outf:
                    outf.write(template_hk.render(report_data))

    return
