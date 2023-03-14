#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# This file contains the 13th version (0.0.13) of the new LSPE-Strip pipeline.
# It produces a complete scan of a polarimeter.
# December 7th 2022, Brescia (Italy)

# Libraries & Modules
import csv
import logging
import sys
import numpy as np

from pathlib import Path

from astropy.time import Time
from jinja2 import Environment, FileSystemLoader
from rich.logging import RichHandler

# MyLibraries & MyModules
import polarimeter as pol
import f_strip as fz


def main():
    """
    Produce a scan of a polarimeter or a list of polarimeters performing a complete analysis including:
    Even-Odd Output, Scientific Data, FFT and correlation Matrices.
    Parameters:
        - **path_file** (``str``): location of the data file, it is indeed the location of the hdf5 file's index
        - **start_datetime** (``str``): start time
        - **end_datetime** (``str``): end time
        - **name_pol** (``str``): name of the polarimeter (can be a list of string)
    """
    logging.basicConfig(level="INFO", format='%(message)s',
                        datefmt="[%X]", handlers=[RichHandler()])  # <3

    if len(sys.argv) < 5:
        # Note: When you run the code with "python procedure.py" you already give an argument:
        # procedure.py is sys.argv[0]
        logging.error("Wrong number of parameters.\n"
                      "Usage: python procedure.py path_datafile start_datetime end_datetime name_pol.\n"
                      "Note that the path of the datafile doesn't need quotation marks in the name, "
                      "i.e. home/data_dir and not 'home/data_dir'.")
        sys.exit(1)

    ####################################################################################################################
    # ARGUMENTS
    path_file = sys.argv[1]  # type: str
    start_datetime = sys.argv[2]  # type: str
    end_datetime = sys.argv[3]  # type: str
    pol_list = list(sys.argv[4:])

    # Flag For Future
    # Set nperseg = np.inf to reach the lowest frequency in FFT
    # Set nperseg = 6* 10**5 to reach 10^-4Hz in FFT
    # Set nperseg = 10**4 to reach 10^-3Hz in FFT
    nperseg = np.inf

    # Common Info for all the polarimeters
    gdate = [Time(start_datetime), Time(end_datetime)]
    date_dir = fz.dir_format(f"{gdate[0]}__{gdate[1]}")

    # rep_output_dir: Where I put the reports
    rep_output_dir = f"../plot/{date_dir}/reports"
    Path(rep_output_dir).mkdir(parents=True, exist_ok=True)

    # plot_dir: Where I find the plots to put in the reports
    plot_dir = f"../"

    ####################################################################################################################
    # CSV FILE
    # General Information about the whole procedure are collected in a csv file
    # csv_output_dir: Where I put the reports
    csv_output_dir = f"../plot/{date_dir}/CSV"
    Path(csv_output_dir).mkdir(parents=True, exist_ok=True)

    csv_general = [
        ["GENERAL REPORT CSV"],
        [""],
        ["Path dataset file", "Start Date Time", "Start Date Time"],
        [f"{path_file}", f"{start_datetime}", f"{end_datetime}"],
        [""],
        ["N Polarimeters"],
        [f"{len(pol_list)}"],
        [""],
        [""],
        ["House Keeping and Thermal Sensors Warning List"],
        [""],
        [""],
    ]

    with open(f'{csv_output_dir}/General_Report_{gdate[0]}__{gdate[1]}.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(csv_general)

    ####################################################################################################################
    ####################################################################################################################
    # START PROCEDURE
    # HouseKeeping & Thermal Sensors
    ####################################################################################################################
    logging.warning(f"The procedure started.\n Going to analyze Strip: House-Keeping parameters and Thermal Sensors")
    strip_name = pol_list[0]
    p = pol.Polarimeter(name_pol=strip_name, path_file=path_file,
                        start_datetime=start_datetime, end_datetime=end_datetime)

    t_warn = []  # List of warnings that will be included in every single-Polarimeter Report

    csv_mean = []
    csv_parameter = []

    ####################################################################################################################
    ####################################################################################################################
    # HOUSE-KEEPING PARAMETERS
    ####################################################################################################################
    logging.info("\n Done. Loading HouseKeeping Parameters now.")
    p.Load_HouseKeeping()

    hk_first_wrong_mean = True
    hk_first_wrong_sampling = True

    good_gen_sampling = True  # Bool that indicates if the sampling of all I,V & O parameters is good

    n_hk_wrong = 0  # type: int # Number of HK with wrong sampling
    hk_wrong = ""  # type: str  # String with the names of wrong sampling parameters

    delta_t = 0  # might be referenced before assignment

    for item_IVO in p.hk_list.keys():
        good_IVO_sampling = True  # Bool that indicates if the sampling of the single I-s, V-s or O-s is good
        for hk_name in p.hk_list[item_IVO]:
            # Setting the expected median of dt between two consecutive timestamps
            exp_med = 1.4
            if item_IVO == "O":
                exp_med = 60.

            hk_holes = fz.find_jump(v=p.hk_t[item_IVO][hk_name], exp_med=exp_med, tolerance=0.1)

            ############################################################################################################
            # Check on the Expected Median
            if not hk_holes["median_ok"]:

                # Captions (mean) for Report & CSV
                if hk_first_wrong_mean:
                    # Report: HK wrong mean caption
                    t_warn.append(f"The Expected mean "
                                  f"for HK parameter sampling is wrong for the parameters:<br />")

                    # CSV file: HK wrong mean caption
                    csv_mean = [["Wrong Sampling Mean time"],
                                ["HK Name"]]
                    hk_first_wrong_mean = False

                # Report: adding HK names
                t_warn.append(f"{hk_name}, ")

                # CSV file: adding HK names
                csv_mean.append([f"{hk_name}"])  # column of names of HK with wrong mean

            ############################################################################################################
            # Check on Sampling Holes
            if hk_holes["n"] != 0:
                good_IVO_sampling = False
                good_gen_sampling = False

                # Caption for CSV
                if hk_first_wrong_sampling:
                    hk_first_wrong_sampling = False

                    # Report: calculating delta_t (used below to write the table in report)
                    delta_t = (p.hk_t[item_IVO][hk_name][:-1] - p.hk_t[item_IVO][hk_name][1:]).sec

                    # CSV file: wrong sampling caption
                    csv_parameter = [[""],
                                     ["HK Sampling Reduction"]]

                msg = f"House-Keeping sampling's reduction found for parameter: {hk_name}.\n"
                logging.warning(msg)
                # Report: adding names
                n_hk_wrong += 1
                hk_wrong += f"{hk_name}, "

        if good_IVO_sampling:
            # Report
            msg = f"House-Keeping parameters {item_IVO}: sampling is good."
            logging.warning(msg + "\n")
            t_warn.append(msg + "<br /><p></p>")

    if not good_gen_sampling:
        # Report: writing a tabular with median, 5th percentile and 95th percentile
        hk_msg = f"<p></p>" \
                 f"House-Keeping Sampling is not good." \
                 f"<p></p> " \
                 f"House-keeping parameter affected: {n_hk_wrong}/28.<br />" \
                 "<p></p>" \
                 "<style>" \
                 "table, th, td {border:1px solid black;}" \
                 "</style>" \
                 "<body>" \
                 "<p></p>" \
                 "<table style='width:100%' align=center>" \
                 "<tr>" \
                 "<th>HK Parameters Affected</th>" \
                 "<th>Median &Delta;t [s]</th>" \
                 "<th>5th Percentile</th>" \
                 "<th>95th Percentile</th>" \
                 "</tr>" \
                 f"<td align=center>{hk_wrong}</td>" \
                 f"<td align=center>{np.median(delta_t)}</td>" \
                 f"<td align=center>{np.percentile(delta_t, 5)}</td>" \
                 f"<td align=center>{np.percentile(delta_t, 95)}</td>" \
                 f"</table></body><p></p><p></p><p></p>"
        t_warn.append(hk_msg)

        # CSV file: writing a tabular with median, 5th percentile and 95th percentile
        csv_parameter.append("")
        csv_parameter.append(["n HK Parameters Affected", "Median Delta t [s]", "5th Percentile", "95th Percentile",
                              "HK Names"])
        csv_parameter.append([f"{n_hk_wrong}", f"{np.median(delta_t)}",
                              f"{np.percentile(delta_t, 5)}", f"{np.percentile(delta_t, 95)}", f"{hk_wrong}"])
        csv_parameter.append("")

    logging.info("Done.\nPlotting House-Keeping parameters.\n")
    p.Norm_HouseKeeping()
    p.Plot_HouseKeeping_VI()
    p.Plot_HouseKeeping_OFF()
    logging.info("\n Done. Now I analyze them.")

    # Report: adding all HK results
    hk_results = p.Analyse_HouseKeeping()
    hk_table = p.HK_table(hk_results)

    # CSV file: adding all HK results
    for item in p.hk_list.keys():
        if item == "V":
            unit = "[&mu;V]"
        elif item == "I":
            unit = "[&mu;A]"
        else:
            unit = "[ADU]"
        csv_parameter.append(["Parameter", f"Max Value {unit}", f"Min Value {unit}",
                              f"Mean {unit}", f"Std_Dev {unit}", "NaN %"])
        for hk_name in p.hk_list[item]:
            csv_parameter.append([f"{hk_name}",
                                  f"{hk_results['max'][item][hk_name]}",
                                  f"{hk_results['min'][item][hk_name]}",
                                  f"{hk_results['mean'][item][hk_name]}",
                                  f"{hk_results['dev_std'][item][hk_name]}",
                                  f"{hk_results['nan_percent'][item][hk_name]}"])

    with open(f'{csv_output_dir}/General_Report_{gdate[0]}__{gdate[1]}.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(csv_mean)
        writer.writerows([""])
        writer.writerows(csv_parameter)
        writer.writerows([""])
    csv_mean = []
    csv_parameter = []

    ####################################################################################################################
    ####################################################################################################################
    # THERMAL SENSORS
    ####################################################################################################################
    logging.info("\n Done. Loading Thermal sensors now.")
    p.Load_Thermal_Sensors()

    ts_first_wrong_mean = True
    ts_first_wrong_sampling = True

    good_gen_sampling = True  # Bool that indicates if the sampling of all TS is good

    for status in range(2):
        for sensor_name in p.thermal_list[f"{status}"]:
            th_holes = fz.find_jump(v=p.thermal_sensors["thermal_times"][f"{status}"],
                                    exp_med=10.,
                                    tolerance=0.1)

            ############################################################################################################
            # Check on the Expected Median
            if not th_holes["median_ok"]:

                # Captions (mean) for Report & CSV
                if ts_first_wrong_mean:
                    # Report: TS wrong mean caption
                    t_warn.append(f"The Expected mean "
                                  f"for Thermal sensors sampling is wrong for the sensors:<br />")

                    # CSV file: TS wrong mean caption
                    csv_mean = [["Wrong Sampling Mean time"],
                                ["TS Name"]]
                    ts_first_wrong_mean = False

                # Report: adding TS names
                t_warn.append(f"{sensor_name}, ")

                # CSV file: adding TS names
                csv_mean.append([f"{sensor_name}"])  # column of names of TS with wrong mean

            ############################################################################################################
            # Check on sampling holes
            if th_holes["n"] != 0:
                good_gen_sampling = False

                # Caption for Report & CSV
                if ts_first_wrong_sampling:
                    # Report: TS caption
                    msg = f"Thermal sensors sampling's reduction found for sensors:"
                    logging.warning(msg)
                    t_warn.append("<p></p>" + msg + "<br />")

                    # CSV: TS caption
                    csv_parameter = [["TS Sampling Reduction"],
                                     ["Thermal Sensor Name"]]
                    ts_first_wrong_sampling = False

                logging.warning(f"\n{sensor_name}.\n")
                # Report: adding names
                t_warn.append(f"{sensor_name}, ")
                # CSV file: adding names
                csv_parameter.append([f"{sensor_name}"])

    if good_gen_sampling:
        # Report
        msg = f"Thermal sensors sampling is good.\n"
        logging.warning(msg)
        t_warn.append("<br />" + msg + "<br />")

    logging.info("Done.\nPlotting thermal sensors dataset.\n")
    p.Norm_Thermal()
    p.Plot_Thermal(status=0)
    p.Plot_Thermal(status=1)
    th_results = p.Analyse_Thermal()
    logging.info("\n Done. Producing Thermal Table now.")

    # Report: adding all HK results
    th_table = p.Thermal_table(th_results)

    # CSV file: adding all TS results
    csv_parameter.append([""])
    csv_parameter.append(["Status", "Sensor Name",
                          "Max value [RAW]", "Min value [RAW]", "Mean [RAW]", "Std_Dev[RAW]", "NaN %[RAW]",
                          "Max value [CAL]", "Min value [CAL]", "Mean [CAL]", "Std_Dev[CAL]", "NaN %[CAL]"])
    calib = ['raw', 'calibrated']
    for status in range(2):
        for sensor_name in p.thermal_list[f"{status}"]:
            csv_parameter.append([f"{status}", f"{sensor_name}",
                                  # Raw
                                  f"{th_results['max'][calib[0]][sensor_name]}",
                                  f"{th_results['min'][calib[0]][sensor_name]}",
                                  f"{th_results['mean'][calib[0]][sensor_name]}",
                                  f"{th_results['dev_std'][calib[0]][sensor_name]}",
                                  f"{th_results['nan_percent'][calib[0]][sensor_name]}",
                                  # Calibrated
                                  f"{th_results['max'][calib[1]][sensor_name]}",
                                  f"{th_results['min'][calib[1]][sensor_name]}",
                                  f"{th_results['mean'][calib[1]][sensor_name]}",
                                  f"{th_results['dev_std'][calib[1]][sensor_name]}",
                                  f"{th_results['nan_percent'][calib[1]][sensor_name]}"])

    with open(f'{csv_output_dir}/General_Report_{gdate[0]}__{gdate[1]}.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(csv_mean)
        writer.writerow([""])
        writer.writerows(csv_parameter)

    ####################################################################################################################
    ####################################################################################################################
    # START PROCEDURE
    # FOR N POLARIMETERS
    ####################################################################################################################
    logging.warning(f"\nGoing to analyze {len(pol_list)} polarimeter.")
    for name_pol in pol_list:
        logging.warning(f"Loading {name_pol}...")
        p = pol.Polarimeter(name_pol=name_pol, path_file=path_file, start_datetime=start_datetime,
                            end_datetime=end_datetime)
        ################################################################################################################
        # STRIP WARNINGS
        p.warnings["time_warning"] = t_warn

        ################################################################################################################
        # Loading Operations
        logging.info("Loading the dataset.\n")
        p.Load_Pol()
        p.Load_Thermal_Sensors()
        p.Load_HouseKeeping()
        ################################################################################################################
        # CSV INFO
        csv_pol_info = [["Pol Name"],
                        [f"{p.name}"],
                        [""],
                        ["Warning list"],
                        [""]]

        with open(f'{csv_output_dir}/General_Report_{gdate[0]}__{gdate[1]}.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(csv_pol_info)
            writer.writerow([""])

        # csv_pol_info = []

        ################################################################################################################
        # END THE PROCEDURE FOR THE POLARIMETER IF
        # 1) NO DATA
        # 2) STRIP OFF -> Data == 0
        ################################################################################################################
        stop_NoData = False
        stop_StripOff = False
        type: str
        for type in p.data.keys():
            for exit in p.data[type].keys():
                if len(p.data[type][exit]) == 0:
                    stop_NoData = True
                # Find a better way to write also this:
                # elif p.data[type][exit] == np.zeros(len(p.data[type][exit])):
                #    stop_StripOff = True

        # No data: Dataset empty
        if stop_NoData:
            # Report: writing the ERROR
            msg = f"No data in the time range wanted. End of the analysis for {name_pol}."
            logging.error(msg)
            p.warnings["time_warning"].append(msg + "<br />")

            # CSV file: writing the ERROR
            with open(f'{csv_output_dir}/General_Report_{gdate[0]}__{gdate[1]}.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["ERROR: No data."])

        # Strip Off
        elif stop_StripOff:
            # Report: writing the ERROR
            msg = f"Strip is off in the time range wanted. End of the analysis for {name_pol}"
            logging.error(msg)
            p.warnings["time_warning"].append(msg + "<br />")

            # CSV file: writing the ERROR
            with open(f'{csv_output_dir}/General_Report_{gdate[0]}__{gdate[1]}.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["WARNING: No data."])

        ################################################################################################################
        # START SAMPLING ANALYSIS
        ################################################################################################################
        else:
            logging.info("Looking for holes in the dataset.\n")
            p.STRIP_SAMPLING_FREQUENCY_HZ(warning=False)
            if p.STRIP_SAMPLING_FREQ != 100:
                # Report: writing sampling frequency
                msg = f"Data-sampling's reduction found. Sampling Frequency: {p.STRIP_SAMPLING_FREQ}."
                logging.warning(msg + "\n")
                p.warnings["time_warning"].append(msg + "<br />")
                p.warnings["eo_warning"].append(msg +
                                                "Possible Even-Odd inversions.<br />"
                                                "Attention: the time on the x-axes of the plots"
                                                "doesn't have sense because of the timestamps normalization. <br />"
                                                )

                # CSV file: writing sampling frequency
                csv_pol_info = [["Sampling Frequency"],
                                [f"{p.STRIP_SAMPLING_FREQ}", "Possible inversions Even-Odd"]]

            else:
                # Report: writing sampling frequency
                msg = "Data-sampling is good: the sampling frequency is 100Hz, " \
                      "hence no holes in scientific output expected."
                logging.warning(msg + "\n")
                p.warnings["time_warning"].append("<p></p>" + msg + "<br /><p></p>")

            logging.info("\nLooking for jumps in the Timestamps.\n")
            jumps = p.Write_Jump(start_datetime=start_datetime)

            if p.STRIP_SAMPLING_FREQ != 100:
                logging.info("\nDone.\nLooking for Even-Odd inversion due to jumps in the Timestamps.\n")
                p.Inversion_EO_Time(jumps_pos=jumps["idx"])

                with open(f'{csv_output_dir}/General_Report_{gdate[0]}__{gdate[1]}.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(csv_pol_info)

            ############################################################################################################
            # SPIKE ANALYSIS
            logging.info(f"Done.\nSpike analysis started.\n")
            s_tab = p.spike_report()

            # Report: writing spikes
            p.warnings["spike_warning"].append(s_tab)

            # CSV file: writing spikes
            csv_pol_info = p.spike_CSV()
            with open(f'{csv_output_dir}/General_Report_{gdate[0]}__{gdate[1]}.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(csv_pol_info)
            ############################################################################################################
            # SCIENTIFIC ANALYSIS

            logging.info(f"Done.\nPreparing the polarimeter {name_pol} for the scientific analysis.\n")

            # Normalization operations
            p.STRIP_SAMPLING_FREQ = 0
            p.Prepare(1)
            p.Norm_Thermal()
            p.Norm_HouseKeeping()
            logging.info(f"Done.\n")

            for type in p.data.keys():
                logging.info(f"Going to Plot {type} Output.")
                p.Plot_Output(type=f"{type}", begin=0, end=-1, show=False)
                logging.info(f"Done.\nStudying Correlations between Thermal Sensors and {type} Output.")
                p.Plot_Correlation_TS(type=type, begin=0, end=-1, show=False)

            logging.info("\nDone.\nEven-Odd Analysis started.\n")
            
            i = 1
            for type in p.data.keys():
                logging.info(f"Going to Plot Even Odd {type} Output and RMS.")
                for smooth in [1, 100, 1000]:
                    p.Plot_EvenOddAll(type=type, even=1, odd=1, all=0, begin=0, end=-1, smooth_len=smooth,
                                      show=False)
                    p.Plot_EvenOddAll(type=type, even=1, odd=1, all=1, begin=0, end=-1, smooth_len=smooth,
                                      show=False)
                    p.Plot_EvenOddAll(type=type, even=0, odd=0, all=1, begin=0, end=-1, smooth_len=smooth,
                                      show=False)
                    logging.info(f"{type}: {3 * i}/18) Output plot done.")

                    p.Plot_RMS_EOA(type=type, window=100, even=1, odd=1, all=0, begin=0, end=-1,
                                   smooth_len=smooth,
                                   show=False)
                    p.Plot_RMS_EOA(type=type, window=100, even=1, odd=1, all=1, begin=0, end=-1,
                                   smooth_len=smooth,
                                   show=False)
                    p.Plot_RMS_EOA(type=type, window=100, even=0, odd=0, all=1, begin=0, end=-1,
                                   smooth_len=smooth,
                                   show=False)
                    logging.info(f"{type}: {3 * i}/18) RMS plot done.")
                    i += 1
                logging.info(f"{type}: Done.\n")

            for type in p.data.keys():
                logging.info(f"Going to Plot {type} Even-Odd Correlation.")
                p.Plot_Correlation_EvenOdd(type, begin=0, end=-1, show=False)
                logging.info(f"Done.\n")
            
            i = 1
            for type in p.data.keys():
                logging.info(f"Going to Plot {type} Even-Odd FFT.")
                _ = p.Plot_FFT_EvenOdd(type=type, even=1, odd=1, all=0, begin=0, end=-1, nseg=nperseg, show=False)
                _ = p.Plot_FFT_EvenOdd(type=type, even=0, odd=0, all=1, begin=0, end=-1, nseg=nperseg, show=False)
                csv_pol_info = p.Plot_FFT_EvenOdd(type=type, even=1, odd=1, all=1, begin=0, end=-1, nseg=nperseg,
                                                  show=False, spike_check=True)
                # CSV file: writing FFT Output spikes
                with open(f'{csv_output_dir}/General_Report_{gdate[0]}__{gdate[1]}.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(csv_pol_info)

                logging.info(f"FFT {type}: Done.")

                logging.info(f"Going to Plot {type} Even-Odd FFT of the RMS.")
                p.Plot_FFT_RMS_EO(type=type, window=100, even=1, odd=1, all=0, begin=0, end=-1, nseg=nperseg,
                                  show=False)
                p.Plot_FFT_RMS_EO(type=type, window=100, even=0, odd=0, all=1, begin=0, end=-1, nseg=nperseg,
                                  show=False)
                p.Plot_FFT_RMS_EO(type=type, window=100, even=1, odd=1, all=1, begin=0, end=-1, nseg=nperseg,
                                  show=False)
                logging.info(f"FFT RMS {type}: Done.")
            i += 1

            logging.info("\nEven Odd Analysis is now completed.\nScientific Data Analysis started.")

            i = 1
            for type in p.data.keys():
                logging.info(f"Going to Plot Scientific Data {type} and RMS.")
                for smooth in [1, 100, 1000]:
                    p.Plot_SciData(type=type, smooth_len=smooth, show=False)
                    logging.info(f"{type}: {i}/6) Data plot done.")

                    p.Plot_RMS_SciData(type=type, window=100, begin=0, end=-1, smooth_len=smooth, show=False)
                    logging.info(f"{type}: {i}/6) RMS plot done.")

                    i += 1

            i = 1
            for type in p.data.keys():
                logging.info(f"Going to Plot Scientific Data {type} FFT and FFT of  RMS.")
                csv_pol_info = p.Plot_FFT_SciData(type=type, begin=0, end=-1, nseg=nperseg,
                                                  show=False, spike_check=True)
                logging.info(f"{type}: {i}/2) Data FFT plot done.")

                # CSV file: writing FFT Scientific Data spikes
                with open(f'{csv_output_dir}/General_Report_{gdate[0]}__{gdate[1]}.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(csv_pol_info)

                p.Plot_FFT_RMS_SciData(type=type, window=100, begin=0, end=-1, nseg=nperseg, show=False)
                logging.info(f"{type}: {i}/2) RMS FFT plot done.")

            logging.info(
                "Scientific Data Analysis is now completed. Correlation Matrices will be now produced.\n")

            ############################################################################################################
            # CORRELATION MATRICES
            ############################################################################################################
            t = 0.4

            csv_pol_info = [
                [""],
                [f"High Correlations (threshold t = {t})"],
                [""],
                ["Data type", "exit i-j", "Corr. Value"]
            ]

            for s in [True, False]:
                for type in p.data.keys():

                    if s:
                        logging.debug(
                            f"\nGoing to plot {type} Correlation Matrix with scientific parameter = {s}\n")
                        csv_pol_info += p.Plot_Correlation_Mat(type=type, scientific=s, show=False, warn_threshold=t)
                        csv_pol_info += p.Plot_Correlation_Mat_RMS(type=type, scientific=s, show=False,
                                                                   warn_threshold=t)

                    if not s:
                        if type == "PWR":
                            csv_pol_info += p.Plot_Correlation_Mat(type=type, scientific=s,
                                                                   show=False, warn_threshold=t)
                            csv_pol_info += p.Plot_Correlation_Mat_RMS(type=type, scientific=s,
                                                                       show=False, warn_threshold=t)

                            for e, o in zip([True, False], [False, True]):
                                logging.debug(f"even={e}, odd = {o}")
                                csv_pol_info += p.Plot_Correlation_Mat(type=type, scientific=s, even=e, odd=o,
                                                                       show=False, warn_threshold=t)
                                csv_pol_info += p.Plot_Correlation_Mat_RMS(type=type, scientific=s, even=e, odd=o,
                                                                           show=False, warn_threshold=t)

            # CSV file: writing FFT Scientific Data spikes
            with open(f'{csv_output_dir}/General_Report_{gdate[0]}__{gdate[1]}.csv', 'a',
                      newline='') as file:
                writer = csv.writer(file)
                writer.writerows(csv_pol_info)

            logging.warning("\nAnalysis completed.\nPreparing warnings for the report now.")

            # WARNINGS
            t_warner = ""
            for bros in p.warnings["time_warning"]:
                t_warner += bros

            t = 0.4
            if len(p.warnings["corr_warning"]) == 0:
                corr_warner = "Nothing to notify. All correlations seem ok."
            else:
                corr_warner = ""
                for bros in p.warnings["corr_warning"]:
                    corr_warner += bros + "<br />"
            corr_warner = f"Correlation Warning Threshold set at {t}.<br />" + corr_warner

            eo_warner = ""
            if len(p.warnings["eo_warning"]) == 0:
                eo_warner = "Nothing to point out.\n"
            else:
                for bros in p.warnings["eo_warning"]:
                    eo_warner += bros + "<br />"

            spike_warner = ""
            for bros in p.warnings["spike_warning"]:
                spike_warner += bros

            # t_warner = ""
            # corr_warner = ""
            # eo_warner = ""
            # spike_warner = ""

            # REPORTS
            logging.info("\nDone. Producing report!")
            data = {
                "name_polarimeter": name_pol,
                "analysis_date": str(f"{start_datetime} - {end_datetime}"),
                "rep_output_dir": rep_output_dir,
                "plot_dir": plot_dir,
                "command_line": " ".join(sys.argv),
                "th_html_tab": th_table,
                "hk_html_tab": hk_table,
                "t_warnings": t_warner,
                "corr_warnings": corr_warner,
                "eo_warnings": eo_warner,
                "spike_warnings": spike_warner,
            }

            # root: Where I find the file.txt with the information to build the report
            root = "../striptease/"
            templates_dir = Path(root)
            env = Environment(loader=FileSystemLoader(templates_dir))
            template = env.get_template('jinja_report.txt')

            filename = Path(f"{rep_output_dir}/report_{name_pol}.html")
            with open(filename, 'w') as outf:
                outf.write(template.render(data))


if __name__ == "__main__":
    main()

# https://code-maven.com/minimal-example-generating-html-with-python-jinja
