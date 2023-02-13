#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# This file contains a 8th version (0.0.8) of the new LSPE-STRIP pipeline to produce a complete scan of a polarimeter.
# December 7th 2022, Brescia (Italy)

# Libraries & Modules
import logging
import sys

from astropy.time import Time
from pathlib import Path
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
        # Note: When I run the code with "python procedure.py" I already give an argument:
        # procedure.py is sys.argv[0]
        logging.error("Wrong number of parameters.\n"
                      "Usage: python procedure.py path_datafile start_datetime end_datetime name_pol.\n"
                      "Note that the path of the datafile doesn't need quotation marks in the name, "
                      "i.e. home/data_dir and not 'home/data_dir'.")
        sys.exit(1)

    path_file = sys.argv[1]  # type: str
    start_datetime = sys.argv[2]  # type: str
    end_datetime = sys.argv[3]  # type: str
    pol_list = list(sys.argv[4:])

    # path_file = "/home/francesco/Scrivania/Tesi/data_test/"  # type: str
    # start_datetime = "2022-12-13 7:00:00"
    # end_datetime = "2022-12-13 9:00:00"
    # pol_list = list(["R0"])

    logging.warning(f"The procedure started.\n Going to analyze {len(pol_list)} polarimeter.")
    for name_pol in pol_list:
        logging.warning(f"Loading {name_pol}...")
        p = pol.Polarimeter(name_pol=name_pol, path_file=path_file, start_datetime=start_datetime,
                            end_datetime=end_datetime)
        # TIME ANALYSIS
        logging.info("Loading the dataset.\n")
        p.Load_Pol()

        logging.info("Looking for holes in the dataset.\n")
        p.STRIP_SAMPLING_FREQUENCY_HZ(warning=False)
        if p.STRIP_SAMPLING_FREQ != 100:
            holes = fz.find_holes(p.times.value)
            if len(holes) != 0:
                msg = "Data-sampling's reduction found.\n"
                logging.warning(msg)
                p.warnings["time_warning"].append(msg)
                p.warnings["eo_warning"].append(msg +
                                                "Possible Even-Odd inversions at the following date times:<br />"
                                                "Attention: the time after the hole in the x-axes of the plots "
                                                "doesn't have sense because of the timestamps normalization. <br />"
                                                "Fix it putting some zero-value fake-outputs in the hole. <br />")
                for h in holes:
                    msg = f"{Time(h, format='mjd').to_datetime().strftime('%Y-%m-%d %H:%M:%S')}."
                    p.warnings["eo_warning"].append(msg)
        else:
            msg = "Data-sampling is good. No holes in scientific output found.\n"
            logging.warning(msg)
            p.warnings["time_warning"].append(msg + "<br /><p></p>")

        # Add here the holes re-run procedure... 2023/02/07

        logging.info("\nDone.\nLooking for jumps in the Timestamps.\n")
        jumps = p.Write_Jump(start_datetime=start_datetime)

        logging.info("\nDone.\nLooking for Even-Odd inversion due to jumps in the Timestamps.\n")
        p.Inversion_EO_Time(jumps_pos=jumps["position"])

        # HOUSE-KEEPING PARAMETERS
        logging.info("\n Done. Loading HouseKeeping Parameters now.")
        p.Load_HouseKeeping()
        good = True
        for item in p.hk_list.keys():
            for hk_name in p.hk_list[item]:
                hk_holes = fz.find_holes(p.hk_t[item][hk_name].value)
                if len(hk_holes) != 0:
                    msg = f"<br />House-Keeping sampling's reduction found for parameter: {hk_name}.\n"
                    logging.warning(msg)
                    p.warnings["time_warning"].append(msg + "<br />")
                    good = False
            if good:
                msg = f"<br />House-Keeping parameter's {item} sampling is good.\n"
                logging.warning(msg)
                p.warnings["time_warning"].append(msg + "<br />")

        logging.info("Done.\nPlotting House-Keeping parameters.\n")
        p.Norm_HouseKeeping()
        p.Plot_HouseKeeping_VI()
        p.Plot_HouseKeeping_OFF()
        logging.info("\n Done. Now I analyze them.")
        hk_results = p.Analyse_HouseKeeping()
        hk_table = p.HK_table(hk_results)

        # THERMAL SENSORS
        logging.info("\n Done. Loading Thermal sensors now.")
        p.Load_Thermal_Sensors()
        good = True
        for status in range(2):
            for sensor_name in p.thermal_list[f"{status}"]:
                th_holes = fz.find_holes(p.thermal_sensors["thermal_times"][f"{status}"].value)
                if len(th_holes) != 0:
                    msg = f"Thermal sensors sampling's reduction found for sensor: {sensor_name}.\n"
                    logging.warning(msg)
                    p.warnings["time_warning"].append(msg + "<br />")
                    good = False
        if good:
            msg = f"Thermal sensors sampling is good.\n"
            logging.warning(msg)
            p.warnings["time_warning"].append(msg + "<br />")

        logging.info("Done.\nPlotting thermal sensors dataset.\n")
        p.Norm_Thermal()
        p.Plot_Thermal()
        th_results = p.Analyse_Thermal()
        logging.info("\n Done. Producing Thermal Table now.")
        th_table = p.Thermal_table(th_results)

        # SCIENTIFIC ANALYSIS

        logging.info(f"Done.\nPreparing the polarimeter {name_pol} for the scientific analysis.\n")

        p.STRIP_SAMPLING_FREQ = 0
        p.Prepare(1)
        logging.info(f"Done.\n")

        for type in p.data.keys():
            logging.info(f"Going to Plot {type} Output.")
            p.Plot_Output(type=f"{type}", begin=0, end=-1, show=False)
            logging.info(f"Done.\nStudying Correlations between Thermal Sensors and {type} Output.")

        logging.info("\nDone.\nEven-Odd Analysis started.\n")
        i = 1
        for type in p.data.keys():
            logging.info(f"Going to Plot Even Odd {type} Output and RMS.")
            for smooth in [1, 100, 1000]:
                p.Plot_EvenOddAll(type=type, even=1, odd=1, all=0, begin=0, end=-1, smooth_len=smooth, show=False)
                p.Plot_EvenOddAll(type=type, even=1, odd=1, all=1, begin=0, end=-1, smooth_len=smooth, show=False)
                p.Plot_EvenOddAll(type=type, even=0, odd=0, all=1, begin=0, end=-1, smooth_len=smooth, show=False)
                logging.info(f"{type}: {3 * i}/18) Output plot done.")

                p.Plot_RMS_EOA(type=type, window=100, even=1, odd=1, all=0, begin=0, end=-1, smooth_len=smooth,
                               show=False)
                p.Plot_RMS_EOA(type=type, window=100, even=1, odd=1, all=1, begin=0, end=-1, smooth_len=smooth,
                               show=False)
                p.Plot_RMS_EOA(type=type, window=100, even=0, odd=0, all=1, begin=0, end=-1, smooth_len=smooth,
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
            for smooth in [1, 100, 1000]:
                p.Plot_SciData(type=type, smooth_len=smooth, show=False)
                logging.info(f"{type}: {i}/6) Data plot done.")

                p.Plot_RMS_SciData(type=type, window=100, begin=0, end=-1, smooth_len=smooth, show=False)
                logging.info(f"{type}: {i}/6) RMS plot done.")

                i += 1
        i = 1
        for type in p.data.keys():
            logging.info(f"Going to Plot Scientific Data {type} FFT and FFT of  RMS.")
            p.Plot_FFT_SciData(type=type, begin=0, end=-1, show=False)
            logging.info(f"{type}: {i}/2) Data FFT plot done.")

            p.Plot_FFT_RMS_SciData(type=type, window=100, begin=0, end=-1, show=False)
            logging.info(f"{type}: {i}/2) RMS FFT plot done.")

        logging.info("Scientific Data Analysis is now completed. Correlation Matrices will be now produced.\n")

        # CORRELATION MATRICES
        t = 0.4
        for s in [True, False]:
            for type in p.data.keys():

                if s:
                    logging.debug(f"\nGoing to plot {type} Correlation Matrix with scientific parameter = {s}\n")
                    p.Plot_Correlation_Mat(type=type, scientific=s, show=False, warn_threshold=t)
                    p.Plot_Correlation_Mat_RMS(type=type, scientific=s, show=False, warn_threshold=t)

                if not s:
                    if type == "PWR":
                        p.Plot_Correlation_Mat(type=type, scientific=s, show=False, warn_threshold=t)
                        p.Plot_Correlation_Mat_RMS(type=type, scientific=s, show=False, warn_threshold=t)

                        for e, o in zip([True, False], [False, True]):
                            logging.debug(f"even={e}, odd = {o}")
                            p.Plot_Correlation_Mat(type=type, scientific=s, even=e, odd=o, show=False, warn_threshold=t)
                            p.Plot_Correlation_Mat_RMS(type=type, scientific=s, even=e, odd=o, show=False,
                                                       warn_threshold=t)

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

        # t_warner = ""
        # corr_warner = ""
        # eo_warner = ""

        # REPORT

        # output_dir: Where I put the reports
        date_dir = fz.dir_format(f"{p.gdate[0]}__{p.gdate[1]}")
        output_dir = f"../plot/{date_dir}/reports"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # plot_dir: Where I find the plots to put in the reports
        plot_dir = f"../"

        # Reports data
        logging.info("\nDone. Producing report!")
        data = {
            "name_polarimeter": name_pol,
            "analysis_date": str(f"{start_datetime} - {end_datetime}"),
            "output_dir": output_dir,
            "plot_dir": plot_dir,
            "command_line": " ".join(sys.argv),
            "th_html_tab": th_table,
            "hk_html_tab": hk_table,
            "t_warnings": t_warner,
            "corr_warnings": corr_warner,
            "eo_warnings": eo_warner,

        }

        # root: Where I find the file.txt with the information to build the report
        root = "../striptease/"
        templates_dir = Path(root)
        env = Environment(loader=FileSystemLoader(templates_dir))
        template = env.get_template('jinja_report.txt')

        filename = Path(f"{output_dir}/report_{name_pol}.html")
        with open(filename, 'w') as outf:
            outf.write(template.render(data))


if __name__ == "__main__":
    main()

# https://code-maven.com/minimal-example-generating-html-with-python-jinja
