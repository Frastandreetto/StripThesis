#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# Libraries & Modules
import logging

from matplotlib import pyplot as plt
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sn  # This should be added to requirements.txt

# MyLibraries & MyModules
import polarimeter as pol
import f_strip as fz


# This file contains the main correlation functions used in the new version of the pipeline
# for functional verification of LSPE-STRIP (2023)
# September 19th 2023, Delft (Netherlands)


def correlation_plot(array1: [], array2: [], dict1: dict, dict2: dict, time1: [], time2: [],
                     data_name1: str, data_name2: str, start_datetime: str, show=False,
                     corr_t=0.4, plot_dir='../plot') -> []:
    """
        Create a Correlation Plot of two dataset: two array, two dictionaries or one array and one dictionary.\n
        Parameters:\n
        - **array1**, **array2** (``array``): arrays ([]) of n1 and n2 elements
        - **dict1**, **dict2** (``dict``): dictionaries ({}) with N1, N2 keys
        - **time1**, **time2** (``array``): arrays ([]) of timestamps: not necessary if the dataset have same length.
        - **data_name1**, **data_name2** (``str``): names of the dataset. Used for titles, labels and to save the png.
        - **start_datetime** (``str``): begin date of dataset. Used for the title of the figure and to save the png.
        - **end_datetime** (``str``): end date of dataset. Used for the title of the figure and to save the png.
        - **show** (``bool``):\n
            *True* -> show the plot and save the figure\n
            *False* -> save the figure only
        - **corr_t** (``int``): if it is overcome by the correlation value of a plot, a warning is produced.\n
        - **plot_dir** (``str``): path where the plots are organized in directories and saved.
        Return a list of warnings that highlight which data are highly correlated.
    """
    # Creating a list to collect the warnings
    warnings = []

    # Data comprehension -----------------------------------------------------------------------------------------------
    # Type check
    # array and timestamps array must be list
    if not (all(isinstance(l, list) for l in (array1, array2, time1, time2))):
        logging.error("Wrong type: note that array1, array2, time1, time2 are list.")
    # dict1 and dict2 must be dictionaries
    if not (all(isinstance(d, dict) for d in (dict1, dict2))):
        logging.error("Wrong type: note that dict1 and dict2 are dictionaries.")

    # Case 1 : two 1D arrays
    # Single plot: scatter correlation between two array
    if array1 != [] and array2 != [] and dict1 == {} and dict2 == {}:
        n_rows = 1
        n_col = 1
        fig_size = (4, 4)

    # Case 2: one 1D array and one dictionary with N keys
    # Plot 1xN: scatter correlation between an array and the exits of a dictionary
    elif array1 != [] and array2 == [] and dict1 != {} and dict2 == {}:
        logging.debug(f"I should be here!")
        # If the object are different, the sampling frequency may be different
        # hence the timestamps array must be provided to interpolate
        if time1 == [] or time2 == []:
            logging.error("Different sampling frequency: provide timestamps array.")
            raise SystemExit(1)
        else:
            n_rows = 1
            n_col = len(dict1.keys())
            fig_size = (4 * n_col, 4 * n_rows)

    # Case 3: two dictionaries with N and M keys
    # Plot NxM: scatter correlation between each dictionary exit
    elif array1 == [] and array2 == [] and dict1 != {} and dict2 != {}:
        n_rows = len(dict1.keys())
        n_col = len(dict2.keys())
        fig_size = (4 * n_col, 4 * n_rows)

    else:
        msg = ("Wrong data. Please insert only two of the four dataset: array1, array2, dict1 or dict2. "
               "Please do not insert array2 if there is not an array1. "
               "Please do not insert dict2 if there is not a dict1.")
        logging.error(f"{msg}")
        raise SystemExit(1)
    # ------------------------------------------------------------------------------------------------------------------

    logging.debug(f"Number of col:{n_col}, of row:{n_rows}, fig size: {fig_size}")
    # Creating the name of the data: used for the fig title and to save the png
    data_name = f"{data_name1}-{data_name2}"
    # Creating the figure with the subplots
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_col, constrained_layout=True, figsize=fig_size)
    # Set the title
    fig.suptitle(f'Correlation {data_name} \n Date: {start_datetime}', fontsize=10)

    # array1 vs array2 -------------------------------------------------------------------------------------------------
    if n_col == 1:

        # Check if the dataset is a TS and store the label for the plots
        label1 = f"{data_name1} Temperature [K]" if data_name1[0] in ("T", "E") else f"{data_name1} Output [ADU]"
        label2 = f"{data_name2} Temperature [K]" if data_name2[0] in ("T", "E") else f"{data_name2} Output [ADU]"

        # Arrays with different length must be interpolated
        if len(array1) != len(array2):
            if time1 == [] or time2 == []:
                logging.error("Different sampling frequency: provide timestamps array.")
                raise SystemExit(1)
            else:
                # Find the longest array (x) and the shortest to be interpolated
                x, short_array, label_x, label_y = (array1, array2, label1, label2) if len(array1) > len(array2) \
                    else (array2, array1, label2, label1)
                x_t, short_t = (time1, time2) if x is array1 else (time2, time1)

                # Interpolation of the shortest array
                y = np.interp(x_t, short_t, short_array)

        # Arrays with same length
        else:
            x = array1
            y = array2
            label_x = label1
            label_y = label2

        axs.plot(x, y, "*", color="firebrick", label="Corr Data")
        # XY-axis
        axs.set_xlabel(f"{label_x}")
        axs.set_ylabel(f"{label_y}")
        # Legend
        axs.legend(prop={'size': 9}, loc=4)

        # Calculate the correlation coefficient matrix -----------------------------------------------------------------
        correlation_matrix = np.corrcoef(x, y)
        # Extract the correlation coefficient between the two datasets from the matrix
        correlation_value = correlation_matrix[0, 1]
        # Print a warning if the correlation value overcomes the threshold, then store it for the report
        if correlation_value > corr_t:
            warn_msg = (f"Found high correlation value: {round(correlation_value, 4)}"
                        f" between {data_name1} and {data_name2}.")
            logging.warning(warn_msg)
            warnings.append(f"|{data_name1}|{data_name2}|{round(correlation_value, 4)}|\n")
    # ------------------------------------------------------------------------------------------------------------------

    elif n_col > 1:
        # dict1 vs dict2 -----------------------------------------------------------------------------------------------
        if n_rows > 1:
            for r, r_exit in enumerate(dict1.keys()):
                for c, c_exit in enumerate(dict2.keys()):
                    if r == 0:
                        axs[r, c].sharey(axs[1, 0])
                        axs[r, c].sharex(axs[1, 0])

                    x = dict1[r_exit]
                    y = dict2[c_exit]
                    axs[r, c].plot(x, y, "*", color="teal", label="Corr Data")

                    # Subplot title
                    axs[r, c].set_title(f'Corr {c_exit} - {r_exit}')
                    # XY-axis
                    # Note: the label here is specific for ADU Output: can be easily generalized if needed...
                    axs[r, c].set_xlabel(f"{data_name1} {r_exit} Output [ADU]")
                    axs[r, c].set_ylabel(f"{data_name2} {c_exit} Output [ADU]")
                    # Legend
                    axs[r, c].legend(prop={'size': 9}, loc=4)

                    # Calculate the correlation coefficient matrix -----------------------------------------------------
                    correlation_matrix = np.corrcoef(x, y)
                    # Extract the correlation coefficient between the two datasets x-y from the matrix
                    correlation_value = correlation_matrix[0, 1]
                    # Print a warning if the correlation value overcomes the threshold, then store it for the report
                    if correlation_value > corr_t:
                        warn_msg = (f'Found high correlation value between '
                                    f'{data_name1} {r_exit} and {data_name2} {c_exit}: {round(correlation_value, 4)}.')
                        logging.warning(warn_msg)
                        warnings.append(f"|{data_name1}{r_exit} |{data_name2}{c_exit}|{round(correlation_value, 4)}|\n")

        # array1 vs dict1 ----------------------------------------------------------------------------------------------
        else:
            for c, exit in enumerate(dict1.keys()):
                x = dict1[exit]
                y = np.interp(time2, time1, array1)
                axs[c].plot(x, y, "*", color="lawngreen", label="Corr Data")

                label_x = f"{data_name2} Output [ADU]"
                label_y = f"{data_name1} Temperature [K]"

                # Subplot title
                axs[c].set_title(f'Corr {exit}')
                # XY-axis
                axs[c].set_xlabel(f"{label_x} ")
                axs[c].set_ylabel(f"{label_y}")
                # Legend
                axs[c].legend(prop={'size': 9}, loc=4)

                # Calculate the correlation coefficient matrix ---------------------------------------------------------
                correlation_matrix = np.corrcoef(x, y)
                # Extract the correlation coefficient between the two datasets from the matrix
                correlation_value = correlation_matrix[0, 1]
                # Print a warning if the correlation value overcomes the threshold, then store it for the report
                if correlation_value > corr_t:
                    warn_msg = (f"Found high correlation value between {data_name1} and {data_name2} "
                                f"in exit {exit}: {round(correlation_value, 4)}.")
                    logging.warning(warn_msg)
                    warnings.append(f"|{data_name1}|{data_name2} {exit}|{round(correlation_value, 4)}|\n")
    else:
        return warnings

    # Procedure to save the png of the plot in the correct dir
    path = f'{plot_dir}/Correlation_Plot/'
    # Check if the dir exists. If not, it will be created.
    Path(path).mkdir(parents=True, exist_ok=True)
    fig.savefig(f'{path}{data_name}_CorrPlot.png')

    # If show is True the plot is visible on video
    if show:
        plt.show()
    plt.close(fig)

    return warnings


def correlation_mat(dict1: {}, dict2: {}, data_name1: str, data_name2: str,
                    start_datetime: str, show=False, corr_t=0.4, plot_dir='../plot') -> []:
    """
       Plot a 4x4 Correlation Matrix of two generic dictionaries (also of one with itself).\n

       Parameters:\n
       - **dict1**, **dict2** (``array``): dataset
       - **data_name** (``str``): name of the dataset. Used for the title of the figure and to save the png.
       - **start_datetime** (``str``): begin date of dataset. Used for the title of the figure and to save the png.
       - **show** (``bool``):\n
            *True* -> show the plot and save the figure\n
            *False* -> save the figure only
       - **corr_t** (``int``): if it is overcome by one of the values of the matrix a warning is produced\n
       - **plot_dir** (``str``): path where the plots are organized in directories and saved.
       Return a list of warnings that highlight which data are highly correlated.
    """
    # Creating the name of the data: used for the fig title and to save the png
    data_name = f"{data_name1}-{data_name2}"

    # Creating a list to collect the warnings
    warnings = []

    # Initialize a boolean variable for the self correlation of a dataset
    self_correlation = False
    # If the second dictionary is not provided we are in a Self correlation case
    if dict2 == {}:
        self_correlation = True
        dict2 = dict1
    # Convert dictionaries to DataFrames
    df1 = pd.DataFrame(dict1)
    df2 = pd.DataFrame(dict2)

    # Initialize an empty DataFrame for correlations
    correlation_matrix = pd.DataFrame(index=df1.columns, columns=df2.columns)

    # Calculate correlations
    for key1 in df1.columns:
        for key2 in df2.columns:
            correlation_matrix.loc[key1, key2] = df1[key1].corr(df2[key2])

            # Print a warning if the correlation value overcomes the threshold, then store it for the report
            if correlation_matrix.loc[key1, key2] > corr_t:
                warn_msg = (f"Found high correlation value between {key1} and {key2}: "
                            f"{round(correlation_matrix.loc[key1, key2], 4)}.")
                logging.warning(warn_msg)
                warnings.append(f"|{data_name1} {key1}|{data_name2} {key2}|{correlation_matrix.loc[key1, key2]}|\n")

    # Self correlation case
    if self_correlation:
        for i in correlation_matrix.keys():
            # Put at Nan the values on the diagonal of the matrix (self correlations)
            correlation_matrix[i][i] = np.nan

    # Convert correlation matrix values to float
    correlation_matrix = correlation_matrix.astype(float)

    # Create a figure to plot the correlation matrices
    fig, axs = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(10, 5))
    # Set the title of the figure
    fig.suptitle(f'Correlation Matrix {data_name} - Date: {start_datetime}', fontsize=12)

    pl_m1 = sn.heatmap(correlation_matrix, annot=True, ax=axs[0], cmap='coolwarm')
    pl_m1.set_title(f"Correlation {data_name}", fontsize=14)
    pl_m2 = sn.heatmap(correlation_matrix, annot=True, ax=axs[1], cmap='coolwarm', vmin=-0.4, vmax=0.4)
    pl_m2.set_title(f"Correlation {data_name} - Fixed Scale", fontsize=14)

    # Procedure to save the png of the plot in the correct dir
    path = f'{plot_dir}/Correlation_Matrix/'
    # Check if the dir exists. If not, it will be created.
    Path(path).mkdir(parents=True, exist_ok=True)
    fig.savefig(f'{path}{data_name}_CorrMat.png')

    # If show is True the plot is visible on video
    if show:
        plt.show()
    plt.close(fig)

    return warnings


def cross_corr_mat(path_file: str, start_datetime: str, end_datetime: str, show=False, corr_t=0.4):
    """
    - **path_file** (``str``): location of the data file, it is indeed the location of the hdf5 file's index
    - **start_datetime** (``str``): begin date of dataset. Used for the title of the figure and to save the png.
    - **end_datetime** (``str``): end date of dataset. Used for the title of the figure and to save the png.
    - **show** (``bool``):\n
            *True* -> show the plot and save the figure\n
            *False* -> save the figure only
       - **corr_t** (``int``): if it is overcome by one of the values of the matrix a warning is produced\n
       Return a list of warnings that highlight which data are highly correlated.

    """
    # Creating a list to collect the warnings
    warnings = []

    # Initializing the data_name
    data_name = ""
    # Creating the path to store the png file containing the matrix
    date_dir = fz.dir_format(f"{start_datetime}__{end_datetime}")
    store_path = f"../plot/{date_dir}/Cross_Corr/"
    # Check if the dir exists. If not, it will be created.
    Path(store_path).mkdir(parents=True, exist_ok=True)

    polar_names = ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "I0", "I1", "I2", "I3", "I4", "I5", "I6",
                   "G0", "G1", "G2", "G3", "G4", "G5", "G6", "O0", "O1", "O2", "O3", "O4", "O5", "O6",
                   "R0", "R1", "R2", "R3", "R4", "R5", "R6", "V0", "V1", "V2", "V3", "V4", "V5", "V6",
                   "W1", "W2", "W3", "W4", "W5", "W6", "Y0", "Y1", "Y2", "Y3", "Y4", "Y5", "Y6"]

    polars = {name: {} for name in polar_names}

    # Loading all polarimeters
    # Time needed on a 3h dataset: ~ 5sec/pol => ~ 5 minutes tot
    for name in polars.keys():
        # Define a Polarimeter
        p = pol.Polarimeter(name_pol=name, path_file=path_file,
                            start_datetime=start_datetime, end_datetime=end_datetime)
        # Loading scientific Output
        logging.info(f"\nLoading Polarimeter {name}.")
        p.Load_Pol()
        # Fill the dictionary with all the data of the polarimeters
        for kind in ["DEM", "PWR"]:
            polars[name][kind] = p.data[kind]

    # Producing 55x55 correlation matrices
    # Total number of matrices: 55 pol x4 single exits x4 all other exits x2 data type => 1760
    # Time needed on a 3h dataset: ~ 7 sec/matr => 3.4 hours
    for type in ["DEM", "PWR"]:
        # Initialize an int that collect the number of repetitions
        times = 1  # type: int
        # Preparing a list of all the names of the polarimeters to parse
        other_names = list(polars.keys())
        for name in polars.keys():
            # Removing the name of the current pol: needed to not repeat the check
            other_names.remove(name)
            # Exit of all the pol of the matrix
            all_exits = ["Q1", "Q2", "U1", "U2"]
            for all_exit in all_exits:
                # Exit of the current pol
                for single_exit in ["Q1", "Q2", "U1", "U2"]:

                    # Considering the same exit on all polarimeters
                    data = {key: val[type][all_exit] for key, val in polars.items()}

                    # Condition to avoid the repetitions of the matrix
                    if single_exit == all_exit:
                        # times == 4 when the four exits have already been parsed
                        if times < 4:
                            # Updating data_name
                            data_name = f"{type}_{all_exit}"
                            times += 1
                        else:
                            pass
                    else:
                        # Changing the data exit of the current pol
                        data[name] = polars[name][type][single_exit]
                        # Updating data name
                        data_name = f"{type}-{name}_{single_exit}-All_{all_exit}"

                    logging.info(f"{data_name}")

                    # Create a DataFrame from the data
                    df = pd.DataFrame(data)
                    # Calculate the correlation matrix
                    correlation_matrix = df.corr()

                    # Remove self-correlations
                    if single_exit == all_exit:
                        for i in correlation_matrix.keys():
                            # Put at Nan the values on the diagonal of the matrix
                            correlation_matrix[i][i] = np.nan

                    # Check on high correlation values
                    # Done only on the changed row
                    for name2 in other_names:
                        if correlation_matrix[name][name2] > corr_t:
                            warn_msg = (f"Found high correlation value between "
                                        f"{name} {single_exit} and {name2} {all_exit}: "
                                        f"{round(correlation_matrix[name][name2], 4)}.")
                            logging.warning(warn_msg)
                            warnings.append(f"|{name} {type} {single_exit}"
                                            f"|{name2} {type} {all_exit}"
                                            f"|{round(correlation_matrix[name][name2], 4)}|\n")

                    # Prepare the figure
                    plt.figure(figsize=(12, 10))
                    plt.xticks(size=4)
                    plt.yticks(size=4)
                    plt.title(f'Correlation Matrix - {data_name}')
                    # Color the matrix
                    sn.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, vmin=-0.8, vmax=0.8,
                               annot_kws={"size": 2})

                    # Saving the figure
                    plt.savefig(f'{store_path}{data_name}_CorrMat.png', dpi=300)
                    # Show the figure on video
                    if show:
                        plt.show()
                    plt.close('all')
    return warnings
