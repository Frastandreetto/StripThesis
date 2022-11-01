#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# This file contains part of the code used in the bachelor thesis of Francesco Andreetto (2020)
# updated to be used on the new version of the software of LSPE-STRIP
# November 1st 2022, Brescia (Italy)

import string
from pathlib import Path


########################################################################################################
# Class for a Polarimeter
########################################################################################################

class Polarimeter:

    def __init__(self, name_pol: string, name_file: Path):

        self.name = name_pol
        self.name_file = name_file

        self.start_time = 0
        self.t = 0
        self.STRIP_SAMPLING_FREQ = 0
        self.norm_mode = 0

        self.data = {}