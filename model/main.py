# -*- coding: utf-8 -*-
'''
Created on Mon Oct 21 14:18:06 2024

@author: jakobsvolba
'''

import numpy as np
import pandas as pd

import functions as func

from model_1 import Model_1

#-------------------------------------------------------------------------------#
#                                                                               #
# initialize model                                                              #
#                                                                               #
#-------------------------------------------------------------------------------#

model = Model_1(_file_name='model_inputs_testing_v2.xlsx')
model.load_data()

#-------------------------------------------------------------------------------#
#                                                                               #
# test model run                                                                #
#                                                                               #
#-------------------------------------------------------------------------------#
fit = 0.2
el_price = 0.4
heatrate_c_run = input("Run model with heatrate curve? (No: [Enter], Yes: y):")
results = model.solve(fit=fit, elec_price=el_price, heatrate_c_run = heatrate_c_run)

func.output_data(results)
func.plot_data(results)
#-------------------------------------------------------------------------------#
#                                                                               #
# model run case 2: feed in tariff                                              #
#                                                                               #
#-------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------#
#                                                                               #
# Process output data                                                           #
#                                                                               #
#-------------------------------------------------------------------------------#
