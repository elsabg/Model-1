# -*- coding: utf-8 -*-
'''
Created on Mon Oct 21 14:18:06 2024

@author: Jakob
'''

import numpy as np
import pandas as pd
from gurobipy import *

import functions as func

from model_1 import Model_1

#-------------------------------------------------------------------------------#
#                                                                               #
# initialize model                                                              #
#                                                                               #
#-------------------------------------------------------------------------------#

model = Model_1(_file_name='model_inputs_feedin.xlsx')
model.load_data()

#-------------------------------------------------------------------------------#
#                                                                               #
# test model run                                                                #
#                                                                               #
#-------------------------------------------------------------------------------#
fit = 0.01
el_price = 0.4

results, variables = model.solve(fit=fit, elec_price=el_price)

func.output_data(results)
#func.plot_data(results)
func.to_xlsx(model)
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
