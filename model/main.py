# -*- coding: utf-8 -*-
'''
Created on Mon Oct 21 14:18:06 2024

@author: Jakob & Elsa
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
fit = 0.05
el_price = 0.4

model.solve(fit=fit, elec_price=el_price)

#-------------------------------------------------------------------------------#
#                                                                               #
# Process output data                                                           #
#                                                                               #
#-------------------------------------------------------------------------------#
func.output_data(model, 2)
#func.plot_data(results)
func.to_xlsx(model)
