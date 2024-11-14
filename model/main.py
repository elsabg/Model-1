# -*- coding: utf-8 -*-
'''
Created on Mon Oct 21 14:18:06 2024

@author: jakobsvolba
'''

import numpy as np
import pandas as pd

import functions as func

from model_1_draft import Model_1

#-------------------------------------------------------------------------------#
#                                                                               #
# initialize model                                                              #
#                                                                               #
#-------------------------------------------------------------------------------#

model = Model_1(_file_name='model_inputs_testing.xlsx')
model.load_data()

#-------------------------------------------------------------------------------#
#                                                                               #
# test model run                                                                #
#                                                                               #
#-------------------------------------------------------------------------------#
pv_rent = 50
el_price = 0.5

results = model.solve(rent=pv_rent, elec_price=el_price)

func.output_data(results)
#-------------------------------------------------------------------------------#
#                                                                               #
# model run case 1: PV capacity rent                                            #
#                                                                               #
#-------------------------------------------------------------------------------#

#read max_govTariff and max pvrent out of 'model_inputs.xlsx'
'''max_values = pd.read_excel('model_inputs.xlsx', decimal=',', sheet_name='tariffs')
maxgovtariff = max_values['Ministry Tariff'][0] # write maxgovtariff and maxpvrent in model inputs
maxpvrent = max_values['PV Rent'][0]
'''
#func.run_model_case_1(model, maxgovtariff, maxpvrent)

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
