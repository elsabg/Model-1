# -*- coding: utf-8 -*-
'''
Created on Mon Oct 21 14:18:06 2024

@author: jakobsvolba
'''

import numpy as np

import functions as func

from model_1_draft import Model_1

#-------------------------------------------------------------------------------#
#                                                                               #
# initialize model                                                              #
#                                                                               #
#-------------------------------------------------------------------------------#

model = Model_1(_file_name='model_inputs.xlsx')

#-------------------------------------------------------------------------------#
#                                                                               #
# test model run                                                                #
#                                                                               #
#-------------------------------------------------------------------------------#

func.test_run_model_case_1(model)

#-------------------------------------------------------------------------------#
#                                                                               #
# model run case 1: PV capacity rent                                            #
#                                                                               #
#-------------------------------------------------------------------------------#

#read max_govTariff out of 'model_inputs.xlsx'
maxgovtariff = 0.5
maxpvrent = 50

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
