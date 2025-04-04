# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 22:08:07 2025

@author: Elsa
"""

import numpy as np
import pandas as pd

import functions as func

from Model_1 import Model_1

#------------------------------------------------------------------------------#
#                                                                              #
# initialize model                                                             #
#                                                                              #
#------------------------------------------------------------------------------#

model = Model_1(_file_name='model_inputs_feedin.xlsx')
model.load_data()

#------------------------------------------------------------------------------#
#                                                                              #
# Model run                                                                    #
#                                                                              #
#------------------------------------------------------------------------------#

ud_penalty = 0
md_level = 0


'''    
for fit in range(1, 42, 5):
    fit = fit / 100
    for el_price in range(1, 42, 5):
        el_price = el_price / 100
        model.solve(fit=fit, elec_price=el_price, 
                    ud_penalty=ud_penalty, md_level = md_level)
        func.output_data(model, 2)
        func.to_xlsx(model, int(fit * 100), 
                     int(el_price * 100))
'''

fit = 0
el_price = 0.4
model.solve(fit=fit, elec_price=el_price, 
            ud_penalty=ud_penalty, md_level = md_level)
func.output_data(model, 2)
func.to_xlsx(model, int(fit * 100), 
             int(el_price * 100))            

#------------------------------------------------------------------------------#
#                                                                              #
# Process output data                                                          #
#                                                                              #
#------------------------------------------------------------------------------#
#func.output_data(model, 2)
