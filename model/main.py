# -*- coding: utf-8 -*-
'''
Created on Mon Oct 21 14:18:06 2024

@author: Jakob & Elsa
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

model = Model_1(_file_name='model_inputs_feedin.xlsx')
model.load_data()

#-------------------------------------------------------------------------------#
#                                                                               #
# test model run                                                                #
#                                                                               #
#-------------------------------------------------------------------------------#

# Single run
if input('Grid Search? y/n: ') == 'n':
    best_fit = 0.01
    best_el_price = 0.4
    model.solve(fit=best_fit, elec_price=best_el_price)
    best_model = model
 
# Grid search
else:
    best_model = 0
    best_model_obj = 0
    best_fit = 0
    best_el_price = 0
    
    for fit in range(0, 11):
        for el_price in range(1, 41):
            fit = fit / 100
            el_price = el_price / 100
            model.solve(fit=fit, elec_price=el_price)
            if model.m.getObjective().getValue() > best_model_obj:
                best_model = model
                best_fit = fit
                best_el_price = el_price
                best_model_obj = model.m.getObjective().getValue()

#-------------------------------------------------------------------------------#
#                                                                               #
# Process output data                                                           #
#                                                                               #
#-------------------------------------------------------------------------------#
func.output_data(best_model, 2)
#func.plot_data(results)
if input("Save results as xlsx? {y/n} -") == "y":
    func.to_xlsx(best_model, int(best_fit * 100), int(best_el_price * 100))
