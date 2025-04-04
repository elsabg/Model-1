# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 22:08:07 2025

@author: Elsa
"""

import numpy as np
import pandas as pd
import os

import functions as func
from model_1 import Model_1

#------------------------------------------------------------------------------#
#                                                                              #
# initialize model                                                             #
#                                                                              #
#------------------------------------------------------------------------------#

model = Model_1(_file_name='model_inputs_feedin.xlsx')
model.load_data()

#------------------------------------------------------------------------------#
#                                                                              #
# Evaluate base case                                                           #
#                                                                              #
#------------------------------------------------------------------------------#

cwd = os.getcwd()
base_path = os.path.join(cwd, '..', 'Base case', 'Output_0_40.xlsx')
file = pd.read_excel(base_path, sheet_name='Summary')
file.set_index('Unnamed: 0', inplace=True)
base_npv = file.loc['NPV', 0]
#------------------------------------------------------------------------------#
#                                                                              #
# Model run                                                                    #
#                                                                              #
#------------------------------------------------------------------------------#

ud_penalty = 0
md_level = 0
prices = [i / 100 for i in range(1, 51, 1)]
fits = []
objs = [] 

for el_price in prices:
    # Check if there is a positive solution
    fit = 0
    model.solve(fit=fit, elec_price=el_price,
                ud_penalty=ud_penalty, md_level=md_level)
    if model.m.getObjective().getValue() < base_npv:
        print(f'No positive solution for {el_price}')
        fits.append(0)
        objs.append(base_npv)
        
    # If yes, run a binary grid search to find it
    elif len(fits) != 0:
        fit_left = fits[-1]
        fit_right = 100
        fit_mid = (fit_left + fit_right) / 2
        model.m.reset()
        model.solve(fit=fit_mid, elec_price=el_price,
                    ud_penalty=ud_penalty, md_level=md_level)

        while abs(model.m.getObjective().getValue() - base_npv) >= 1000:
            if model.m.getObjective().getValue() >= base_npv:
                fit_left = fit_mid
            else:
                fit_right = fit_mid
            fit_mid = (fit_right + fit_left) / 2
            model.m.reset()
            model.solve(fit=fit_mid, elec_price=el_price,
                        ud_penalty=ud_penalty, md_level=md_level)
       
        fits.append(fit_mid)
        objs.append(model.m.getObjective().getValue())
        func.output_data(model, 2)
        func.to_xlsx(model, int(fit_mid * 100), int(el_price * 100))
        print('positou')
        
    else:
        fit_left = 0
        fit_right = 1
        fit_mid = (fit_left + fit_right) / 2
        model.m.reset()
        model.solve(fit=fit_mid, elec_price=el_price,
                    ud_penalty=ud_penalty, md_level=md_level) 
        while abs(model.m.getObjective().getValue() - base_npv) >= 1000:
            if model.m.getObjective().getValue() >= base_npv:
                fit_left = fit_mid
            else:
                fit_right = fit_mid
            fit_mid = (fit_right + fit_left) / 2
            model.m.reset()
            model.solve(fit=fit_mid, elec_price=el_price,
                        ud_penalty=ud_penalty, md_level=md_level)
                
        fits.append(fit_mid)
        objs.append(model.m.getObjective().getValue())
        func.output_data(model, 2)
        func.to_xlsx(model, int(fit_mid * 100), int(el_price * 100))
            
    
    
#------------------------------------------------------------------------------#
#                                                                              #
# Add summary file                                                             #
#                                                                              #
#------------------------------------------------------------------------------#

summary = pd.DataFrame((prices, fits, [base_npv]*len(fits), objs), 
                       index=['Prices', 'Feed-in Tariffs', 
                              'Base NPV', 'NPV'])
summary.to_excel('Summary.xlsx')