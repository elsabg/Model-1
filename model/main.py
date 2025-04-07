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

def single_run(in_path, fit, elec_price, out_path,
               md_level=0, ud_penalty=0):
    model = Model_1(_file_name=in_path)
    model.load_data()
    model.solve(fit=fit, elec_price=elec_price, 
                md_level = md_level, ud_penalty=ud_penalty)
    func.output_data(model, 2)
    func.to_xlsx(model, int(fit * 100), int(elec_price * 100), 
                 out_path, multi=0)

def multi_run(in_path, fits, elec_prices, out_path,
              md_level=0, ud_penalty=0):
    for elec_price in elec_prices:
        for fit in fits:
            model = Model_1(_file_name=in_path)
            model.load_data()
            model.solve(fit=fit, elec_price=elec_price, 
                        md_level = md_level, ud_penalty=ud_penalty)
            func.output_data(model, 2)
            func.to_xlsx(model, int(fit * 100), int(elec_price * 100), 
                         out_path)    

def fit_search(in_path, out_path,
               md_level=0, ud_penalty=0):
    
    # Initialize model
    model = Model_1(_file_name=in_path)
    model.load_data()
    
    
    # Define base case
    base_path = os.path.join(os.getcwd(), "Outputs",
                             "1. Base Case", "Output_0_40.xlsx")
    file = pd.read_excel(base_path, sheet_name='Summary')
    file.set_index('Unnamed: 0', inplace=True)
    base_npv = file.loc['NPV', 0]


    # Grid search
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
            func.to_xlsx(model, int(fit_mid * 100), int(el_price * 100), out_path)
            
        else:
            fit_left = 0
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
            func.to_xlsx(model, int(fit_mid * 100), int(el_price * 100), out_path)
            
    summary = pd.DataFrame((prices, fits, [base_npv]*len(fits), objs), 
                           index=['Prices', 'Feed-in Tariffs', 
                                  'Base NPV', 'NPV'])
    summary.to_excel(os.path.join(out_path, 'Summary.xlsx'))           



cwd = os.getcwd()

# Initial Solution
in_path = os.path.join(cwd, 'Inputs', 'model_inputs_inelas.xlsx')
out_path = os.path.join(cwd, 'Outputs', '0. Initial Solution')
single_run(in_path=in_path, fit=0, elec_price=0.4, out_path=out_path)

# Base Case
in_path = os.path.join(cwd, 'Inputs', 'model_inputs_inelas_noFI_noPV.xlsx')
out_path = os.path.join(cwd, 'Outputs', '1. Base Case')
single_run(in_path=in_path, fit=0, elec_price=0.4, out_path=out_path)

# FiT search with no PV
in_path = os.path.join(cwd, 'Inputs', 'model_inputs_inelas_noPV.xlsx')
out_path = os.path.join(cwd, 'Outputs', '2. No PV')
fit_search(in_path, out_path)

# FiT search with no PV
in_path = os.path.join(cwd, 'Inputs', 'model_inputs_inelas.xlsx')
out_path = os.path.join(cwd, 'Outputs', '3. With PV')
fit_search(in_path, out_path)