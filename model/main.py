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
               md_level=0, ud_penalty=0, re_level=0, voll=0.7):
    global model
    model = Model_1(_file_name=in_path)
    model.load_data()
    model.solve(fit=fit, elec_price=elec_price, 
                md_level = md_level, ud_penalty=ud_penalty, 
                re_level=re_level, voll=voll)
    func.output_data(model, 2)
    func.to_xlsx(model, int(fit * 100), int(elec_price * 100), 
                 out_path, multi=0)

def multi_run(in_path, fits, elec_prices, out_path,
              md_level=0, ud_penalty=0, re_level=0, voll=0.7):
    for elec_price in elec_prices:
        for fit in fits:
            model = Model_1(_file_name=in_path)
            model.load_data()
            model.solve(fit=fit, elec_price=elec_price, 
                        md_level = md_level, ud_penalty=ud_penalty, 
                        re_level=re_level, voll=voll)
            func.output_data(model, 2)
            func.to_xlsx(model, int(fit * 100), int(elec_price * 100), 
                         out_path)    

def fit_search(in_path, out_path, prices,
               md_level=0, ud_penalty=0, re_level=0, voll=0.7):
    
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
    global fits
    global fit_mid
    global fit_right
    global fit_left
    
    output_files = os.listdir(out_path)
    
    if "Summary.xlsx" in output_files:
        prev_summary = pd.read_excel(os.path.join(out_path, "Summary.xlsx"),
                                     sheet_name=None)
        last_re = float(list(prev_summary.keys())[-1])
        
        if last_re == re_level:
            last_re_summary = prev_summary[str(last_re)].set_index('Unnamed: 0')
            fits = last_re_summary.loc['Feed-in Tariffs'].dropna()
            fits = list(fits)
            objs = last_re_summary.loc['NPV'].dropna()
            objs = list(objs)
            
        elif last_re < re_level:
            fits = []
            objs = []
            
        else:
            return
        
    else:
        fits = []
        objs = []
    
    fits = []
    objs = []
    for el_price in prices[len(fits)::]:
        # Check if there is a positive solution
        fit = 0
        model.solve(fit=fit, elec_price=el_price,
                    ud_penalty=ud_penalty, md_level=md_level,
                    re_level=re_level, voll=voll)
        if model.m.getObjective().getValue() < base_npv:
            print(f'No positive solution for {el_price}')
            fits.append(0)
            objs.append(base_npv)
            
            summary = pd.DataFrame((prices, fits, [base_npv]*len(fits), objs), 
                                   index=['Prices', 'Feed-in Tariffs', 
                                          'Base NPV', 'NPV'])
            try:
                with pd.ExcelWriter(os.path.join(out_path, 'Summary.xlsx'), 
                                    mode='a', engine='openpyxl', 
                                    if_sheet_exists='replace') as writer:
                    summary.to_excel(writer, sheet_name=str(re_level))
                
            except FileNotFoundError:
                with pd.ExcelWriter(os.path.join(out_path, 'Summary.xlsx'), 
                                    mode='w', engine='openpyxl') as writer:
                    summary.to_excel(writer, sheet_name=str(re_level))
            
        # If yes, run a binary grid search to find it
        elif len(fits) != 0:
            fit_left = fits[-1]
            fit_right = 4
            fit_mid = (fit_left + fit_right) / 2
            model.m.reset()
            worse = True
            print(f'-----FiT: {fit_mid}, price: {el_price}-----')
            model.solve(fit=fit_mid, elec_price=el_price,
                        ud_penalty=ud_penalty, md_level=md_level, 
                        re_level=re_level, voll=voll)
    
            while worse:
                model_feed_in = sum(model.feed_in[i, y, d, h].X 
                                    for (i, y, d, h) in model.feed_in.keys())
                model_obj = model.m.getObjective().getValue()
                
                if model_feed_in <= 5000 and model_obj >= base_npv:
                    fit_mid = 'inf'
                    break
                
                if (fit_right - fit_mid) < 0.01 and model_obj >= base_npv:
                    print("------------breaking-----------------")
                    break
                
                if model.m.getObjective().getValue() >= base_npv:
                    fit_left = fit_mid
                else:
                    fit_right = fit_mid
                fit_mid = (fit_right + fit_left) / 2
                model.m.reset()
                print(f'-----FiT: {fit_mid}, price: {el_price}-----')
                model.solve(fit=fit_mid, elec_price=el_price,
                            ud_penalty=ud_penalty, md_level=md_level, 
                            re_level=re_level, voll=voll)
                if (abs(model.m.getObjective().getValue() - base_npv) >= 10000
                    and model.m.getObjective().getValue() >= base_npv):
                    worse = False
                
            if fit_mid == 'inf':
                break
            
            fits.append(fit_mid)
            objs.append(model.m.getObjective().getValue())
            func.output_data(model, 2)
            func.to_xlsx(model, int(fit_mid * 100), int(el_price * 100), 
                         os.path.join(out_path))
            
            summary = pd.DataFrame((prices, fits, [base_npv]*len(fits), objs), 
                                   index=['Prices', 'Feed-in Tariffs', 
                                          'Base NPV', 'NPV'])
            try:
                with pd.ExcelWriter(os.path.join(out_path, 'Summary.xlsx'), 
                                    mode='a', engine='openpyxl', 
                                    if_sheet_exists='replace') as writer:
                    summary.to_excel(writer, sheet_name=str(re_level))
                
            except FileNotFoundError:
                with pd.ExcelWriter(os.path.join(out_path, 'Summary.xlsx'), 
                                    mode='w', engine='openpyxl') as writer:
                    summary.to_excel(writer, sheet_name=str(re_level))
            
        else:
            fit_left = 0
            fit_right = 4
            fit_mid = (fit_left + fit_right) / 2
            model.m.reset()
            worse = True
            print(f'-----FiT: {fit_mid}, price: {el_price}-----')
            model.solve(fit=fit_mid, elec_price=el_price,
                        ud_penalty=ud_penalty, md_level=md_level,
                        re_level = re_level, voll=voll)
            
            while worse:
                model_feed_in = sum(model.feed_in[i, y, d, h].X 
                                    for (i, y, d, h) in model.feed_in.keys())
                model_obj = model.m.getObjective().getValue()
                
                if model_feed_in <= 5000 and model_obj >= base_npv:
                    fit_mid = 'inf'
                    break
                
                if (fit_right - fit_mid) < 0.01 and model_obj >= base_npv:
                    break
                
                if model.m.getObjective().getValue() >= base_npv:
                    fit_left = fit_mid
                else:
                    fit_right = fit_mid
                fit_mid = (fit_right + fit_left) / 2
                model.m.reset()
                print(f'-----FiT: {fit_mid}, price: {el_price}-----')
                model.solve(fit=fit_mid, elec_price=el_price,
                            ud_penalty=ud_penalty, md_level=md_level, 
                            re_level=re_level, voll=voll)
                if (abs(model.m.getObjective().getValue() - base_npv) >= 10000
                    and model.m.getObjective().getValue() >= base_npv):
                    worse = False
                    
            fits.append(fit_mid)
            objs.append(model.m.getObjective().getValue())
            func.output_data(model, 2)
            func.to_xlsx(model, int(fit_mid * 100), int(el_price * 100), 
                         out_path)
        
            
            summary = pd.DataFrame((prices, fits, [base_npv]*len(fits), objs), 
                                   index=['Prices', 'Feed-in Tariffs', 
                                          'Base NPV', 'NPV'])
            try:
                with pd.ExcelWriter(os.path.join(out_path, 'Summary.xlsx'), 
                                    mode='a', engine='openpyxl', 
                                    if_sheet_exists='replace') as writer:
                    summary.to_excel(writer, sheet_name=str(re_level))
                
            except FileNotFoundError:
                with pd.ExcelWriter(os.path.join(out_path, 'Summary.xlsx'), 
                                    mode='w', engine='openpyxl') as writer:
                    summary.to_excel(writer, sheet_name=str(re_level))

cwd = os.getcwd()
day_weights = [199, 106, 60]
#### OR (after running the model at least once) ####
#list(model.d_weights)
####################################################


outFile_sum = os.path.join(cwd, 'Outputs')
summary_path_1 = os.path.join(outFile_sum, '3. With PV', 'Summary.xlsx')
summary_path_2 = os.path.join(outFile_sum, '2. No PV', 'Summary.xlsx')
summary_path_3 = os.path.join(outFile_sum, '9. No PV w Bat', 'Summary.xlsx')

'''
# Initial Solution
in_path = os.path.join(cwd, 'Inputs', 'model_inputs_inelas.xlsx')
out_path = os.path.join(cwd, 'Outputs', '0. Initial Solution')

single_run(in_path=in_path, fit=0, elec_price=0.4, out_path=out_path, 
           re_level = 0)

outFile = os.path.join(out_path, 'Output_0_40.xlsx')
func.change_excel(outFile)
'''
# Base Case
in_path = os.path.join(cwd, 'Inputs', 'model_inputs_inelas_noFI_noPV.xlsx')
out_path = os.path.join(cwd, 'Outputs', '1. Base Case')
single_run(in_path=in_path, fit=0, elec_price=0.4, out_path=out_path)
'''
re_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

# FiT search with no PV
prices = np.arange(0, 0.5, 0.01)
for re_level in re_levels:
    in_path = os.path.join(cwd, 'Inputs', 'model_inputs_inelas_noPV.xlsx')
    out_path = os.path.join(cwd, 'Outputs', '2. No PV')
    fit_search(in_path, out_path, prices, re_level=re_level)

# FiT search with PV
prices = np.arange(0, 0.5, 0.01)
for re_level in re_levels:
    in_path = os.path.join(cwd, 'Inputs', 'model_inputs_inelas.xlsx')
    out_path = os.path.join(cwd, 'Outputs', '3. With PV')
    fit_search(in_path, out_path, prices, re_level=re_level)

# FiT search with no PV but with Batteries
prices = np.arange(0, 0.5, 0.01)
for re_level in re_levels:
    in_path = os.path.join(cwd, 'Inputs', 'model_inputs_inelas_NoPV_wBat.xlsx')
    out_path = os.path.join(cwd, 'Outputs', '9. No PV w Bat')
    fit_search(in_path, out_path, prices, re_level=re_level)

'''
# Fixed RE
in_path = os.path.join(cwd, 'Inputs', 'model_inputs_inelas_NoPV.xlsx')
out_path = os.path.join(cwd, 'Outputs', '8. Fixed RE', 'No PV')
fits = np.arange(0, 0.3, 0.02)
elec_prices = np.arange(0.35, 0.49, 0.01)
re_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
'''
for re_level in re_levels:
    multi_run(in_path=in_path, fits=fits, elec_prices=elec_prices, 
              out_path=out_path, re_level=re_level)

for re_level in re_levels:
    re_path = os.path.join(out_path, 'Output Files', str(int(re_level * 100)))
    files = os.listdir(re_path)
    for file in files:
        func.change_excel(os.path.join(re_path, file))

func.eval_summary(os.path.join(out_path, 'Output Files'), 
                  day_weights, max_fits = summary_path_2)   
'''
in_path = os.path.join(cwd, 'Inputs', 'model_inputs_inelas.xlsx')
out_path = os.path.join(cwd, 'Outputs', '8. Fixed RE', 'With PV')
fits = np.arange(0, 0.07, 0.01)
elec_prices = np.arange(0.27, 0.32, 0.01)
re_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
'''
for re_level in re_levels:
    multi_run(in_path=in_path, fits=fits, elec_prices=elec_prices, 
              out_path=out_path, re_level=re_level)

for re_level in re_levels:
    re_path = os.path.join(out_path, 'Output Files', str(int(re_level * 100)))
    files = os.listdir(re_path)
    for file in files:
        func.change_excel(os.path.join(re_path, file))

func.eval_summary(os.path.join(out_path, 'Output Files'), 
                  day_weights, max_fits = summary_path_1)   
'''
in_path = os.path.join(cwd, 'Inputs', 'model_inputs_inelas_NoPV_wBat.xlsx')
out_path = os.path.join(cwd, 'Outputs', '8. Fixed RE', 'No PV w Bat')

fits = np.arange(0, 0.31, 0.02)
elec_prices = np.arange(0.29, 0.46, 0.02)
re_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
'''
for re_level in re_levels:
    multi_run(in_path=in_path, fits=fits, elec_prices=elec_prices, 
              out_path=out_path, re_level=re_level)

for re_level in re_levels:
    re_path = os.path.join(out_path, 'Output Files', str(int(re_level * 100)))
    files = os.listdir(re_path)
    for file in files:
        func.change_excel(os.path.join(re_path, file))

func.eval_summary(os.path.join(out_path, 'Output Files'),
                  day_weights, max_fits = summary_path_3)   
'''