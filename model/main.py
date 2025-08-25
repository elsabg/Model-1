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
               md_level=0, ud_penalty=0, re_level=0, 
               voll=0.7, total_budget=np.inf, interest=0.1):
    
    global model
    os.makedirs(out_path, exist_ok=True)
    
    model = Model_1(_file_name=in_path)
    model.load_data()
    model.solve(fit=fit, elec_price=elec_price, 
                md_level = md_level, ud_penalty=ud_penalty, 
                re_level=re_level, voll=voll, total_budget=total_budget,
                interest=interest)
    func.output_data(model, 2)
    func.to_xlsx(model, round(fit * 100), round(elec_price * 100), 
                 out_path, multi=0)

def multi_run(in_path, fits, elec_prices, out_path,
              md_level=0, ud_penalty=0, re_level=0, 
              voll=0.7, total_budget=np.inf, index='budget', interest=0.1):
    
    global model
    os.makedirs(out_path, exist_ok=True)
    
    for elec_price in elec_prices:
        for fit in fits:
            model = Model_1(_file_name=in_path)
            model.load_data()
            model.solve(fit=fit, elec_price=elec_price, 
                        md_level = md_level, ud_penalty=ud_penalty, 
                        re_level=re_level, voll=voll, 
                        total_budget=total_budget, interest=interest)
            func.output_data(model, 2)
            func.to_xlsx(model, round(fit * 100), round(elec_price * 100), 
                         out_path, 1, index)    

def fit_search(in_path, out_path, prices,
               md_level=0, ud_penalty=0, re_level=0, voll=0.7,
               total_budget=np.inf, search='budget', interest=0.1):
    
    os.makedirs(out_path, exist_ok=True)
    index = search
    
    # Initialize model
    model = Model_1(_file_name=in_path)
    model.load_data()
    
    
    # Define base case
    base_path = os.path.join(os.getcwd(), "Outputs",
                             "0. Current Case", "Output_0_40.xlsx")
    file = pd.read_excel(base_path, sheet_name='Summary')
    file.set_index('Unnamed: 0', inplace=True)
    base_npv = file.loc['NPV', 0]

    # Grid search
    global fits
    global fit_mid
    global fit_right
    global fit_left
    
    try:
        output_files = os.listdir(out_path)
        
        if "Summary.xlsx" in output_files:
            prev_summary = pd.read_excel(os.path.join(out_path, "Summary.xlsx"),
                                         sheet_name=None)
            last_re = list(prev_summary.keys())[-1]
            
            if index == 're':
                if float(last_re) == re_level:
                    last_re_summary = prev_summary[
                        last_re
                        ].set_index('Unnamed: 0')
                    fits = last_re_summary.loc['Feed-in Tariffs'].dropna()
                    fits = list(fits)
                    objs = last_re_summary.loc['NPV'].dropna()
                    objs = list(objs)
                    
                elif float(last_re) < re_level:
                    fits = []
                    objs = []
                    
                else:
                    return
                
            else:
                fits = []
                objs = []
                
        else:
            fits = []
            objs = []
                
    except FileNotFoundError:
        fits = []
        objs = []

    for el_price in prices[len(fits)::]:
        # Check if there is a positive solution
        fit = 0
        model.solve(fit=fit, elec_price=el_price,
                    ud_penalty=ud_penalty, md_level=md_level,
                    re_level=re_level, voll=voll, 
                    total_budget=total_budget, interest=interest)
        
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
                    if index == 're':
                        summary.to_excel(writer, sheet_name=str(re_level))
                        
                    elif index == 'budget':
                        summary.to_excel(writer, 
                                         sheet_name=str(total_budget))
                    elif index == 'i':
                        summary.to_excel(writer, 
                                         sheet_name=str(interest))
                
            except FileNotFoundError:
                with pd.ExcelWriter(os.path.join(out_path, 'Summary.xlsx'), 
                                    mode='w', engine='openpyxl') as writer:
                    if index == 're':
                        summary.to_excel(writer, sheet_name=str(re_level))
                        
                    elif index == 'budget':
                        summary.to_excel(writer, 
                                         sheet_name=str(total_budget))
                    elif index == 'i':
                        summary.to_excel(writer, 
                                         sheet_name=str(interest))
                        
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
                        re_level=re_level, voll=voll, 
                        total_budget=total_budget, interest=interest)
    
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
                            re_level=re_level, voll=voll,
                            total_budget=total_budget, interest=interest)
                if (abs(model.m.getObjective().getValue() - base_npv) <= 10000
                    and model.m.getObjective().getValue() >= base_npv):
                    worse = False
                
            if fit_mid == 'inf':
                break
            
            fits.append(fit_mid)
            objs.append(model.m.getObjective().getValue())
            func.output_data(model, 2)
            func.to_xlsx(model, round(fit_mid * 100), round(el_price * 100), 
                         os.path.join(out_path), index=search)
            
            summary = pd.DataFrame((prices, fits, [base_npv]*len(fits), objs), 
                                   index=['Prices', 'Feed-in Tariffs', 
                                          'Base NPV', 'NPV'])
            try:
                with pd.ExcelWriter(os.path.join(out_path, 'Summary.xlsx'), 
                                    mode='a', engine='openpyxl', 
                                    if_sheet_exists='replace') as writer:
                    if index == 're':
                        summary.to_excel(writer, sheet_name=str(re_level))
                        
                    elif index == 'budget':
                        summary.to_excel(writer, 
                                         sheet_name=str(total_budget))
                    elif index == 'i':
                        summary.to_excel(writer, 
                                         sheet_name=str(interest))
                        
                
            except FileNotFoundError:
                with pd.ExcelWriter(os.path.join(out_path, 'Summary.xlsx'), 
                                    mode='w', engine='openpyxl') as writer:
                    if index == 're':
                        summary.to_excel(writer, sheet_name=str(re_level))
                        
                    elif index == 'budget':
                        summary.to_excel(writer, 
                                         sheet_name=str(total_budget))
                    elif index == 'i':
                        summary.to_excel(writer, 
                                         sheet_name=str(interest))
            
        else:
            fit_left = 0
            fit_right = 4
            fit_mid = (fit_left + fit_right) / 2
            model.m.reset()
            worse = True
            print(f'-----FiT: {fit_mid}, price: {el_price}-----')
            model.solve(fit=fit_mid, elec_price=el_price,
                        ud_penalty=ud_penalty, md_level=md_level,
                        re_level = re_level, voll=voll,
                        total_budget=total_budget, interest=interest)
            
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
                            re_level=re_level, voll=voll,
                            total_budget=total_budget, interest=interest)
                if (abs(model.m.getObjective().getValue() - base_npv) <= 10000
                    and model.m.getObjective().getValue() >= base_npv):
                    worse = False
                
            if fit_mid == 'inf':
                break
                    
            fits.append(fit_mid)
            objs.append(model.m.getObjective().getValue())
            func.output_data(model, 2)
            func.to_xlsx(model, round(fit_mid * 100), round(el_price * 100), 
                         out_path, index=search)
        
            
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
                    if index == 're':
                        summary.to_excel(writer, sheet_name=str(re_level))
                        
                    elif index == 'budget':
                        summary.to_excel(writer, 
                                         sheet_name=str(total_budget))
                    elif index == 'i':
                        summary.to_excel(writer, 
                                         sheet_name=str(interest))

cwd = os.getcwd()
day_weights = [199, 106, 60]
#### OR (after running the model at least once) ####
#list(model.d_weights)
####################################################


outFile_sum = os.path.join(cwd, 'Outputs')
'''
# Current Case
in_path = os.path.join(cwd, 'Inputs', 'inputs.xlsx')
out_path = os.path.join(cwd, 'Outputs', '0. Current Case')
single_run(in_path=in_path, fit=0, elec_price=0.4, out_path=out_path,
           total_budget=np.inf)
'''

# Current budget
current_path = os.path.join(cwd, 
                            'Outputs', 
                            '0. Current Case', 
                            'Output_0_40.xlsx')
current_df = pd.read_excel(current_path, sheet_name=None)
current_cfs = current_df['Costs and Revenues'].set_index('Unnamed: 0')
interest = current_df['Summary'].set_index('Unnamed: 0').loc['Interest Rate'][0]

current_budget = 0
for y in range(len(current_cfs.columns)):
    capex_y = current_cfs.loc['Total Capital Costs'][y]
    current_budget += capex_y * (1 / (1 + interest) ** y)

# Baseline Model
in_path = os.path.join(cwd, 'Inputs', 'inputs_RE.xlsx')
out_path = os.path.join(cwd, 'Outputs', '1. Baseline')

#   Creating Bugdet Range
budgets = np.arange(250000, 2000001, 250000)
budgets = budgets.tolist()
budgets.insert(current_budget % 1e5 * 1e5)
budgets.remove(500000)


prices = np.arange(0, 0.41, 0.01)
prices_gs = np.arange(0, 0.41, 0.01)
fits = np.arange(0, 0.26, 0.01)

out_path_gs = os.path.join(cwd, 'Outputs', '1. Baseline', 'Grid Search')

for budget in budgets:
    fit_search(in_path, out_path, prices, re_level=0,
               total_budget=budget, search='budget')
    multi_run(in_path=in_path, fits=fits, elec_prices=prices_gs, 
              out_path=out_path_gs, re_level=0, 
              total_budget=budget)

summary_path_1 = os.path.join(outFile_sum, '1. Baseline', 'Summary.xlsx')
func.eval_summary(os.path.join(cwd, 'Outputs', '1. Baseline', 
                               'Grid Search', 'Output Files'),
                  max_fits = summary_path_1)


# RE Sensitivity
in_path = os.path.join(cwd, 'Inputs', 'inputs_RE.xlsx')
out_path = os.path.join(cwd, 'Outputs', '2. Budget')

re_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
prices = np.arange(0, 0.40, 0.01)
prices_gs = np.arange(0, 0.3, 0.01)
fits = np.arange(0, 0.3, 0.01)

for budget in budgets:

    out_path_gs = os.path.join(out_path, 'Sensitivity', str(budget))
    out_path_fr = os.path.join(out_path, 'Feasible Region', str(budget))

    for re_level in re_levels:
        fit_search(in_path, out_path_fr, prices, re_level=re_level,
                   total_budget=budget, search='re')
        multi_run(in_path=in_path, fits=fits, elec_prices=prices_gs, 
                  out_path=out_path_gs, re_level=re_level, 
                  total_budget=budget, index='re')

for budget in budgets:
    summary_path_2_b = os.path.join(outFile_sum, '2. Budget', 'Feasible Region',
                                    str(budget), 'Summary.xlsx')
    
    func.eval_summary(os.path.join(cwd, 'Outputs', '2. Budget',
                                   'Sensitivity', str(budget), 'Output Files'),
                      max_fits = summary_path_2_b)