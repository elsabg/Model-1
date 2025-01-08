# -*- coding: utf-8 -*-
'''
Created on Mon Oct 21 14:18:06 2024

@author: jakobsvolba
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

model = Model_1(_file_name='model_inputs_testing_v4.xlsx')
model.load_data()


lcoe_pv = func.calc_lcoe_pv('model_inputs_testing_v4.xlsx')
fit = lcoe_pv
ud_penalty = 0.1
el_price = 0.39

#-------------------------------------------------------------------------------#
#                                                                               #
# single model run                                                              #
#                                                                               #
#-------------------------------------------------------------------------------#
run_model = input("Single Model Run? (No: [Enter],Yes: [y], Only show results: [r]):")
if run_model == 'y':
    heatrate_c_run = input("Run model with heatrate curve? (No: [Enter], Yes: [y]):")
    dem_elasticity_c_run = input("Run model with demand elasticity? (No: [Enter], Yes: [y]):")

    data = func.single_modelrun(model, fit, el_price, ud_penalty, heatrate_c_run, dem_elasticity_c_run)
    func.show_singlerun_data(data)
elif run_model == 'r':
    data = func.get_results('results.xlsx')
    func.show_singlerun_data(data)

#-------------------------------------------------------------------------------#
#                                                                               #
# multiple model runs                                                           #
#                                                                               #
#-------------------------------------------------------------------------------#
data_weights = pd.read_excel('model_inputs_testing_v3.xlsx', sheet_name='day_weights')
day_weights = data_weights['Weight'].to_numpy()

num_runs = 30
ud_penalty_max = 0.5
feedin_max = 0.1

ud_runs = input("Unmet Demand multiple Model runs? (No: [Enter],Yes: [y], Only show results: [r]):")

if ud_runs == 'y':
    func.ud_modelruns(model, ud_penalty_max, num_runs, el_price, fit, day_weights)

elif ud_runs == 'r':
    func.print_ud_curve(ud_penalty_max, num_runs, 1)


pv_fit_runs = input("PV Fit multiple Model runs? (No: [Enter],Yes: [y], Only show results: [r]):")

if pv_fit_runs == 'y':
    func.pv_fit_modelruns(model, feedin_max, num_runs, el_price, ud_penalty, day_weights)
elif pv_fit_runs == 'r':
    func.print_pv_fit_curve(feedin_max, num_runs, 1)








