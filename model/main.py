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

model = Model_1(_file_name='model_inputs_testing_v3.xlsx')
model.load_data()

#-------------------------------------------------------------------------------#
#                                                                               #
# model run                                                                     #
#                                                                               #
#-------------------------------------------------------------------------------#
fit = 0.2
el_price = 0.4
heatrate_c_run = 'y'
dem_elasticity_c_run = 'y'
run_model = input("Run model? (Yes: [Enter], No: n):")
if run_model != 'n':
    heatrate_c_run = input("Run model with heatrate curve? (No: [Enter], Yes: y):")
    dem_elasticity_c_run = input("Run model with demand elasticity? (No: [Enter], Yes: y):")

    results = model.solve(fit=fit, elec_price=el_price,
                    heatrate_c_run = heatrate_c_run, dem_elasticity_c_run = dem_elasticity_c_run)

    save_results = input("Save results? (Yes: y, No: [Enter]):")
    if save_results == 'y':
        func.save_results(results)

    (ret, inst, added, disp_gen, disp_pv, disp_feedin,
     unmetD, bat_in, bat_out, state_of_charge, num_households,
     heat_rate_binary, price_binary, quantity_binary, total_demand) = results

    data = {
    'retired_capacity': pd.DataFrame(ret),
    'installed_capacity': pd.DataFrame(inst),
    'added_capacity': pd.DataFrame(added),
    'num_households': pd.DataFrame(num_households),
    'price_binary': pd.DataFrame(price_binary),
    'quantity_binary': pd.DataFrame(quantity_binary)

    }
    for y in range(15):
        data['dispatched_generation_' + str(y + 1)] = pd.DataFrame(disp_gen[y])
        data['dispatched_pv_' + str(y + 1)] =  pd.DataFrame(disp_pv[y])
        data['dispatched_feedin_' + str(y + 1)] =  pd.DataFrame(disp_feedin[y])
        data['unmet_demand_' + str(y + 1)] =  pd.DataFrame(unmetD[y])
        data['battery_input_' + str(y + 1)] =  pd.DataFrame(bat_in[y])
        data['battery_output_' + str(y + 1)] =  pd.DataFrame(bat_out[y])
        data['state_of_charge_' + str(y + 1)] =  pd.DataFrame(state_of_charge[y])
        data['total_demand_' + str(y + 1)] =  pd.DataFrame(total_demand[y])
        #data['price_binary_' + str(y + 1)] = pd.DataFrame(price_binary[y])
        for d in range(3):
            data['heat_rate_binary_' + str(y + 1)+'_'+str(d + 1)] = pd.DataFrame(heat_rate_binary[y][d])

else:
    data = func.get_results('results.xlsx')
#-------------------------------------------------------------------------------#
#                                                                               #
# Process output data                                                           #
#                                                                               #
#-------------------------------------------------------------------------------#
func.show_tables(func.get_tabels(data))
if heatrate_c_run or dem_elasticity_c_run == 'y':
    func.show_binaries(func.get_binaries(data), 1, 2) # winter in year 1

save_plots = input("Save plots? (Yes: y, No: [Enter]):")
if save_plots == 'y':
    for y in range (data['num_households'].shape[1]):
        func.plot_days(func.get_timeseries(data, y), y, 'save')

while(1):
    showyear = input("Year:(1-15):")
    func.plot_days(func.get_timeseries(data, int(showyear)-1), int(showyear)-1)



