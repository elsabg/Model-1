# -*- coding: utf-8 -*-
'''
Created on Tue Oct 21 13:42:03 2024

@author: Jakob & Elsa
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gurobipy import *
import os

def output_data(resultsArray):
    '''Process output data'''

    ret, inst, added, disp_gen, bat_in, bat_out, num_households, feed_in, total_demand = resultsArray #no unmed demand

    disp_gen = pd.DataFrame(
        disp_gen, columns=[i for i in range(disp_gen.shape[1])]
    )
    
    feed_in = pd.DataFrame(
        feed_in, columns=[i for i in range(feed_in.shape[1])]
    )

    bat_in = pd.DataFrame(
        bat_in, columns=[i for i in range(bat_in.shape[1])]
    )

    bat_out = pd.DataFrame(
        bat_out, columns=[i for i in range(bat_out.shape[1])]
    )

    num_households = pd.DataFrame(
        num_households, columns=[i for i in range(num_households.shape[1])]
    )

    inst = pd.DataFrame(
        inst, columns=[i for i in range(inst.shape[1])]
    )

    added = pd.DataFrame(
        added, columns=[i for i in range(added.shape[1])]
    )

    ret = pd.DataFrame(
        ret, columns=[i for i in range(ret.shape[1])]
    )

    names = ['Diesel Generator', 'Owned PV', 'Owned Batteries']
    inst.index = names
    added.index = names
    ret.index = names


    print('\n-----------installed capacity-----------\n')
    print(inst.round(2))
    print('\n-----------added capacity-----------\n')
    print(added.round(2))
    print('\n-----------retired capacity-----------\n')
    print(ret.round(2))
    print('\n-----------dispatched Energy Generator year 1-----------\n')
    print(disp_gen.round(2))
    print('\n-----------feed in year 1-----------\n')
    print(feed_in.round(2))
    print('\n-----------battery Input year 1-----------\n')
    print(bat_in.round(2))
    print('\n-----------battery Output year 1-----------\n')
    print(bat_out.round(2))
    print('\n-----------Number of connected household types-----------\n')
    print(num_households)
    return


def plot_data(resultsArray):
    '''plot some output data'''

    ret, inst, added, disp_gen, bat_in, bat_out, num_households, feed_in, total_demand = resultsArray #no unmet demand

    fig, ax = plt.subplots()
    ax.bar(np.arange(24), bat_in[0], 0.5, label='Battery Input', color = 'green')
    ax.bar(np.arange(24), bat_out[0], 0.5, label='Battery Output', color = 'red')
    ax.bar(np.arange(24) + 0.5, feed_in[0], 0.5, label='Feed in', color = 'orange')
    ax.plot(np.arange(24), total_demand[0], label='Total Demand', color = 'black')
    ax.plot(np.arange(24), disp_gen['Diesel generator', 0, 0], label='DG', color = 'blue')
    ax.plot(np.arange(24), disp_gen['Owned PV', 0, 0], label='PV', color='magenta')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Energy')
    ax.set_title('Generation and res. Load Profile over a day (year 12)')

    ax.legend()
    plt.show()

def to_xlsx(model):
    #import model ranges
    years = model.years
    days = model.days
    hours = model.hours
    house = model.house
    techs_g = model.techs_g
    techs = model.techs
    
    #Set up yearly dataframes for hourly decision variables
    disp_dg = pd.DataFrame(np.zeros((days, hours)))
    disp_pv = pd.DataFrame(np.zeros((days, hours)))
    bat_in = pd.DataFrame(np.zeros((days, hours)))
    bat_out = pd.DataFrame(np.zeros((days, hours)))
    soc = pd.DataFrame(np.zeros((days, hours)))
    feed_in_1 = pd.DataFrame(np.zeros((days, hours)))
    feed_in_2 = pd.DataFrame(np.zeros((days, hours)))
    feed_in_3 = pd.DataFrame(np.zeros((days, hours)))
    feed_in_4 = pd.DataFrame(np.zeros((days, hours)))
    feed_in_5 = pd.DataFrame(np.zeros((days, hours)))
    
    for y in range(years):
        # Populate the hourly dataframes
        for d in range(days):
            for h in range(hours):
                disp_dg[h][d] = model.disp['Diesel Generator', y, d, h].X
                disp_pv[h][d] = model.disp['Owned PV', y, d, h].X
                bat_in[h][d] = model.b_in[y, d, h].X
                bat_out[h][d] = model.b_out[y, d, h].X
                soc[h][d] = model.soc[y, d, h].X
                feed_in_1[h][d] = model.feed_in['Type 1', y, d, h].X
                feed_in_2[h][d] = model.feed_in['Type 2', y, d, h].X
                feed_in_3[h][d] = model.feed_in['Type 3', y, d, h].X
                feed_in_4[h][d] = model.feed_in['Type 4', y, d, h].X
                feed_in_5[h][d] = model.feed_in['Type 5', y, d, h].X
        
        # Create dataframes of yearly variables
        cost_names = ['Total Revenues',
                      'Total Capital Costs',
                      'Total Operation Variable Costs',
                      'Total Operation Fixed Costs',
                      'Total Profits']
        costs = pd.DataFrame([[model.tr[y].X, model.tcc[y].X, 
                              model.tovc[y].X, model.tofc[y].X, 
                              model.tp[y].X]], 
                             columns=cost_names)
        
        
        cap_cols = ['Added Capacity',
                    'Installed Capacity',
                    'Retired Capacity']
        cap_ind = [g for g in techs]
        cap = pd.DataFrame(np.zeros((len(techs), len(cap_cols))),
                           columns=cap_cols,
                           index=cap_ind)
        for g in techs:
            cap.loc[g, 'Added Capacity'] = model.added_cap[g, y].X
            cap.loc[g, 'Installed Capacity'] = model.inst_cap[g, y].X
            cap.loc[g, 'Retired Capacity'] = model.ret_cap[g, y].X
            

        num_house = pd.DataFrame([[model.h_weight['Type 1', y].X,
                                  model.h_weight['Type 2', y].X,
                                  model.h_weight['Type 3', y].X,
                                  model.h_weight['Type 4', y].X,
                                  model.h_weight['Type 5', y].X]],
                                 columns=house)
                    
        # Create new folder within directory for output files
        folder_name = 'Output Files'
        current_directory = os.getcwd()
        folder_path = os.path.join(current_directory, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        # Write dataframes to excel, with one file per year
        with pd.ExcelWriter(os.path.join(folder_path, f"Year {y}.xlsx"), 
                            engine='openpyxl') as writer:
            
            # Hourly variables, each with its own sheet
            disp_dg.to_excel(writer, sheet_name='DG Dispatch')
            disp_pv.to_excel(writer, sheet_name='PV Dispatch')
            bat_in.to_excel(writer, sheet_name='Battery Input')
            bat_out.to_excel(writer, sheet_name='Battery Output')
            soc.to_excel(writer, sheet_name='State of Charge')
            feed_in_1.to_excel(writer, sheet_name='Feed in from Type 1')
            feed_in_2.to_excel(writer, sheet_name='Feed in from Type 2')
            feed_in_3.to_excel(writer, sheet_name='Feed in from Type 3')
            feed_in_4.to_excel(writer, sheet_name='Feed in from Type 4')
            feed_in_5.to_excel(writer, sheet_name='Feed in from Type 5')
            
            # Yearly variables grouped into sheets
            costs.to_excel(writer, sheet_name='Costs and Revenues')
            cap.to_excel(writer, sheet_name='Capacities')
            num_house.to_excel(writer, sheet_name='Connected Households')
    
    
