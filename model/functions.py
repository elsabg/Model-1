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

def output_data(model, t=0):
    '''Process output data'''
    
    ret = np.ones((len(model.techs), model.years)) # retired capacity
    inst = np.zeros((len(model.techs), model.years)) # installed capacity
    added = np.zeros((len(model.techs), model.years)) # added capacity
    disp_gen = np.zeros((model.days, model.hours))
    bat_in = np.zeros((model.days, model.hours))
    bat_out = np.zeros((model.days, model.hours))
    num_households = np.ones((len(model.house), model.years))
    feed_in_energy = np.zeros((model.days, model.hours))

    for y in range(model.years):
        for g in model.techs:
            ret[model.techs.tolist().index(g)][y] = model.ret_cap[g, y].X
            inst[model.techs.tolist().index(g)][y] = model.inst_cap[g, y].X
            added[model.techs.tolist().index(g)][y] = model.added_cap[g, y].X

    for d in range(model.days):
        for h in range(model.hours):
            disp_gen[d][h] = model.disp['Diesel Generator', t, d, h].X
            bat_in[d][h] = model.b_in[t, d, h].X
            bat_out[d][h] = model.b_out[t, d, h].X
            feed_in_energy[d][h] = sum(model.feed_in[i, t, d, h].X 
                                       for i in model.house)
    for house in model.house:
        for y in range(model.years):
            num_households[model.house.tolist().index(house)][y] = model.h_weight[house, y].X

    disp_gen = pd.DataFrame(
        disp_gen, columns=[i for i in range(disp_gen.shape[0])]
    )
    
    feed_in = pd.DataFrame(
        feed_in_energy, columns=[i for i in range(feed_in_energy.shape[1])]
    )

    bat_in = pd.DataFrame(
        bat_in, columns=[i for i in range(bat_in.shape[1])]
    )

    bat_out = pd.DataFrame(
        bat_out, columns=[i for i in range(bat_out.shape[1])]
    )
    '''
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

    names = ['Diesel Generator', 'Owned PV', 'Batteries (kw)', 'Batteries Capacity (kWh)']
    inst.index = names
    added.index = names
    ret.index = names
    names_housetypes = ['Consumer Residential', 'Prosumer Residential', 'Solar Farm', 'Water Pumping Station', 'Prosumer Business']
    num_households.index = names_housetypes


    print(f'\n-----------installed capacity-----------\n')
    print(inst.round(2))
    print(f'\n-----------added capacity-----------\n')
    print(added.round(2))
    print(f'\n-----------retired capacity-----------\n')
    print(ret.round(2))
    print(f'\n-----------dispatched power from DG year {t}-----------\n')
    print(disp_gen.round(2))
    print(f'\n-----------feed in year {t}-----------\n')
    print(feed_in.round(2))
    print(f'\n-----------battery Input year {t}-----------\n')
    print(bat_in.round(2))
    print(f'\n-----------battery Output year {t}-----------\n')
    print(bat_out.round(2))
    print(f'\n-----------Number of connected household per type-----------\n')
    print(num_households)


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

    # Stack the other bars on top
    p2 = ax1.bar(hours, bat_out, bottom=disp_gen, label='Battery Output', color='red')
    p3 = ax1.bar(hours, feed_in, bottom=disp_gen + bat_out, label='Feed in', color='orange')
    p4 = ax1.bar(hours, unmetD, bottom=disp_gen + bat_out + feed_in, label='Unmet Demand', color='purple')
    p5 = ax1.bar(hours, -bat_in, label='Battery Input', color='green')

    ax1.plot(hours, total_demand[2], label='Total Demand', color='black', linestyle='-', marker='o')

    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Energy')
    ax1.set_title('Stacked Bar Chart of Energy Data over a Winter Day (Year 10)')
    ax1.legend(loc='upper left')

    ax1.legend()

    fig2, ax2 = plt.subplots()
    ax2.plot(hours, state_of_charge, label='State of Charge', color='blue', linestyle='-', marker='o')

    q1 = ax2.bar(hours, bat_in, label='Battery Input', color='green')
    p2 = ax2.bar(hours, -bat_out, label='Battery Output', color='red')
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('State of Charge')
    ax2.set_title('State of Charge over a Winter Day (Year 10)')
    ax2.legend(loc='upper left')

    #plt.tight_layout()
    plt.show()

def to_xlsx(model):
    ############################################################################
    # Import model ranges                                                      #
    ############################################################################
    years = model.years
    days = model.days
    hours = model.hours
    house = model.house
    techs_g = model.techs_g
    techs = model.techs
    
    ############################################################################
    # Create yearly dataframes for hourly decision variables                   #
    ############################################################################
    
    # Set up the dataframes
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
    total_demand = pd.DataFrame(np.zeros((days, hours)))
    
    # Populate the hourly dataframes
    for y in range(years):
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
                total_demand[h][d] = sum(model.surplus[i][d][h]
                                         * model.h_weight[i, y].X
                                         for i in house)
                
        ############################################################################
        # Create dataframes for yearly decision variables                          #
        ############################################################################
        
        cost_names = ['Total Revenues',
                      'Total Capital Costs',
                      'Total Operation Variable Costs',
                      'Total Operation Fixed Costs',
                      'Total Profits']
        label_names = ['Diesel Generator',
                       'PV',
                       'Batteries',
                       'Feed-in',
                       'Total']
        costs = pd.DataFrame([[model.tr[y].X, model.tcc[y].X, 
                              model.tovc[y].X, model.tofc[y].X, 
                              model.tp[y].X]], 
                             columns=cost_names)
        
        # Yearly capacities
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
            
        
        # Yearly connected households
        num_house = pd.DataFrame([[model.h_weight['Type 1', y].X,
                                  model.h_weight['Type 2', y].X,
                                  model.h_weight['Type 3', y].X,
                                  model.h_weight['Type 4', y].X,
                                  model.h_weight['Type 5', y].X]],
                                 columns=house)

        ############################################################################
        # Export to Excel and save in current directory                        #
        ############################################################################
                            
        # Create new folder within current directory for output files
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
            total_demand.to_excel(writer, sheet_name='Yearly demand')
    
    
