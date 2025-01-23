# -*- coding: utf-8 -*-
'''
Created on Tue Oct 21 13:42:03 2024

@author: Jakob & Elsa
'''

import numpy as np
import pandas as pd
import os

def get_dfs(model, t):
    ''' get DataFrames from solved model'''
    
    ret = np.ones((len(model.techs), model.years)) # retired capacity
    inst = np.zeros((len(model.techs), model.years)) # installed capacity
    added = np.zeros((len(model.techs), model.years)) # added capacity
    disp_gen = np.zeros((model.days, model.hours))
    disp_pv = np.zeros((model.days, model.hours))
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
            disp_pv[d][h] = model.disp['Owned PV', t, d, h].X
            bat_in[d][h] = model.b_in[t, d, h].X
            bat_out[d][h] = model.b_out[t, d, h].X
            feed_in_energy[d][h] = sum(model.feed_in[i, t, d, h].X 
                                       for i in model.house)
    for house in model.house:
        for y in range(model.years):
            num_households[model.house.tolist().index(house)][y] = model.h_weight[house, y].X
    
    disp_gen = pd.DataFrame(
        disp_gen, columns=[i for i in range(disp_gen.shape[1])]
    )
    
    disp_pv = pd.DataFrame(
        disp_pv, columns=[i for i in range(disp_gen.shape[1])]
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
            
    return ret, inst, added, disp_gen, disp_pv, bat_in, bat_out, num_households, feed_in


def output_data(model, t=0):
    '''Process output data'''
    ret, inst, added, disp_gen, disp_pv, bat_in, bat_out, num_households, feed_in = get_dfs(model, t)
    
    print('\n-----------installed capacity-----------\n')
    print(inst.round(2))
    print('\n-----------added capacity-----------\n')
    print(added.round(2))
    print('\n-----------retired capacity-----------\n')
    print(ret.round(2))
    print(f'\n-----------dispatched power from DG in year {t}-----------\n')
    print(disp_gen.round(2))
    print(f'\n-----------dispatched power from PV in year {t}-----------\n')
    print(disp_pv.round(2))
    print(f'\n-----------feed in year {t}-----------\n')
    print(feed_in.round(2))
    print(f'\n-----------battery Input year {t}-----------\n')
    print(bat_in.round(2))
    print(f'\n-----------battery Output year {t}-----------\n')
    print(bat_out.round(2))
    print('\n-----------Number of connected household per type-----------\n')
    print(num_households)


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
    
    # Set up additional dataframes
    soc = pd.DataFrame(np.zeros((days * years, hours)))
    feed_in_energy = {f'Type {i+1}': 
               pd.DataFrame(np.zeros((days, hours)))
               for i in range(len(house))}
    total_demand = pd.DataFrame(np.zeros((days, hours)))
    
    # Populate the hourly dataframes
    for y in range(years):
        ret, inst, added, disp_dg, disp_pv, bat_in, bat_out, num_house, feed_in = get_dfs(model, y)
        for d in range(days):
            for h in range(hours):
                soc[h][d] = model.soc[y, d, h].X
                for i in range(len(house)):
                    feed_in_energy[f'Type {i+1}'][h][d] = model.feed_in[f'Type {i+1}', y, d, h].X
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
            bat_in.to_excel(writer, sheet_name='Battery Input ')
            bat_out.to_excel(writer, sheet_name='Battery Output')
            soc.to_excel(writer, sheet_name='State of Charge')
            for i in range(len(house)):
                feed_in_energy[f'Type {i+1}'].to_excel(writer, 
                                                       sheet_name 
                                                       = f'Feed in from Type {i+1}')
            
            # Yearly variables grouped into sheets
            costs.to_excel(writer, sheet_name='Costs and Revenues')
            cap.to_excel(writer, sheet_name='Capacities')
            num_house.to_excel(writer, sheet_name='Connected Households')
            total_demand.to_excel(writer, sheet_name='Yearly demand')
    
    
