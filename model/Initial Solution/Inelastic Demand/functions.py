# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 22:12:55 2025

@author: Elsa
"""

import numpy as np
import pandas as pd
import os

def get_dfs(model, t):
    ''' get DataFrames from solved model'''
    
    ############################################################################
    # Set up empty arrays                                                      #
    ############################################################################
    cost_names = ['Total Revenues',
                  'Total Capital Costs',
                  'Total Operation Variable Costs',
                  'Total Operation Fixed Costs',
                  'Total Profits']
    
    ret = np.ones((len(model.techs), model.years)) # retired capacity
    inst = np.zeros((len(model.techs), model.years)) # installed capacity
    added = np.zeros((len(model.techs), model.years)) # added capacity
    disp_gen = np.zeros((model.days, model.hours))
    disp_pv = np.zeros((model.days, model.hours))
    bat_in = np.zeros((model.days, model.hours))
    bat_out = np.zeros((model.days, model.hours))
    soc = np.zeros((model.days, model.hours))
    num_households = np.zeros((len(model.house), model.years))
    feed_in_energy = np.zeros((model.days, model.hours))
    costs = np.zeros((len(cost_names), model.years))
    net_demand = np.zeros((model.days, model.hours))
    total_demand = np.zeros((model.days, model.hours))
    net_surplus = np.zeros((model.days, model.hours))
    ud = np.zeros((model.days, model.hours))
    
    ############################################################################
    # Fill arrays from solved model                                            #
    ############################################################################
    
    # One-time DataFrames
    for y in range(model.years):
        for g in model.techs:
            ret[model.techs.tolist().index(g)][y] = model.ret_cap[g, y].X
            inst[model.techs.tolist().index(g)][y] = model.inst_cap[g, y].X
            added[model.techs.tolist().index(g)][y] = model.added_cap[g, y].X
        costs[0][y] = model.tr[y].X
        costs[1][y] = model.tcc[y].X
        costs[2][y] = model.tovc[y].X
        costs[3][y] = model.tofc[y].X
        costs[4][y] = model.tp[y].X

    # Yearly DataFrames
    for d in range(model.days):
        for h in range(model.hours):
            disp_gen[d][h] = model.disp['Diesel Generator', t, d, h].X
            disp_pv[d][h] = model.disp['Owned PV', t, d, h].X
            bat_in[d][h] = model.b_in[t, d, h].X
            bat_out[d][h] = model.b_out[t, d, h].X
            soc[d][h] = model.soc[t, d, h].X
            feed_in_energy[d][h] = sum(model.feed_in[i, t, d, h].X 
                                       for i in model.house)
            net_demand[d][h] = sum(model.surplus[i][d][h]
                                     * model.h_weight[i, t].X
                                     for i in model.house)
            for i in model.house:
                if model.surplus[i][d][h] >= 0:
                    net_surplus[d][h] += (model.surplus[i][d][h] 
                                          * model.h_weight[i, t].X)
                else:
                    total_demand[d][h] += (model.surplus[i][d][h]
                                         * model.h_weight[i, t].X)
            ud[d][h] = model.ud[t, d, h].X
            
    for h in model.house:
        for y in range(model.years):
            num_households[model.house.tolist().index(h)][y] = model.h_weight[h, y].X
    
    ############################################################################
    # Convert arrays to dataframes                                             #
    ############################################################################
    
    disp_gen = pd.DataFrame(
        disp_gen, columns=[i for i in range(model.hours)]
    )
    
    disp_pv = pd.DataFrame(
        disp_pv, columns=[i for i in range(model.hours)]
    )
    feed_in = pd.DataFrame(
        feed_in_energy, columns=[i for i in range(model.hours)]
    )

    bat_in = pd.DataFrame(
        bat_in, columns=[i for i in range(model.hours)]
    )

    bat_out = pd.DataFrame(
        bat_out, columns=[i for i in range(model.hours)]
    )
    
    soc = pd.DataFrame(
        soc, columns=[i for i in range(model.hours)]
        )
    
    net_demand = pd.DataFrame(
        net_demand, columns = [i for i in range(model.hours)])

    total_demand = pd.DataFrame(
        total_demand, columns = [i for i in range(model.hours)])

    net_surplus = pd.DataFrame(
        net_surplus, columns = [i for i in range(model.hours)])
    
    ud = pd.DataFrame(
        ud, columns = [i for i in range(model.hours)])
    
    num_households = pd.DataFrame(
        num_households, columns=[i for i in range(model.years)],
        index = ['Consumers', 'Prosumers']
    )

    inst = pd.DataFrame(
        inst, columns=[i for i in range(model.years)]
    )

    added = pd.DataFrame(
        added, columns=[i for i in range(model.years)]
    )

    ret = pd.DataFrame(
        ret, columns=[i for i in range(model.years)]
    )
    
    costs = pd.DataFrame(
        costs, columns = [i for i in range(model.years)]
        )
    


    # Fix string indices
    inst.index = model.techs.tolist()
    added.index = model.techs.tolist()
    ret.index = model.techs.tolist()
    costs.index = cost_names
    
    ############################################################################
    # Return the DataFrames                                                    #
    ############################################################################
    
    dfs = [ret, inst, added, disp_gen, disp_pv, 
           bat_in, bat_out, num_households, feed_in,
           costs, soc, net_demand, total_demand, net_surplus, ud]
        
    return dfs


def output_data(model, t=0):
    '''Print output data in console'''
    
    dfs = get_dfs(model, t)
    ret = dfs[0]
    inst = dfs[1]
    added = dfs[2]
    disp_gen = dfs[3]
    disp_pv = dfs[4]
    bat_in = dfs[5]
    bat_out = dfs[6]
    num_households = dfs[7]
    feed_in = dfs[8]
    
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


def to_xlsx(model, fit, elec_price):
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
    # Create empty DataFrames for hourly decisions                             #
    ############################################################################
    y_d_index = [f'{y}.'+f'{d}'
                for y in range(years)
                for d in range(days)]
    
    disp_gen = pd.DataFrame(index = y_d_index, 
                            columns = [h for h in range(hours)])
    disp_pv = pd.DataFrame(index = y_d_index, 
                           columns = [h for h in range(hours)])
    bat_in = pd.DataFrame(index = y_d_index, 
                          columns = [h for h in range(hours)])
    bat_out = pd.DataFrame(index = y_d_index, 
                           columns = [h for h in range(hours)])
    soc = pd.DataFrame(index = y_d_index, 
                       columns = [h for h in range(hours)])
    feed_in = pd.DataFrame(index = y_d_index, 
                           columns = [h for h in range(hours)])
    net_demand = pd.DataFrame(index = y_d_index, 
                                columns = [h for h in range(hours)])
    total_demand = pd.DataFrame(index = y_d_index,
                              columns = [h for h in range(hours)])
    net_surplus = pd.DataFrame(index = y_d_index,
                              columns = [h for h in range(hours)])
    ud = pd.DataFrame(index=y_d_index,
                      columns = [h for h in range(hours)])
    
    ############################################################################
    # Import One-time DataFrames                                               #
    ############################################################################

    dfs = get_dfs(model, 0)
    ret = dfs[0]
    inst = dfs[1]
    added = dfs[2]
    num_households = dfs[7]
    costs = dfs[9]
    
    ############################################################################
    # Populate yearly DataFrames                                               #
    ############################################################################
    
    for y in range(0, years):
        dfs = get_dfs(model, y)
        disp_gen_y = dfs[3]
        disp_pv_y = dfs[4]
        bat_in_y = dfs[5]
        bat_out_y = dfs[6]
        feed_in_y = dfs[8]
        soc_y = dfs[10]
        net_demand_y = dfs[11]
        total_demand_y = dfs[12]
        net_surplus_y = dfs[13]
        ud_y = dfs[14]
        
        for d in range(days):
            disp_gen.loc[f'{y}.'+f'{d}'] = disp_gen_y.loc[d]
            disp_pv.loc[f'{y}.'+f'{d}'] = disp_pv_y.loc[d]
            bat_in.loc[f'{y}.'+f'{d}'] = bat_in_y.loc[d]
            bat_out.loc[f'{y}.'+f'{d}'] = bat_out_y.loc[d]
            feed_in.loc[f'{y}.'+f'{d}'] = feed_in_y.loc[d]
            soc.loc[f'{y}.'+f'{d}'] = soc_y.loc[d]
            total_demand.loc[f'{y}.'+f'{d}'] = total_demand_y.loc[d]
            net_demand.loc[f'{y}.'+f'{d}'] = net_demand_y.loc[d]
            net_surplus.loc[f'{y}.'+f'{d}'] = net_surplus_y.loc[d]
            ud.loc[f'{y}.'+f'{d}'] = ud_y.loc[d]
            
    ############################################################################
    # Export to Excel and save in current directory                            #
    ############################################################################
                        
    # Write output file in current working directory    
    with pd.ExcelWriter(f"Output_{fit}_{elec_price}.xlsx", 
                        engine='openpyxl') as writer:
        
        costs.to_excel(writer, sheet_name='Costs and Revenues')
        num_households.to_excel(writer, sheet_name='Connected Households')
        inst.to_excel(writer, sheet_name='Installed Capacities')
        added.to_excel(writer, sheet_name='Added Capacities')
        ret.to_excel(writer, sheet_name='Retired Capacities')
        disp_gen.to_excel(writer, sheet_name='DG Dispatch')
        disp_pv.to_excel(writer, sheet_name='PV Dispatch')
        bat_in.to_excel(writer, sheet_name='Battery Input ')
        bat_out.to_excel(writer, sheet_name='Battery Output')
        soc.to_excel(writer, sheet_name='State of Charge')
        feed_in.to_excel(writer, sheet_name='Fed-in Capacity')
        total_demand.to_excel(writer, sheet_name='Yearly demand')
        net_demand.to_excel(writer, sheet_name='Net demand')
        net_surplus.to_excel(writer, sheet_name='Net surplus')
        ud.to_excel(writer, sheet_name='Unmet Demand')