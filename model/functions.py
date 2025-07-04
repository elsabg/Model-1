# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 22:12:55 2025

@author: Elsa
"""

import numpy as np
import pandas as pd
import os

from openpyxl import load_workbook

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


def to_xlsx(model, fit, elec_price, out_path, multi=1, index='re'):
    assert index == 're' or index == 'budget' or index =='i', 'Wrong search input'
    ############################################################################
    # Import model ranges                                                      #
    ############################################################################
    years = model.years
    days = model.days
    hours = model.hours
    house = model.house
    techs_g = model.techs_g
    techs = model.techs
    voll = model.voll
    interest = model.i
    
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
    # Create summary DataFrame                                                 #
    ############################################################################
    
    # Summary Information
    waste = 0
    total_waste = 0
    unmet_d = 0
    total_ud = 0
    met_d = 0
    house_surplus = pd.DataFrame(columns=['surplus'])
    disc_surplus = 0
    d_weights = list(model.d_weights)
    
    for y in range(model.years):
        feed_in_y = 0
        for d in range(model.days):
            feed_in_y += sum(feed_in.loc[f'{y}.'+f'{d}']) * d_weights[d]
            for h in range(model.hours):
                unmet_d += model.ud[y, d, h].X * d_weights[d]
                met_d += -1 * (model.ud[y, d, h].X) * d_weights[d]
                total_ud += model.ud[y, d, h].X * d_weights[d]
                waste += net_surplus[h][d] * d_weights[d]
                for i in model.house:
                    
                    if model.surplus[i][d][h] >= 0:
                        total_waste += (model.surplus[i][d][h] 
                                        * model.max_house_str[i]
                                        * d_weights[d])
            
                    total_waste += (-1 * model.feed_in[i, y, d, h].X 
                                    * d_weights[d])
                    
                    waste += (-1 * model.feed_in[i, y, d, h].X
                              * d_weights[d])
                    
                    total_ud += (max(-1 * model.surplus[i][d][h], 0)
                                 * (model.max_house_str[i]
                                    - model.h_weight[i, y].X)
                                 * d_weights[d])
                    
                    met_d += (max(-1 * model.surplus[i][d][h], 0)
                              * model.max_house_str[i]
                              * d_weights[d])
        
        house_surplus.loc[y] = (met_d * (voll - elec_price / 100) 
                                + feed_in_y * fit / 100)
        disc_surplus += house_surplus.loc[y][0] * (1 / (1 + interest) ** y)
    
    # Summary DataFrame
    summary_info = [model.i, model.ud_penalty, model.md_level, 
                    model.re_level, model.obj, waste, total_waste,
                    unmet_d, total_ud, model.voll, disc_surplus]
    summary_index = ['Interest Rate', 'Unmet Demand Penalty',
                     'Required Level of Met Demand', 'Minimum Feed-in %',
                     'NPV', 'Wasted Prosumer Surplus', 
                     'Total Wasted Prosumer Surplus',
                     'Unmet Demand', 'Total Unmet Demand', 'VoLL',
                     'Household Surplus']
    summary = pd.DataFrame(summary_info, index = summary_index)
            
    ############################################################################
    # Export to Excel and save in current directory                            #
    ############################################################################
                        
    # Create new folder within current directory for output files
    if multi == 1:
        folder_name = 'Output Files'
        if index == 're':
            index_folder = str(int(model.re_level * 100))
        elif index == 'budget':
            index_folder = str(model.total_budget)
        elif index == 'i':
            index_folder = str(model.i * 100)
            
        folder_path = os.path.join(out_path, folder_name, index_folder)
        os.makedirs(folder_path, exist_ok=True)
    
    else:
        folder_path = out_path
        os.makedirs(folder_path, exist_ok=True)
    
    
    # Write dataframes to excel
    with pd.ExcelWriter(os.path.join(folder_path, 
                                     f"Output_{fit}_{elec_price}.xlsx"), 
                        engine='openpyxl') as writer:
        
        summary.to_excel(writer, sheet_name='Summary')
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
        house_surplus.to_excel(writer, sheet_name='Household Surplus')

def eval_summary(outPath, day_weights, years = 15, days = 3, max_fits=None):
    metrics = pd.DataFrame(columns = ['RE target', 'FiT', 'Price', 
                                      'Unmet Demand', 'Wasted Surplus',
                                      'Household Surplus'])
    metrics.set_index('RE target', inplace=True)
    
    assert len(day_weights) == days, 'Weights do not match days'
    if max_fits != None:
        max_fits_df = pd.read_excel(max_fits, sheet_name=None)
        
    re_levels = os.listdir(outPath)
    
    for re_level in re_levels:
        if re_level == '0':
            max_fits_re = max_fits_df[re_level]
        else:
            max_fits_re = max_fits_df[str(round(float(re_level) / 100 , 1))]
            
        max_fits_re.set_index('Unnamed: 0', inplace=True)
        
        row = max_fits_re.loc['Prices']
        
        files = os.listdir(os.path.join(outPath, re_level))
        
        waste_perc = np.nan
        ud_perc = np.nan
        best_fit = np.nan
        best_el_price = np.nan
        best_surp = 0
            
        for file in files:
            fit = int(file.split('_')[1]) / 100
            price = int(file.split('_')[2].split('.')[0]) / 100
            #print(f'FiT: {fit}, Price: {price}')
            price_col = row[row == round(price, 2)].index.tolist()
            try:
                max_fit = max_fits_re[price_col[0]]['Feed-in Tariffs']
            except IndexError:
                max_fit = 10
                print(f'{price} not in summary')
                
            if fit < max_fit or max_fit == np.nan:
                outFile = os.path.join(outPath, re_level, file)
                summary = pd.read_excel(outFile, sheet_name = None)
                summary['Summary'].set_index('Unnamed: 0', inplace = True)
                surp = summary['Summary'].loc["Household Surplus"][0]
                
                if surp >= best_surp:
                    waste = summary['Summary'].loc["Total Wasted Prosumer Surplus"][0]
                    net_surplus_df = summary['Net surplus']
                    if 'Unnamed: 0' in net_surplus_df.columns:
                        net_surplus_df.set_index('Unnamed: 0', inplace = True)
                    net_surplus = 0
                    for y in range(years):
                        for d in range(days):
                            net_surplus += (net_surplus_df.loc[float(f'{y}.{d}')].sum()
                                            * day_weights[d])
                    
                    waste_perc = waste / net_surplus
                    
                    unmet_demand = summary['Summary'].loc["Total Unmet Demand"][0]
                    demand_df = summary['Yearly demand']
                    demand_df.set_index('Unnamed: 0', inplace = True)
                    demand= 0
                    for y in range(years):
                        for d in range(days):
                            demand += (demand_df.loc[float(f'{y}.{d}')].sum()
                                       * day_weights[d])
                            
                    ud_perc = unmet_demand / - demand
                    
                    best_fit = fit
                    best_el_price = price
                    best_surp = surp
        
        if best_surp == 0:
            best_surp = np.nan
        metrics.loc[re_level] = [best_fit, best_el_price, ud_perc,
                                 waste_perc, best_surp]
                    
    outFile = os.path.join(outPath, '..', 'Evaluation Metrics.xlsx')
    metrics.to_excel(outFile)
    
def change_excel(outFile):
    
    try:
        wb = load_workbook(outFile)
        fit = int(outFile.split('_')[1]) / 100
        el_price = int(outFile.split('_')[2].split('.')[0]) / 100
        sheet = wb['Summary']
        df = pd.read_excel(outFile, sheet_name=None)
        
        d_weights=[199, 106, 60]
    
        unmet_demand = df['Unmet Demand'].set_index('Unnamed: 0')
        df_fi = df['Fed-in Capacity'].set_index('Unnamed: 0')
        df_demand = df['Yearly demand'].set_index('Unnamed: 0')
        household_surplus = 0
        for i in range(0, 15):
            fi_y = 0
            met_d_y = 0
            hs_y = 0
            for j in range(0, 3):
                fi_y += df_fi.loc[float(f'{i}'+'.'+f'{j}')].sum() * d_weights[j]
                met_d_y += (-1 * df_demand.loc[float(f'{i}'+'.'+f'{j}')].sum() * d_weights[j]
                          - unmet_demand.loc[float(f'{i}'+'.'+f'{j}')].sum()) * d_weights[j]
            
            hs_y = met_d_y * (0.7 - el_price) + (fi_y * fit) 
            household_surplus += hs_y * (1 / ((1.1) ** i))
            
        sheet['B12'] = household_surplus
    
        # Save the workbook
        wb.save(outFile)
    
    except:
        print('No work:', outFile)
