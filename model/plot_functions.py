# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 22:18:29 2025

@author: Elsa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as mpatches
from scipy.interpolate import griddata
import seaborn as sns
from matplotlib.lines import Line2D

import os

def rep_day(outFile, year, day, multi=1):
    '''Plot some output data and save to folder'''
    
    sns.set(font_scale=1)
    sns.set_style("whitegrid")
    
    new_plots_folder = os.path.join(outFile, "..")

    if multi == 1:
        re_folder = os.path.basename(os.path.dirname(outFile))
        new_plots_folder = os.path.join(new_plots_folder, "..", "..",
                                        "Daily Generation", re_folder)
        os.makedirs(new_plots_folder, exist_ok=True)

    out = pd.read_excel(outFile, sheet_name=None)

    file = os.path.basename(outFile)
    fit = int(file.split('_')[1])
    el_price = int(file.split('_')[2].split('.')[0])

    cost = out['Costs and Revenues'].set_index('Unnamed: 0')
    house = out['Connected Households'].set_index('Unnamed: 0')
    inst = out['Installed Capacities'].set_index('Unnamed: 0')
    added = out['Added Capacities'].set_index('Unnamed: 0')
    ret = out['Retired Capacities'].set_index('Unnamed: 0')
    disp_dg = out['DG Dispatch'].set_index('Unnamed: 0')
    disp_pv = out['PV Dispatch'].set_index('Unnamed: 0')
    bat_in = out['Battery Input '].set_index('Unnamed: 0')
    bat_out = out['Battery Output'].set_index('Unnamed: 0')
    soc = out['State of Charge'].set_index('Unnamed: 0')
    feed_in = out['Fed-in Capacity'].set_index('Unnamed: 0')
    tot_dem = out['Yearly demand'].set_index('Unnamed: 0')
    net_dem = out['Net demand'].set_index('Unnamed: 0')
    net_sur = out['Net surplus'].set_index('Unnamed: 0')
    ud = out['Unmet Demand'].set_index('Unnamed: 0')
    
    sur = net_sur - feed_in

    fig, ax = plt.subplots(figsize=(12, 8))

    # Representative day
    index = float(f'{year}' + '.' + f'{day}')
    ax.bar(np.arange(24), bat_in.loc[index] * -1,
           width=0.5, label='Battery Input', color='#b1b1b1')
    ax.bar(np.arange(24), disp_dg.loc[index],
           bottom = feed_in.loc[index],
           width=0.5, label='DG', color='#d14b4b')
    ax.bar(np.arange(24), disp_pv.loc[index],
           bottom=disp_dg.loc[index] + feed_in.loc[index],
           width=0.5, label='PV', color='#f9e395')
    ax.bar(np.arange(24), feed_in.loc[index],
           bottom= 0,
           width=0.5, label='Feed in', color='#dbe4ed')
    ax.bar(np.arange(24), bat_out.loc[index],
           bottom=(disp_dg.loc[index]
                   + disp_pv.loc[index] + feed_in.loc[index]),
           width=0.5, label='Battery Output', color='#c2deaf')
    ax.bar(np.arange(24), ud.loc[index],
           bottom=(disp_dg.loc[index]
                   + disp_pv.loc[index] + feed_in.loc[index]
                   + bat_out.loc[index]),
           width=0.5, label='Unmet Demand', color='#f2b382')

    ax.plot(np.arange(24), tot_dem.loc[index].to_numpy() * -1,
            label='Total Demand', color='#595755')
    ax.plot(np.arange(24), sur.loc[index].to_numpy(),
            label='Surplus', linestyle='dashed',
            color='#595755')

    ax.set_xlabel('Hour')
    ax.set_ylabel('Energy')
    '''
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.97), ncol=4,
              bbox_transform=fig.transFigure,
              frameon=False)
    '''
    if multi == 0:
        ax.set_yticks([i for i in range (0, 901, 100)])
    
    plt.subplots_adjust(top=.85)
    plot_path = os.path.join(new_plots_folder, 
                             f"Daily_gen_{fit}_{el_price}.png")
    plt.savefig(plot_path)
    plt.close()
    

def inst_cap(outFile, multi=1):
    '''Plot installed capacities and save to folder'''
    
    sns.set(font_scale=1.15)
    sns.set_style("whitegrid")
    new_plots_folder = os.path.join(outFile, "..")

    if multi == 1:
        re_folder = os.path.basename(os.path.dirname(outFile))
        new_plots_folder = os.path.join(new_plots_folder, "..", "..",
                                        "Installed capacities", re_folder)
        os.makedirs(new_plots_folder, exist_ok=True)

    inst = pd.read_excel(outFile, sheet_name='Installed Capacities')
    inst.set_index('Unnamed: 0', inplace=True)

    file = os.path.basename(outFile)
    fit = int(file.split('_')[1])
    el_price = int(file.split('_')[2].split('.')[0])


    fig, ax = plt.subplots()

    ax.bar(np.arange(15), inst.loc['Diesel Generator'],
           width=0.5, label='Diesel generator', color="#d14b4b")
    ax.bar(np.arange(15), inst.loc['Owned PV'],
           width=0.5, label='Owned PV',
           bottom=inst.loc['Diesel Generator'], color="#f9e395")
    ax.bar(np.arange(15), inst.loc['Owned Batteries'],
           width=0.5, label='Owned batteries',
           bottom=inst.loc['Owned PV'] + inst.loc['Diesel Generator'],
           color="#c2deaf")

    ax.set_xlabel('Year')
    ax.set_ylabel('Capacity installed in kW')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, .94), ncol=3,
              bbox_transform=fig.transFigure, frameon=False)
    
    if multi == 1:
        ax.set_yticks([i for i in range (0, 1001, 250)])
      
    plt.subplots_adjust(top=.85)
    plot_path = os.path.join(new_plots_folder, f"Installed_Capacities_{fit}_{el_price}.png")
    plt.savefig(plot_path)
    plt.close()
    

def gen_year(outFile, multi):
    
    sns.set(font_scale=1.15)
    sns.set_style("whitegrid")
    new_plots_folder = os.path.join(outFile, "..")

    if multi == 1:
        re_folder = os.path.basename(os.path.dirname(outFile))
        new_plots_folder = os.path.join(new_plots_folder, "..", "..",
                                        "Yearly generation", re_folder)
        os.makedirs(new_plots_folder, exist_ok=True)

    out = pd.read_excel(outFile, sheet_name=None)

    file = os.path.basename(outFile)
    fit = int(file.split('_')[1])
    el_price = int(file.split('_')[2].split('.')[0])
    
    disp_dg = out['DG Dispatch'].set_index('Unnamed: 0')
    disp_pv = out['PV Dispatch'].set_index('Unnamed: 0')
    bat_out = out['Battery Output'].set_index('Unnamed: 0')
    bat_in = out['Battery Input '].set_index('Unnamed: 0')
    feed_in = out['Fed-in Capacity'].set_index('Unnamed: 0')
    unmet = out['Unmet Demand'].set_index('Unnamed: 0')
    
    weights = {0: 199,
               1: 106,
               2: 60}
    
    year_dg = []
    for y in range(15):
        tot_dg_y = 0
        for d in range(3):
            tot_dg_y += sum(disp_dg.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
        year_dg.append(tot_dg_y)
        
    year_pv = []
    for y in range(15):
        tot_pv_y = 0
        for d in range(3):
            tot_pv_y += sum(disp_pv.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
        year_pv.append(tot_pv_y)
    
    year_bat_in = []
    for y in range(15):
        tot_bat_in_y = 0
        for d in range(3):
            tot_bat_in_y += sum(bat_in.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
        year_bat_in.append(- tot_bat_in_y)

    year_bat_out = []
    for y in range(15):
        tot_bat_out_y = 0
        for d in range(3):
            tot_bat_out_y += sum(bat_out.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
        year_bat_out.append(tot_bat_out_y)

    year_fed_in = []
    for y in range(15):
        tot_fed_in_y = 0
        for d in range(3):
            tot_fed_in_y += sum(feed_in.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
        year_fed_in.append(tot_fed_in_y)

    year_ud = []
    for y in range(15):
        tot_ud_y = 0
        for d in range(3):
            tot_ud_y += sum(unmet.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
        year_ud.append(tot_ud_y)
        
        
    fig, ax = plt.subplots()
    
    ax.bar(np.arange(15), year_bat_in,
           width=0.5, label='Battery Input', color='#b1b1b1')
    ax.bar(np.arange(15), year_dg,
           bottom = year_fed_in,
           width=0.5, label='DG', color='#d14b4b')
    ax.bar(np.arange(15), year_pv,
           bottom=np.add(year_dg, year_fed_in),
           width=0.5, label='PV', color='#f9e395')
    ax.bar(np.arange(15), year_fed_in,
           bottom= 0,
           width=0.5, label='Feed in', color='#dbe4ed')
    ax.bar(np.arange(15), year_bat_out,
           bottom=np.add(np.add(year_dg, year_fed_in), year_pv),
           width=0.5, label='Battery Output', color='#c2deaf')
    ax.bar(np.arange(15), year_ud,
           bottom=np.add(np.add(np.add(year_dg, year_fed_in), year_pv),
                         year_bat_out),
           width=0.5, label='Unmet Demand', color='#f2b382')
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Energy')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, .97), ncol=3,
              bbox_transform=fig.transFigure, frameon=False)
    
    if multi == 1:
       ax.set_yticks([i for i in range (-1500000, 3000001, 500000)]) 
    
    plt.subplots_adjust(top=.85)
    plot_path = os.path.join(new_plots_folder, 
                             f"Yearly_gen_{fit}_{el_price}.png")
    plt.savefig(plot_path)
    plt.close()
        
def add_ret(outFile, multi):
    
    sns.set(font_scale=1.15)
    sns.set_style("whitegrid")
    new_plots_folder = os.path.join(outFile, "..")

    if multi == 1:
        re_folder = os.path.basename(os.path.dirname(outFile))
        new_plots_folder = os.path.join(new_plots_folder, "..", "..",
                                        "Added and retired capacities",
                                        re_folder)
        os.makedirs(new_plots_folder, exist_ok=True)

    out = pd.read_excel(outFile, sheet_name=None)
    added = out['Added Capacities'].set_index("Unnamed: 0")
    ret = out['Retired Capacities'].set_index("Unnamed: 0")

    file = os.path.basename(outFile)
    fit = int(file.split('_')[1])
    el_price = int(file.split('_')[2].split('.')[0])
    
    fig, ax = plt.subplots()

    ax.bar(np.arange(15), added.loc['Diesel Generator'],
           width=0.5, label='Added diesel generator', color="#d14b4b")
    ax.bar(np.arange(15), added.loc['Owned PV'],
           width=0.5, label='Added PV',
           bottom=added.loc['Diesel Generator'], color="#f9e395")
    ax.bar(np.arange(15), added.loc['Owned Batteries'],
           width=0.5, label='Added batteries',
           bottom=added.loc['Owned PV'] + added.loc['Diesel Generator'],
           color="#c2deaf")
    
    ax.bar(np.arange(15), ret.loc['Diesel Generator'] * -1,
           width=0.5, label='Retired diesel generator', color="#d14b4b",
           hatch='\\')
    ax.bar(np.arange(15), ret.loc['Owned PV'] * -1,
           width=0.5, label='Retired PV',
           bottom=added.loc['Diesel Generator'] * -1, color="#f9e395",
           hatch='\\')
    ax.bar(np.arange(15), ret.loc['Owned Batteries'] * -1,
           width=0.5, label='Retired batteries',
           bottom=-ret.loc['Owned PV'] - ret.loc['Diesel Generator'],
           color="#c2deaf", hatch='\\')
    
    ax.plot(np.arange(15), [0]*15, color='#595755')

    ax.set_xlabel('Year')
    ax.set_ylabel('Capacity added and retired in kW')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, .97), ncol=3,
              bbox_transform=fig.transFigure,
              frameon=False)

    if multi == 1:
        ax.set_yticks([i for i in range (-500, 551, 250)]) 
     
    plt.subplots_adjust(top=.85)
    plot_path = os.path.join(new_plots_folder, f"Add_Ret_Capacities_{fit}_{el_price}.png")
    plt.savefig(plot_path)
    plt.close()
    
def get_npv(casePath):
    
    new_plots_folder = os.path.join(casePath, "NPV.png")
    outFile = os.path.join(casePath, "Output Files")

    files = os.listdir(outFile)

    npvs = []
    fits = []
    prices = []
    
    for file in files:
        fit = int(file.split('_')[1])
        price = int(file.split('_')[2].split('.')[0])
        summary = pd.read_excel(os.path.join(outFile, file), 
                                sheet_name='Summary')
        summary.set_index("Unnamed: 0", inplace=True)
        npv = summary.loc["Household Surplus"]
        npvs.append(npv)
        fits.append(fit)
        prices.append(price)
    
    x_grid = np.linspace(min(prices), max(prices), 5)  # Adjust resolution (50x50 grid)
    y_grid = np.linspace(min(fits), max(fits), 5)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    # Interpolate Z values onto the structured grid
    Z_grid = griddata((prices, fits), npvs, (X_grid, Y_grid), method='cubic')
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis')
    
    ax.set_xlabel("Price")
    ax.set_ylabel("FiT")
    ax.set_zlabel("NPV")
    
    plt.show()
    plt.savefig(new_plots_folder, dpi=300, bbox_inches='tight')
    
    
def fit_v_price(casePath, search='re'):
    sns.set(font_scale=1)
    global keys
    
    new_plots_folder = os.path.join(casePath, "FiTs v Prices.png")
    outFile = os.path.join(casePath, "Summary.xlsx")
    
    out = pd.read_excel(outFile, sheet_name=None)
    fig, ax = plt.subplots()
    colors = ["#595755", "#6d597a", "#DA4167" ,
              "#f2b382", "#f4d35e", "#85a4c4", "#c2deaf" ]
    i = 0
    show_infeasible_label = True
    keys = list(out.keys())
    if search == 're':
        if len(keys) >= 7:
            keys = keys[:7]
            
    elif search == 'budget':
        if len(keys) > 7:
            keys = [str(i) for i in range(500000, 2000001, 250000)]
        elif len(keys) == 7:
            keys = keys[:7]
            
    for key in keys:
        out[key].set_index('Unnamed: 0', inplace=True)
        fits = out[key].loc['Feed-in Tariffs'].to_list()
        unfeas_fits = [fit for fit in fits if fit == 0]
        prices = out[key].loc['Prices'].to_list()
        
        if search == 're':
            label = f'{int(float(key) * 100)}%'
        elif search == 'budget':
            label = f'USD {float(key)/ 10e6} M'
            
        ax.plot(prices[len(unfeas_fits) - 1 ::], fits[len(unfeas_fits) - 1 ::], 
                marker='o', linestyle='-', color=colors[i], 
                zorder=2 if show_infeasible_label else 1,
                label=label)
        ax.scatter(prices[len(unfeas_fits) -1 : len(unfeas_fits)], 
                   unfeas_fits[len(unfeas_fits) - 1 ::], 
                   marker='x', color='red', zorder=3, 
                   label='Infeasible' if show_infeasible_label else "")
        i+=1
        show_infeasible_label = False
    
    plt.axvline(x=0.4, color='gray', linestyle='--', linewidth=1)
    
    ax.set_xlabel('Price in USD')
    ax.set_ylabel('Maximum Feed-in Tariff in USD')
    if search == 're':
        ax.legend(title = 'Minimum Renewable Energy Generation Target',
                  loc='upper center',
                  bbox_to_anchor=(0.5, 1.25),
                  ncol=4,
                  frameon=False)
    elif search == 'budget':
        ax.legend(title = 'Maximum Disocunted Budget',
                  loc='upper center',
                  bbox_to_anchor=(0.5, 1.25),
                  ncol=4,
                  frameon=False)
    
    ax.set_xticks(np.arange(0.25, 0.48, 0.02))
    
    sns.set_style("whitegrid")
    
    plt.subplots_adjust(top=0.75)  # push plot down to fit legend
    plt.tight_layout()

    plt.savefig(new_plots_folder)
    plt.close()
    
    
def surp_heatmap(casePath, re_level, max_fits=None): # summary file
    
    sns.set(font_scale=1.3)
    
    new_plots_folder = os.path.join(casePath, f"Surplus heatmap {re_level}.png")
    filesPath = os.path.join(casePath, 'Output Files', str(int(re_level * 100)))
    files = os.listdir(filesPath)
    
    data = {'Prices ($)': [],
            'FiTs ($)': [],
            'Surpluses': []}
    
    for file in files:
        price = int(file.split('_')[2].split('.')[0]) / 100
        data['Prices ($)'].append(price)
        fit = int(file.split('_')[1]) / 100
        data['FiTs ($)'].append(fit)

        out = pd.read_excel(os.path.join(filesPath, file), sheet_name='Summary')
        out.set_index("Unnamed: 0", inplace=True)
        data['Surpluses'].append(out.loc["Household Surplus"][0])
    
    price_index = list(dict.fromkeys(data['Prices ($)']))
    price_index.sort()
    fit_index = list(dict.fromkeys(data['FiTs ($)']))
    fit_index.sort()
    
    df = pd.DataFrame(data)
    
    heatmap_data = df.pivot(index='FiTs ($)', 
                            columns='Prices ($)', 
                            values='Surpluses')
    
    if max_fits != None:
        max_fits = pd.read_excel(max_fits, sheet_name = str(re_level))
        max_fits.set_index("Unnamed: 0", inplace=True)
        mask = np.zeros_like(heatmap_data, dtype=bool)
        for i, price in enumerate(price_index):
            row = max_fits.loc['Prices']
            global price_col
            price_col = row[row == price].index.tolist()
            global max_fit
            try:
                max_fit = max_fits[price_col[0]]['Feed-in Tariffs']
                max_fit = round(max_fit, 2)
                for j, fit in enumerate(fit_index):
                    if fit > max_fit or max_fit == "Nan" or max_fit == 0:
                        mask[j, i] = True
            except IndexError:
                print(f're_level={re_level}, price={price}')
                
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(heatmap_data / 1000000, fmt=".1f", 
                cmap="YlGnBu", cbar_kws={'label': 'Surplus value (M $)'},
                mask=mask, annot=False)
    
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(new_plots_folder)
    plt.close()
    
def ud_comp(casePaths, re_levels): # re_levels in %
    
    assert type(casePaths)== list, "casePaths should be a list"
    
    new_plots_folder = os.path.join(casePaths[0], '..', 'UD comparison.png')
    
    cases = []
    sns.set(font_scale = 2)
    colors = ["#f4d35e", "#85a4c4", "#c2deaf" ]
    i = 0
    bar_data = {}
    
    name_mapping = {
    'No PV': 'Feed-in Only',
    'No PV w Bat': 'Feed-in + Batteries',
    'With PV': 'Feed-in + PV + Batteries'
    }
    
    bar_data['RE target'] = re_levels
    for casePath in casePaths:
        original_name = os.path.basename(casePath)
        readable_name = name_mapping.get(original_name, original_name)
        cases.append(readable_name)
        case_summary = pd.read_excel(os.path.join(casePath, 
                                                   'Evaluation Metrics.xlsx'))
            
        if len(case_summary['Unmet Demand']) < len(re_levels):
            temp = case_summary['Unmet Demand'] * 100
            temp = list(temp)
            temp += [np.nan] * (len(re_levels) 
                               - len(case_summary['Unmet Demand']))
            bar_data[cases[-1]] = temp
        elif len(case_summary['Unmet Demand']) == len(re_levels):
            bar_data[cases[-1]] = case_summary['Unmet Demand'] * 100
        i += 1
        
    df_data = pd.DataFrame(bar_data)
    df_data_melted = df_data.melt('RE target', var_name='Unmet Demand', 
                                    value_name='Value')
    plt.figure(figsize=(15, 7))
    hue_order = ['Feed-in Only', 
                 'Feed-in + Batteries', 
                 'Feed-in + PV + Batteries']
    color_map = dict(zip(hue_order, colors))
    
    ax = sns.barplot(x='RE target', y='Value', hue='Unmet Demand', 
                     data=df_data_melted, palette=colors, hue_order=hue_order)
    
    re_targets = df_data['RE target'].values
    x_locs = ax.get_xticks()
    width_per_bar = 0.8 / len(hue_order) 


    for i, re_target in enumerate(re_targets):
        for j, case in enumerate(hue_order):
            y_val = df_data[case].iloc[i]
            if pd.isna(y_val):
                x_pos = x_locs[i] - 0.4 + width_per_bar * (j + 0.5)
                ax.scatter(x_pos, -0.1, marker='x', color=color_map[case], 
                           s=100, linewidths=2)
    
    handles, labels = ax.get_legend_handles_labels()
    black_x = Line2D([], [], color='black', marker='x', linestyle='None',
                     markersize=10, label='Infeasible', mew=2)
    handles.append(black_x)
    labels.append('Infeasible')
                
    plt.xlabel('RE Target (%)')
    plt.ylabel('Unmet Demand (%)')
    ax.legend(handles=handles, labels=labels,
              title='Case',
              loc='upper center',
              bbox_to_anchor=(0.5, 1.23),
              ncol=4,
              frameon=False)
    plt.subplots_adjust(top=.85) 
    plt.subplots_adjust(bottom = .13)
    plt.yticks(np.arange(0, 81, 20))
    
    plt.savefig(new_plots_folder)
    plt.close()
    
def ws_comp(casePaths, re_levels): 
    
    assert type(casePaths)== list, "casePaths should be a list"
    
    new_plots_folder = os.path.join(casePaths[0], '..', 'WS comparison.png')
    
    cases = []
    sns.set(font_scale = 2)
    colors = ["#f4d35e", "#85a4c4", "#c2deaf" ]
    i = 1
    bar_data = {}
    
    name_mapping = {
    'No PV': 'Feed-in Only',
    'No PV w Bat': 'Feed-in + Batteries',
    'With PV': 'Feed-in + PV + Batteries'
    }
    
    bar_data['RE target'] = re_levels
    
    for casePath in casePaths:
        original_name = os.path.basename(casePath)
        readable_name = name_mapping.get(original_name, original_name)
        cases.append(readable_name)
        case_summary = pd.read_excel(os.path.join(casePath, 
                                                   'Evaluation Metrics.xlsx'))
        if len(case_summary['Wasted Surplus']) <= len(re_levels):
            temp = case_summary['Wasted Surplus'] * 100
            temp = list(temp)
            temp += [np.nan] * (len(re_levels) 
                               - len(case_summary['Wasted Surplus']))
            bar_data[cases[-1]] = temp
        elif len(case_summary['Wasted Surplus']) == len(re_levels):
            bar_data[cases[-1]] = case_summary['Wasted Surplus'] * 100
            
        bar_data[cases[-1]] = case_summary['Wasted Surplus'] * 100
        i += 1
        
    df_data = pd.DataFrame(bar_data)
    df_data_melted = df_data.melt('RE target', var_name='Wasted Surplus', 
                                    value_name='Value')
    plt.figure(figsize=(15, 7))
    
    hue_order = ['Feed-in Only', 
                 'Feed-in + Batteries', 
                 'Feed-in + PV + Batteries']
    color_map = dict(zip(hue_order, colors))
    
    ax = sns.barplot(x='RE target', y='Value', hue='Wasted Surplus', 
                     data=df_data_melted, palette=colors, hue_order=hue_order)
    
    re_targets = df_data['RE target'].values
    x_locs = ax.get_xticks()
    width_per_bar = 0.8 / len(hue_order) 

    for i, re_target in enumerate(re_targets):
        for j, case in enumerate(hue_order):
            y_val = df_data[case].iloc[i]
            if pd.isna(y_val):
                x_pos = x_locs[i] - 0.4 + width_per_bar * (j + 0.5)
                ax.scatter(x_pos, -0.1, marker='x', color=color_map[case], 
                           s=100, linewidths=2)
    
    handles, labels = ax.get_legend_handles_labels()
    black_x = Line2D([], [], color='black', marker='x', linestyle='None',
                     markersize=10, label='Infeasible', mew=2)
    handles.append(black_x)
    labels.append('Infeasible')

    plt.xlabel('RE Target (%)')
    plt.ylabel('Wasted Surplus (%)')
    ax.legend(handles=handles, labels=labels,
              title='Case',
              loc='upper center',
              bbox_to_anchor=(0.5, 1.23),
              ncol=4,
              frameon=False)
    plt.subplots_adjust(top=.85) 
    plt.subplots_adjust(bottom = .13)
    
    plt.yticks(np.arange(0, 81, 20))
    
    plt.savefig(new_plots_folder)
    plt.close()
    
def re_comp(casePath, comp=None):
    
    sns.set(font_scale=1.5)
    
    new_plots_folder_ud = os.path.join(casePath, 'UD+WS+RE comparison.png')
    new_plots_folder_hs = os.path.join(casePath, 'HS+RE comparison.png')
    new_plots_folder_pf = os.path.join(casePath, 'P+FiT+RE comparison.png')
    
    fig_ud, ax_ud = plt.subplots(figsize=(16, 6))
    fig_hs, ax_hs = plt.subplots()
    fig_pf, ax_pf = plt.subplots()

    tot_dg = []
    tot_pv = []
    tot_fi = []
    tot_ud = []
    tot_bat = []
    pie_handles= []
    i=0
    
    # Get summary from evaluation metrics files
    eval_sum = pd.read_excel(os.path.join(casePath, 
                                          'Grid Search',
                                          'Evaluation Metrics.xlsx'))
    
    eval_sum = eval_sum.sort_values(by='RE target')
    
    # Plot
    x = np.array(eval_sum['RE target'])
    width = 3

    ax_ud.bar(x - width/2, eval_sum['Unmet Demand'] * 100,
              width=width,
              color='#f4d35e',
              label='Unmet Demand')

    ax_ud.bar(x + width/2, eval_sum['Wasted Surplus'] * 100,
              width=width,
              color='#85a4c4',
              label='Wasted Surplus')
    
    y_min, y_max = ax_ud.get_ylim()
    ax_ud.set_ylim(y_min, y_max * 1.5)
    
    if comp != None:
        pie_labels = ["DG", "PV", "Fed-in", "Unmet", "Battery"]
        
        for index, row in eval_sum.iterrows():
            re_level = row['RE target']
            fit_b = eval_sum['FiT'].loc[index]
            fit_b = round(fit_b, 2)
            p_b = round(eval_sum['Price'].loc[index], 2)
            
            
            out = pd.read_excel(os.path.join(casePath,
                                             'Grid Search',
                                             'Output Files',
                                             str(int(re_level)),
                                             f'Output_{int(fit_b*100)}_{int(p_b*100)}.xlsx'),
                                sheet_name=None)
            
            disp_dg = out['DG Dispatch'].set_index('Unnamed: 0')
            disp_pv = out['PV Dispatch'].set_index('Unnamed: 0')
            feed_in = out['Fed-in Capacity'].set_index('Unnamed: 0')
            unmet = out['Unmet Demand'].set_index('Unnamed: 0')
            bat = out['Battery Output'].set_index('Unnamed: 0')
            
            weights_df = out['Summary'].set_index('Unnamed: 0')
            weights = weights_df[0].loc['Day Weights'].split(',')
            weights = [int(w) for w in weights]
            
            year_dg = []
            for y in range(15):
                tot_dg_y = 0
                for d in range(3):
                    tot_dg_y += sum(disp_dg.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
                year_dg.append(tot_dg_y)
            tot_dg.append(sum(year_dg) / 1e6)
            
            year_pv = []
            for y in range(15):
                tot_pv_y = 0
                for d in range(3):
                    tot_pv_y += sum(disp_pv.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
                year_pv.append(tot_pv_y)
            tot_pv.append(sum(year_pv) / 1e6)
            
            year_fed_in = []
            for y in range(15):
                tot_fed_in_y = 0
                for d in range(3):
                    tot_fed_in_y += sum(feed_in.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
                year_fed_in.append(tot_fed_in_y)
            tot_fi.append(sum(year_fed_in) / 1e6)
        
            year_ud = []
            for y in range(15):
                tot_ud_y = 0
                for d in range(3):
                    tot_ud_y += sum(unmet.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
                year_ud.append(tot_ud_y)
            tot_ud.append(sum(year_ud) / 1e6) 
            
            year_bat_out = []
            for y in range(15):
                tot_bat_out_y = 0
                for d in range(3):
                    tot_bat_out_y += sum(bat.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
                year_bat_out.append(tot_bat_out_y)
            tot_bat.append(sum(year_bat_out) / 1e6)
            
            pie_data = [tot_dg[i], tot_pv[i], tot_fi[i],
                        tot_ud[i], tot_bat[i]]
            pie_colors = ["#d14b4b", "#f9e395", "#dbe4ed",
                          "#f2b382", "#b1b1b1"]
            
            # Create an inset axis at data coordinates (x, y_position)
            y_max = max(eval_sum['Unmet Demand'].loc[index] * 100, 
                        eval_sum['Wasted Surplus'].loc[index] * 100)
            inset_ax = ax_ud.inset_axes([re_level - 6, y_max + 3, 9, 9], 
                                        transform=ax_ud.transData) 

            i+=1
    
            inset_ax.pie(pie_data, colors=pie_colors, startangle=90)
            inset_ax.set_aspect('equal')
            inset_ax.axis('off')
            
        pie_handles = [mpatches.Patch(color=c, label=l) for c, l in zip(pie_colors, pie_labels)]
        
        ax_ud.legend(handles=pie_handles,
                     title="Energy (%)",
                     loc="upper right",
                     bbox_to_anchor=(0.5, 1.2))
        
        
    ax_ud.set_xlabel('RE Target (%)')
    ax_ud.set_ylabel('(%)')
    bar_legend = ax_ud.get_legend_handles_labels()
    bar_handles, bar_labels = bar_legend
    
    combined_handles = pie_handles + bar_handles
    combined_labels = pie_labels + bar_labels
    blank_handle = mpatches.Patch(color='none', label='')
    
    ax_ud.legend(combined_handles, combined_labels,
                 title="Pie Legend  |  Bar Legend",
                 loc='upper center',
                 bbox_to_anchor=(0.5, 1.3),
                 ncol=5,
                 frameon=False)
    
    fig_ud.subplots_adjust(top=0.8, left=0.15) 
    
    fig_ud.savefig(new_plots_folder_ud)
    plt.close(fig_ud)
    
    ax_hs.plot(np.array(eval_sum['RE target']),
               np.array(eval_sum['Household Surplus'] / 1e6),
               linewidth = 3,
               color = '#f4d35e',
               label = 'Household Surplus')
    ax_hs.set_xlabel('RE Target (%)')
    ax_hs.set_ylabel('Househol Surplus (M $)')
    ax_hs.legend(loc='upper center',
                 bbox_to_anchor=(0.5, 1.27),
                 ncol=1,
                 frameon=False)
    fig_hs.subplots_adjust(top=0.8, left=0.15) 
    
    fig_hs.savefig(new_plots_folder_hs)
    plt.close(fig_hs)
    
    ax_pf.plot(np.array(eval_sum['RE target']),
               np.array(eval_sum['Price']),
               linewidth = 3,
               color = '#f4d35e',
               label = 'Price')
    ax_pf.plot(np.array(eval_sum['RE target']),
               np.array(eval_sum['FiT']),
               linewidth = 3,
               color = '#85a4c4',
               label = 'FiT')
    ax_pf.set_xlabel('RE Target (%)')
    ax_pf.set_ylabel('Amount ($)')
    ax_pf.legend(loc='upper center',
                 bbox_to_anchor=(0.5, 1.27),
                 ncol=1,
                 frameon=False)
    fig_pf.subplots_adjust(top=0.8, left=0.15) 
    
    fig_pf.savefig(new_plots_folder_pf)
    plt.close(fig_pf)
    
    

def energy_sensitivity(casePath, sensitivity, s_range, re_level):

    assert sensitivity == 'budget' or sensitivity == 'interest', 'Wrong sensitivity parameter'
    try:
        s_range = s_range.tolist()
    except AttributeError:
        pass
    
    sns.set(font_scale=1.15)
    
    # Create new figure
    new_plots_folder = os.path.join(casePath, 
                                    f'Energy {str(int(re_level * 100))}.png')
    fig, ax = plt.subplots()
    
    tot_dg = []
    tot_pv = []
    tot_fi = []
    tot_ud = []
    tot_bat = []
    
    for s in s_range:        
        # find summary of evaluation metrics
        summary_df = pd.read_excel(os.path.join(casePath, 
                                                'Sensitivity',
                                                str(s),
                                                'Evaluation Metrics.xlsx'))
        
        summary_df.set_index('RE target', inplace=True)
        fit_b = summary_df['FiT'].loc[int(re_level * 100)]
        fit_b = round(fit_b, 2)
        p_b = round(summary_df['Price'].loc[int(re_level * 100)], 2)
        
        print(f'fit:{fit_b}, price:{p_b}')
        out = pd.read_excel(os.path.join(casePath,
                                         'Sensitivity',
                                         str(s),
                                         'Output Files',
                                         str(int(re_level * 100)),
                                         f'Output_{int(fit_b*100)}_{int(p_b*100)}.xlsx'),
                            sheet_name=None)
        
        disp_dg = out['DG Dispatch'].set_index('Unnamed: 0')
        disp_pv = out['PV Dispatch'].set_index('Unnamed: 0')
        feed_in = out['Fed-in Capacity'].set_index('Unnamed: 0')
        unmet = out['Unmet Demand'].set_index('Unnamed: 0')
        bat = out['Battery Output'].set_index('Unnamed: 0')
        
        weights_df = out['Summary'].set_index('Unnamed: 0')
        weights = weights_df[0].loc['Day Weights'].split(',')
        weights = [int(w) for w in weights]
        
        year_dg = []
        for y in range(15):
            tot_dg_y = 0
            for d in range(3):
                tot_dg_y += sum(disp_dg.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
            year_dg.append(tot_dg_y)
        tot_dg.append(sum(year_dg) / 1e6)
        
        year_pv = []
        for y in range(15):
            tot_pv_y = 0
            for d in range(3):
                tot_pv_y += sum(disp_pv.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
            year_pv.append(tot_pv_y)
        tot_pv.append(sum(year_pv) / 1e6)
        
        year_fed_in = []
        for y in range(15):
            tot_fed_in_y = 0
            for d in range(3):
                tot_fed_in_y += sum(feed_in.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
            year_fed_in.append(tot_fed_in_y)
        tot_fi.append(sum(year_fed_in) / 1e6)
    
        year_ud = []
        for y in range(15):
            tot_ud_y = 0
            for d in range(3):
                tot_ud_y += sum(unmet.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
            year_ud.append(tot_ud_y)
        tot_ud.append(sum(year_ud) / 1e6) 
        
        year_bat_out = []
        for y in range(15):
            tot_bat_out_y = 0
            for d in range(3):
                tot_bat_out_y += sum(bat.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
            year_bat_out.append(tot_bat_out_y)
        tot_bat.append(sum(year_bat_out) / 1e6)
     
    if sensitivity == 'budget':
        s_range = [i / 1e6 for i in s_range]
    tot_dg = np.array(tot_dg)
    tot_pv = np.array(tot_pv)
    tot_fi = np.array(tot_fi)
    tot_ud = np.array(tot_ud)
    tot_bat = np.array(tot_bat)
    
    ax.bar(s_range,
           tot_dg,
           label = 'Dispatch from DG',
           color = "#d14b4b", width = 0.1
           )
    
    ax.bar(s_range,
           tot_pv,
           label = 'Dispatch from PV',
           color = "#f9e395", width = 0.1,
           bottom = tot_dg)
    
    ax.bar(s_range,
           tot_fi,
           label = 'Fed-in Capacity',
           color = "#c2deaf",  width = 0.1,
           bottom = tot_dg + tot_pv)
    
    ax.bar(s_range,
           tot_ud,
           label = 'Unmet Demand',
           color = "#f2b382", width = 0.1,
           bottom = tot_dg + tot_pv + tot_fi)
    
    ax.bar(s_range,
           tot_bat,
           label = 'Batteries',
           color = "#b1b1b1", width = 0.1,
           bottom = tot_dg + tot_pv + tot_fi + tot_ud)
    
    if sensitivity == 'budget':
        ax.set_xlabel('Budget (USD M)')
    elif sensitivity == 'interest':
        ax.set_xlabel('Interest Rate')
    ax.set_ylabel('Energy (GWh)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, .97), ncol=3,
              bbox_transform=fig.transFigure,
              frameon=False)

    
    plt.subplots_adjust(top=.85)
    
    plt.savefig(new_plots_folder)
    plt.close()


def capacity_sensitivity(casePath, sensitivity, s_range, re_level):

    assert sensitivity == 'budget' or sensitivity == 'interest' or sensitivity == 're_level', 'Wrong sensitivity parameter'
    
    try:
        s_range = s_range.tolist()
    except AttributeError:
        pass
    
    sns.set(font_scale=1.15)
    
    # Create new figure
    new_plots_folder = os.path.join(casePath, 
                                    f'Capacity {str(int(re_level * 100))}.png')
    fig, ax = plt.subplots()
    
    tot_dg = []
    tot_pv = []
    tot_bat = []
    
    for s in s_range:        
        # find summary of evaluation metrics
        summary_df = pd.read_excel(os.path.join(casePath, 
                                                'Sensitivity',
                                                str(s),
                                                'Evaluation Metrics.xlsx'))
        
        summary_df.set_index('RE target', inplace=True)
        fit_b = summary_df['FiT'].loc[int(re_level * 100)]
        fit_b = round(fit_b, 2)
        p_b = round(summary_df['Price'].loc[int(re_level * 100)], 2)
        print(f'fit: {fit_b}, price: {p_b}, s: {s}, re: {re_level}')
        
        out = pd.read_excel(os.path.join(casePath,
                                         'Sensitivity',
                                         str(s),
                                         'Output Files',
                                         str(int(re_level * 100)),
                                         f'Output_{int(fit_b*100)}_{int(p_b*100)}.xlsx'),
                            sheet_name='Installed Capacities')
        out.set_index('Unnamed: 0', inplace=True)
        
        cap_dg = out.loc['Diesel Generator'].mean()
        cap_pv = out.loc['Owned PV'].mean()
        cap_bat = out.loc['Owned Batteries'].mean()
        
        tot_dg.append(cap_dg)
        tot_pv.append(cap_pv)
        tot_bat.append(cap_bat)
     
    if sensitivity == 'budget':
        s_range = [i / 1e6 for i in s_range]
    tot_dg = np.array(tot_dg)
    tot_pv = np.array(tot_pv)
    tot_bat = np.array(tot_bat)
    
    ax.bar(s_range,
           tot_dg,
           label = 'Disel Generator',
           color = "#d14b4b", width = 0.1
           )
    
    ax.bar(s_range,
           tot_pv,
           label = 'PV',
           color = "#f9e395", width = 0.1,
           bottom = tot_dg)
    

    ax.bar(s_range,
           tot_bat,
           label = 'Batteries',
           color = "#b1b1b1", width = 0.1,
           bottom = tot_dg + tot_pv)
    
    if sensitivity == 'budget':
        ax.set_xlabel('Budget (USD M)')
    elif sensitivity == 'interest' or sensitivity == 're_level':
        ax.set_xlabel('Interest Rate')
    ax.set_ylabel('Average Installed Capacity (kW)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, .97), ncol=3,
              bbox_transform=fig.transFigure,
              frameon=False)

    
    plt.subplots_adjust(top=.85)
    
    plt.savefig(new_plots_folder)
    plt.close()
    
def p_FiT_sensitivity(casePath, s_range, re_level):
    sns.set(font_scale=1.15)
    
    re_level = int(re_level * 100)
    # Create new figure
    new_plots_folder = os.path.join(casePath, 
                                    f'Price and FiT {str(re_level)}.png')
    fig, ax = plt.subplots()
    fig, ax = plt.subplots()
    
    prices = []
    fits = []
    
    if type(s_range) == list:
        s_range = np.array(s_range)
    
    for s in s_range:
        re_levels = os.listdir(os.path.join(casePath, 
                                            'Sensitivity',
                                            str(s),
                                            'Output Files'))
        
        # find summary of evaluation metrics
        summary_df = pd.read_excel(os.path.join(casePath, 
                                                'Sensitivity',
                                                str(s),
                                                'Evaluation Metrics.xlsx'))
        summary_df.sort_values(by='RE target')
        summary_df.set_index('RE target', inplace=True)
        
        fit_b = summary_df['FiT'].loc[int(re_level)]
        fit_b = round(fit_b, 2)
        p_b = round(summary_df['Price'].loc[int(re_level)], 2)
        
        prices.append(p_b)
        fits.append(fit_b)
        
    ax.plot(s_range / 1e6, prices, color='#d14b4b', label='Prices')
    ax.plot(s_range / 1e6, fits, color='#f9e395', label='FiTs')        
    ax.set_xlabel('Budget in $M')
    ax.set_ylabel('$')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, .97), ncol=3,
              bbox_transform=fig.transFigure,
              frameon=False)

    
    plt.subplots_adjust(top=.85)
    
    plt.savefig(new_plots_folder)
    plt.close()
    
def sensitivity_v_RE(casePath, sensitivity):
    
    assert sensitivity == 'budget' or sensitivity == 'interest' or sensitivity == 're_level', 'Wrong sensitivity parameter'

    sns.set(font_scale=1.15)
    s_range = os.listdir(os.path.join(casePath, 'Sensitivity'))
    
    # Create new figure
    new_plots_folder = os.path.join(casePath, f'{sensitivity} v RE.png')
    fig, ax = plt.subplots()
    
    if sensitivity == 'budget':
        nameHandle = 'Budget (M$)'
    else:
        nameHandle = 'Interest (%)'
    data = {f'{nameHandle}': [],
        'RE level': [],
        'Surpluses': []}
    
    for s in s_range:
        re_levels = os.listdir(os.path.join(casePath, 
                                            'Sensitivity',
                                            str(s),
                                            'Output Files'))
        re_levels = [float(re) / 100 for re in re_levels]
        
        # find summary of evaluation metrics
        summary_df = pd.read_excel(os.path.join(casePath, 
                                                'Sensitivity',
                                                str(s),
                                                'Evaluation Metrics.xlsx'))
        summary_df.set_index('RE target', inplace=True)
        
        for re_level in re_levels:
            fit_b = summary_df['FiT'].loc[int(re_level * 100)]
            fit_b = round(fit_b, 2)
            p_b = round(summary_df['Price'].loc[int(re_level * 100)], 2)
            
            data['RE level'].append(re_level)
            if sensitivity == 'budget':
                data[f'{nameHandle}'].append(float(s) / 1e6)
            else:
                data[f'{nameHandle}'].append(float(s))
            
            try:
                out = pd.read_excel(os.path.join(casePath,
                                             'Sensitivity',
                                             str(s),
                                             'Output Files',
                                             str(int(re_level * 100)),
                                             f'Output_{int(fit_b*100)}_{int(p_b*100)}.xlsx'),
                                sheet_name='Summary')
                out.set_index('Unnamed: 0', inplace=True)
                
                surplus = out.loc["Household Surplus"][0]
                data['Surpluses'].append(surplus)
                
            except ValueError:
                data['Surpluses'].append(0)
                
    df = pd.DataFrame(data)
    heatmap_data = df.pivot(index=f'{nameHandle}', 
                            columns='RE level', 
                            values='Surpluses')
    
    
    mask = np.zeros_like(heatmap_data, dtype=bool)
    null = df[df['Surpluses'] == 0]
    if len(null) != 0:
        for _, row in null.iterrows():
            null_s = row[f'{nameHandle}']
            null_re = row['RE level']
            
            if null_s in heatmap_data.index and null_re in heatmap_data.columns:
                row_idx = heatmap_data.index.get_loc(null_s)
                col_idx = heatmap_data.columns.get_loc(null_re)
                mask[row_idx, col_idx] = True

    
    sns.heatmap(heatmap_data / 1e6, fmt=".1f", 
                cmap="YlGnBu", cbar_kws={'label': 'Surplus value (M $)'},
                annot=False, mask=mask)
    
    plt.tight_layout()
    plt.savefig(new_plots_folder)
    plt.close()

def re_sensitivity(casePath, re_levels):
    
    sns.set(font_scale=1.15)
    
    new_plots_folder_cap = os.path.join(casePath, 
                                        'Capacity.png')
    new_plots_folder_ene = os.path.join(casePath, 
                                        'Energy.png')
    fig_c, ax_c = plt.subplots()
    fig_e, ax_e = plt.subplots()
    
    tot_dg = []
    tot_d = []
    tot_pv = []
    tot_bat = []
    tot_fi = []
    tot_ud = []
    
    tot_dg_cap = []
    tot_pv_cap = []
    tot_bat_cap = []
    
    summary_df = pd.read_excel(os.path.join(casePath, 
                                            'Grid Search',
                                            'Evaluation Metrics.xlsx'))
    
    summary_df = summary_df.sort_values(by='RE target')
    for index, row in summary_df.iterrows():
        re_level = row['RE target']
        re_level_s = str(int(re_level))
        fit = row['FiT']
        fit_s = str(int(fit * 100))
        p = row['Price']
        p_s = str(int(p * 100))
        
        out = pd.read_excel(os.path.join(casePath, 
                                         'Grid Search',
                                         'Output Files',
                                         re_level_s,
                                         f'Output_{fit_s}_{p_s}.xlsx'),
                            sheet_name=None)
        
        demand = out['Yearly demand'].set_index('Unnamed: 0')
        disp_dg = out['DG Dispatch'].set_index('Unnamed: 0')
        disp_pv = out['PV Dispatch'].set_index('Unnamed: 0')
        feed_in = out['Fed-in Capacity'].set_index('Unnamed: 0')
        unmet = out['Unmet Demand'].set_index('Unnamed: 0')
        bat = out['Battery Output'].set_index('Unnamed: 0')
        
        weights_df = out['Summary'].set_index('Unnamed: 0')
        weights = weights_df[0].loc['Day Weights'].split(',')
        weights = [int(w) for w in weights]
        
        cap = out['Installed Capacities']
        cap.set_index('Unnamed: 0', inplace=True)
        
        cap_dg = cap.loc['Diesel Generator'].mean()
        cap_pv = cap.loc['Owned PV'].mean()
        cap_bat = cap.loc['Owned Batteries'].mean()
        
        tot_dg_cap.append(cap_dg)
        tot_pv_cap.append(cap_pv)
        tot_bat_cap.append(cap_bat)
        
        year_d = []
        for y in range(15):
            tot_d_y = 0
            for d in range(3):
                tot_d_y += sum(demand.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
            year_d.append(tot_d_y)
        tot_d.append(sum(year_d) / 1e6)
        
        year_dg = []
        for y in range(15):
            tot_dg_y = 0
            for d in range(3):
                tot_dg_y += sum(disp_dg.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
            year_dg.append(tot_dg_y)
        tot_dg.append(sum(year_dg) / 1e6)
        
        year_pv = []
        for y in range(15):
            tot_pv_y = 0
            for d in range(3):
                tot_pv_y += sum(disp_pv.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
            year_pv.append(tot_pv_y)
        tot_pv.append(sum(year_pv) / 1e6)
        
        year_fed_in = []
        for y in range(15):
            tot_fed_in_y = 0
            for d in range(3):
                tot_fed_in_y += sum(feed_in.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
            year_fed_in.append(tot_fed_in_y)
        tot_fi.append(sum(year_fed_in) / 1e6)
    
        year_ud = []
        for y in range(15):
            tot_ud_y = 0
            for d in range(3):
                tot_ud_y += sum(unmet.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
            year_ud.append(tot_ud_y)
        tot_ud.append(sum(year_ud) / 1e6) 
        
        year_bat_out = []
        for y in range(15):
            tot_bat_out_y = 0
            for d in range(3):
                tot_bat_out_y += sum(bat.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
            year_bat_out.append(tot_bat_out_y)
        tot_bat.append(sum(year_bat_out) / 1e6)
     
    tot_dg_cap = np.array(tot_dg_cap)
    tot_pv_cap = np.array(tot_pv_cap)
    tot_bat_cap = np.array(tot_bat_cap)
    
    ax_c.bar(re_levels,
            tot_dg_cap,
            label = 'Disel Generator',
            color = "#d14b4b", width = 0.02
            )
    
    ax_c.bar(re_levels,
           tot_pv_cap,
           label = 'PV',
           color = "#f9e395", width = 0.02,
           bottom = tot_dg_cap)
    

    ax_c.bar(re_levels,
           tot_bat_cap,
           label = 'Batteries',
           color = "#b1b1b1", width = 0.02,
           bottom = tot_dg_cap + tot_pv_cap)
    

    ax_c.set_xlabel('RE target')
    ax_c.set_ylabel('Average Installed Capacity (kW)')
    ax_c.legend(loc='upper center', bbox_to_anchor=(0.5, .97), ncol=3,
              bbox_transform=fig_c.transFigure,
              frameon=False)

    fig_c.subplots_adjust(top=.85)
    fig_c.savefig(new_plots_folder_cap)
    plt.close(fig_c)

    tot_dg = np.array(tot_dg)
    tot_pv = np.array(tot_pv)
    tot_fi = np.array(tot_fi)
    tot_ud = np.array(tot_ud)
    tot_bat = np.array(tot_bat)
    tot_d = np.array(tot_d)
    
    ax_e.bar(re_levels,
           tot_dg,
           label = 'Dispatch from DG',
           color = "#d14b4b", width = 0.04
           )
    
    ax_e.bar(re_levels,
           tot_pv,
           label = 'Dispatch from PV',
           color = "#f9e395", width = 0.04,
           bottom = tot_dg)
    
    ax_e.bar(re_levels,
           tot_fi,
           label = 'Fed-in Capacity',
           color = "#c2deaf",  width = 0.04,
           bottom = tot_dg + tot_pv)
    
    ax_e.bar(re_levels,
           tot_ud,
           label = 'Unmet Demand',
           color = "#f2b382", width = 0.04,
           bottom = tot_dg + tot_pv + tot_fi)
    
    ax_e.bar(re_levels,
           tot_bat,
           label = 'Batteries',
           color = "#b1b1b1", width = 0.04,
           bottom = tot_dg + tot_pv + tot_fi + tot_ud)
    
    ax_e.plot(re_levels, tot_d * -1, color="#595755", label='Total Demand')
    
    ax_e.set_xlabel('RE target')
    ax_e.set_ylabel('Energy (GWh)')
    ax_e.legend(loc='upper center', bbox_to_anchor=(0.5, .97), ncol=3,
              bbox_transform=fig_e.transFigure,
              frameon=False)

    
    fig_e.subplots_adjust(top=.85)
    fig_e.savefig(new_plots_folder_ene)
    plt.close(fig_e)
    

def min_v_act_RE(casePath):
    
    sns.set(font_scale=1.5)
    new_plots_folder = os.path.join(casePath, 
                                        'RE target.png')
    fig, ax = plt.subplots()
    
    act_re = []
    act_re_disp = []
    tot_pv = []
    tot_bat_out = []
    tot_bat_in = []
    tot_fi = []
    tot_d = []
    tot_dg = []
    tot_ud = []
    
    re_levels = []
    
    eval_sum = pd.read_excel(os.path.join(casePath, 
                                          'Grid Search',
                                          'Evaluation Metrics.xlsx'))
    eval_sum = eval_sum.sort_values(by='RE target')
    for index, row in eval_sum.iterrows():
        re_level = row['RE target']
        re_levels.append(re_level)
        re_level_s = str(int(re_level))
        fit = row['FiT']
        fit_s = str(int(fit * 100))
        p = row['Price']
        p_s = str(int(p * 100))
        
        out = pd.read_excel(os.path.join(casePath, 
                                         'Grid Search',
                                         'Output Files',
                                         re_level_s,
                                         f'Output_{fit_s}_{p_s}.xlsx'),
                            sheet_name=None)

        demand = out['Yearly demand'].set_index('Unnamed: 0')
        disp_pv = out['PV Dispatch'].set_index('Unnamed: 0')
        feed_in = out['Fed-in Capacity'].set_index('Unnamed: 0')
        bat_out = out['Battery Output'].set_index('Unnamed: 0')
        bat_in = out['Battery Input '].set_index('Unnamed: 0')
        ud = out['Unmet Demand'].set_index('Unnamed: 0')
        disp_dg = out['DG Dispatch'].set_index('Unnamed: 0')
        
        weights_df = out['Summary'].set_index('Unnamed: 0')
        weights = weights_df[0].loc['Day Weights'].split(',')
        weights = [int(w) for w in weights]

        year_d = []
        for y in range(15):
            tot_d_y = 0
            for d in range(3):
                tot_d_y += sum(demand.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
            year_d.append(tot_d_y)
        tot_d.append(sum(year_d))
        
        year_dg = []
        for y in range(15):
            tot_dg_y = 0
            for d in range(3):
                tot_dg_y += sum(disp_dg.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
            year_dg.append(tot_dg_y)
        tot_dg.append(sum(year_dg))
        
        year_ud = []
        for y in range(15):
            tot_ud_y = 0
            for d in range(3):
                tot_ud_y += sum(ud.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
            year_ud.append(tot_ud_y)
        tot_ud.append(sum(year_ud))
        
        year_pv = []
        for y in range(15):
            tot_pv_y = 0
            for d in range(3):
                tot_pv_y += sum(disp_pv.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
            year_pv.append(tot_pv_y)
        tot_pv.append(sum(year_pv))
        
        year_fed_in = []
        for y in range(15):
            tot_fed_in_y = 0
            for d in range(3):
                tot_fed_in_y += sum(feed_in.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
            year_fed_in.append(tot_fed_in_y)
        tot_fi.append(sum(year_fed_in))
        
        year_bat_out = []
        for y in range(15):
            tot_bat_out_y = 0
            for d in range(3):
                tot_bat_out_y += sum(bat_out.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
            year_bat_out.append(tot_bat_out_y)
        tot_bat_out.append(sum(year_bat_out))
        
        year_bat_in = []
        for y in range(15):
            tot_bat_in_y = 0
            for d in range(3):
                tot_bat_in_y += sum(bat_in.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
            year_bat_in.append(tot_bat_in_y)
        tot_bat_in.append(sum(year_bat_in))
        
        act_re.append((sum(year_fed_in) + sum(year_pv)) / 
                      (- sum(year_d) + sum(bat_in) - sum(bat_out)))
        act_re_disp.append((sum(year_fed_in) + sum(year_pv)) / 
                           (sum(year_fed_in) + sum(year_pv) + sum(year_dg)))
     
    ax.plot(re_levels,
            act_re,
            label = 'RE as % of demand',
            linewidth=3)
    ax.plot(re_levels,
            act_re_disp,
            label = 'RE as % of dispatch',
            linewidth=3)
    
    ax.set_xlabel('RE target')
    ax.set_ylabel('Actual RE')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, .97), ncol=2,
              bbox_transform=fig.transFigure,
              frameon=False)

    fig.subplots_adjust(top=.85)
    fig.savefig(new_plots_folder)
    plt.close(fig)
#------------------------------------------------------------------------------#    
# Run the functions for the different cases                                    #
#------------------------------------------------------------------------------#

cwd = os.getcwd()
outFile = os.path.join(cwd, "Outputs")
'''
# Current Case
outFile_0 = os.path.join(outFile, '0. Current Case', 'Output_0_40.xlsx')
add_ret(outFile_1, multi=0)
gen_year(outFile_1, multi=0)
rep_day(outFile_1, multi=0, year=10, day=1)
inst_cap(outFile_1, multi=0)
get_houses(outFile_1, multi=0)
'''

# Baseline
outFile_1 = os.path.join(outFile, '1. Baseline')

#fit_v_price(outFile_1)

outFile_1_1 = os.path.join(outFile_1, 'Grid Search')
outFile_sum = os.path.join(cwd, 'Outputs')
summary_path_1 = os.path.join(outFile_sum, '1. Baseline', 'Summary.xlsx')
re_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
'''
for re_level in re_levels:
    surp_heatmap(outFile_1_1, re_level=re_level, max_fits=summary_path_1)
'''
#re_comp(outFile_1, comp=1)
#re_sensitivity(outFile_1, re_levels)
#min_v_act_RE(outFile_1)

summary_df = pd.read_excel(os.path.join(outFile_1, 
                                        'Grid Search',
                                        'Evaluation Metrics.xlsx'))

summary_df = summary_df.sort_values(by='RE target')
for index, row in summary_df.iterrows():
    re_level = row['RE target']
    re_level_s = str(int(re_level))
    fit = row['FiT']
    fit_s = str(int(fit * 100))
    p = row['Price']
    p_s = str(int(p * 100))
    
    outName = f'Output_{fit_s}_{p_s}.xlsx'
    outPath = os.path.join(outFile_1, 'Grid Search', 'Output Files', re_level_s, outName)
    add_ret(outPath, 1)
    inst_cap(outPath, 1)
    gen_year(outPath, 1)

'''
# Budget
outFile_2 = os.path.join(outFile, '2. Budget')
budgets = np.arange(100000, 2000001, 2000000)
budgets = os.listdir(os.path.join(outFile_2, 'Sensitivity'))
budgets = [int(float(b)) for b in budgets]
budgets.sort()

for budget in budgets:
    fit_v_price(os.path.join(outFile_2, 'Feasible Region', str(budget)))

re_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

for re_level in re_levels:
    energy_sensitivity(outFile_2, 'budget', budgets, re_level)
    capacity_sensitivity(outFile_2, 'budget', budgets, re_level)
    p_FiT_sensitivity(outFile_2, budgets, re_level)
    
sensitivity_v_RE(outFile_2, 'budget')
'''
'''
# Interest
outFile_3 = os.path.join(outFile, '3. Interest')
interests = [5, 10]

for interest in interests:
    fit_v_price(os.path.join(outFile_3, 'Feasible Region', str(interest)))

re_levels = [0, 0.1, 0.2, 0.3, 0.4]
for re_level in re_levels:
    energy_sensitivity(outFile_3, 'interest', interests, re_level)
    capacity_sensitivity(outFile_3, 'interest', interests, re_level)

sensitivity_v_RE(outFile_3, 'interest')
'''