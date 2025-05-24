# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 22:18:29 2025

@author: Elsa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import seaborn as sns

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
    
    new_plots_folder = os.path.join(outFile, "..")

    if multi == 1:
        re_folder = os.path.basename(os.path.dirname(outFile))
        new_plots_folder = os.path.join(new_plots_folder, "..", "..",
                                        "Installed capaities", re_folder)
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
              bbox_transform=fig.transFigure)
    
    if multi == 1:
        ax.set_yticks([i for i in range (0, 751, 250)])
      
    plt.subplots_adjust(top=.85)
    plot_path = os.path.join(new_plots_folder, f"Installed_Capacities_{fit}_{el_price}.png")
    plt.savefig(plot_path)
    plt.close()
    

def get_houses(outFile, multi=1):
    
    new_plots_folder = os.path.join(outFile, "..")

    if multi == 1:
        re_folder = os.path.basename(os.path.dirname(outFile))
        new_plots_folder = os.path.join(new_plots_folder, "..", "..",
                                        "Connected households", re_folder)
        os.makedirs(new_plots_folder, exist_ok=True)

    out = pd.read_excel(outFile, sheet_name="Connected Households")
    out.set_index('Unnamed: 0', inplace=True)

    file = os.path.basename(outFile)
    fit = int(file.split('_')[1])
    el_price = int(file.split('_')[2].split('.')[0])
    
    fig, ax = plt.subplots()

    max_house = {"Consumers": [635] * 15,"Prosumers": [440] * 15}
    colors = {"Consumers": '#f2b382', "Prosumers": '#85a4c4'}
    for house in out.index:
        ax.plot(out.loc[house], label=house, color=colors[house])
        ax.plot(max_house[house], linestyle='dashed', 
                label=f'Maximum {house}', color=colors[house])
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Connected households')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, .97), ncol=2,
              bbox_transform=fig.transFigure)
    
    ax.set_yticks([i for i in range (0, 701, 100)])  
    
    plt.subplots_adjust(top=.85)
    plot_path = os.path.join(new_plots_folder, 
                             f"Connected_households_{fit}_{el_price}.png")
    plt.savefig(plot_path)
    plt.close()

def gen_year(outFile, multi):
    
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
    
    weights = {0: 91,
               1: 153,
               2: 121}
    
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
              bbox_transform=fig.transFigure)
    
    if multi == 1:
       ax.set_yticks([i for i in range (0, 3500001, 500000)]) 
    
    plt.subplots_adjust(top=.85)
    plot_path = os.path.join(new_plots_folder, 
                             f"Yearly_gen_{fit}_{el_price}.png")
    plt.savefig(plot_path)
    plt.close()
        
def add_ret(outFile, multi):
    
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
              bbox_transform=fig.transFigure)

    if multi == 1:
        ax.set_yticks([i for i in range (-500, 751, 250)]) 
     
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
    
    
def fit_v_price(casePath):
    sns.set(font_scale=1.15)
    
    new_plots_folder = os.path.join(casePath, "FiTs v Prices.png")
    outFile = os.path.join(casePath, "Summary.xlsx")
    
    out = pd.read_excel(outFile, sheet_name=None)
    fig, ax = plt.subplots()
    colors = ["#595755", "#6d597a", "#DA4167" ,
              "#f2b382", "#f4d35e", "#85a4c4", "#c2deaf" ]
    i = 0
    show_infeasible_label = True
    keys = list(out.keys())
    if len(keys) >= 7:
        keys = keys[:7]
        
    for key in keys:
        out[key].set_index('Unnamed: 0', inplace=True)
        fits = out[key].loc['Feed-in Tariffs'].to_list()
        unfeas_fits = [fit for fit in fits if fit == 0]
        prices = out[key].loc['Prices'].to_list()
    
        ax.plot(prices[len(unfeas_fits) - 1 ::], fits[len(unfeas_fits) - 1 ::], 
                marker='o', linestyle='-', color=colors[i], 
                zorder=2 if show_infeasible_label else 1,
                label=f'{int(float(key) * 100)}%')
        ax.scatter(prices[len(unfeas_fits) -1 : len(unfeas_fits)], 
                   unfeas_fits[len(unfeas_fits) - 1 ::], 
                   marker='x', color='red', zorder=3, 
                   label='Infeasible' if show_infeasible_label else "")
        i+=1
        show_infeasible_label = False
    
    plt.axvline(x=0.4, color='gray', linestyle='--', linewidth=1)
    
    ax.set_xlabel('Price in USD')
    ax.set_ylabel('Maximum Feed-in Tariff in USD')
    ax.legend(title = 'Minimum Renewable Energy Generation Target',
              loc='upper center',
              bbox_to_anchor=(0.5, 1.25),
              ncol=4,
              frameon=False)
    
    ax.set_xticks(np.arange(0.3, 0.5, 0.02))
    
    sns.set_style("whitegrid")
    
    plt.subplots_adjust(top=0.75)  # push plot down to fit legend
    plt.tight_layout()

    plt.savefig(new_plots_folder)
    plt.close()
    
'''
def fi_level(casePath): #takes the case folder as input
    
    new_plots_folder = os.path.join(casePath, "Feed in Levels.png")
    summaryPath = os.path.join(casePath, "Summary.xlsx")
    summary = pd.read_excel(summaryPath, sheet_name=None)

    re_levels = list(summary.keys())
    if len(re_levels) >= 5:
        re_levels = re_levels[-5::]
    
    fig, ax = plt.subplots()
    colors = ["#f9e395", "#595755", "#c2deaf", "#85a4c4", "#f2b382" ]
    i=0
    
    
    for re_level in re_levels:
        outPath = os.path.join(casePath, 'Output Files', 
                               str(int(float(re_level)*100)))
        outFiles = os.listdir(outPath)
        fi_levels = []
        prices = []
        fits = summary[re_level].set_index('Unnamed: 0').loc['Feed-in Tariffs']
        fits = fits.to_list()
        fits = [fit for fit in fits if fit > 0]
        
        for outFile in outFiles:
            price = int(outFile.split('_')[2].split('.')[0]) / 100
            out = pd.read_excel(os.path.join(outPath, outFile), sheet_name=None)
            feed_in = out['Fed-in Capacity'].set_index('Unnamed: 0')
            dg = out['DG Dispatch'].set_index('Unnamed: 0')
            pv = out['PV Dispatch'].set_index('Unnamed: 0')
            b_out = out['Battery Output'].set_index('Unnamed: 0')
        
            total_feed_in = feed_in.values.sum()
            total_dg = dg.values.sum()
            total_pv = pv.values.sum()
            total_b_out = b_out.values.sum()
        
            feed_in_level = total_feed_in / (total_feed_in + total_dg
                                             + total_pv + total_b_out)
            fi_levels.append(feed_in_level * 100)
            prices.append(price)
            
        levels_df = pd.DataFrame(fi_levels, index=prices)
        levels_df.sort_index(inplace=True)
        fi_levels = levels_df[0].to_list()
        prices = levels_df.index.to_list()
        sizes = [((fi - min(fi_levels)) / (max(fi_levels) - min(fi_levels)) 
                 * 500 + 50) 
                 for fi in fi_levels]
        ax.scatter(prices, fits, s=sizes, color=colors[i])
        ax.plot(prices, fits, color=colors[i], marker='o', label=re_level)
        i+=1

    ax.set_xlabel('Price in USD')
    ax.set_ylabel('Feed-in tariff')
    #ax.set_ylabel('Fed-in capacity as % of total dispatch')
    ax.legend()
    
    plt.savefig(new_plots_folder)
    plt.close()
'''
    
def unmet_demand(casePath):
    new_plots_folder = os.path.join(casePath, "Unmet Demand.png")
    summaryPath = os.path.join(casePath, "Summary.xlsx")
    summary = pd.read_excel(summaryPath, sheet_name=None)

    re_levels = list(summary.keys())
    if len(re_levels) >= 5:
        re_levels = re_levels[-5::]
    
    fig, ax = plt.subplots()
    colors = ["#f9e395", "#595755", "#c2deaf", "#85a4c4", "#f2b382" ]
    i=0
    
    for re_level in re_levels:
        outPath = os.path.join(casePath, 'Output Files', 
                               str(int(float(re_level)*100)))
        outFiles = os.listdir(outPath)
        uds = []
        t_uds = []
        prices = []
        fits = summary[re_level].set_index('Unnamed: 0').loc['Feed-in Tariffs']
        fits = fits.to_list()
        fits = [fit for fit in fits if fit > 0]
        
        for outFile in outFiles:
            price = int(outFile.split('_')[2].split('.')[0]) / 100
            out = pd.read_excel(os.path.join(outPath, outFile), 
                                sheet_name="Summary").set_index('Unnamed: 0')
            ud = out.loc['Unmet Demand']
            tot_ud = out.loc['Total Unmet Demand']
            
            uds.append(ud[0])
            t_uds.append(tot_ud)
            prices.append(price)
            
        ud_df = pd.DataFrame({0: uds, 1: t_uds}, index=prices)
        ud_df.sort_index(inplace=True)
        uds = ud_df[0].to_list()
        t_uds = ud_df[1].to_list()
        prices = ud_df.index.to_list()
        
        ax.plot(prices, t_uds, linestyle='--', color=colors[i], 
                marker='o', label=f'Total UD, {re_level}')
        ax.plot(prices, uds, color=colors[i], 
                marker='o', label=f'Connected UD, {re_level}')
        fit_labels = [f'FiT={fit:.2f}' for fit in fits]
        for x, y, label in zip(prices, uds, fit_labels):
            ax.text(x, y-0.5, label, fontsize=9, ha='right')
        '''
        sizes = [((ud - min(uds)) / (max(uds) - min(uds)) 
                 * 500 + 50) 
                 for ud in uds]
        ax.scatter(prices, fits, s=sizes, color=colors[i])
        ax.plot(prices, fits, color=colors[i], marker='o', label=re_level)
        '''
        i+=1
        
    ax.set_xlabel('Price in USD')
    ax.set_ylabel('Unmet Demand in kWh')
    ax.legend()
    
    plt.savefig(new_plots_folder)
    plt.close()
    
def wasted_surplus(casePath):
    global ws
    global wss
    
    new_plots_folder = os.path.join(casePath, "Wasted Surplus.png")
    summaryPath = os.path.join(casePath, "Summary.xlsx")
    summary = pd.read_excel(summaryPath, sheet_name=None)

    re_levels = list(summary.keys())
    if len(re_levels) >= 5:
        re_levels = re_levels[-5::]
    
    fig, ax = plt.subplots()
    colors = ["#595755", "#6d597a",  "#85a4c4", "#f2b382", "#c2deaf" ]
    i=0
    
    for re_level in re_levels:
        outPath = os.path.join(casePath, 'Output Files', 
                               str(int(float(re_level)*100)))
        outFiles = os.listdir(outPath)
        wss = []
        t_wss = []
        prices = []
        fits = summary[re_level].set_index('Unnamed: 0').loc['Feed-in Tariffs']
        fits = fits.to_list()
        fits = [fit for fit in fits if fit > 0]
        
        for outFile in outFiles:
            price = int(outFile.split('_')[2].split('.')[0]) / 100
            out = pd.read_excel(os.path.join(outPath, outFile), 
                                sheet_name="Summary").set_index('Unnamed: 0')
            ws = out.loc['Wasted Prosumer Surplus']
            tot_ws = out.loc['Total Wasted Prosumer Surplus']
            
            wss.append(ws[0])
            t_wss.append(tot_ws)
            prices.append(price)
            
        ws_df = pd.DataFrame({0: wss, 1: t_wss}, index=prices)
        ws_df.sort_index(inplace=True)
        wss = ws_df[0].to_list()
        t_wss = ws_df[1].to_list()
        prices = ws_df.index.to_list()
        
        
        ax.plot(prices, t_wss, linestyle='--', color=colors[i], 
                marker='o', label=f'Total WS, {re_level}')
        ax.plot(prices, wss, color=colors[i], 
                marker='o', label=f'Connected WS, {re_level}')
        '''
        sizes = [((ws - min(wss)) / (max(wss) - min(wss)) 
                 * 500 + 50) 
                 for ws in wss]
        ax.scatter(prices, fits, s=sizes, color=colors[i])
        ax.plot(prices, fits, color=colors[i], marker='o', label=re_level)
        '''
        i+=1
        
    ax.set_xlabel('Price in USD')
    ax.set_ylabel('Wasted Prosumer Surplus in kWh')
    ax.legend()
    
    plt.savefig(new_plots_folder)
    plt.close()

def ud_heatmap(casePath, re_level):
    
    sns.set(font_scale=1.1)
    
    new_plots_folder = os.path.join(casePath, f"UD heatmap_{re_level}.png")
    filesPath = os.path.join(casePath, 'Output Files', str(int(re_level * 100)))
    files = os.listdir(filesPath)
    
    data = {'Prices': [],
            'FiTs': [],
            'Unmet Demand': []}
    for file in files:
        price = int(file.split('_')[2].split('.')[0]) / 100
        data['Prices'].append(price)
        fit = int(file.split('_')[1]) / 100
        data['FiTs'].append(fit)

        out = pd.read_excel(os.path.join(filesPath, file), sheet_name='Summary')
        out.set_index("Unnamed: 0", inplace=True)
        data['Unmet Demand'].append(out.loc["Unmet Demand"][0])
    
    
    df = pd.DataFrame(data)
    
    # Pivot the data to make F columns, P rows, and U the values
    heatmap_data = df.pivot(index='Prices', 
                            columns='FiTs', 
                            values='Unmet Demand')
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'U value'})
    
    plt.tight_layout()
    plt.savefig(new_plots_folder)
    plt.close()
    
def surp_heatmap(casePath, re_level, max_fits=None): # summary file
    
    sns.set(font_scale=1.1)
    
    global price_index
    global fit_index
    global heatmap_data
    global data
    
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
    
    heatmap_data = df.pivot(index='Prices ($)', 
                            columns='FiTs ($)', 
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
                        mask[i, j] = True
            except IndexError:
                print(f're_level={re_level}, price={price}')
                
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data / 10000, fmt=".1f", 
                cmap="YlGnBu", cbar_kws={'label': 'Surplus value'},
                mask=mask, annot=True)
    
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(new_plots_folder)
    plt.close()
    
def ud_comp(casePaths, re_levels): # re_levels in %
    
    assert type(casePaths)== list, "casePaths should be a list"
    
    new_plots_folder = os.path.join(casePaths[0], '..', 'UD comparison.png')
    
    cases = []
    sns.set(font_scale = 1.8)
    colors = ["#f4d35e", "#85a4c4", "#c2deaf" ]
    i = 0
    bar_data = {}
    
    name_mapping = {
    'No PV': 'Feed-in Only',
    'No PV w Bat': 'Feed-in and Battery',
    'With PV': 'Feed-in and PV'
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
    plt.figure(figsize=(12, 7))
    hue_order = ['Feed-in Only', 'Feed-in and Battery', 'Feed-in and PV']
    sns.barplot(x='RE target', y='Value', hue='Unmet Demand', 
                data=df_data_melted, palette=colors, hue_order=hue_order)
    
    plt.xlabel('RE Target (%)')
    plt.ylabel('Unmet Demand (%)')
    plt.legend(title = 'Case',
               loc='upper center',
               bbox_to_anchor=(0.5, 1.3),
               ncol=3,
               frameon=False)
    plt.tight_layout()
    
    plt.yticks(np.arange(0, 80, 20))
    
    plt.savefig(new_plots_folder)
    plt.close()
    
def ws_comp(casePaths): 
    
    assert type(casePaths)== list, "casePaths should be a list"
    
    new_plots_folder = os.path.join(casePaths[0], '..', 'WS comparison.png')
    
    cases = []
    sns.set(font_scale = 1.8)
    colors = ["#f4d35e", "#85a4c4", "#c2deaf" ]
    i = 1
    bar_data = {}
    
    name_mapping = {
    'No PV': 'Feed-in Only',
    'No PV w Bat': 'Feed-in and Battery',
    'With PV': 'Feed-in and PV'
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
    plt.figure(figsize=(12, 7))
    
    hue_order = ['Feed-in Only', 'Feed-in and Battery', 'Feed-in and PV']
    
    sns.barplot(x='RE target', y='Value', hue='Wasted Surplus', 
                data=df_data_melted, palette=colors, hue_order=hue_order)
    
    plt.xlabel('RE Target (%)')
    plt.ylabel('Wasted Surplus (%)')
    plt.legend(title = 'Case',
               loc='upper center',
               bbox_to_anchor=(0.5, 1.3),
               ncol=3,
               frameon=False)
    plt.tight_layout()
    
    plt.yticks(np.arange(0, 81, 20))
    
    plt.savefig(new_plots_folder)
    plt.close()
    
def re_comp(casePaths):
    
    assert type(casePaths)== list, "casePaths should be a list"
    
    sns.set(font_scale=1.1)
    
    new_plots_folder_p = os.path.join(casePaths[0], '..', 'P+RE comparison.png')
    new_plots_folder_f = os.path.join(casePaths[0], '..', 'FiT+RE comparison.png')
    new_plots_folder_hs = os.path.join(casePaths[0], '..', 'HS+RE comparison.png')
    
    fig_p, ax_p = plt.subplots()
    fig_f, ax_f = plt.subplots()
    fig_hs, ax_hs = plt.subplots()
    
    i = 0
    
    name_mapping = {
    'No PV': 'Feed-in Only',
    'No PV w Bat': 'Feed-in and Battery',
    'With PV': 'Feed-in and PV'
    }
    
    color_map = {
    'Feed-in Only': '#f4d35e',
    'Feed-in and Battery': '#85a4c4',
    'Feed-in and PV': '#c2deaf'
    }
    
    cases = []
    
    for casePath in casePaths:
        original_name = os.path.basename(casePath)
        readable_name = name_mapping.get(original_name, original_name)
        cases.append(readable_name)
        case_summary = pd.read_excel(os.path.join(casePath, 
                                                   'Evaluation Metrics.xlsx'))
        color = color_map[readable_name]
        ax_p.plot(np.array(case_summary['RE target']),
                  np.array(case_summary['Price']),
                  label = readable_name,
                  color=color,
                  linewidth=3)
        ax_f.plot(np.array(case_summary['RE target']),
                  np.array(case_summary['FiT']),
                  label = readable_name,
                  color=color,
                  linewidth=3)
        ax_hs.plot(np.array(case_summary['RE target']),
                   np.array(case_summary['Household Surplus']),
                   label = readable_name,
                   color=color,
                   linewidth=3)
        
        i += 1
    
    # Get current handles and labels
    handles_p, labels_p = ax_p.get_legend_handles_labels()
    handles_f, labels_f = ax_f.get_legend_handles_labels()
    handles_hs, labels_hs = ax_hs.get_legend_handles_labels()
    
    desired_order = ['Feed-in Only', 'Feed-in and Battery', 'Feed-in and PV']
    
    new_handles_p = [handles_p[labels_p.index(lbl)] for lbl in desired_order]
    new_handles_f = [handles_f[labels_f.index(lbl)] for lbl in desired_order]
    new_handles_hs = [handles_hs[labels_hs.index(lbl)] for lbl in desired_order]    
    
    ax_p.set_xlabel('RE Target (%)')
    ax_p.set_ylabel('Electricity Price ($/kWh)')
    ax_p.legend(new_handles_p,
               desired_order,
               title = 'Case',
               loc='upper center',
               bbox_to_anchor=(0.5, 1.3),
               ncol=3,
               frameon=False)
    fig_p.tight_layout()
    
    fig_p.savefig(new_plots_folder_p)
    plt.close()
    
    ax_f.set_xlabel('RE Target (%)')
    ax_f.set_ylabel('Feed-in Tariff ($/kWh)')
    ax_f.legend(new_handles_f,
               desired_order,
               title = 'Case',
               loc='upper center',
               bbox_to_anchor=(0.5, 1.3),
               ncol=3,
               frameon=False)
    fig_f.tight_layout()
    
    fig_f.savefig(new_plots_folder_f)
    plt.close()
    
    ax_hs.set_xlabel('RE Target (%)')
    ax_hs.set_ylabel('Household Surplus ($)')
    ax_hs.legend(new_handles_hs,
               desired_order,
               title = 'Case',
               loc='upper center',
               bbox_to_anchor=(0.5, 1.3),
               ncol=3,
               frameon=False)
    fig_hs.tight_layout()
    
    fig_hs.savefig(new_plots_folder_hs)
    plt.close()
    
# Run the functions for the different cases
cwd = os.getcwd()
outFile = os.path.join(cwd, "Outputs")
'''
# Initial Solution
outFile_0 = os.path.join(outFile, '0. Initial Solution', 'Output_0_40.xlsx')
add_ret(outFile_0, multi=0)
gen_year(outFile_0, multi=0)
rep_day(outFile_0, multi=0, year=10, day=1)
inst_cap(outFile_0, multi=0)
get_houses(outFile_0, multi=0)

# Base Case
outFile_1 = os.path.join(outFile, '1. Base Case', 'Output_0_40.xlsx')
add_ret(outFile_1, multi=0)
gen_year(outFile_1, multi=0)
rep_day(outFile_1, multi=0, year=10, day=1)
inst_cap(outFile_1, multi=0)
get_houses(outFile_1, multi=0)
'''

# No PV
outFile_2 = os.path.join(outFile, '2. No PV', 'Output Files')
re_levels = os.listdir(outFile_2)
'''
for re_level in re_levels:
    files = os.listdir(os.path.join(outFile_2, re_level))

    for file in files:
        outFile_2_1 = os.path.join(outFile, '2. No PV', 'Output Files', re_level, file)
        add_ret(outFile_2_1, multi=1)
        gen_year(outFile_2_1, multi=1)
        rep_day(outFile_2_1, multi=1, year=10, day=1)
        inst_cap(outFile_2_1, multi=1)
        get_houses(outFile_2_1, multi=1)
'''
outFile_2_2 = os.path.join(outFile, '2. No PV')
#fit_v_price(outFile_2_2)
#fi_level(outFile_2_2)
#unmet_demand(outFile_2_2)
#wasted_surplus(outFile_2_2)
'''
# With PV
outFile_3 = os.path.join(outFile, '3. With PV', 'Output Files')
re_levels = os.listdir(outFile_3)

for re_level in re_levels:
    files = os.listdir(os.path.join(outFile_3, re_level))
    for file in files:
        outFile_3_1 = os.path.join(outFile, '3. With PV', 'Output Files', re_level, file)
        add_ret(outFile_3_1, multi=1)
        gen_year(outFile_3_1, multi=1)
        rep_day(outFile_3_1, multi=1, year=10, day=1)
        inst_cap(outFile_3_1, multi=1)
        get_houses(outFile_3_1, multi=1)

outFile_3_2 = os.path.join(outFile, '3. With PV')
fit_v_price(outFile_3_2)
#fi_level(outFile_3_2)
#unmet_demand(outFile_3_2)
#wasted_surplus(outFile_3_2)
'''
# No PV with Batteries
outFile_9 = os.path.join(outFile, '9. No PV w Bat', 'Output Files')
re_levels = os.listdir(outFile_9)
'''
for re_level in re_levels:
    files = os.listdir(os.path.join(outFile_9, re_level))

    for file in files:
        outFile_9_1 = os.path.join(outFile, '9. No PV w Bat', 'Output Files', re_level, file)
        add_ret(outFile_9_1, multi=1)
        gen_year(outFile_9_1, multi=1)
        rep_day(outFile_9_1, multi=1, year=10, day=1)
        inst_cap(outFile_9_1, multi=1)
        get_houses(outFile_9_1, multi=1)
'''
outFile_9_2 = os.path.join(outFile, '9. No PV w Bat')
#fit_v_price(outFile_9_2)
#fi_level(outFile_2_2)
#unmet_demand(outFile_2_2)
#wasted_surplus(outFile_2_2)
'''

# Initial Solution with VHR
outFile_4= os.path.join(outFile, '4. Initial Solution (Variable HR)', 'Output_0_40.xlsx')
add_ret(outFile_4, multi=0)
gen_year(outFile_4, multi=0)
rep_day(outFile_4, multi=0, year=10, day=1)
inst_cap(outFile_4, multi=0)
get_houses(outFile_4, multi=0)

# Base Case with VHR
outFile_5 = os.path.join(outFile, '5. Base Case (Variable HR)', 'Output_0_40.xlsx')
add_ret(outFile_5, multi=0)
gen_year(outFile_5, multi=0)
rep_day(outFile_5, multi=0, year=10, day=1)
inst_cap(outFile_5, multi=0)
get_houses(outFile_5, multi=0)

'''
outFile_8 = os.path.join(outFile, '8. Fixed RE')
outFile_8_1 = os.path.join(outFile_8, 'With PV')
outFile_8_2 = os.path.join(outFile_8, 'No PV')
outFile_8_3 = os.path.join(outFile_8, 'No PV w Bat')
summary_path_1 = os.path.join(outFile, '3. With PV', 'Summary.xlsx')
summary_path_2 = os.path.join(outFile, '2. No PV', 'Summary.xlsx')
summary_path_3 = os.path.join(outFile, '9. No PV w Bat', 'Summary.xlsx')

re_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.6]

for re_level in re_levels:
    #surp_heatmap(outFile_8_1, re_level=re_level, max_fits=summary_path_1)
    #surp_heatmap(outFile_8_2, re_level=re_level, max_fits=summary_path_2)
    #surp_heatmap(outFile_8_3, re_level=re_level, max_fits=summary_path_3)
    re_level = str(int(re_level * 100))
    filePath_1 = os.path.join(outFile_8_1, 'Output Files', re_level)
    filePath_2 = os.path.join(outFile_8_2, 'Output Files', re_level)
    filePath_3 = os.path.join(outFile_8_3, 'Output Files', re_level)
    files_1 = os.listdir(filePath_1)
    files_2 = os.listdir(filePath_2)
    files_3 = os.listdir(filePath_3)
'''
    for file in files_1:
        outFile_8_1_1 = os.path.join(filePath_8_1, file)
        add_ret(outFile_8_1_1, multi=1)
        gen_year(outFile_8_1_1, multi=1)
        rep_day(outFile_8_1_1, multi=1, year=10, day=1)
        inst_cap(outFile_8_1_1, multi=1)
        get_houses(outFile_8_1_1, multi=1)
    
    for file in files_2:
        outFile_8_2_1 = os.path.join(filePath_8_2, file)
        add_ret(outFile_8_2_1, multi=1)
        gen_year(outFile_8_2_1, multi=1)
        rep_day(outFile_8_2_1, multi=1, year=10, day=1)
        inst_cap(outFile_8_2_1, multi=1)
        get_houses(outFile_8_2_1, multi=1)
        
    for file in files_3:
        outFile_8_3_1 = os.path.join(filePath_3, file)
        add_ret(outFile_8_3_1, multi=1)
        gen_year(outFile_8_3_1, multi=1)
        rep_day(outFile_8_3_1, multi=1, year=10, day=1)
        inst_cap(outFile_8_3_1, multi=1)
        get_houses(outFile_8_3_1, multi=1)
 '''
casePaths = [outFile_8_1, outFile_8_2, outFile_8_3]
re_levels = [0, 10, 20, 30, 40, 50, 60]
ud_comp(casePaths, re_levels)
ws_comp(casePaths)
re_comp(casePaths) 