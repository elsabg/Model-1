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
import os

def rep_day(outFile, year, day, multi=1):
    '''Plot some output data and save to folder'''
    
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

    fig, ax = plt.subplots()

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
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.97), ncol=4,
              bbox_transform=fig.transFigure)
    if multi == 1:
        ax.set_yticks([i for i in range (0, 701, 100)])
    
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
        npv = summary.loc["NPV"]
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
    
    new_plots_folder = os.path.join(casePath, "FiTs v Prices.png")
    outFile = os.path.join(casePath, "Summary.xlsx")
    
    out = pd.read_excel(outFile, sheet_name=None)
    fig, ax = plt.subplots()
    colors = ["#f9e395", "#595755", "#c2deaf", "#85a4c4", "#f2b382" ]
    i = 0
    show_infeasible_label = True
    keys = list(out.keys())
    if len(keys) >= 5:
        keys = keys[-5::]
        
    for key in keys:
        out[key].set_index('Unnamed: 0', inplace=True)
        fits = out[key].loc['Feed-in Tariffs'].to_list()
        unfeas_fits = [fit for fit in fits if fit == 0]
        prices = out[key].loc['Prices'].to_list()
    
        ax.plot(prices[len(unfeas_fits) - 2 ::], fits[len(unfeas_fits) - 2 ::], 
                marker='o', linestyle='-', color=colors[i], zorder=1,
                label=float(key))
        ax.scatter(prices[len(unfeas_fits) -2 : len(unfeas_fits)], 
                   unfeas_fits[len(unfeas_fits) - 2 ::], 
                   marker='x', color='red', zorder=2, 
                   label='Infeasible' if show_infeasible_label else "")
        i+=1
        show_infeasible_label = False
    
    ax.set_xlabel('Price in USD')
    ax.set_ylabel('Feed-in Tariff in USD')
    ax.legend()
    
    ax.set_xticks([i / 100 for i in range(len(unfeas_fits) - 2, 
                                          len(prices)+1, 5)])
    plt.savefig(new_plots_folder)
    plt.close()
    

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
        
        for outFile in outFiles:
            price = int(outFile.split('_')[2].split('.')[0])
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
            fi_levels.append(feed_in_level)
            prices.append(price)
            levels_df = pd.DataFrame(fi_levels, index=prices)
            levels_df.sort_index(inplace=True)
            fi_levels = levels_df[0].to_list()
            prices = levels_df.index.to_list()
        ax.plot(fi_levels, prices, color=colors[i], marker='o', label=re_level)
        i+=1

    ax.set_xlabel('Price in USD')
    ax.set_ylabel('Fed-in capacity as % of total dispatch')
    ax.legend()
    
    plt.savefig(new_plots_folder)
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
fit_v_price(outFile_2_2)
fi_level(outFile_2_2)


# With PV
outFile_3 = os.path.join(outFile, '3. With PV', 'Output Files')
re_levels = os.listdir(outFile_3)
'''
for re_level in re_levels:
    files = os.listdir(os.path.join(outFile_3, re_level))
    for file in files:
        outFile_3_1 = os.path.join(outFile, '3. With PV', 'Output Files', re_level, file)
        add_ret(outFile_3_1, multi=1)
        gen_year(outFile_3_1, multi=1)
        rep_day(outFile_3_1, multi=1, year=10, day=1)
        inst_cap(outFile_3_1, multi=1)
        get_houses(outFile_3_1, multi=1)
'''
outFile_3_2 = os.path.join(outFile, '3. With PV')
fit_v_price(outFile_3_2)
fi_level(outFile_3_2)