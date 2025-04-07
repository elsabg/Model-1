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

def rep_day(outFile, year, day):
    '''Plot some output data and save to folder'''

    new_plots_folder = os.path.join(os.getcwd(), "Representative days")
    os.makedirs(new_plots_folder, exist_ok=True)

    script_dir = os.getcwd()
    file_path = os.path.join(script_dir, "Output Files", outFile)
    out = pd.read_excel(file_path, sheet_name=None)

    fit = int(outFile.split('_')[1])
    el_price = int(outFile.split('_')[2].split('.')[0])

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
    ax.set_title(f'Representative generation profile for FiT={fit}c and grid price={el_price}c',
                 pad=40)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.97), ncol=4,
              bbox_transform=fig.transFigure)
    ax.set_yticks([i for i in range (0, 1401, 200)])  
    
    plt.subplots_adjust(top=.85)
    plot_path = os.path.join(new_plots_folder, 
                             f"Representative_day_{fit}_{el_price}.png")
    plt.savefig(plot_path)
    plt.close()


def inst_cap(fit, el_price):
    '''Plot installed capacities and save to folder'''

    new_plots_folder = os.path.join(os.getcwd(), "Installed capacities")
    os.makedirs(new_plots_folder, exist_ok=True)

    fit = int(fit * 100)
    el_price = int(el_price * 100)

    script_dir = os.getcwd()
    outFile = f'Output_{fit}_{el_price}.xlsx'
    file_path = os.path.join(script_dir, "Output Files", outFile)
    inst = pd.read_excel(file_path, sheet_name='Installed Capacities')
    inst.set_index('Unnamed: 0', inplace=True)

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
    ax.set_title(f'Yearly installed capacities for FiT={fit}c and grid price={el_price}c',
                 pad=40)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, .94), ncol=3,
              bbox_transform=fig.transFigure)
    ax.set_yticks([i for i in range (0, 2251, 250)])  

    plt.subplots_adjust(top=.85)
    plot_path = os.path.join(new_plots_folder, f"Installed_Capacities_{fit}_{el_price}.png")
    plt.savefig(plot_path)
    plt.close()
    

def get_houses(file):
    cwd = os.getcwd()
    files_path = os.path.join(cwd, "Output Files")
    out = pd.read_excel(os.path.join(files_path, file), 
                        sheet_name="Connected Households")
    out.set_index('Unnamed: 0', inplace=True)
    fig, ax = plt.subplots()
    fit = int(file.split('_')[1])
    el_price = int(file.split('_')[2].split('.')[0])
    max_house = {"Consumers": [635] * 15,"Prosumers": [440] * 15}
    colors = {"Consumers": '#dbe4ed', "Prosumers": '#f2b382'}
    for house in out.index:
        ax.plot(out.loc[house], label=house, color=colors[house])
        ax.plot(max_house[house], linestyle='dashed', 
                label=f'Maximum {house}', color=colors[house])
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Connected households')
    ax.set_title(f'Connected households for FiT={fit}c and grid price={el_price}c',
                 pad=38)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, .97), ncol=2,
              bbox_transform=fig.transFigure)
    
    ax.set_yticks([i for i in range (0, 701, 100)])  
    
    plt.subplots_adjust(top=.85)
    new_plots_folder = os.path.join(os.getcwd(), "Connected households")
    os.makedirs(new_plots_folder, exist_ok=True)
    plot_path = os.path.join(new_plots_folder, 
                             f"Connected_households_{fit}_{el_price}.png")
    plt.savefig(plot_path)
    plt.close()

def gen_year(file):
    
    new_plots_folder = os.path.join(os.getcwd(), "Yearly Generation")
    os.makedirs(new_plots_folder, exist_ok=True)

    script_dir = os.getcwd()
    file_path = os.path.join(script_dir, "Output Files", file)
    out = pd.read_excel(file_path, sheet_name=None)

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
    ax.set_title(f'Yearly generation for FiT={fit}c and grid price={el_price}c',
                 pad=40)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, .97), ncol=3,
              bbox_transform=fig.transFigure)
    
    ax.set_yticks([i for i in range (0, 7000001, 1000000)])
    
    plt.subplots_adjust(top=.85)
    plot_path = os.path.join(new_plots_folder, 
                             f"Yearly_gen_{fit}_{el_price}.png")
    plt.savefig(plot_path)
    plt.close()
        
def add_ret(file):
    
    script_dir = os.getcwd()
    new_plots_folder = os.path.join(script_dir, "Added and Retired capacities")
    file_path = os.path.join(script_dir, "Output Files", file)
    os.makedirs(new_plots_folder, exist_ok=True)
    out = pd.read_excel(file_path, sheet_name=None)
    
    fit = int(file.split('_')[1])
    el_price = int(file.split('_')[2].split('.')[0])
    added = out['Added Capacities'].set_index("Unnamed: 0")
    ret = out['Retired Capacities'].set_index("Unnamed: 0")
    
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
    ax.set_title(f'Yearly added and retired capacities for FiT={fit}c and grid price={el_price}c',
                 pad=40)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, .97), ncol=3,
              bbox_transform=fig.transFigure)

    ax.set_yticks([i for i in range (-1000, 1751, 250)])  

    plt.subplots_adjust(top=.85)
    plot_path = os.path.join(new_plots_folder, f"Add_Ret_Capacities_{fit}_{el_price}.png")
    plt.savefig(plot_path)
    plt.close()
    
def get_npv(i): #takes interest rate as argument

    script_dir = os.getcwd()
    file_path = os.path.join(script_dir, "Output Files")
    files = os.listdir(file_path)

    npvs = []
    fits = []
    prices = []
    
    for file in files:
        fit = int(file.split('_')[1])
        price = int(file.split('_')[2].split('.')[0])
        summary = pd.read_excel(os.path.join(file_path, file), 
                                sheet_name='Costs and Revenues')
        summary.set_index("Unnamed: 0", inplace=True)
        profits = summary.loc["Total Profits"]
        npv=0
        for y in range(15):
            npv += profits[y] * (1 / ((1 + i) ** y))
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
    plt.savefig("NPV.png", dpi=300, bbox_inches='tight')
    
def fit_v_price(file):
    out = pd.read_excel(file).set_index('Unnamed: 0')
    fits = out.loc['Feed-in Tariffs'].to_list()
    unfeas_fits = [fit for fit in fits if fit == 0]
    prices = out.loc['Prices'].to_list()
    base_npv = out.loc['Base NPV', 0]
    
    fig, ax = plt.subplots()
    ax.plot(prices[len(unfeas_fits) - 2 ::], fits[len(unfeas_fits) - 2 ::], 
            marker='o', linestyle='-', color='#595755', zorder=1)
    ax.scatter(prices[len(unfeas_fits) -2 : len(unfeas_fits)], 
               unfeas_fits[len(unfeas_fits) - 2 ::], 
               marker='x', color='red', zorder=2)
    
    ax.set_xlabel('Price in USD')
    ax.set_ylabel('Feed-in Tariff in USD')
    ax.set_title(f'Price and FiT pairs for NPV={base_npv}')
    
    ax.set_xticks([i / 100 for i in range(len(unfeas_fits) - 2, 51, 5)])
    plt.savefig('fits_v_prices.png')
    plt.close()
    
# Run the functions for different FiT values
cwd = os.getcwd()
files_path = os.path.join(cwd, "Output Files")
files = os.listdir(files_path)


fit_v_price('Summary.xlsx')


for file in files:
    fit = int(file.split('_')[1])
    el_price = int(file.split('_')[2].split('.')[0])
    add_ret(file)
    gen_year(file)
    rep_day(file, 10, 1)
    inst_cap(fit/100, el_price/100)
    get_houses(file)
