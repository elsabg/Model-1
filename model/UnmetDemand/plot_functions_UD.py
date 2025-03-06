# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 22:18:29 2025

@author: Elsa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

    fig, ax = plt.subplots()

    # Representative day
    index = float(f'{year}' + '.' + f'{day}')
    ax.bar(np.arange(24), bat_in.loc[index] * -1,
           width=0.5, label='Battery Input', color='green')
    ax.bar(np.arange(24), disp_dg.loc[index],
           width=0.5, label='DG', color='blue')
    ax.bar(np.arange(24), disp_pv.loc[index],
           bottom=disp_dg.loc[index],
           width=0.5, label='PV', color='magenta')
    ax.bar(np.arange(24), feed_in.loc[index],
           bottom=disp_dg.loc[index] + disp_pv.loc[index],
           width=0.5, label='Feed in', color='cyan')
    ax.bar(np.arange(24), bat_out.loc[index],
           bottom=(disp_dg.loc[index]
                   + disp_pv.loc[index] + feed_in.loc[index]),
           width=0.5, label='Battery Output', color='red')
    ax.bar(np.arange(24), ud.loc[index],
           bottom=(disp_dg.loc[index]
                   + disp_pv.loc[index] + feed_in.loc[index]
                   + bat_out.loc[index]),
           width=0.5, label='Unmet Demand', color='orange')

    ax.plot(np.arange(24), tot_dem.loc[index].to_numpy() * -1,
            label='Total Demand', color='black')
    ax.plot(np.arange(24), net_sur.loc[index].to_numpy(),
            label='Total Surplus', linestyle='dashed',
            color='black')

    ax.set_xlabel('Hour')
    ax.set_ylabel('Energy')
    ax.set_title(f'Representative day for FiT={fit}c and grid price={el_price}c')

    ax.legend(loc='upper right')

    plot_path = os.path.join(new_plots_folder, f"Representative_day_{fit}_{el_price}.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Saved plot: {plot_path}")

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
           width=0.5, label='Diesel generator', color="blue")
    ax.bar(np.arange(15), inst.loc['Owned PV'],
           width=0.5, label='Owned PV',
           bottom=inst.loc['Diesel Generator'], color="magenta")
    ax.bar(np.arange(15), inst.loc['Owned Batteries'],
           width=0.5, label='Owned batteries',
           bottom=inst.loc['Owned PV'] + inst.loc['Diesel Generator'],
           color="green")

    ax.set_xlabel('Year')
    ax.set_ylabel('Capacity installed in kW')
    ax.set_title(f'Yearly installed capacities for FiT={fit}c and grid price={el_price}c')

    ax.legend(loc='upper right')

    # Save figure to "Corresponding Plots (Thanks to Julien)" folder
    plot_path = os.path.join(new_plots_folder, f"Installed_Capacities_{fit}_{el_price}.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Saved plot: {plot_path}")


# Run the functions for different FiT values
for fit in [0.01, 0.1, 0.25, 0.3, 0.4]:
    rep_day(f'Output_{int(fit * 100)}_40.xlsx', 10, 1)
    inst_cap(fit, 0.4)
