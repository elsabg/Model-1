# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 12:04:12 2025

@author: Elsa
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def rep_day(outFile, year, day):
    '''plot some output data'''
    
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
    dem = out['Yearly demand'].set_index('Unnamed: 0')

    fig, ax = plt.subplots()
    
    # representative day
    index = float(f'{year}'+'.'+f'{day}')
    ax.bar(np.arange(24), bat_in.loc[index] * -1, 
           0.5, label='Battery Input', color = 'green')
    ax.bar(np.arange(24), bat_out.loc[index], 
           0.5, label='Battery Output', color = 'red')
    ax.bar(np.arange(24), feed_in.loc[index], 
           0.5, label='Feed in', color = 'orange')
    ax.bar(np.arange(24), disp_dg.loc[index], 
           0.5, label='DG', color = 'blue')
    ax.bar(np.arange(24), disp_pv.loc[index], 
           0.5, label='PV', color='magenta')

    ax.plot(np.arange(24), dem.loc[index].to_numpy() * -1, 
            label='Total Demand', color = 'black')
    
    ax.set_xlabel('Hour')
    ax.set_ylabel('Energy')
    ax.set_title(f'Representative day for FiT={fit}c and grid price={el_price}c')

    ax.legend(loc='upper right')
    plt.show()

rep_day('Output_10_40.xlsx', 10, 1)