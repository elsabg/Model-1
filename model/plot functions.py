# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 12:04:12 2025

@author: Elsa
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_data(model):
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

    ax.legend()
    plt.show()