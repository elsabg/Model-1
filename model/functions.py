# -*- coding: utf-8 -*-
'''
Created on Tue Oct 21 13:42:03 2024

@author: jakobsvolba
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def output_data(resultsArray):
    '''Process output data'''

    ret, inst, added, disp_gen, bat_in, bat_out, num_households, feed_in, total_demand = resultsArray #no unmed demand

    disp_gen = pd.DataFrame(
        disp_gen, columns=[i for i in range(disp_gen.shape[1])]
    )
    
    
    feed_in = pd.DataFrame(
        feed_in, columns=[i for i in range(feed_in.shape[1])]
    )
    
    '''
    unmetD = pd.DataFrame(
        unmetD, columns=[i for i in range(unmetD.shape[1])]
    )
    '''

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


    print('\n-----------installed capacity-----------\n')
    print(inst.round(2))
    print('\n-----------added capacity-----------\n')
    print(added.round(2))
    print('\n-----------retired capacity-----------\n')
    print(ret.round(2))
    print('\n-----------dispatched Energy Generator year 1-----------\n')
    print(disp_gen.round(2))
    print('\n-----------feed in year 1-----------\n')
    print(feed_in.round(2))
    #print('\n-----------unmet Demand year 1-----------\n')
    #print(unmetD.round(2))
    print('\n-----------battery Input year 1-----------\n')
    print(bat_in.round(2))
    print('\n-----------battery Output year 1-----------\n')
    print(bat_out.round(2))
    print('\n-----------Number of connected household types-----------\n')
    print(num_households)
    '''write output data to excel file'''
    return


def plot_data(resultsArray):
    '''plot some output data'''

    ret, inst, added, disp_gen, bat_in, bat_out, num_households, feed_in, total_demand = resultsArray #no unmet demand

    fig, ax = plt.subplots()
    ax.bar(np.arange(24), bat_in[0], 0.5, label='Battery Input', color = 'green')
    ax.bar(np.arange(24), bat_out[0], 0.5, label='Battery Output', color = 'red')
    ax.bar(np.arange(24) + 0.5, feed_in[0], 0.5, label='Feed in', color = 'orange')
    #ax.bar(np.arange(24), unmetD[0], 0.5, label='Unmet Demand', color = 'blue')
    ax.plot(np.arange(24), total_demand[0], label='Total Demand', color = 'black')
    ax.plot(np.arange(24), disp_gen['Diesel generator', 0, 0], label='DG', color = 'blue')
    ax.plot(np.arange(24), disp_gen['Owned PV', 0, 0], label='PV', color='magenta')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Energy')
    ax.set_title('Generation and res. Load Profile over a day (year 12)')

    ax.legend()
    plt.show()


