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

    ret, inst, added, disp_gen, unmetD, bat_in, bat_out, num_households, feed_in, total_demand = resultsArray

    '''
    disp_gen = pd.DataFrame(
        disp_gen, columns=[i for i in range(disp_gen.shape[0])]
    )

    feed_in = pd.DataFrame(
        feed_in, columns=[i for i in range(feed_in.shape[1])]
    )

    unmetD = pd.DataFrame(
        unmetD, columns=[i for i in range(unmetD.shape[1])]
    )

    bat_in = pd.DataFrame(
        bat_in, columns=[i for i in range(bat_in.shape[1])]
    )

    bat_out = pd.DataFrame(
        bat_out, columns=[i for i in range(bat_out.shape[1])]
    )
    '''
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
    names_housetypes = ['Consumer Residential', 'Prosumer Residential', 'Solar Farm', 'Water Pumping Station', 'Prosumer Business']
    num_households.index = names_housetypes


    print('\n-----------installed capacity-----------\n')
    print(inst.round(2))
    print('\n-----------added capacity-----------\n')
    print(added.round(2))
    print('\n-----------retired capacity-----------\n')
    print(ret.round(2))
    print('\n-----------Number of connected household types-----------\n')
    print(num_households)
    return

def plot_data(resultsArray):
    '''plot some output data'''

    ret, inst, added, disp_gen, unmetD, bat_in, bat_out, num_households, feed_in, total_demand = resultsArray

    fig, ax = plt.subplots()
    hours = np.arange(24)

    # Create the bottom bar
    p1 = ax.bar(hours, disp_gen, label='Dispatched Generation', color='blue')

    # Stack the other bars on top
    p2 = ax.bar(hours, bat_out, bottom=disp_gen, label='Battery Output', color='red')
    p3 = ax.bar(hours, feed_in, bottom=disp_gen + bat_out, label='Feed in', color='orange')
    p4 = ax.bar(hours, unmetD, bottom=disp_gen + bat_out + feed_in, label='Unmet Demand', color='purple')
    p5 = ax.bar(hours, -bat_in, label='Battery Input', color='green')

    ax.plot(hours, total_demand[2], label='Total Demand', color='black', linestyle='-', marker='o')

    ax.set_xlabel('Hour')
    ax.set_ylabel('Energy')
    ax.set_title('Stacked Bar Chart of Energy Data over a Winter Day (Year 1)')
    ax.legend(loc='upper left')

    ax.legend()
    plt.show()


