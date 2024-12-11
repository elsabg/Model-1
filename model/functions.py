# -*- coding: utf-8 -*-
'''
Created on Tue Oct 21 13:42:03 2024

@author: jakobsvolba
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def show_tables(tables):
    '''Process output data'''

    (ret, inst, added, num_households) = tables

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

    names = ['Diesel Generator', 'Owned PV', 'Batteries (kw)', 'Batteries Capacity (kWh)']
    inst.index = names
    added.index = names
    ret.index = names
    names_housetypes = ['Consumer Residential', 'Prosumer Residential', 'Solar Farm', 'Water Pumping Station', 'Consumer Business', 'Prosumer Business']
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

def show_binaries(binaries, year, day):
    '''Show binaries'''

    (heat_rate_binary, price_binary) = binaries
    heat_rate_binary = pd.DataFrame(
        heat_rate_binary[year][day], columns=[i for i in range(heat_rate_binary.shape[3])] #heat_rate_binary.shape[2])]
    )
    price_binary = pd.DataFrame(
        price_binary[year], columns=[i for i in range(price_binary.shape[2])]
    )
    print('\n-----------heat rate binary-----------\n')
    print(heat_rate_binary)
    print('\n-----------price binary-----------\n')
    print(price_binary)
    return

def plot_day(timeseriesArray, year, day):
    '''plot some output data'''

    (disp_gen, disp_pv, disp_feedin, unmetD, bat_in, bat_out,
     state_of_charge, total_demand) = timeseriesArray

    fig1, ax1 = plt.subplots()
    hours = np.arange(24)

    # Create the bottom bar
    p1 = ax1.bar(hours, disp_gen[day], label='Dispatched Generation', color='blue')

    # Stack the other bars on top
    p2 = ax1.bar(hours, bat_out[day], bottom=disp_gen[day], label='Battery Output', color='red')
    p3 = ax1.bar(hours, disp_pv[day], bottom=disp_gen[day] + bat_out[day], label='Owned PV', color = 'orange')
    p4 = ax1.bar(hours, disp_feedin[day], bottom=disp_gen[day] + bat_out[day] + disp_pv[day], label='Feed in', color='yellow')
    p5 = ax1.bar(hours, unmetD[day], bottom=disp_gen[day] + bat_out[day] + disp_pv[day] + disp_feedin[day], label='Unmet Demand', color='purple')
    p6 = ax1.bar(hours, -bat_in[day], label='Battery Input', color='green')

    ax1.plot(hours, total_demand[day], label='Total Demand', color='black', linestyle='-', marker='o')

    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Energy')
    ax1.set_title('Stacked Bar Chart of Energy Data over a Day '+ str(day + 1) +' (Year '+str(year)+')')
    ax1.legend(loc='upper left')

    ax1.legend()
    plt.show()

def plot_soc(timeseriesArray, year, day):

    (disp_gen, disp_pv, disp_feedin, unmetD, bat_in, bat_out,
     state_of_charge, total_demand) = timeseriesArray

    fig2, ax2 = plt.subplots()
    hours = np.arange(24)
    ax2.plot(hours, state_of_charge[day], label='State of Charge', color='blue', linestyle='-', marker='o')

    q1 = ax2.bar(hours, bat_in[day], label='Battery Input', color='green')
    p2 = ax2.bar(hours, -bat_out[day], label='Battery Output', color='red')
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('State of Charge')
    ax2.set_title('State of Charge over a Day '+ str(day + 1) +' (Year '+str(year)+')')
    ax2.legend(loc='upper left')


    plt.show()

def save_results(results):
    (ret, inst, added, disp_gen, disp_pv, disp_feedin,
     unmetD, bat_in, bat_out, state_of_charge, num_households,
     heat_rate_binary, price_binary, total_demand) = results
    save_array2d_to_excel(ret, 'results.xlsx', 'retired_capacity')
    save_array2d_to_excel(inst, 'results.xlsx', 'installed_capacity')
    save_array2d_to_excel(added, 'results.xlsx', 'added_capacity')

    [save_array2d_to_excel(disp_gen[y], 'results.xlsx', 'dispatched_generation_' + str(y + 1)) for y in range(len(disp_gen))]
    [save_array2d_to_excel(disp_pv[y], 'results.xlsx', 'dispatched_pv_' + str(y + 1)) for y in range(len(disp_pv))]
    [save_array2d_to_excel(disp_feedin[y], 'results.xlsx', 'dispatched_feedin_' + str(y + 1)) for y in range(len(disp_feedin))]
    [save_array2d_to_excel(unmetD[y], 'results.xlsx', 'unmet_demand_' + str(y + 1)) for y in range(len(unmetD))]
    [save_array2d_to_excel(bat_in[y], 'results.xlsx', 'battery_input_' + str(y + 1)) for y in range(len(bat_in))]
    [save_array2d_to_excel(bat_out[y], 'results.xlsx', 'battery_output_' + str(y + 1)) for y in range(len(bat_out))]
    [save_array2d_to_excel(state_of_charge[y], 'results.xlsx', 'state_of_charge_' + str(y + 1)) for y in range(len(state_of_charge))]
    [save_array2d_to_excel(total_demand[y], 'results.xlsx', 'total_demand_' + str(y + 1)) for y in range(len(total_demand))]

    save_array2d_to_excel(num_households, 'results.xlsx', 'num_households')
    save_array2d_to_excel(price_binary, 'results.xlsx', 'price_binary')
    #save_array2d_to_excel(heat_rate_binary, 'results.xlsx', 'heat_rate_binary')
    for y in range (15):
        [save_array2d_to_excel(heat_rate_binary[:, y, d, :], 'results.xlsx', 'heat_rate_binary_' + str(y + 1)+'_'+str(d + 1)) for d in range(3)]
        #save_array2d_to_excel(price_binary[y], 'results.xlsx', 'price_binary_' + str(y + 1))

def save_array2d_to_excel(array2d, file_name, s_name):
    # Convert the ret array to a pandas DataFrame
    df_array2d = pd.DataFrame(array2d.reshape(-1, array2d.shape[1]))

    with pd.ExcelWriter(file_name, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df_array2d.to_excel(writer, sheet_name=s_name, index=False)

def get_results(file_name):
    return pd.read_excel(file_name, decimal=',', sheet_name=None)


def get_timeseries(data, year):
    disp_gen = data['dispatched_generation_'+ str(year + 1)].iloc[:,:].to_numpy()
    disp_pv = data['dispatched_pv_'+ str(year + 1)].iloc[:, :].to_numpy()
    disp_feedin = data['dispatched_feedin_'+ str(year + 1)].iloc[:,:].to_numpy()
    unmetD = data['unmet_demand_'+ str(year + 1)].iloc[:,:].to_numpy()
    bat_in = data['battery_input_'+ str(year + 1)].iloc[:, :].to_numpy()
    bat_out = data['battery_output_'+ str(year + 1)].iloc[:, :].to_numpy()
    state_of_charge = data['state_of_charge_'+ str(year + 1)].iloc[:, :].to_numpy()
    total_demand = data['total_demand_'+ str(year + 1)].iloc[:, :].to_numpy()

    return [disp_gen, disp_pv, disp_feedin, unmetD, bat_in, bat_out, state_of_charge, total_demand]

def get_tabels(data):
    ret = data['retired_capacity'].iloc[:,:].to_numpy()
    inst = data['installed_capacity'].iloc[:,:].to_numpy()
    added = data['added_capacity'].iloc[:,:].to_numpy()
    num_households = data['num_households'].iloc[:,:].to_numpy()

    return [ret, inst, added, num_households]

def get_binaries(data):
    heat_rate_binary = np.zeros((15, 3, 8, 2))
    price_binary = data['price_binary'].iloc[:, :].to_numpy()
    for y in range(15):
        #price_binary[y] = data['price_binary_' + str(y + 1)].iloc[:, :].to_numpy()
        for d in range(3):
            heat_rate_binary[y][d] = data['heat_rate_binary_' + str(y + 1) + '_' + str(d + 1)].iloc[:, :].to_numpy()
    return [heat_rate_binary, price_binary]
