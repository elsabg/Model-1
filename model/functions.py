# -*- coding: utf-8 -*-
'''
Created on Tue Oct 21 13:42:03 2024

@author: jakobsvolba
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#--------------------------------------------------------------------------------#
# Print Results                                                                  #
#--------------------------------------------------------------------------------#

def show_tables(tables):
    '''Print Tabels Installed Added Retired Capacity and Number of Households'''

    (ret, inst, added, num_households) = tables

    num_households = pd.DataFrame(
        num_households, columns=[i + 1 for i in range(num_households.shape[1])]
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
    names_housetypes = ['Consumer Residential', 'Prosumer Residential']#, 'Solar Farm', 'Water Pumping Station', 'Consumer Business', 'Prosumer Business']
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
    '''Show the binary variables of Heatrate and Price Elasticity'''

    (heat_rate_binary, price_binary, quantity_binary) = binaries
    heat_rate_binary = pd.DataFrame(
        heat_rate_binary[year][day], columns=[i for i in range(heat_rate_binary.shape[3])] #heat_rate_binary.shape[2])]
    )
    price_binary = pd.DataFrame(
        price_binary, columns=[i for i in range(price_binary.shape[1])]
    )
    quantity_binary = pd.DataFrame(
        quantity_binary, columns=[i for i in range(quantity_binary.shape[1])]
    )


    #print('\n-----------heat rate binary-----------\n')
    #print(heat_rate_binary)
    print('\n-----------price binary-----------\n')
    print(price_binary)
    print('\n-----------quantity binary-----------\n')
    print(quantity_binary)
    return

#--------------------------------------------------------------------------------#
# Plot Results                                                                   #
#--------------------------------------------------------------------------------#

def plot_days(timeseriesArray, year, s='plot'):
    '''plot or save all days of a year'''
    (disp_gen, disp_pv, disp_feedin, unmetD, bat_in, bat_out,
     state_of_charge, total_demand) = timeseriesArray

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    hours = np.arange(24)
    for i in range(2):
        p1 = axs[i, 1].bar(hours, disp_gen[2 * i], label='Dispatched Generation', color='blue')

        # Stack the other bars on top
        p2 = axs[i, 1].bar(hours, bat_out[2 * i], bottom=disp_gen[2 * i], label='Battery Output', color='red')
        p3 = axs[i, 1].bar(hours, disp_pv[2 * i], bottom=disp_gen[2 * i] + bat_out[2 * i], label='Owned PV', color='orange')
        p4 = axs[i, 1].bar(hours, disp_feedin[2 * i], bottom=disp_gen[2 * i] + bat_out[2 * i] + disp_pv[2 * i], label='Feed in',
                           color='yellow')
        p5 = axs[i, 1].bar(hours, unmetD[2 * i], bottom=disp_gen[2 * i] + bat_out[2 * i] + disp_pv[2 * i] + disp_feedin[2 * i],
                           label='Unmet Demand', color='purple')
        p6 = axs[i, 1].bar(hours, -bat_in[2 * i], label='Battery Input', color='green')

        axs[i, 1].plot(hours, total_demand[2 * i], label='Total Demand', color='black', linestyle='-', marker='o')

        axs[i, 1].set_xlabel('Hour of Day (h)')
        axs[i, 1].set_ylabel('Energy (kWh)')


        p1 = axs[i, 0].bar(hours, disp_gen[1], label='Dispatched Generation', color='blue')

        # Stack the other bars on top
        p2 = axs[i, 0].bar(hours, bat_out[1], bottom=disp_gen[1], label='Battery Output', color='red')
        p3 = axs[i, 0].bar(hours, disp_pv[1], bottom=disp_gen[1] + bat_out[1], label='Owned PV',
                           color='orange')
        p4 = axs[i, 0].bar(hours, disp_feedin[1], bottom=disp_gen[1] + bat_out[1] + disp_pv[1],
                           label='Feed in',
                           color='yellow')
        p5 = axs[i, 0].bar(hours, unmetD[1],
                           bottom=disp_gen[1] + bat_out[1] + disp_pv[1] + disp_feedin[1],
                           label='Unmet Demand', color='purple')
        p6 = axs[i, 0].bar(hours, -bat_in[1], label='Battery Input', color='green')

        axs[i, 0].plot(hours, total_demand[1], label='Total Demand', color='black', linestyle='-', marker='o')

        axs[i, 0].set_xlabel('Hour of Day (h)')
        axs[i, 0].set_ylabel('Energy (kWh)')

        if i == 0:
            axs[i, 0].set_title('Year ' + str(year+1) + ': Spring')
            axs[i, 1].set_title('Year ' + str(year+1) + ': Summer')
        else:
            axs[i, 0].set_title('Year ' + str(year+1) + ': Autumn')
            axs[i, 1].set_title('Year ' + str(year+1) + ': Winter')
    axs[0, 0].legend(loc='upper left')
    plt.tight_layout()
    if s == 'save':
        plt.savefig('plots/timeseries/timeseries_year'+str(year)+'.png')
    else:
        plt.show()



def plot_soc(timeseriesArray, year, day):
    '''plot the SoC of a specific day'''
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

#--------------------------------------------------------------------------------#
# Save/Read results to/from excel file                                           #
#--------------------------------------------------------------------------------#

def save_results(results):
    '''Save the results arrays to an excel file'''
    (ret, inst, added, disp_gen, disp_pv, disp_feedin,
     unmetD, bat_in, bat_out, state_of_charge, num_households,
     heat_rate_binary, price_binary, quantity_binary, total_demand,
     res_Demand, pros_feedin, pros_demandarray) = results
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
    save_array2d_to_excel(quantity_binary, 'results.xlsx', 'quantity_binary')
    #[save_array2d_to_excel(pros_feedin[d], 'results.xlsx', 'pros_feedin_'+str(d+1)) for d in range(3)]
    #[save_array2d_to_excel(res_Demand[d], 'results.xlsx', 'res_Demand_'+str(d+1)) for d in range(3)]
    for y in range (15):
        [save_array2d_to_excel(heat_rate_binary[y][d], 'results.xlsx', 'heat_rate_binary_' + str(y + 1)+'_'+str(d + 1)) for d in range(3)]
        #save_array2d_to_excel(price_binary[y], 'results.xlsx', 'price_binary_' + str(y + 1))

def save_array2d_to_excel(array2d, file_name, s_name):
    '''Save a 2D array to an excel file'''
    # Convert the ret array to a pandas DataFrame
    if array2d.ndim == 1:
        df_array2d = pd.DataFrame(array2d.reshape(1, -1))
    else:
        df_array2d = pd.DataFrame(array2d.reshape(-1, array2d.shape[1]))

    with pd.ExcelWriter(file_name, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df_array2d.to_excel(writer, sheet_name=s_name, index=False)

def get_results(file_name):
    '''Read the results from an excel file'''
    return pd.read_excel(file_name, decimal=',', sheet_name=None)


def get_timeseries(data, year):
    '''Get the timeseries arrays from the dataframe'''
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
    '''Get the tables from the dataframe'''
    ret = data['retired_capacity'].iloc[:,:].to_numpy()
    inst = data['installed_capacity'].iloc[:,:].to_numpy()
    added = data['added_capacity'].iloc[:,:].to_numpy()
    num_households = data['num_households'].iloc[:,:].to_numpy()

    return [ret, inst, added, num_households]

def get_binaries(data):
    '''Get the binary variables from the dataframe'''
    heat_rate_binary = np.zeros((15, 3, 8, 2))
    price_binary = data['price_binary'].iloc[:, :].to_numpy()
    quantity_binary = data['quantity_binary'].iloc[:, :].to_numpy()
    for y in range(15):
        #price_binary[y] = data['price_binary_' + str(y + 1)].iloc[:, :].to_numpy()
        for d in range(3):
            heat_rate_binary[y][d] = data['heat_rate_binary_' + str(y + 1) + '_' + str(d + 1)].iloc[:, :].to_numpy()
    return [heat_rate_binary, price_binary, quantity_binary]

#--------------------------------------------------------------------------------#
# single model run                                                               #
#--------------------------------------------------------------------------------#

def single_modelrun(model, fit, el_price, ud_penalty, heatrate_c_run, dem_elasticity_c_run):
    '''Single model run with given parameters'''
    results = model.solve(fit=fit, elec_price=el_price, ud_penalty=ud_penalty,
                          heatrate_c_run=heatrate_c_run, dem_elasticity_c_run=dem_elasticity_c_run)

    save_res = input("Save results? (Yes: y, No: [Enter]):")
    if save_res == 'y':
        save_results(results)

    (ret, inst, added, disp_gen, disp_pv, disp_feedin,
     unmetD, bat_in, bat_out, state_of_charge, num_households,
     heat_rate_binary, price_binary, quantity_binary, total_demand,
     res_Demand, pros_feedin, pros_demandarray) = results

    data = {
        'retired_capacity': pd.DataFrame(ret),
        'installed_capacity': pd.DataFrame(inst),
        'added_capacity': pd.DataFrame(added),
        'num_households': pd.DataFrame(num_households),
        'price_binary': pd.DataFrame(price_binary),
        'quantity_binary': pd.DataFrame(quantity_binary)

    }
    for y in range(15):
        data['dispatched_generation_' + str(y + 1)] = pd.DataFrame(disp_gen[y])
        data['dispatched_pv_' + str(y + 1)] = pd.DataFrame(disp_pv[y])
        data['dispatched_feedin_' + str(y + 1)] = pd.DataFrame(disp_feedin[y])
        data['unmet_demand_' + str(y + 1)] = pd.DataFrame(unmetD[y])
        data['battery_input_' + str(y + 1)] = pd.DataFrame(bat_in[y])
        data['battery_output_' + str(y + 1)] = pd.DataFrame(bat_out[y])
        data['state_of_charge_' + str(y + 1)] = pd.DataFrame(state_of_charge[y])
        data['total_demand_' + str(y + 1)] = pd.DataFrame(total_demand[y])
        # data['price_binary_' + str(y + 1)] = pd.DataFrame(price_binary[y])
        for d in range(3):
            data['heat_rate_binary_' + str(y + 1) + '_' + str(d + 1)] = pd.DataFrame(heat_rate_binary[y][d])
            #data['pros_feedin_'+str(d+1)] = pd.DataFrame(pros_feedin[d])
            #data['res_Demand_'+str(d+1)] = pd.DataFrame(res_Demand[d])

    return data

def show_singlerun_data(data):
    '''Process the output data of a single model run'''
    show_tables(get_tabels(data))
    show_binaries(get_binaries(data), 1, 2)  # winter in year 1

    save_plots = input("Save plots? (Yes: y, No: [Enter]):")
    if save_plots == 'y':
        for y in range(data['num_households'].shape[1]):
            plot_days(get_timeseries(data, y), y, 'save')

    while True:
        showyear = input("Year:(1-15):")
        plot_days(get_timeseries(data, int(showyear) - 1), int(showyear) - 1)

#--------------------------------------------------------------------------------#
# generate multi run plots                                                       #
#--------------------------------------------------------------------------------#


def sum_year(nparray, year, d_weights):
    '''Sum up values of np array over a year'''
    sum = 0
    for d in range(nparray.shape[1]):
        sum += np.sum(nparray[year][d]) * d_weights[d]
    return sum

def pv_fit_modelruns(model, fit_max, num_runs, el_price, ud_penalty, day_weights):
    fit = np.round(np.linspace(0, fit_max, num_runs), 2)
    for i in range(num_runs):
        results = model.solve(fit=fit[i], elec_price=el_price, ud_penalty=ud_penalty,
                              heatrate_c_run = 'y', dem_elasticity_c_run = 'n')
        (ret, inst, added, disp_gen, disp_pv, disp_feedin,
         unmetD, bat_in, bat_out, state_of_charge, num_households,
         heat_rate_binary, price_binary, quantity_binary, total_demand,
         res_Demand, pros_feedin, pros_demandarray) = results

        disp_pv_year = np.array(
            [sum_year(disp_pv, y, day_weights) for y in range(num_households.shape[1])])
        disp_feedin_year = np.array(
            [sum_year(disp_feedin, y, day_weights) for y in range(num_households.shape[1])])
        save_array2d_to_excel(disp_pv_year, 'multirun_results.xlsx', 'disp_pv_year_' + str(fit[i]))
        save_array2d_to_excel(disp_feedin_year, 'multirun_results.xlsx', 'disp_feedin_year_' + str(fit[i]))

    save_fig = input("Save figure? (No: [Enter], Yes: y):")
    if save_fig == 'y':
        for i in range(15):
            print_pv_fit_curve(fit_max, num_runs, i, 'save')
    print_pv_fit_curve(fit_max, num_runs, 1)

def print_pv_fit_curve(fit_max, num_runs, year, s = 'plot'):
    '''Print the PV fit curve'''
    fit = np.round(np.linspace(0, fit_max, num_runs), 2)
    data = pd.read_excel('multirun_results.xlsx', sheet_name=None)
    disp_pv = pd.DataFrame()
    disp_feedin = pd.DataFrame()
    for i in range(num_runs):
        disp_pv['disp_pv_fit_'+str(fit[i])] = data['disp_pv_year_'+str(fit[i])].iloc[:, :].to_numpy().flatten()
        disp_feedin['disp_feedin_fit_'+str(fit[i])] = data['disp_feedin_year_'+str(fit[i])].iloc[:, :].to_numpy().flatten()
    fig, ax = plt.subplots()
    ax.plot(fit, disp_pv.iloc[year - 1] / 1000, label='Installed PV', marker='o')
    ax.plot(fit, disp_feedin.iloc[year -1] / 1000, label='Prosumer Feed-in', marker='o')
    ax.set_xlabel('Feed in Tarif ($/kWh)')
    ax.set_ylabel('Dispatched PV Energy per Year (MWh)')
    ax.legend(loc='upper left')
    ax.set_title('Year '+str(year+1))
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)

    if s == 'save':
        plt.savefig('plots/pv_feedin/pv_dispatch_'+str(year)+'.png')
    else:
        plt.show()


def ud_modelruns(model, ud_penalty_max, num_runs, el_price, fit, day_weights):
    ud_penalty = np.round(np.linspace(0, ud_penalty_max, num_runs), 2)
    for i in range(num_runs):
        results = model.solve(fit=fit, elec_price=el_price, ud_penalty=ud_penalty[i],
                              heatrate_c_run = 'y', dem_elasticity_c_run = 'n')
        (ret, inst, added, disp_gen, disp_pv, disp_feedin,
         unmetD, bat_in, bat_out, state_of_charge, num_households,
         heat_rate_binary, price_binary, quantity_binary, total_demand,
         res_Demand, pros_feedin, pros_demandarray) = results

        unmetD_year = np.array(
            [[sum_year(unmetD, y, day_weights), num_households[0, y], num_households[1, y]]
             for y in range(num_households.shape[1])])
        save_array2d_to_excel(unmetD_year, 'multirun_results.xlsx', 'unmetD_year_' + str(ud_penalty[i]))

    save_fig = input("Save figure? (No: [Enter], Yes: [y]):")
    if save_fig == 'y':
        for i in range(15):
            print_ud_curve(ud_penalty_max, num_runs, i, 'save')
    print_ud_curve(ud_penalty_max, num_runs, 1)

def print_ud_curve(ud_penalty_max, num_runs, year, s = 'plot'):
    '''Print the PV fit curve'''
    ud_penalty = np.round(np.linspace(0, ud_penalty_max, num_runs), 2)
    data = pd.read_excel('multirun_results.xlsx', sheet_name=None)
    unmetD = pd.DataFrame()
    num_consumers = pd.DataFrame()
    num_prosumers = pd.DataFrame()
    for i in range(num_runs):
        unmetD['unmetD_fit_'+str(ud_penalty[i])] = data['unmetD_year_'+str(ud_penalty[i])].iloc[:, 0].to_numpy()
        num_consumers['num_consumers_fit_'+str(ud_penalty[i])] = data['unmetD_year_'+str(ud_penalty[i])].iloc[:, 1].to_numpy()
        num_prosumers['num_prosumers_fit_'+str(ud_penalty[i])] = data['unmetD_year_'+str(ud_penalty[i])].iloc[:, 2].to_numpy()
    fig, ax1 = plt.subplots()
    ax1.plot(ud_penalty, unmetD.iloc[year - 1] / 1000, label='Unmet Demand', marker='o')
    ax1.set_xlabel('Unmet Demand Penalty ($/kWh)')
    ax1.set_ylabel('Unmet Energy Demand per year (MWh)')
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(left=0)

    ax2 = ax1.twinx()
    ax2.plot(ud_penalty, num_consumers.iloc[year - 1], label='Consumers', color='black', linestyle='', marker='o')
    ax2.plot(ud_penalty, num_prosumers.iloc[year - 1], label='Prosumers', color='black', linestyle='', marker='x')
    ax2.set_ylabel('Number of Households')
    ax2.set_ylim(bottom=0)

    fig.legend()
    ax1.set_title('Year '+str(year+1))
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)

    if s == 'save':
        plt.savefig('plots/unmet_demand/ud_plot_'+str(year)+'.png')
    else:
        plt.show()


def calc_lcoe_pv(filename):
    '''Calculate the LCOE of PV'''
    parameters = pd.read_excel(filename, sheet_name='parameters')
    tech_values = pd.read_excel(filename, sheet_name='tech')
    cap_factors = pd.read_excel(filename, sheet_name='cap_factors').to_numpy()
    mean_values = np.zeros(3)
    for i in range(3):
        mean_values[i] = np.mean(cap_factors[i][1:])
    cap_fac = mean_values[0] * 0.25 + mean_values[1] * 0.5 + mean_values[2] * 0.25
    interest_r = parameters['Interest rate'][0]
    lifetime = tech_values['Lifetime'][1]
    capex = tech_values['UCC'][1]
    ofc = tech_values['UOFC'][1]
    ovc = tech_values['UOVC'][1]

    alpha = (((1 + interest_r)**lifetime)*interest_r)/(((1 + interest_r)**lifetime)-1)
    return ((capex * alpha + ofc) / (cap_fac * 8760)) + ovc
