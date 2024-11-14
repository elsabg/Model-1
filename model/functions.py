# -*- coding: utf-8 -*-
'''
Created on Tue Oct 21 13:42:03 2024

@author: jakobsvolba
'''

import numpy as np
import pandas as pd

def run_model_case_1(model, max_govtariff, max_pvrent):
    '''Run model case 1 for a range of government tariffs and PV rents'''

    el_price = np.linspace(0, max_govtariff, 20)
    pv_rent = np.linspace(1, max_pvrent, 20)


    for i in el_price:
        for j in pv_rent:
            model.solve(rent=j, elec_price=i)




def output_data(resultsArray):
    '''Process output data'''

    ret, inst, added, disp_gen, unmetD, bat_in, bat_out, num_households = resultsArray

    disp_gen = pd.DataFrame(
        disp_gen, columns=[i for i in range(disp_gen.shape[1])]
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


    print('\n-----------installed capacity-----------\n')
    print(inst.round(2))
    print('\n-----------added capacity-----------\n')
    print(added.round(2))
    print('\n-----------retired capacity-----------\n')
    print(ret.round(2))
    print('\n-----------dispatched Energy Generator year 1-----------\n')
    print(disp_gen.round(2))
    print('\n-----------unmet Demand year 1-----------\n')
    print(unmetD.round(2))
    print('\n-----------battery Input year 1-----------\n')
    print(bat_in.round(2))
    print('\n-----------battery Output year 1-----------\n')
    print(bat_out.round(2))
    print('\n-----------Number of connected household types-----------\n')
    print(num_households)
    '''write output data to excel file'''
    return

