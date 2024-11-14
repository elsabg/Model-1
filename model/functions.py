# -*- coding: utf-8 -*-
'''
Created on Tue Oct 21 13:42:03 2024

@author: jakobsvolba
'''

import numpy as np

def run_model_case_1(model, max_govtariff, max_pvrent):
    '''Run model case 1 for a range of government tariffs and PV rents'''

    el_price = np.linspace(0, max_govtariff, 20)
    pv_rent = np.linspace(1, max_pvrent, 20)


    for i in el_price:
        for j in pv_rent:
            model.solve(rent=j, elec_price=i)
            write_output_data(model)


def test_run_model_case_1(model):
    '''test run with fixed values'''

    pv_rent = 50
    el_price = 0.5
    model.solve(rent=pv_rent, elec_price=el_price)
    #write_output_data(model)

def write_output_data(model):
    '''write output data to excel file'''
    return

