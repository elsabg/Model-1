# -*- coding: utf-8 -*-
'''
Created on Tue Oct  14:18:06 2024

@author: jakobsvolba
'''

import numpy as np

from model_1_draft import Model_1



#-------------------------------------------------------------------------------#
#                                                                               #
# initialize model                                                              #
#                                                                               #
#-------------------------------------------------------------------------------#
model = Model_1(_file_name='model_inputs.xlsx')

#read max_govTariff out of 'model_inputs.xlsx'

el_price = np.linspace(0, max_govTariff, 20)

#-------------------------------------------------------------------------------#
#                                                                               #
# model run case 1: PV capacity rent                                            #
#                                                                               #
#-------------------------------------------------------------------------------#
pv_rent = np.linspace(1, max_pvRent, 20)

for i in el_price:
    for j in pv_rent:
        model.solve(rent=j, elec_price=i)
        # write output data to a file


#-------------------------------------------------------------------------------#
#                                                                               #
# model run case 2: feed in tariff                                              #
#                                                                               #
#-------------------------------------------------------------------------------#




#-------------------------------------------------------------------------------#
#                                                                               #
# Process output data                                                           #
#                                                                               #
#-------------------------------------------------------------------------------#
