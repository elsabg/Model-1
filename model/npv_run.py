# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 10:59:50 2025

@author: Elsa
"""

import pandas as pd
import numpy as np

from model_1 import Model_1

#------------------------------------------------------------------------------#
#                                                                              #
# initialize model                                                             #
#                                                                              #
#------------------------------------------------------------------------------#

model = Model_1(_file_name='model_inputs_feedin.xlsx')
model.load_data()

#------------------------------------------------------------------------------#
#                                                                              #
# FiT search run                                                               #
#                                                                              #
#------------------------------------------------------------------------------#

# All prices
prices = [p for p in range(0, 41)]
fits = []

for price in prices:
    highest_fit = 0
    model.solve(fit=highest_fit, elec_price=price)
    while model.m.getObjective().getValue() }= 0:
        fits.append(highest_fit - 1)
        break
    highest_fit += 1
    
fvp = pd.DataFrame(np.array([prices, fits]))
fvp.index = (['prices', 'fits'])
fvp.to_excel('FeedIn_vs_Price.xlsx')

# Single price
price = 40
highest_fit = 0
model.solve(fit=highest_fit, elec_price=price)
while model.m.getObjective().getValue() >= 0:
    highest_fit += 1
    model.solve(fit=highest_fit, elec_price=price)
print(f'Highest Fit for price {price} is {highest_fit}')
