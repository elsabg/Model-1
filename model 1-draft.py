# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 21:51:57 2024

@author: Elsa
"""

import numpy as np
import pandas as pd
import math
from gurobipy import *

class Model_1:
    
    def __init__(self, _file_name):
        self._file_name=_file_name
        
    #loading model parameters
    def load_data(self):
        self.data = pd.read_excel(self._file_name, sheet_name=None)
        
        self.years = int(self.data['parameters']['Planning Horizon'][0])
        self.days = int(self.data['parameters']['Days'][0])
        self.hours = int(self.data['parameters']['Hours'][0])
        self.d_weights = self.data['day_weights']['Weights'].to_numpy()
        self.i = int(self.data['parameters']['Interest Rate'][0])
        
        #Capacity parameters
            #Household capacities
        self.max_house = self.data['rent_cap']['No available'].to_numpy()
        self.avg_pv_cap = self.data['rent_cap']['Avg PV capacity'].to_numpy()
            #DGC capacities
        self.init_cap = self.data['tech']['Initial Capacity'].to_numpy()
        self.life_0 = self.data['tech']['Remaining Lifetime'].to_numpy()
        self.life = self.data['tech']['Lifetime'].to_numpy()
            #Capcity factors
        self.cap_fact = self.data['cap_factors'].to_numpy()
        self.min_soc = int(self.data['parameters']['min SoC'][0])
        self.bat_eff = int(self.data['parameters']['Battery Eff'][0])
        
        #Costs and tariffs
        self.ucc = self.data['tech']['UCC'].to_numpy()
        self.uofc = self.data['tech']['UOFC'].to_numpy()
        self.uovc = self.data['tech']['UOVC'].to_numpy()
        self.heat_r = self.data['heat_rate'].to_numpy()
        self.diesel_p = self.data['tariffs']['Diesel Price'].to_numpy()
        self.max_tariff = self.data['tariffs']['Ministry Tariff'].to_numpy()
        
        #Electricity demand
        self.demand = self.data['elec_demand'].to_numpy()
        
        #Sets
        self.techs = self.data['tech'].columns.to_numpy()
        
    def solve(self, rent, elec_price):
        #Decision variables for grid search
        self.rent = rent
        self.elec_price = elec_price
        
        m=Model('Model_1_case_1')
        
        #Model decision varaibles
        added_cap=m.addVars(self.techs,)

    def testfunction(self):
        '''test commit messages'''
        print('This is a super test commit')
