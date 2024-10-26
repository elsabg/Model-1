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
        'read the excel file'

        self.data = pd.read_excel(self._file_name, sheet_name=None)
        
        self.years = int(self.data['parameters']['Planning horizon'][0])
        self.days = int(self.data['parameters']['Days'][0])
        self.hours = int(self.data['parameters']['Hours'][0])
        self.d_weights = self.data['day_weights']['Weight'].to_numpy()
        self.i = int(self.data['parameters']['Interest rate'][0])
        
        #Capacity parameters
            #Household capacities
        self.max_house = self.data['rent_cap'].loc[0].iloc[1::].to_numpy()
        self.avg_pv_cap = self.data['rent_cap'].loc[1].iloc[1::].to_numpy()
            #DGC capacities
        self.init_cap = self.data['tech']['Initial capacity'].to_numpy()
        self.life_0 = self.data['tech']['Remaining lifetime'].to_numpy()
        self.life = self.data['tech']['Lifetime'].to_numpy()
            #Capacity factors
        self.cap_fact = self.data['cap_factors'].to_numpy()
        self.min_soc = int(self.data['parameters']['min SoC'][0])
        self.bat_eff = int(self.data['parameters']['Battery Eff'][0])
        
            #Costs and tariffs
        self.ucc = self.data['tech']['UCC'].to_numpy()
        self.uofc = self.data['tech']['UOFC'].to_numpy()
        self.uovc = self.data['tech']['UOVC'].to_numpy()
        self.heat_r_k = self.data['heat_rate']['HR'].to_numpy()
        self.diesel_p = self.data['tariffs']['Diesel Price'].to_numpy()
        self.max_tariff = self.data['tariffs']['Ministry Tariff'].to_numpy()
        
            #Electricity demand
        self.demand = self.data['elec_demand'].to_numpy()
        
            #Sets
        self.techs = self.data['tech'].columns.to_numpy()[1:-1]
        self.techs_g = ['Diesel Generator', 'Owned PV', 'Rented PV']
        self.techs_o = ['Diesel Generator', 'Owned PV', 'Owned Batteries']
        self.house = self.data['rent_cap'].columns.to_numpy()[1::]
        
    def solve(self, rent, elec_price):
        'Create and solve the model'

        #Decision variables for grid search
        self.rent = rent
        self.elec_price = elec_price
        
        m=Model('Model_1_case_1')

        #----------------------------------------------------------------------#
        #                                                                      #
        # Decision Variables                                                   #
        #                                                                      #
        #----------------------------------------------------------------------#

        added_cap = m.addVars(self.techs, self.years, name='addedCap')
        added_cap_e = m.addVars(self.years, name='addedCapE')
        b_in = m.addVars(self.years, self.days, self.hours, name='bIn')
        b_out = m.addVars(self.years, self.days, self.hours, name='bOut')
        inst_cap = m.addVars(self.techs, self.years, name='instCap')
        inst_cap_e = m.addVars(self.years, name='instCapE')
        disp = m.addVars(self.techs, self.years, 
                         self.days, self.hours, 
                         name='disp')
        ret_cap = m.addVars(self.techs, self.years, name='retiredCap')
        ret_cap_e = m.addVars(self.years, name='retiredCapE')
        soc = m.addVars(self.years, self.days, self.hours, name='SoC')
        p_DGC = m.addVars(self.years, name='priceDGC')
        ud = m.addVars(self.years, self.days, self.hours, name='unmetDemand')
        h_weight = m.addVars(self.house, name='houseWeight')

        #case 1-specific decision variables
        rent = m.addVar(name='rent')
        ren_cap = m.addVars(self.techs, self.years, name='renCap')

        #heat rate variables
        b = m.addVars(self.heat_r_k, self.years, 
                      self.days, self.hours, 
                      vtype=GRB.BINARY, name='b')

        #----------------------------------------------------------------------#
        #                                                                      #
        # Objective function                                                   #
        #                                                                      #
        #----------------------------------------------------------------------#

        #Setting up the costs and revenues for each year
        tr = np.zeros(self.years) #total yearly revenues
        tcc = np.zeros(self.years) #total yearly capital costs
        tovc = np.zeros(self.years) #total yearly operation variable costs
        tofc = np.zeros(self.years) #total yearly operation fixed costs
        tcud = np.zeros(self.years) #total yearly cost of unmet demand


        for y in range(self.years):
            tr[y] = quicksum(
                (
                    (
                        (
                            disp[g][y][d][h] * self.p_DGC[y]
                        ) for h in range(self.hours)
                    ) * self.d_weights[d]
                    for d in range(self.days)
                ) for g in self.techs
            ) 
            tcc[y] = quicksum(
                (
                    (
                        added_cap[g][y] * self.ucc[g]
                    ) for g in techs
                ) + added_cap_e[y] * self.ucc[4]
            )
            tovc[y] = quicksum(
                (
                    (
                        (
                            (
                                disp[g][y][d][h]
                            ) for h in range(self.hours)
                        ) * self.d_weights[d]
                    ) for d in range(self.days)
                ) * self.uovc[g] for g in self.techs
            ) + quicksum(
                (
                    (
                        (
                            self.heat_r * disp[g][y][d][h] * self.diesel_p[y]
                        ) for h in range(self.hours)
                    ) * self.d_weights[d]
                ) for d in range(self.days)
            )
            tofc[y] = quicksum(
                (
                    (
                        inst_cap[g][y] * self.uofc[g]
                    ) for g in self.techs
                ) + ren_cap[y] * rent
            )
        
        # Net Present Value of Total Profits
        tp_npv = quicksum(
            (
                (
                    tr[y] - tcc[y] - tofc[y] - tovc[y] #yearly profits
                ) * ( 1 / ((1 + self.i) ** y)) # discount factor
            ) for y in range(self.years)
        )

        m.setObjective(tp_npv, GRB.MAXIMIZE)

        #----------------------------------------------------------------------#
        #                                                                      #
        # Constraints                                                          #
        #                                                                      #
        #----------------------------------------------------------------------#

        # Supply-Demand Balance
        m.addConstrts(
            (
                (quicksum(disp[g][y][d][h] for g in self.techs) 
                + ud[y][d][h] + b_out[y][d][h] ==
                quicksum(h_weight[i][y] * self.demand[i][y][g][h] 
                for i in self.house) + b_in[y][d][h])
                for h in range(self.hours)
                for d in range(self.days)
                for y in range(self.years)
            ),
            "Supply-demand balance"
        )
        m.addConstrts(
            (
                (h_weight[i][y] =< self.max_house[i][y])
                for i in self.house
                for y in range(self.years)
            ),
            "Maximum connected houses"
        )

        # Generator Capacity
        m.addConstrts(
            (
                (inst_cap[g][y] == 
                inst_cap[g][y-1] + added_cap[g][y]
                - ret_cap[g][y])
                for g in self.techs_o
                for y in range(1, self.years)
            ), 
            "Tracking capacity"
        )
        m.addConstrts(
            (
                (inst_cap[g][0] == self.init_cap[g])
                for g in self.techs_o
            ),
            "Initial capacity"
        )

        # Generator retirement
        m.addConstrts(
            (
                (ret_cap[g][self.life_0 - 1] == self.init_cap[g])
                for g in self.techs_o
            ),
            "Retirement of initial capacity"
        )
        m.addConstrts(
            (
                (ret_cap[g][y] == 0)
                for g in self.techs_o
                for y in range(self.life_0) 
            ),
            "Retirement before initial capacity"
        )
        m.addConstrts(
            (
                ((ret_cap[g][y] == added_cap[g][y - self.life[g]])
                for y in range(self.life_0 + 1, self.years))
                for g in self.techs_o
            ),
            "Retirement after initial capacity"
        )

        # Rented Capacity
        m.addConstrts(
            (
                (inst_cap['Rented PV'][y] == rent_cap[g][y])
                for y in range (1, self.years)
            ),
            "Tracking rented capacity" 
        )
        m.addConstrt(
            (inst_cap['Rented PV'][y] == 0),
            "Initial rented capacity"
        )
        m.addConstrts(
            (
                (ren_cap[g][y] =< 
                quicksum((self.h_weight[i][y] * self.avg_pv_cap[i])
                for i in self.house)
                for y in range(self.years))
            ),
            "Maximum rented capacity"
        )

        # Dispatch
        m.addConstrts(
            (
                (disp['Diesel Generator'][y][d][h] =< 
                inst_cap['Diesel Generator'][y])
                for h in range(self.hours)
                for d in range(self.days)
                for y in range(self.years)
            ),
            "Maximum DG dispatch"
        )
        m.addConstrts(
            (
                (disp[g][y][d][h] =<
                self.cap_fact[y][d][h] * inst_cap[g][y])
                for g in ['Owned PV', "Rented PV"]
                for h in range(self.hours)
                for d in range(self.days)
                for y in range(self.years)
            ),
            "Maximum PV dispatch"
        )
        m.addConstrts(
            (
                (b_in[y][d][h] =<
                inst_cap['Batteries'][y])
                for y in range(self.years)
            ),
            "Maximum battery input"
        )
        m.addConstrts(
            (
                (b_out[y][d][h] =<
                inst_cap['Batteries'][y])
                for y in range(self.years)
            ),
            "Maximum battery output"
        )
