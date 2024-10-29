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
        self.init_cap = (self.data['tech']['Initial capacity'].to_numpy())[:-1]
        self.life_0 = (self.data['tech']['Remaining lifetime'].to_numpy())[:-1]
        self.init_cap_e = (self.data['tech']['Initial capacity'].to_numpy())[-1]
        self.life_0_e = (self.data['tech']['Remaining lifetime'].to_numpy())[-1]
        self.life = (self.data['tech']['Lifetime'].to_numpy())[:-1]
        self.life_e = (self.data['tech']['Lifetime'].to_numpy())[-1]
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
        self.demand_1 = self.data['elec_demand (1)'].iloc[:, 1:].to_numpy()
        self.demand_2 = self.data['elec_demand (2)'].iloc[:, 1:].to_numpy()
        self.demand_3 = self.data['elec_demand (3)'].iloc[:, 1:].to_numpy()
        self.demand_4 = self.data['elec_demand (4)'].iloc[:, 1:].to_numpy()
        self.demand_5 = self.data['elec_demand (5)'].iloc[:, 1:].to_numpy()
        self.demand = [self.demand_1, self.demand_2, 
                       self.demand_3, self.demand_4,
                       self.demand_5]
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

        '''
        Year 0 is outside of the planning horizon. The decisions start at year 
        1, while year 0 only holds initial capacities.
        '''

        added_cap = m.addVars(self.techs, self.years + 1, name='addedCap')
        added_cap_e = m.addVars(self.years + 1, name='addedCapE')
        b_in = m.addVars(self.years + 1, self.days, self.hours, name='bIn')
        b_out = m.addVars(self.years + 1, self.days, self.hours, name='bOut')
        inst_cap = m.addVars(self.techs, self.years, name='instCap')
        inst_cap_e = m.addVars(self.years + 1, name='instCapE')
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

        #heat rate binary variables
        b = m.addVars(len(self.heat_r_k), self.years, 
                      self.days, self.hours, 
                      vtype=GRB.BINARY, name='b')
        heat_r = np.zeros(self.years + 1, self.days, self.hours)

        #----------------------------------------------------------------------#
        #                                                                      #
        # Objective function                                                   #
        #                                                                      #
        #----------------------------------------------------------------------#

        #Setting up the costs and revenues for each year
        tr = np.zeros(self.years + 1) #total yearly revenues
        tcc = np.zeros(self.years + 1) #total yearly capital costs
        tovc = np.zeros(self.years + 1) #total yearly operation variable costs
        tofc = np.zeros(self.years + 1) #total yearly operation fixed costs
        tcud = np.zeros(self.years + 1) #total yearly cost of unmet demand


        for y in range(1, self.years + 1):
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
                    ) for g in self.techs
                ) + added_cap_e[y] * self.ucc[-1]
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
        m.addConstrs(
            (
                h_weight[i][y] <= self.max_house[i][y]
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
                for y in range(self.life + 1, self.years))
                for g in self.techs_o
            ),
            "Retirement after initial capacity"
        )
        m.addConstrts(
            (
                ((ret_cap[g][y] == 0)
                for y in range(self.life_0 + 1, self.life))
                for g in self.techs_o
            ),
            "Retirement between initial capacity and life"
        )

        # Rented Capacity
        m.addConstrts(
            (
                (inst_cap['Rented PV'][y] == ren_cap[g][y])
                for y in range (1, self.years)
            ),
            "Tracking rented capacity" 
        )
        m.addConstrt(
            (inst_cap['Rented PV'][y] == 0),
            "Initial rented capacity"
        )
        m.addConstrs(
            (
                ren_cap[g][y] <= 
                quicksum(self.h_weight[i][y] * self.avg_pv_cap[i] for i in self.house)
                for y in range(self.years)
            ),
            "Maximum rented capacity"
        )

        # Dispatch
        m.addConstrts(
            (
                (disp['Diesel Generator'][y][d][h] <= 
                inst_cap['Diesel Generator'][y])
                for h in range(self.hours)
                for d in range(self.days)
                for y in range(self.years)
            ),
            "Maximum DG dispatch"
        )
        m.addConstrts(
            (
                (disp[g][y][d][h] <=
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
                (b_in[y][d][h] <=
                inst_cap['Batteries'][y])
                for y in range(self.years)
                for d in range(self.days)
                for h in range(self.hours)
            ),
            "Maximum battery input"
        )
        m.addConstrts(
            (
                (b_out[y][d][h] <=
                inst_cap['Batteries'][y])
                for y in range(self.years)
                for d in range(self.days)
                for h in range(self.hours)
            ),
            "Maximum battery output"
        )

        # Heat Rate
        bigM = max(self.inst_cap['Diesel Generator'])

        m.addConstrts(
            (
                (disp['Diesel Generator'][y][d][h] >= 0)
                for y in range(self.years)
                for d in range(self.days)
                for h in range(self.hours)
            ), 
            'heat rate 1.1'
        )
        m.addConstrts(
            (
                (disp['Diesel Generator'][y][d][h] <= 
                inst_cap['Diesel Generator'][y] * 0.25 
                + bigM * (1-b[0][y][d][h]))
                for y in range(self.years)
                for d in range(self.days)
                for h in range(self.hours)
            ), 
            'heat rate 1.2'
        )
        m.addConstrts(
            (
                (disp['Diesel Generator'][y][d][h] >= 
                inst_cap['Diesel Generator'][y] * 0.25 
                - bigM * (1-b[1][y][d][h]))
                for y in range(self.years)
                for d in range(self.days)
                for h in range(self.hours)
            ), 
            'heat rate 2.1'
        )
        m.addConstrts(
            (
                (disp['Diesel Generator'][y][d][h] <= 
                inst_cap['Diesel Generator'][y] * 0.5 
                + bigM * (1-b[1][y][d][h]))
                for y in range(self.years)
                for d in range(self.days)
                for h in range(self.hours)
            ), 
            'heat rate 2.2'
        )
        m.addConstrts(
            (
                (disp['Diesel Generator'][y][d][h] >= 
                inst_cap['Diesel Generator'][y] * 0.5 
                - bigM * (1-b[2][y][d][h]))
                for y in range(self.years)
                for d in range(self.days)
                for h in range(self.hours)
            ), 
            'heat rate 3.1'
        )
        m.addConstrts(
            (
                (disp['Diesel Generator'][y][d][h] <= 
                inst_cap['Diesel Generator'][y] * 0.75 
                + bigM * (1-b[2][y][d][h]))
                for y in range(self.years)
                for d in range(self.days)
                for h in range(self.hours)
            ), 
            'heat rate 3.2'
        )
        m.addConstrts(
            (
                (disp['Diesel Generator'][y][d][h] >= 
                inst_cap['Diesel Generator'][y] * 0.75 
                - bigM * (1-b[4][y][d][h]))
                for y in range(self.years)
                for d in range(self.days)
                for h in range(self.hours)
            ), 
            'heat rate 4.1'
        )
        m.addConstrts(
            (
                (disp['Diesel Generator'][y][d][h] <= 
                inst_cap['Diesel Generator'][y] 
                + bigM * (1-b[4][y][d][h]))
                for y in range(self.years)
                for d in range(self.days)
                for h in range(self.hours)
            ), 
            'heat rate 4.2'
        )
        m.addConstrts(
            (
                quicksum(b[j][y][d][h] for j in range(4)) == 1
                for y in range(self.years)
                for d in range(self.days)
                for h in range(self.hours)
            ),
            'heat rate binary'
        )

        # Storage
        m.addConstrts(
            (
                (soc[y][d][h] == soc[y-1][d][h] 
                + self.bat_eff * b_in[y][d][h]
                - b_out[y][d][h] / self.bat_eff)
                for y in range(1, self.years)
                for d in range(self.days)
                for h in range(self.hours)
            ),
            'SoC tracking'
        )
        m.addConstrts(
            (
                (soc[y][d][0] == soc[y][d][23])
                for y in range(self.years)
                for d in range(self.days)
            ),
            ' SoC of representative periods'
        )
        m.addConstrts(
            (
                (self.min_soc * self.inst_cap_e[y] <=
                soc[y][d][h])
                for y in range(self.years)
                for d in range(self.days)
                for h in range(self.hours)
            ),
            'SoC capacity 1'
        )
        m.addConstrts(
            (
                (self.inst_cap_e[y] >=
                soc[y][d][h])
                for y in range(self.years)
                for d in range(self.days)
                for h in range(self.hours)
            ),
            'SoC capacity 2'
        )
        m.addConstrts(
            (
                (inst_cap_e[y] == 
                inst_cap_e[y-1] + added_cap_e[y]
                - ret_cap_e[y])
                for y in range(1, self.years)
            ), 
            "Tracking storage capacity"
        )
        m.addConstrt(
            (
                inst_cap_e[0] == self.init_cap_e
            ),
            "Initial storage capacity"
        )

        # Storage retirement
        m.addConstrt(
            (
                ret_cap_e[self.life_0_e - 1] == self.init_cap_e
            ),
            "Retirement of initial storage capacity"
        )
        m.addConstrts(
            (
                (ret_cap_e[y] == 0)
                for y in range(self.life_0_e) 
            ),
            "Retirement before initial capacity"
        )
        m.addConstrts(
            (
                ((ret_cap_e[y] == added_cap_e[y - self.life_e])
                for y in range(self.life_0_e + 1, self.years))
            ),
            "Retirement after initial capacity"
        )

        # Tariff constraints
        m.addConstrts(
            (
                (p_DGC[y] <= self.max_tariff[y])
                for y in range(self.years)
            ),
            'maximum tariff'
        )

        m.optimize()

        #----------------------------------------------------------------------#
        #                                                                      #
        # Output                                                               #
        #                                                                      #
        #----------------------------------------------------------------------#

        ret = np.zeros(len(self.techs), self.years + 1) # retired capacity
        inst = np.zeros(len(self.techs), self.years + 1) # installed capacity
        added = np.zeros(len(self.techs), self.years + 1) # added capacity
        rented = np.zeros(len(self.techs), self.years + 1) # rented capacity
        ret_e = np.zeros(self.years + 1) # retired energy capacity
        inst_e = np.zeros(self.years + 1) # installed energy capacity
        added_e = np.zeros(self.years + 1) # added energy capacity

        for y in range(self.years + 1):
            for g in range(len(self.techs)):
                ret[g][y] = ret_cap[g][y].X
                inst[g][y] = inst_cap[g][y].X
                added[g][y] = added_cap[g][y].X
                rented[g][y] = ren_cap[g][y].X
            ret_e[y] = ret_cap_e[y].X
            inst_e[y] = inst_cap_e[y].X
            added_e[y] = added_cap_e[y].X
        
        ret = pd.DataFrame(
            ret, columns=[i for i in range(self.years + 1)]
            )
        inst = pd.DataFrame(
            inst, columns=[i for i in range(self.years + 1)]
            )
        added = pd.DataFrame(
            added, columns=[i for i in range(self.years + 1)]
            )
        rented = pd.DataFrame(
            rented, columns=[i for i in range(self.years + 1)]
            )
        ret_e = pd.Series(
            ret_e, columns=[i for i in range(self.years + 1)]
            )
        inst_e = pd.Series(
            inst_e, columns=[i for i in range(self.years + 1)]
            )
        added_e = pd.Series(
            added_e, columns=[i for i in range(self.years + 1)]
            )
        
        print(ret)
        print(inst)
        print(added)
        print(rented)
        print(ret_e)
        print(inst_e)
        print(added_e)