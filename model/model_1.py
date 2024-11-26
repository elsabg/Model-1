# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 21:51:57 2024

@author: Elsa & Jakob
"""

import numpy as np
import pandas as pd
from gurobipy import *

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

class Model_1:

    def __init__(self, _file_name):
        self._file_name=_file_name

    #loading model parameters
    def load_data(self):
        'read the excel file'

        self.data = pd.read_excel(self._file_name, decimal=',', sheet_name=None)
        self.tech_df = self.data['tech'].set_index('Unnamed: 0')

        #-------------------------------------------------------------------------------#
        # Time Parameters                                                               #
        #-------------------------------------------------------------------------------#

        self.years = int(self.data['parameters']['Planning horizon'][0])
        self.days = int(self.data['parameters']['Days'][0])
        self.hours = int(self.data['parameters']['Hours'][0])
        self.d_weights = self.data['day_weights']['Weight'].to_numpy()

        #-------------------------------------------------------------------------------#
        # Capacity Parameters                                                           #
        #-------------------------------------------------------------------------------#

        #Initial Generation Capacities
        self.init_cap = self.tech_df['Initial capacity'].iloc[:-1].to_dict()
        self.init_cap_e = (self.data['tech']['Initial capacity'].to_numpy())[-1]

        #Household capacities
        self.max_house = self.data['rent_cap'].loc[0].iloc[1::].to_numpy()
        self.avg_pv_cap = self.data['rent_cap'].loc[1].iloc[1::].to_numpy()
        self.cap_fact = self.data['cap_factors'].iloc[:, 1:].to_numpy()

        #Capacities accessible via strings
        self.max_house_str = {
            'Type 1': self.max_house[0],
            'Type 2': self.max_house[1],
            'Type 3': self.max_house[2],
            'Type 4': self.max_house[3],
            'Type 5': self.max_house[4]
        }
        self.avg_pv_cap_str = {
            'Type 1': self.avg_pv_cap[0],
            'Type 2': self.avg_pv_cap[1],
            'Type 3': self.avg_pv_cap[2],
            'Type 4': self.avg_pv_cap[3],
            'Type 5': self.avg_pv_cap[4]
        }

        #-------------------------------------------------------------------------------#
        # Lifetime                                                                      #
        #-------------------------------------------------------------------------------#

        #Remaining lifetime
        self.life_0 = self.tech_df['Remaining lifetime'].iloc[:-1].to_dict()
        self.life_0_e = (self.data['tech']['Remaining lifetime'].to_numpy())[-1]

        #Technology lifetime
        self.life = self.tech_df['Lifetime'].iloc[:-1].to_dict()
        self.life_e = (self.data['tech']['Lifetime'].to_numpy())[-1]

        #-------------------------------------------------------------------------------#
        # Costs                                                                         #
        #-------------------------------------------------------------------------------#

        #Technology costs
        self.ucc = self.tech_df['UCC'].to_dict()
        self.uofc = self.tech_df['UOFC'].to_dict()
        self.uovc = self.tech_df['UOVC'].to_dict()

        #Unmet demand
        self.ud_penalty = self.data['parameters']['Unmet demand penalty'][0]

        #fixed heat rate value
        self.heat_r_v = 0.25

        #heat rate curve
        self.heat_r_k = self.data['heat_rate']['HR'].to_numpy()

        self.diesel_p = self.data['tariffs']['Diesel Price'].to_numpy()

        #-------------------------------------------------------------------------------#
        # Electricity Demand                                                            #
        #-------------------------------------------------------------------------------#

        #Household Types
        self.house = self.data['rent_cap'].columns.to_numpy()[1::]

        #Demand
        self.demand_1 = self.data['elec_demand (1)'].iloc[:, 1:].to_numpy()
        self.demand_2 = self.data['elec_demand (2)'].iloc[:, 1:].to_numpy()
        self.demand_3 = self.data['elec_demand (3)'].iloc[:, 1:].to_numpy()
        self.demand_4 = self.data['elec_demand (4)'].iloc[:, 1:].to_numpy()
        self.demand_5 = self.data['elec_demand (5)'].iloc[:, 1:].to_numpy()


        # Residual Demand (without PV)
        self.res_demand = {
            'Type 1': self.demand_1.tolist(),
            'Type 2': self.demand_2.tolist(),
            'Type 3': self.demand_3.tolist(),
            'Type 4': self.demand_4.tolist(),
            'Type 5': self.demand_5.tolist()
        }

        # feed in energy from prosumers
        self.pros_feedin = {
            'Type 1': self.demand_1.tolist(),
            'Type 2': self.demand_2.tolist(),
            'Type 3': self.demand_3.tolist(),
            'Type 4': self.demand_4.tolist(),
            'Type 5': self.demand_5.tolist()
        }

        for h_type in self.res_demand:
            for i in range(len(self.res_demand[h_type])):
                for j in range(len(self.res_demand[h_type][i])):
                    self.res_demand[h_type][i][j] = max(0, (self.res_demand[h_type][i][j]
                        - self.cap_fact[i][j] * self.avg_pv_cap_str[h_type]))

                    self.pros_feedin[h_type][i][j] = max(0, (self.cap_fact[i][j] * self.avg_pv_cap_str[h_type]
                        - self.res_demand[h_type][i][j]))


        #-------------------------------------------------------------------------------#
        # Battery and other Parameters                                                  #
        #-------------------------------------------------------------------------------#

        self.min_soc = self.data['parameters']['min SoC'][0]
        self.bat_eff = self.data['parameters']['Battery Eff'][0]

        self.i = self.data['parameters']['Interest rate'][0]
        self.max_tariff = self.data['tariffs']['Ministry Tariff'].to_numpy()

        self.cap_steps = self.data['capacity_steps']['Diesel Generator'].to_numpy()

        #------------------------------------------------------------------------------#
        # Sets                                                                         #
        #------------------------------------------------------------------------------#

        self.techs = self.data['tech'].iloc[:-1, 0].to_numpy()
        self.techs_g = self.techs[:3] # ['Diesel Generator', 'Owned PV', 'Feed In Prosumers']
        self.techs_g_o = self.techs[:2] # ['Diesel Generator', 'Owned PV']
        self.techs_o = np.array(['Diesel Generator', 'Owned PV', 'Owned Batteries'])



    def solve(self, fit, elec_price):
        'Create and solve the model'

        self.fit = fit
        self.elec_price = elec_price

        m = Model('Model_1_case_1')

        '''
        Year 0 is outside of the planning horizon. The decisions start at year
        1, while year 0 only holds initial capacities.
        '''

        #----------------------------------------------------------------------#
        #                                                                      #
        # Decision Variables                                                   #
        #                                                                      #
        #----------------------------------------------------------------------#

        added_cap = m.addVars(self.techs_o, self.years + 1, name='addedCap', lb = 0, vtype = GRB.INTEGER)
        added_cap_e = m.addVars(self.years + 1, name='addedCapE', lb = 0, vtype = GRB.INTEGER)

        inst_cap = m.addVars(self.techs_o, self.years + 1, name='instCap', lb=0)
        inst_cap_e = m.addVars(self.years + 1, name='instCapE', lb=0)

        disp = m.addVars(self.techs_g, self.years + 1, self.days, self.hours, name='disp', lb=0)

        feed_in = m.addVars(self.house, self.years + 1, self.days, self.hours, name='feedIn', lb = 0)

        b_in = m.addVars(self.years + 1, self.days, self.hours, name='bIn', lb = 0)
        b_out = m.addVars(self.years + 1, self.days, self.hours, name='bOut', lb = 0)

        ret_cap = m.addVars(self.techs_o, self.years + 1, name='retiredCap', lb = 0)
        ret_cap_e = m.addVars(self.years + 1, name='retiredCapE',  lb = 0)

        soc = m.addVars(self.years + 1, self.days, self.hours, name='SoC', lb = 0)

        ud = m.addVars(self.years + 1, self.days, self.hours, name='unmetDemand', lb = 0)

        h_weight = m.addVars(self.house, self.years + 1, name='houseWeight', lb = 0, vtype=GRB.INTEGER)

        int_cap_steps = m.addVars(len(self.cap_steps), self.years +1, name = 'binCapSteps', vtype=GRB.INTEGER, lb = 0)

        bin_heat_rate = m.addVars(len(self.heat_r_k), self.years + 1,
                      self.days, self.hours,
                      vtype=GRB.BINARY, name='binHeatRate')


        #----------------------------------------------------------------------#
        #                                                                      #
        # Objective function                                                   #
        #                                                                      #
        #----------------------------------------------------------------------#

        tr = [0] * (self.years + 1) #total yearly revenues
        tcc = [0] * (self.years + 1) #total yearly capital costs
        tovc = [0] * (self.years + 1) #total yearly operation variable costs
        tofc = [0] * (self.years + 1) #total yearly operation fixed costs
        tcud = [0] * (self.years + 1) #total yearly cost of unmet demand

        for y in range(1, self.years + 1):

            # Revenue
            tr[y] = quicksum(
                ((disp[g, y, d, h] + b_out[y, d, h] - b_in[y, d, h]) * self.d_weights[d])
                for g in self.techs_g
                for d in range(self.days)
                for h in range(self.hours)
            ) * self.elec_price

            # Capital Costs
            tcc[y] = quicksum(
                    (
                        added_cap[g, y] * self.ucc[g]
                    ) for g in self.techs_g_o
                ) + added_cap_e[y] * self.ucc['Owned Batteries']


            # Operation Variable Costs with fixed DG heat rate value
            tovc[y] = quicksum(
                disp[g, y, d, h] * self.d_weights[d] * self.uovc[g]
                for g in self.techs_g_o
                for d in range(self.days)
                for h in range(self.hours)
            ) + quicksum(
                (b_out[y, d, h] + b_in[y, d, h]) * self.d_weights[d] * self.uovc['Owned Batteries']
                for d in range(self.days)
                for h in range(self.hours)
            ) + quicksum(
                self.heat_r_v * disp['Diesel Generator', y, d, h] * self.diesel_p[y - 1] * self.d_weights[d]
                for d in range(self.days)
                for h in range(self.hours)
            ) + quicksum(
                quicksum(feed_in[i, y, d, h] for i in self.house) * self.fit
                for d in range(self.days)
                for h in range(self.hours)
            )

            '''
            # Operation Variable Costs with DG heat rate curve
            tovc[y] = (quicksum(
                disp[g, y, d, h] * self.d_weights[d] * self.uovc[g]
                for g in self.techs_g_o
                for d in range(self.days)
                for h in range(self.hours)
            ) + quicksum(
                (b_out[y, d, h] + b_in[y, d, h]) * self.d_weights[d] * self.uovc['Owned Batteries']
                for d in range(self.days)
                for h in range(self.hours)
            ) + quicksum(
                quicksum(self.heat_r_k[i] * bin_heat_rate[i, y, d, h] for i in range(len(self.heat_r_k)))
                    * disp['Diesel Generator', y, d, h] * self.diesel_p[y - 1] * self.d_weights[d]
                for d in range(self.days)
                for h in range(self.hours)
            ) + quicksum(
                quicksum(feed_in[i, y, d, h] for i in self.house) * self.fit
                for d in range(self.days)
                for h in range(self.hours)
            ))
            '''

            # Operation Fixed Costs
            tofc[y] = quicksum(
                (
                    inst_cap[g, y] * self.uofc[g]
                    for g in self.techs_o
                )
            )

            # Cost of Unmet Demand
            tcud[y] = quicksum(
                ud[y, d, h] * self.d_weights[d] * self.ud_penalty
                for d in range(self.days)
                for h in range(self.hours)
            )


        # Net Present Value of Total Profits
        tp_npv = quicksum(
            (
                (
                    tr[y] - tcc[y] - tofc[y] - tcud[y]- tovc[y] #yearly profits
                ) * ( 1 / ((1 + self.i) ** y)) # discount factor
            ) for y in range(1, self.years + 1)
        )

        m.setObjective(tp_npv, GRB.MAXIMIZE)

        #----------------------------------------------------------------------#
        #                                                                      #
        # Constraints                                                          #
        #                                                                      #
        #----------------------------------------------------------------------#

        #----------------------------------------------------------------------#
        # Demand and Dispatch                                                  #
        #----------------------------------------------------------------------#

        m.addConstrs(
            (
                (quicksum(
                    disp[g, y, d, h] for g in self.techs_g)
                    + ud[y, d, h] + b_out[y, d, h] ==
                quicksum(
                    h_weight[i, y] * self.res_demand[i][d][h]
                    for i in self.house) + b_in[y, d, h])
                for h in range(self.hours)
                for d in range(self.days)
                for y in range(1, self.years + 1)
            ),
            "Supply-demand balance"
        )

        m.addConstrs(
            (
                (disp['Diesel Generator', y, d, h] <=
                 inst_cap[('Diesel Generator', y)])
                for h in range(self.hours)
                for d in range(self.days)
                for y in range(1, self.years + 1)
            ),
            "Maximum DG dispatch"
        )
        m.addConstrs(
            (
                (disp['Owned PV', y, d, h] <=
                 self.cap_fact[d][h] * inst_cap['Owned PV', y])
                for h in range(self.hours)
                for d in range(self.days)
                for y in range(1, self.years + 1)
            ),
            "Maximum PV dispatch"
        )
        m.addConstrs(
            (
                (b_in[y, d, h] <=
                 inst_cap['Owned Batteries', y])
                for y in range(1, self.years + 1)
                for d in range(self.days)
                for h in range(self.hours)
            ),
            "Maximum battery input"
        )
        m.addConstrs(
            (
                (b_out[y, d, h] <=
                 inst_cap['Owned Batteries', y])
                for y in range(1, self.years + 1)
                for d in range(self.days)
                for h in range(self.hours)
            ),
            "Maximum battery output"
        )

        m.addConstrs(
            (
                h_weight[i, y] <= self.max_house_str[i]
                for i in self.house
                for y in range(1, self.years + 1)
            ),
            "Maximum connected houses"
        )

        #----------------------------------------------------------------------#
        # Generation Capacity                                                  #
        #----------------------------------------------------------------------#

        m.addConstrs(
            (
                (inst_cap[g, y] ==
                inst_cap[g, y-1] + added_cap[g, y]
                - ret_cap[g, y])
                for g in self.techs_o
                for y in range(1, self.years + 1)
            ),
            "Tracking capacity"
        )
        m.addConstrs(
            (
                (inst_cap[g, 0] == self.init_cap[g])
                for g in self.techs_o
            ),
            "Initial capacity"
        )

        m.addConstrs(
            (
                (added_cap['Diesel Generator', y] ==
                    quicksum(int_cap_steps[i, y] * self.cap_steps[i] for i in range(len(self.cap_steps))))
                for y in range(1, self.years + 1)
            ),
            "Steps for added diesel generator capacity"
        )

        # ----------------------------------------------------------------------#
        # Feed in PV from Prosumers                                             #
        # ----------------------------------------------------------------------#

        m.addConstrs(
            (
                disp['Feed In Prosumers', y, d, h] <=
                    quicksum(feed_in[i, y, d, h] for i in self.house)
                for y in range(1, self.years + 1)
                for d in range(self.days)
                for h in range(self.hours)
            ),
            "Link dispatch to feed in"
        )

        m.addConstrs(
            (
                feed_in[i, y, d, h] <= h_weight[i, y] * self.pros_feedin[i][d][h]
                for i in self.house
                for y in range(1, self.years + 1)
                for d in range(self.days)
                for h in range(self.hours)
            ),
            "max Feed in"
        )


        #----------------------------------------------------------------------#
        # Battery Operation                                                    #
        #----------------------------------------------------------------------#

        m.addConstrs(
            (
                (soc[y, d, h] == soc[y, d, h - 1]
                 + self.bat_eff * b_in[y, d, h]
                 - b_out[y, d, h] / self.bat_eff)
                for y in range(1, self.years + 1)
                for d in range(self.days)
                for h in range(1, self.hours)
            ),
            'SoC tracking'
        )
        m.addConstrs(
            (
                (soc[y, d, 0] == soc[y, d, 23]
                 + self.bat_eff * b_in[y, d, 0]
                 - b_out[y, d, 0] / self.bat_eff)
                for y in range(1, self.years + 1)
                for d in range(self.days)
            ),
            ' SoC of hour 0'
        )
        m.addConstrs(
            (
                (self.min_soc * inst_cap_e[y] <=
                 soc[y, d, h])
                for y in range(1, self.years + 1)
                for d in range(self.days)
                for h in range(self.hours)
            ),
            'SoC capacity 1'
        )
        m.addConstrs(
            (
                (inst_cap_e[y] >=
                 soc[y, d, h])
                for y in range(1, self.years + 1)
                for d in range(self.days)
                for h in range(self.hours)
            ),
            'SoC capacity 2'
        )
        m.addConstrs(
            (
                (inst_cap_e[y] ==
                 inst_cap_e[y - 1] + added_cap_e[y]
                 - ret_cap_e[y])
                for y in range(1, self.years + 1)
            ),
            "Tracking storage capacity"
        )
        m.addConstr(
            (
                    inst_cap_e[0] == self.init_cap_e
            ),
            "Initial storage capacity"
        )

        #----------------------------------------------------------------------#
        # Generation Retirement                                                #
        #----------------------------------------------------------------------#

        m.addConstrs(
            (
                (ret_cap[g, self.life_0[g] + 1] == self.init_cap[g])
                for g in self.techs_o
            ),
            "Retirement of initial capacity"
        )
        m.addConstrs(
            (
                (ret_cap[g, y] == 0)
                for g in self.techs_o
                for y in range(1, self.life_0[g] + 1)   # range(self.life_0) returns values only up to life_0 - 1
            ),
            "Retirement before initial capacity"
        )
        m.addConstrs(
            (
                (ret_cap[g, y] == added_cap[g, y - self.life[g]])
                for g in self.techs_o
                for y in range(self.life[g] + 1, self.years + 1)
            ),
            "Retirement after initial capacity"
        )

        m.addConstrs(
            (
                (ret_cap[g, y] == 0)
                for g in self.techs_o
                for y in range(self.life_0[g] + 2, min(self.life[g] + 1, self.years + 1))
            ),
            "Retirement between initial capacity and life"
        )

        #----------------------------------------------------------------------#
        # Battery Retirement                                                   #
        #----------------------------------------------------------------------#

        m.addConstr(
            (
                    ret_cap_e[self.life_0_e + 1] == self.init_cap_e
            ),
            "Retirement of initial storage capacity"
        )
        m.addConstrs(
            (
                (ret_cap_e[y] == 0)
                for y in range(1, self.life_0_e + 1)
            ),
            "Retirement before initial capacity"
        )
        m.addConstrs(
            (
                ((ret_cap_e[y] == added_cap_e[y - self.life_e])
                 for y in range(self.life_0_e + 2, self.years + 1))
            ),
            "Retirement after initial capacity"
        )

        m.addConstrs(
            (
                (ret_cap_e[y] == 0)
                for y in range(self.life_0_e + 2, min(self.life_e + 1, self.years + 1))
            ),
            "Retirement between initial capacity and life"
        )

        #----------------------------------------------------------------------#
        # Heat Rate Curve                                                      #
        #----------------------------------------------------------------------#

        '''
        bigM = 700 # find the max value of bigM

        m.addConstrs(
            (
                quicksum(bin_heat_rate[i, y, d, h] for i in range(len(self.heat_r_k))) == 1
                for y in range(1, self.years + 1)
                for d in range(self.days)
                for h in range(self.hours)
            ),
            "Sum Binary set = 1"
        )
        m.addConstrs(
            (
                (disp['Diesel Generator', y, d, h] <=
                inst_cap['Diesel Generator', y] * 0.25
                + bigM * (1-bin_heat_rate[0, y, d, h]))
                for y in range(1, self.years + 1)
                for d in range(self.days)
                for h in range(self.hours)
            ),
            'heat rate 1.2'
        )
        m.addConstrs(
            (
                (disp['Diesel Generator', y, d, h] >=
                inst_cap['Diesel Generator', y] * 0.25
                - bigM * (1-bin_heat_rate[1, y, d, h]))
                for y in range(1, self.years + 1)
                for d in range(self.days)
                for h in range(self.hours)
            ),
            'heat rate 2.1'
        )
        m.addConstrs(
            (
                (disp['Diesel Generator', y, d, h] <=
                inst_cap['Diesel Generator', y] * 0.5
                + bigM * (1-bin_heat_rate[1, y, d, h]))
                for y in range(1, self.years + 1)
                for d in range(self.days)
                for h in range(self.hours)
            ),
            'heat rate 2.2'
        )
        m.addConstrs(
            (
                (disp['Diesel Generator', y, d, h] >=
                inst_cap['Diesel Generator', y] * 0.5
                - bigM * (1-bin_heat_rate[2, y, d, h]))
                for y in range(1, self.years + 1)
                for d in range(self.days)
                for h in range(self.hours)
            ),
            'heat rate 3.1'
        )
        m.addConstrs(
            (
                (disp['Diesel Generator', y, d, h] <=
                inst_cap['Diesel Generator', y] * 0.75
                + bigM * (1-bin_heat_rate[2, y, d, h]))
                for y in range(1, self.years + 1)
                for d in range(self.days)
                for h in range(self.hours)
            ),
            'heat rate 3.2'
        )
        m.addConstrs(
            (
                (disp['Diesel Generator', y, d, h] >=
                inst_cap['Diesel Generator', y] * 0.75
                - bigM * (1-bin_heat_rate[3, y, d, h]))
                for y in range(1, self.years + 1)
                for d in range(self.days)
                for h in range(self.hours)
            ),
            'heat rate 4.1'
        )
        m.addConstrs(
            (
                (disp['Diesel Generator', y, d, h] <=
                inst_cap['Diesel Generator', y]
                + bigM * (1 - bin_heat_rate[3, y, d, h]))
                for y in range(1, self.years + 1)
                for d in range(self.days)
                for h in range(self.hours)
            ),
            'heat rate 4.2'
        )
        '''

        #----------------------------------------------------------------------#
        # Optimization                                                         #
        #----------------------------------------------------------------------#

        m.optimize()

        #----------------------------------------------------------------------#
        #                                                                      #
        # Return Output                                                        #
        #                                                                      #
        #----------------------------------------------------------------------#

        ret = np.zeros((len(self.techs_o), self.years + 1)) # retired capacity
        inst = np.zeros((len(self.techs_o), self.years + 1)) # installed capacity
        added = np.zeros((len(self.techs_o), self.years + 1)) # added capacity
        disp_gen = np.zeros((self.days, self.hours))
        unmetD = np.zeros((self.days, self.hours))
        bat_in = np.zeros((self.days, self.hours))
        bat_out = np.zeros((self.days, self.hours))
        num_households = np.zeros((len(self.house), self.years + 1))
        feed_in_energy = np.zeros((self.days, self.hours))

        for y in range(self.years + 1):
            for g in self.techs_o:
                ret[self.techs_o.tolist().index(g)][y] = ret_cap[g, y].X
                inst[self.techs_o.tolist().index(g)][y] = inst_cap[g, y].X
                added[self.techs_o.tolist().index(g)][y] = added_cap[g, y].X

        for d in range(self.days):
            for h in range(self.hours):
                disp_gen[d][h] = disp['Diesel Generator', 1, d, h].X
                unmetD[d][h] = ud[12, d, h].X
                bat_in[d][h] = b_in[12, d, h].X
                bat_out[d][h] = b_out[12, d, h].X
                feed_in_energy[d][h] = disp['Feed In Prosumers', 12, d, h].X

        for house in self.house:
            for y in range(self.years + 1):
                num_households[self.house.tolist().index(house)][y] = np.abs(h_weight[house, y].X)


        total_demand = np.zeros((self.days, self.hours))

        for house in self.house:
            for d in range(self.days):
                for h in range(self.hours):
                    total_demand[d][h] += self.res_demand[house][d][h] * num_households[self.house.tolist().index(house)][12]



        return_array = [ret, inst, added, disp_gen, unmetD, bat_in, bat_out, num_households, feed_in_energy, total_demand]

        return return_array
