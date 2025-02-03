# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 21:51:57 2024

@author: Elsa & Jakob
"""

import numpy as np
import pandas as pd
from fontTools.misc.bezierTools import epsilon
from gurobipy import *
import matplotlib.pyplot as plt

import customer_demand as cd

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
        self.init_cap = self.tech_df['Initial capacity'].to_dict()

        #Household capacities
        self.max_house = self.data['rent_cap'].loc[0].iloc[1::].to_numpy()
        self.avg_pv_cap = self.data['rent_cap'].loc[1].iloc[1::].to_numpy()
        self.cap_fact = self.data['cap_factors'].iloc[:, 1:].to_numpy()

        #Capacities accessible via strings
        self.max_house_str = {
            'Type 1': self.max_house[0],
            'Type 2': self.max_house[1],
            #'Type 3': self.max_house[2],
            #'Type 4': self.max_house[3],
            #'Type 5': self.max_house[4],
            #'Type 6': self.max_house[5]
        }
        self.avg_pv_cap_str = {
            'Type 1': self.avg_pv_cap[0],
            'Type 2': self.avg_pv_cap[1],
            #'Type 3': self.avg_pv_cap[2],
            #'Type 4': self.avg_pv_cap[3],
            #'Type 5': self.avg_pv_cap[4],
            #'Type 6': self.avg_pv_cap[5]
        }

        #-------------------------------------------------------------------------------#
        # Lifetime                                                                      #
        #-------------------------------------------------------------------------------#

        #Remaining lifetime
        self.life_0 = self.tech_df['Remaining lifetime'].to_dict()

        #Technology lifetime
        self.life = self.tech_df['Lifetime'].to_dict()

        #-------------------------------------------------------------------------------#
        # Costs                                                                         #
        #-------------------------------------------------------------------------------#

        #Technology costs
        self.ucc = self.tech_df['UCC'].to_dict()

        self.uofc = self.tech_df['UOFC'].to_dict()
        self.uovc = self.tech_df['UOVC'].to_dict()

        #Unmet demand
        #self.ud_penalty = self.data['parameters']['Unmet demand penalty'][0]

        #fixed heat rate value
        self.heat_r_v = 0.30

        #heat rate curve
        self.heat_r_k = self.data['heat_rate']['HR'].to_numpy()

        self.diesel_p = self.data['tariffs']['Diesel Price'].to_numpy()

        #-------------------------------------------------------------------------------#
        # Electricity Demand                                                            #
        #-------------------------------------------------------------------------------#

        #Household Types
        self.house = self.data['rent_cap'].columns.to_numpy()[1::]

        #demand elasticity
        self.elasticity = self.data['parameters']['demand_elasticity'][0]

        #Demand
        self.demand_1 = self.data['elec_demand (1)'].iloc[:, 1:].to_numpy()
        self.demand_2 = self.data['elec_demand (2)'].iloc[:, 1:].to_numpy()
        #self.demand_3 = self.data['elec_demand (3)'].iloc[:, 1:].to_numpy()
        #self.demand_4 = self.data['elec_demand (4)'].iloc[:, 1:].to_numpy()
        #self.demand_5 = self.data['elec_demand (5)'].iloc[:, 1:].to_numpy()
        #self.demand_6 = self.data['elec_demand (6)'].iloc[:, 1:].to_numpy()


        # Residual Demand (without PV)
        self.res_demand = {
            'Type 1': self.demand_1.tolist(),
            'Type 2': self.demand_2.tolist(),
            #'Type 3': self.demand_3.tolist(),
            #'Type 4': self.demand_4.tolist(),
            #'Type 5': self.demand_5.tolist(),
            #'Type 6': self.demand_6.tolist()
        }

        # feed in energy from prosumers
        self.pros_feedin = {
            'Type 1': np.zeros((self.demand_1.shape[0], self.demand_1.shape[1])),
            'Type 2': np.zeros((self.demand_1.shape[0], self.demand_1.shape[1])),
            #'Type 3': self.demand_3.tolist(),
            #'Type 4': self.demand_4.tolist(),
            #'Type 5': self.demand_5.tolist(),
            #'Type 6': self.demand_6.tolist()
        }


        #-------------------------------------------------------------------------------#
        # Battery and other Parameters                                                  #
        #-------------------------------------------------------------------------------#

        self.min_soc = self.data['parameters']['min SoC'][0]
        self.bat_eff = self.data['parameters']['Battery Eff'][0]

        self.bat_cap_min = 500 # kWh
        self.cap_power_ratio = 6    # 6 hours of storage
        self.ucc['Owned Batteries'] = self.ucc['Owned Batteries'] * self.cap_power_ratio # from cost per kWh to cost per kW

        self.i = self.data['parameters']['Interest rate'][0]
        self.max_tariff = self.data['tariffs']['Ministry Tariff'].to_numpy()

        self.cap_steps = self.data['capacity_steps']['Diesel Generator'].to_numpy()

        self.pros_soc_max = 4  # kwh
        self.pros_soc_min = self.pros_soc_max * self.min_soc

        #battery landuse and available land
        self.pv_landuse = 8 #m^2/kw
        self.pv_land = 10000 #m^2 (1/200 of toatl availiable land)

        #-------------------------------------------------------------------------------#
        # Calculations                                                                  #
        #-------------------------------------------------------------------------------#

        cd.calc_pros_demand_feedin(self)

        # cd.calc_res_demand(self)
        # cd.calc_pros_feedin(self)
        '''
        self.max_feedin = np.zeros(self.days)
        for h_type in self.pros_feedin:
            for d in range(self.days):
                for h in range(self.hours):
                    self.max_feedin[d] += self.pros_feedin[h_type][d][h] * self.max_house_str[h_type]

        self.max_prosdemand = np.zeros(self.days)
        for d in range(self.days):
            for h in range(self.hours):
                self.max_prosdemand[d] += self.res_demand['Type 2'][d][h] * self.max_house_str['Type 2']
        '''
        self.max_feedin = 0
        for h_type in self.pros_feedin:
            for d in range(self.days):
                for h in range(self.hours):
                    self.max_feedin += self.pros_feedin[h_type][d][h] * self.d_weights[d]

        self.max_prosdemand = 0
        for d in range(self.days):
            for h in range(self.hours):
                self.max_prosdemand += self.res_demand['Type 2'][d][h] * self.d_weights[d]


        # historic demand
        self.hist_demand = np.zeros(self.days)

        for d in range(self.days):
            self.hist_demand[d] = (sum(cd.mc_demand(self, self.max_house_str, 0, d, h) for h in range(self.hours))
                                   * self.d_weights[d]) # year 0 for access max_house_str without index year

        self.steps = 5

        self.hist_price = 0.4

        self.disp_steps_year, self.disp_steps_month, self.price_steps = cd.calc_disp_price_steps(self)

        #cd.plot_households(self)
        cd.plot_demand(self)
        #------------------------------------------------------------------------------#
        # Sets                                                                         #
        #------------------------------------------------------------------------------#

        self.techs = self.data['tech'].iloc[:-1, 0].to_numpy()
        self.techs_g = self.techs[:3] # ['Diesel Generator', 'Owned PV', 'Feed In Prosumers']
        self.techs_g_o = self.techs[:2] # ['Diesel Generator', 'Owned PV']
        self.techs_o = np.array(['Diesel Generator', 'Owned PV', 'Owned Batteries'])

    def solve(self, fit, elec_price, ud_penalty, heatrate_c_run, dem_elasticity_c_run):
        'Create and solve the model'

        self.fit = fit
        self.elec_price = elec_price
        self.ud_penalty = ud_penalty
        self.heatrate_c_run = heatrate_c_run
        self.dem_elasticity_c_run = dem_elasticity_c_run

        m = Model('Model_1_case_1')
        #m.setParam('MIPGap', 0.005)
        #m.setParam('ScaleFlag', 1)
        '''
        Year 0 is outside of the planning horizon. The decisions start at year
        1, while year 0 only holds initial capacities.
        '''

        #----------------------------------------------------------------------#
        #                                                                      #
        # Decision Variables                                                   #
        #                                                                      #
        #----------------------------------------------------------------------#

        added_cap = m.addVars(self.techs_o, self.years + 1, name='addedCap', lb=0 ) #, vtype=GRB.INTEGER)

        inst_cap = m.addVars(self.techs_o, self.years + 1, name='instCap', lb=0)

        disp = m.addVars(self.techs_g, self.years + 1, self.days, self.hours, name='disp', lb=0)

        feed_in = m.addVars(self.house, self.years + 1, self.days, self.hours, name='feedIn', lb = 0)

        b_in = m.addVars(self.years + 1, self.days, self.hours, name='bIn', lb = 0)
        b_out = m.addVars(self.years + 1, self.days, self.hours, name='bOut', lb = 0)

        ret_cap = m.addVars(self.techs_o, self.years + 1, name='retiredCap', lb = 0)

        soc = m.addVars(self.years + 1, self.days, self.hours, name='SoC', lb = 0)

        ud = m.addVars(self.years + 1, self.days, self.hours, name='unmetDemand', lb = 0)

        h_weight = m.addVars(self.house, self.years, name='houseWeight', lb = 0, vtype=GRB.INTEGER)

        int_cap_steps = m.addVars(len(self.cap_steps), self.years + 1, name = 'binCapSteps', vtype=GRB.INTEGER, lb = 0)

        bin_heat_rate = m.addVars(len(self.heat_r_k), self.years,
                      self.days, self.hours // 3,
                      vtype=GRB.BINARY, name='binHeatRate')

        bin_price_curve = m.addVars(self.steps,  #self.years,
                                    vtype=GRB.BINARY, name='binPriceCurve')

        bin_battery = m.addVars(self.years + 1, vtype=GRB.BINARY, name='binBattery')

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
            if self.dem_elasticity_c_run == 'y':
                tr[y] = quicksum(
                    ((quicksum(disp[g, y, d, h] for g in self.techs_g) + b_out[y, d, h] - b_in[y, d, h]) *
                     self.d_weights[d]) *  quicksum(self.price_steps[i] * bin_price_curve[i] for i in range(self.steps))
                    for d in range(self.days)
                    for h in range(self.hours)
                )

            else:
                tr[y] = quicksum(
                    ((quicksum(disp[g, y, d, h] for g in self.techs_g) + b_out[y, d, h] - b_in[y, d, h]) *
                     self.d_weights[d])
                    for d in range(self.days)
                    for h in range(self.hours)
                ) * self.elec_price


            # Capital Costs
            tcc[y] = quicksum(
                    (
                        added_cap[g, y] * self.ucc[g]
                    ) for g in self.techs_o
                )

            if self.heatrate_c_run == 'y':
                # Operation Variable Costs with DG heat rate curve
                tovc[y] = (quicksum(
                    quicksum(disp[g, y, d, h] * self.uovc[g] for g in self.techs_g_o) * self.d_weights[d] # battery uovc = 0
                    for d in range(self.days)
                    for h in range(self.hours)
                ) + quicksum(
                    (b_out[y, d, h] + b_in[y, d, h]) * self.d_weights[d] * self.uovc['Owned Batteries']
                    for d in range(self.days)
                    for h in range(self.hours)
                ) + quicksum(
                    quicksum(self.heat_r_k[i] * bin_heat_rate[i, y - 1, d, h // 3] for i in range(len(self.heat_r_k)))
                    * disp['Diesel Generator', y, d, h] * self.diesel_p[y - 1] * self.d_weights[d]
                    for d in range(self.days)
                    for h in range(self.hours)
                ) + quicksum(
                    quicksum(feed_in[i, y, d, h] for i in self.house) * self.fit * self.d_weights[d]
                    for d in range(self.days)
                    for h in range(self.hours)
                ))
            else:
                # Operation Variable Costs with fixed DG heat rate value
                tovc[y] = quicksum(
                    quicksum(disp[g, y, d, h] * self.uovc[g] for g in self.techs_g_o) * self.d_weights[d] # battery uovc = 0
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
                    quicksum(feed_in[i, y, d, h] for i in self.house) * self.fit * self.d_weights[d]
                    for d in range(self.days)
                    for h in range(self.hours)
                )


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
                    tr[y] - tcc[y] - tofc[y]- tovc[y] - tcud[y] #yearly profits
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

        if self.dem_elasticity_c_run == 'y':
            m.addConstrs(
                (
                    quicksum(disp[g, y, d, h] for g in self.techs_g) + b_out[y, d, h] + ud[y, d, h] ==
                    cd.elastic_mc_demand(self, bin_price_curve, y, d, h) + b_in[y, d, h]
                    for h in range(self.hours)
                    for d in range(self.days)
                    for y in range(1, self.years + 1)
                ),
                "Supply-demand balance"
            )
        else:
            m.addConstrs(
                (
                    quicksum(disp[g, y, d, h] for g in self.techs_g) + b_out[y, d, h] + ud[y, d, h] ==
                    cd.mc_demand(self, h_weight, y, d, h) + b_in[y, d, h]
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
                for y in range(self.years)
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

        m.addConstrs(
            (
                inst_cap['Owned PV', y] * self.pv_landuse <= self.pv_land
                for y in range(1, self.years + 1)
            ),
            "Restrict Available Land for PV Use"
        )

        big_M3 = 10000

        m.addConstrs(
            (
                added_cap['Owned Batteries', y] <= big_M3 * bin_battery[y]
                for y in range(1, self.years + 1)
            ),
            "Big M for Battery minimal Capacity"
        )

        m.addConstrs(
            (
                added_cap['Owned Batteries', y] >= (self.bat_cap_min / self.cap_power_ratio) * bin_battery[y]
                for y in range(1, self.years + 1)
            ),
            "Battery minimal Capacity"
        )

        # ----------------------------------------------------------------------#
        # Feed in PV from Prosumers                                             #
        # ----------------------------------------------------------------------#

        m.addConstrs(
            (
                disp['Feed In Prosumers', y, d, h] ==
                    quicksum(feed_in[i, y, d, h] for i in self.house)
                for y in range(1, self.years + 1)
                for d in range(self.days)
                for h in range(self.hours)
            ),
            "Link dispatch to feed in"
        )

        m.addConstrs(
            (
                h_weight['Type 2', y-1] - quicksum(ud[y, d, h] * self.d_weights[d]
                for h in range(self.hours)
                for d in range(self.days)
                ) / self.max_prosdemand
                >= quicksum( quicksum(feed_in[i, y, d, h]
                for i in self.house
                for h in range(self.hours)) * self.d_weights[d]
                for d in range(self.days)
                )/self.max_feedin
                for y in range(1, self.years + 1)
            ),
            "Unmet Demand balance Feed IN"
        )




        if dem_elasticity_c_run == 'y':
            m.addConstrs(
                (
                    feed_in[i, y, d, h] <= self.max_house_str[i] * self.pros_feedin[i][d][h]
                    for i in self.house
                    for y in range(1, self.years + 1)
                    for d in range(self.days)
                    for h in range(self.hours)
                ),
                "max Feed in"
            )

        else:
            m.addConstrs(
                (
                    feed_in[i, y, d, h] <= h_weight[i, y - 1] * self.pros_feedin[i][d][h]
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
                (self.min_soc * inst_cap['Owned Batteries', y] * self.cap_power_ratio <=
                 soc[y, d, h])
                for y in range(1, self.years + 1)
                for d in range(self.days)
                for h in range(self.hours)
            ),
            'SoC capacity 1'
        )
        m.addConstrs(
            (
                (inst_cap['Owned Batteries', y] * self.cap_power_ratio >=
                 soc[y, d, h])
                for y in range(1, self.years + 1)
                for d in range(self.days)
                for h in range(self.hours)
            ),
            'SoC capacity 2'
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
        # Heat Rate Curve                                                      #
        #----------------------------------------------------------------------#

        if self.heatrate_c_run == 'y':
            bigM_1 = 700  # find the max value of bigM

            m.addConstrs(
                (
                    quicksum(bin_heat_rate[i, y, d, h] for i in range(len(self.heat_r_k))) == 1
                    for y in range(self.years)
                    for d in range(self.days)
                    for h in range(self.hours // 3)
                ),
                "Sum Binary set = 1"
            )
            # epsilon = 1e-6  # Small positive value
            m.addConstrs(
                (
                    (disp['Diesel Generator', y, d, h] <=
                     inst_cap['Diesel Generator', y] * 0.25
                     + bigM_1 * (1 - bin_heat_rate[0, y, d, h // 3]) - epsilon)
                    for y in range(self.years)
                    for d in range(self.days)
                    for h in range(self.hours)
                ),
                'heat rate 1.2'
            )


            m.addConstrs(
                (
                    (disp['Diesel Generator', y, d, h] >=
                     inst_cap['Diesel Generator', y] * 0.25
                     - bigM_1 * (1 - bin_heat_rate[1, y, d, h // 3]))
                    for y in range(self.years)
                    for d in range(self.days)
                    for h in range(self.hours)
                ),
                'heat rate 2.1'
            )

        # ----------------------------------------------------------------------#
        # Price Curve (el. Demand)                                              #
        # ----------------------------------------------------------------------#

        if self.dem_elasticity_c_run == 'y':
            bigM_2 = 10 ** 8

            m.addConstr(
                (
                    quicksum(bin_price_curve[i] for i in range(self.steps)) == 1
                ),
                "Sum Binary set = 1"
            )

            year = 1

            for i in range(self.steps - 1):
                m.addConstr(
                    (
                        (cd.demand_sum_year(self, year, disp, ud, b_out, b_in) <=
                         self.disp_steps_year[i] + (1 - bin_price_curve[self.steps - 1 - i]) * bigM_2)
                    ),
                    "Year "+str(y)+"Price curve " + str(i) + ".up"
                )

                m.addConstr(
                    (
                        cd.demand_sum_year(self, year, disp, ud, b_out, b_in) >= self.disp_steps_year[i] - bigM_2 * (
                                1 - bin_price_curve[self.steps - 2 - i])
                    ),
                    "Year "+str(y)+" Price curve " + str(i + 1) + ".low"
                )

        #----------------------------------------------------------------------#
        # Optimization                                                         #
        #----------------------------------------------------------------------#

        #m.computeIIS()
        #m.write("model.ilp")
        m.optimize()



        #----------------------------------------------------------------------#
        #                                                                      #
        # Return Output                                                        #
        #                                                                      #
        #----------------------------------------------------------------------#

        ret = np.zeros((4, self.years + 1)) # retired capacity
        inst = np.zeros((4, self.years + 1)) # installed capacity
        added = np.zeros((4, self.years + 1)) # added capacity
        disp_gen = np.zeros((self.years, self.days, self.hours))
        disp_pv = np.zeros((self.years, self.days, self.hours))
        disp_feedin = np.zeros((self.years, self.days, self.hours))
        unmetD = np.zeros((self.years, self.days, self.hours))
        bat_in = np.zeros((self.years, self.days, self.hours))
        bat_out = np.zeros((self.years, self.days, self.hours))
        state_of_charge = np.zeros((self.years, self.days, self.hours))
        num_households = np.zeros((len(self.house), self.years))
        heat_rate_binary = np.zeros((self.years, self.days, self.hours // 3, len(self.heat_r_k)))
        price_binary = np.zeros(self.steps)
        quantity_binary = np.zeros(self.steps)
        total_demand = np.zeros((self.years, self.days, self.hours))

        for y in range(self.years + 1):
            for g in self.techs_o:
                ret[self.techs_o.tolist().index(g)][y] = ret_cap[g, y].X
                inst[self.techs_o.tolist().index(g)][y] = inst_cap[g, y].X
                added[self.techs_o.tolist().index(g)][y] = added_cap[g, y].X
            ret[3][y] = ret_cap['Owned Batteries', y].X * self.cap_power_ratio
            inst[3][y] = inst_cap['Owned Batteries', y].X * self.cap_power_ratio
            added[3][y] = added_cap['Owned Batteries', y].X * self.cap_power_ratio

        for y in range(self.years):
            for d in range(self.days):
                for h in range(self.hours):
                    disp_gen[y][d][h] = disp['Diesel Generator', y + 1, d, h].X
                    disp_pv[y][d][h] = disp['Owned PV', y + 1, d, h].X
                    disp_feedin[y][d][h] = disp['Feed In Prosumers', y + 1, d, h].X
                    unmetD[y][d][h] = ud[y + 1, d, h].X
                    bat_in[y][d][h] = b_in[y + 1, d, h].X
                    bat_out[y][d][h] = b_out[y + 1, d, h].X
                    state_of_charge[y][d][h] = soc[y + 1 , d, h].X
                    if self.heatrate_c_run == 'y':
                        for i in range(len(self.heat_r_k)):
                            heat_rate_binary[y][d][h // 3][i] = bin_heat_rate[i, y, d, h // 3].X




        for y in range(self.years):
            if self.dem_elasticity_c_run == 'y':
                for house in self.house:
                    num_households[self.house.tolist().index(house)][y] = np.abs(self.max_house_str[house])

                for i in range(self.steps):
                    for d in range(self.days):
                        price_binary[i] = self.price_steps[i] * bin_price_curve[i].X
                        quantity_binary[i] = self.disp_steps_year[i] * bin_price_curve[i].X

                for d in range(self.days):
                    for h in range(self.hours):
                        hourly_demand = 0
                        for house in self.house:
                            hourly_demand += (
                                    (sum(self.disp_steps_month[self.steps - 1 - i][d] * bin_price_curve[i].X for i in range(self.steps))
                                        / self.hist_demand[d]) * self.res_demand[house][d][h] * self.max_house_str[house])
                        total_demand[y][d][h] = hourly_demand


            else:
                for house in self.house:
                    num_households[self.house.tolist().index(house)][y] = h_weight[house, y].X

                for d in range(self.days):
                    for h in range(self.hours):
                        hourly_demand = 0
                        for house in self.house:
                            hourly_demand += self.res_demand[house][d][h] * h_weight[house, y].X
                        total_demand[y][d][h] = hourly_demand
        pros_demandarray = cd.fill_pros_demandarray(self, unmetD, disp_feedin, num_households)
        #print(pros_demandarray)

        return_array = [ret, inst, added, disp_gen, disp_pv, disp_feedin,
                        unmetD, bat_in, bat_out, state_of_charge, num_households,
                        heat_rate_binary, price_binary, quantity_binary, total_demand,
                        self.res_demand, self.pros_feedin, pros_demandarray]

        return return_array
