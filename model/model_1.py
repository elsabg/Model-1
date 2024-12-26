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

        #----------------------------------------------------------------------#
        # Time Parameters                                                      #
        #----------------------------------------------------------------------#

        self.years = int(self.data['parameters']['Planning horizon'][0])
        self.days = int(self.data['parameters']['Days'][0])
        self.hours = int(self.data['parameters']['Hours'][0])
        self.d_weights = self.data['day_weights']['Weight'].to_numpy()

        #----------------------------------------------------------------------#
        # Capacity Parameters                                                  #
        #----------------------------------------------------------------------#

        #Initial Generation Capacities
        self.init_cap = self.tech_df['Initial capacity'].iloc[:-1].to_dict()
        '''
        self.init_cap_e = (self.data['tech']['Initial capacity'].to_numpy())[-1]
        '''
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

        #----------------------------------------------------------------------#
        # Lifetime                                                             #
        #----------------------------------------------------------------------#

        #Remaining lifetime
        self.life_0 = self.tech_df['Remaining lifetime'].iloc[:-1].to_dict()
        '''
        self.life_0_e = (self.data['tech']['Remaining lifetime'].to_numpy())[-1]
        '''
        #Technology lifetime
        self.life = self.tech_df['Lifetime'].iloc[:-1].to_dict()
        '''
        self.life_e = (self.data['tech']['Lifetime'].to_numpy())[-1]
        '''

        #----------------------------------------------------------------------#
        # Costs                                                                #
        #----------------------------------------------------------------------#

        #Technology costs
        self.ucc = self.tech_df['UCC'].to_dict()
        self.uofc = self.tech_df['UOFC'].to_dict()
        self.uovc = self.tech_df['UOVC'].to_dict()
        
        '''
        #Unmet demand
        self.ud_penalty = self.data['parameters']['Unmet demand penalty'][0]
        '''

        #heat rate curve
        self.heat_r_k = self.data['heat_rate']['HR'].to_numpy()
        
        self.diesel_p = self.data['tariffs']['Diesel Price'].to_numpy()

        #----------------------------------------------------------------------#
        # Electricity Demand                                                   #
        #----------------------------------------------------------------------#

        #Household Types
        self.house = self.data['rent_cap'].columns.to_numpy()[1::]

        # Demand
        self.demand_1 = self.data['elec_demand (1)'].iloc[:, 1:].to_numpy()
        self.demand_2 = self.data['elec_demand (2)'].iloc[:, 1:].to_numpy()
        self.demand_3 = self.data['elec_demand (3)'].iloc[:, 1:].to_numpy()
        self.demand_4 = self.data['elec_demand (4)'].iloc[:, 1:].to_numpy()
        self.demand_5 = self.data['elec_demand (5)'].iloc[:, 1:].to_numpy()

        self.demand = {
            'Type 1': self.demand_1.tolist(),
            'Type 2': self.demand_2.tolist(),
            'Type 3': self.demand_3.tolist(),
            'Type 4': self.demand_4.tolist(),
            'Type 5': self.demand_5.tolist()
        }
        
        
        # Surplus
        # Positive surplus can be fed-in, negative surplus is additional demand
        self.surplus = {
            'Type 1': self.demand_1.tolist(),
            'Type 2': self.demand_2.tolist(),
            'Type 3': self.demand_3.tolist(),
            'Type 4': self.demand_4.tolist(),
            'Type 5': self.demand_5.tolist()
        }
        for h in self.surplus: # house type
            for i in range(len(self.surplus[h])): # days
                for j in range(len(self.surplus[h][i])): #hours
                    self.surplus[h][i][j] = (self.cap_fact[i][j] 
                                             * self.avg_pv_cap_str[h]
                                             - self.surplus[h][i][j])

        #----------------------------------------------------------------------#
        # Battery and other Parameters                                         #
        #----------------------------------------------------------------------#

        self.min_soc = self.data['parameters']['min SoC'][0]
        self.bat_eff = self.data['parameters']['Battery Eff'][0]

        self.i = self.data['parameters']['Interest rate'][0]
        self.max_tariff = self.data['tariffs']['Ministry Tariff'].to_numpy()

        self.cap_steps = self.data['capacity_steps'][
            'Diesel Generator'].to_numpy()

        #----------------------------------------------------------------------#
        # Sets                                                                 #
        #----------------------------------------------------------------------#

        # All technologies (['Disel Generator', 'Owned PV', 'Owned Batteries'])
        self.techs = self.data['tech'].iloc[:-1, 0].to_numpy() 
        # Generation technologies (['Diesel Generator', 'Owned PV'])
        self.techs_g = self.techs[:2] 



    def solve(self, fit, elec_price):
        'Create and solve the model'

        self.fit = fit
        self.elec_price = elec_price

        m = Model('Model_1')

        #----------------------------------------------------------------------#
        #                                                                      #
        # Decision Variables                                                   #
        #                                                                      #
        #----------------------------------------------------------------------#

        added_cap = m.addVars(self.techs, self.years, 
                              name='addedCap', lb = 0, vtype = GRB.INTEGER)
        '''
        added_cap_e = m.addVars(self.years, 
                                name='addedCapE', lb = 0, vtype = GRB.INTEGER)
        '''
        inst_cap = m.addVars(self.techs, self.years, name='instCap', lb=0)
        '''
        inst_cap_e = m.addVars(self.years, name='instCapE', lb=0)
        '''
        disp = m.addVars(self.techs_g, self.years, self.days, self.hours, 
                         name='disp', lb=0)

        feed_in = m.addVars(self.house, self.years, self.days, self.hours, 
                            name='feedIn', lb = 0)

        b_in = m.addVars(self.years, self.days, self.hours, 
                         name='bIn', lb = 0)
        b_out = m.addVars(self.years, self.days, self.hours, 
                          name='bOut', lb = 0)

        ret_cap = m.addVars(self.techs, self.years, 
                            name='retiredCap', lb = 0)
        '''
        ret_cap_e = m.addVars(self.years, name='retiredCapE',  lb = 0)
        '''

        soc = m.addVars(self.years, self.days, self.hours, 
                        name='SoC', lb = 0)

        ud = m.addVars(self.years, self.days, self.hours, 
                       name='unmetDemand', lb = 0)

        h_weight = m.addVars(self.house, self.years, 
                             name='houseWeight', lb = 0, vtype=GRB.INTEGER)

        int_cap_steps = m.addVars(len(self.cap_steps), self.years, 
                                  name = 'binCapSteps', vtype=GRB.INTEGER, 
                                  lb = 0)

        bin_heat_rate = m.addVars(range(5), self.years,
                      self.days, self.hours,
                      vtype=GRB.BINARY, name='binHeatRate')
        
        d_cons = m.addVars(self.years, self.days, self.hours,
                           name='diesel consumption')
        
        
        #Auxiliary variables
        aux_min = m.addVars(self.house, self.years, self.days, self.hours,
                             name='min_auxilliary')

        aux_max = m.addVars(self.house, self.years, self.days, self.hours,
                             name='max_auxilliary', lb=0)
        
        z_bin = m.addVars(range(2), self.house, self.years, self.days, self.hours,
                          vtype=GRB.BINARY, name='z binary')
        #----------------------------------------------------------------------#
        #                                                                      #
        # Objective function                                                   #
        #                                                                      #
        #----------------------------------------------------------------------#

        tr = [0] * (self.years) #total yearly revenues
        tcc = [0] * (self.years) #total yearly capital costs
        tovc = [0] * (self.years) #total yearly operation variable costs
        tofc = [0] * (self.years) #total yearly operation fixed costs
        tcud = [0] * (self.years) #total yearly cost of unmet demand

        for y in range(self.years):

            # Revenue
            tr[y] = self.elec_price * quicksum((
                (sum(disp[g, y, d, h] for g in self.techs_g) 
                 + b_out[y, d, h] - b_in[y, d, h]) 
                * self.d_weights[d])
                for d in range(self.days)
                for h in range(self.hours)
            )

            # Capital Costs
            tcc[y] = (quicksum(
                (added_cap[g, y] * self.ucc[g]) for g in self.techs))

            '''
            # Operation Variable Costs with fixed DG heat rate value
            tovc[y] = quicksum(
                disp[g, y, d, h] * self.d_weights[d] * self.uovc[g]
                for g in self.techs_g
                for d in range(self.days)
                for h in range(self.hours)
            ) + quicksum(
                (b_out[y, d, h] + b_in[y, d, h]) 
                * self.d_weights[d] * self.uovc['Owned Batteries']
                for d in range(self.days)
                for h in range(self.hours)
            ) + quicksum(
                self.heat_r_v * disp['Diesel Generator', y, d, h] 
                * self.diesel_p[y - 1] * self.d_weights[d]
                for d in range(self.days)
                for h in range(self.hours)
            ) + quicksum(
                quicksum(feed_in[i, y, d, h] for i in self.house) * self.fit
                for d in range(self.days)
                for h in range(self.hours)
            )

            '''
            # Operation Variable Costs with DG heat rate curve
            tovc[y] = (quicksum((quicksum((self.uovc[g] * disp[g, y, d, h]) 
                                        for g in self.techs_g)
                               + quicksum((self.fit * feed_in[i, y, d, h]) 
                                          for i in self.house)
                               + self.uovc['Owned Batteries'] 
                               * (b_in[y, d, h])) 
                               * self.d_weights[d]
                               for h in range(self.hours)
                               for d in range(self.days))
                       + quicksum(d_cons[y, d, h] * self.diesel_p 
                                  * self.d_weights[d]
                                  for h in range(self.hours)
                                  for d in range(self.days))
                       )

            # Operation Fixed Costs
            tofc[y] = quicksum((inst_cap[g, y] * self.uofc[g]) 
                               for g in self.techs
            )
            
            '''
            # Cost of Unmet Demand
            tcud[y] = quicksum(
                ud[y, d, h] * self.d_weights[d] * self.ud_penalty
                for d in range(self.days)
                for h in range(self.hours)
            )
            '''

        # Net Present Value of Total Profits without unmet demand penalty
        tp_npv = quicksum(((tr[y] - tcc[y] - tofc[y] - tovc[y]) 
                           * ( 1 / ((1 + self.i) ** y))) #discount factor
                          for y in range(self.years)
                          )
        print(tp_npv)
        m.setObjective(tp_npv[0], GRB.MAXIMIZE)

        #----------------------------------------------------------------------#
        #                                                                      #
        # Constraints                                                          #
        #                                                                      #
        #----------------------------------------------------------------------#

        #----------------------------------------------------------------------#
        # Demand-Supply Balance                                                #
        #----------------------------------------------------------------------#
        
        # Auxiliary Constraints
        M = 1000
        m.addConstrs(((aux_min[i, y, d, h] <=
                       feed_in[i, y, d, h] + z_bin[0, i, y, d, h] * M)
                      for i in self.house
                      for y in range(self.years)
                      for d in range(self.days)
                      for h in range(self.hours)
                      ),
                     name='min aux 1.1')
        m.addConstrs(((aux_min[i, y, d, h] >=
                       feed_in[i, y, d, h] - z_bin[0, i, y, d, h] * M)
                      for i in self.house
                      for y in range(self.years)
                      for d in range(self.days)
                      for h in range(self.hours)
                      ),
                     name='min aux 1.2')
        m.addConstrs(((feed_in[i, y, d, h] <=
                       h_weight[i, y] * self.surplus[i][d][h]
                       + z_bin[0, i, y, d, h] * M)
                      for i in self.house
                      for y in range(self.years)
                      for d in range(self.days)
                      for h in range(self.hours)
                      ),
                     name='min aux 1.3')
        
        m.addConstrs(((aux_min[i, y, d, h] <=
                       h_weight[i, y] * self.surplus[i][d][h] + 
                       (1 - z_bin[0, i,y,d,h]) * M)
                      for i in self.house
                      for y in range(self.years)
                      for d in range(self.days)
                      for h in range(self.hours)
                      ),
                     name='min aux 2.1')
        m.addConstrs(((aux_min[i, y, d, h] >=
                       h_weight[i, y] * self.surplus[i][d][h] - 
                       (1 - z_bin[0, i, y, d, h]) * M)
                      for i in self.house
                      for y in range(self.years)
                      for d in range(self.days)
                      for h in range(self.hours)
                      ),
                     name='min aux 2.2')
        m.addConstrs(((h_weight[i, y] * self.surplus[i][d][h] <=
                       feed_in[i, y, d, h] +
                       (1 - z_bin[0, i, y, d, h]) * M)
                      for i in self.house
                      for y in range(self.years)
                      for d in range(self.days)
                      for h in range(self.hours)
                      ),
                     name='min aux 2.3')
        
        # Supply-demand balance constraint
        m.addConstrs(((b_out[y, d, h] 
                        + sum(disp[g, y, d, h] for g in self.techs_g) 
                        + sum(aux_min[i, y, d, h] for i in self.house) == 
                        b_in[y, d, h]) # no unmet demand
                      for h in range(self.hours)
                      for d in range(self.days)
                      for y in range(self.years)
                      ),
                     "Supply-demand balance"
                     )
        
        # Auxiliary constraints
        m.addConstrs(((aux_max[i, y, d, h] <=
                       h_weight[i, y] * self.surplus[i][d][h]
                       + z_bin[1, i, y, d, h] * M)
                      for i in self.house
                      for y in range(self.years)
                      for d in range(self.days)
                      for h in range(self.hours)
                      ),
                     name='max aux 1.1')
        m.addConstrs(((aux_max[i, y, d, h] >=
                       h_weight[i, y] * self.surplus[i][d][h]
                       - z_bin[1, i, y, d, h] * M)
                      for i in self.house
                      for y in range(self.years)
                      for d in range(self.days)
                      for h in range(self.hours)
                      ),
                     name='max aux 1.2')
        m.addConstrs(((h_weight[i, y] * self.surplus[i][d][h] >=
                       - z_bin[1, i, y, d, h] * M)
                      for i in self.house
                      for y in range(self.years)
                      for d in range(self.days)
                      for h in range(self.hours)
                      ),
                     name='max aux 1.3'
                     )
        
        m.addConstrs(((aux_max[i, y, d, h] <=
                       (1 - z_bin[1, i, y, d, h]) * M)
                      for i in self.house
                      for y in range(self.years)
                      for d in range(self.days)
                      for h in range(self.hours)
                      ),
                     name='max aux 2.1'
                     )
        m.addConstrs(((aux_max[i, y, d, h] >=
                       - (1 - z_bin[1, i, y, d, h]) * M)
                      for i in self.house
                      for y in range(self.years)
                      for d in range(self.days)
                      for h in range(self.hours)
                      ),
                     name='max aux 2.2'
                     )
        m.addConstrs(((h_weight[i, y] * self.surplus[i][d][h] <=
                       (1 - z_bin[1, i, y, d, h]) * M)
                      for i in self.house
                      for y in range(self.years)
                      for d in range(self.days)
                      for h in range(self.hours)
                      ),
                     name='max aux 2.3'
                     )
        
        # Feed-in capacity constraints
        m.addConstrs(((feed_in[i, y, d, h] <=
                       aux_max[i, y, d, h])
                       for i in self.house
                       for h in range(self.hours)
                       for d in range(self.days)
                       for y in range(self.years)
                       ),
                      "Feed in cap"
            )
        
        m.addConstrs(((h_weight[i, y] <= self.max_house_str[i]) 
                       for i in self.house 
                       for y in range(self.years)
                       ),
                      "Max house cap"
            )
        
        #----------------------------------------------------------------------#
        # Generation Capacity                                                  #
        #----------------------------------------------------------------------#

        m.addConstrs(
            (
                (inst_cap[g, y] ==
                inst_cap[g, y - 1] + added_cap[g, y]
                - ret_cap[g, y])
                for g in self.techs
                for y in range(1, self.years)
            ),
            "Tracking capacity"
        )
        
        m.addConstrs(
            (
                (inst_cap[g, 0] == 
                 self.init_cap[g] + added_cap[g, 0]
                 - ret_cap[g, 0])
                for g in self.techs
            ),
            "Initial capacity"
        )
        
        
        '''
        m.addConstrs(
            (
                (added_cap['Diesel Generator', y] ==
                    quicksum(int_cap_steps[i, y] * self.cap_steps[i] 
                             for i in range(len(self.cap_steps))))
                for y in range(1, self.years + 1)
            ),
            "Steps for added diesel generator capacity"
        )
        '''
        
        #----------------------------------------------------------------------#
        # Generation Retirement                                                #
        #----------------------------------------------------------------------#

        m.addConstrs(
            (
                (ret_cap[g, self.life_0[g]] == self.init_cap[g])
                for g in self.techs
            ),
            "Retirement of initial capacity"
        )
        
        m.addConstrs(
            (
                (ret_cap[g, y] == 0)
                for g in self.techs
                for y in range(self.life_0[g])   # range(self.life_0) returns values only up to life_0
            ),
            "Retirement before initial capacity"
        )
        
        m.addConstrs(
            (
                (ret_cap[g, y] == added_cap[g, y - self.life[g]])
                for g in self.techs
                for y in range(self.life[g], self.years)
            ),
            "Retirement after initial capacity"
        )

        m.addConstrs(
            (
                (ret_cap[g, y] == 0)
                for g in self.techs
                for y in range(self.life_0[g] + 1, 
                               min(self.life[g], self.years))
            ),
            "Retirement between initial capacity and life"
        )
        
        #----------------------------------------------------------------------#
        # Dispatch                                                             #
        #----------------------------------------------------------------------#

        m.addConstrs(
            (
                (disp['Diesel Generator', y, d, h] <=
                 inst_cap[('Diesel Generator', y)])
                for h in range(self.hours)
                for d in range(self.days)
                for y in range(self.years)
            ),
            "Maximum DG dispatch"
        )
        
        m.addConstrs(
            (
                (disp['Owned PV', y, d, h] <=
                 self.cap_fact[d, h] * inst_cap['Owned PV', y])
                for h in range(self.hours)
                for d in range(self.days)
                for y in range(self.years)
            ),
            "Maximum PV dispatch"
        )
        
        m.addConstrs(
            (
                (b_in[y, d, h] <=
                 inst_cap['Owned Batteries', y])
                for y in range(self.years)
                for d in range(self.days)
                for h in range(self.hours)
            ),
            "Maximum battery input"
        )
        
        m.addConstrs(
            (
                (b_out[y, d, h] <=
                 inst_cap['Owned Batteries', y])
                for y in range(self.years)
                for d in range(self.days)
                for h in range(self.hours)
            ),
            "Maximum battery output"
        )
        
        #----------------------------------------------------------------------#
        # Heat Rate                                                            #
        #----------------------------------------------------------------------#
        M = 1000
        e = 0.01
        
        m.addConstrs(
            ((d_cons[y, d, h] >= 
              self.heat_r_k[0] * disp['Diesel Generator', y, d, h] 
              - bin_heat_rate[0, y, d, h] * M) 
             for y in range(self.years)
             for d in range(self.days)
             for h in range(self.hours)
             ),
            "Heat Rate 1.1"
        )
        
        m.addConstrs(
            ((d_cons[y, d, h] <= 
              self.heat_r_k[0] * disp['Diesel Generator', y, d, h] 
              + bin_heat_rate[0, y, d, h] * M)
             for y in range(self.years)
             for d in range(self.days)
             for h in range(self.hours)
             ),
            "Heat Rate 1.2"
        )
        
        m.addConstrs(
            ((disp['Diesel Generator', y, d, h] >=
              0.3 * inst_cap['Diesel Generator', y]
              - (1 - bin_heat_rate[0, y, d, h] * M))
             for y in range(self.years)
             for d in range(self.days)
             for h in range(self.hours)
             ),
            "Boundary 1" 
        )
        
        m.addConstrs(
            ((d_cons[y, d, h] >= 
              self.heat_r_k[1] * disp['Diesel Generator', y, d, h] 
              - bin_heat_rate[1, y, d, h] * M)
             for y in range(self.years)
             for d in range(self.days)
             for h in range(self.hours)
             ),
            "Heat Rate 2.1"
        )
        
        m.addConstrs(
            ((d_cons[y, d, h] <= 
              self.heat_r_k[1] * disp['Diesel Generator', y, d, h]
              + bin_heat_rate[1, y, d, h] * M)
             for y in range(self.years)
             for d in range(self.days)
             for h in range(self.hours)
             ),
            "Heat Rate 2.2"
        )
        
        m.addConstrs(
            ((disp['Diesel Generator', y, d, h] <= 
              0.3 * inst_cap['Diesel Generator', y]
              + bin_heat_rate[2, y, d, h] * M - e)
             for y in range(self.years) 
             for d in range(self.days) 
             for h in range(self.hours) 
             ), 
            "Boundary 2.1"
        )
        
        m.addConstrs(
            ((disp['Diesel Generator', y, d, h] >= 
              0.6 * inst_cap['Diesel Generator', y] 
              + bin_heat_rate[3, y, d, h] * M)
             for y in range(self.years)
             for d in range(self.days)
             for h in range(self.hours)
             ),
            "Boundary 2.2"
        )
        
        m.addConstrs(
            ((d_cons[y, d, h] >= 
              self.heat_r_k[2] * disp['Diesel Generator', y, d, h]
              - bin_heat_rate[4, y, d, h] * M)
             for y in range(self.years)
             for d in range(self.days)
             for h in range(self.hours)
             ),
            "Heat Rate 3.1"
        )
        
        m.addConstrs(
            ((d_cons[y, d, h] <= 
              self.heat_r_k[2] * disp['Diesel Generator', y, d, h]
              + bin_heat_rate[4, y, d, h] * M)
             for y in range(self.years)
             for d in range(self.days)
             for h in range(self.hours)
             ),
            "Heat Rate 3.2"
        )
        
        m.addConstrs(
            ((disp['Diesel Generator', y, d, h] <=
              0.6 * inst_cap['Diesel Generator', y]
              - (1 - bin_heat_rate[4, y, d, h] * M - e))
             for y in range(self.years)
             for d in range(self.days)
             for h in range(self.hours)
             ),
            "Boundary 3" 
        )
        
        m.addConstrs(
            ((bin_heat_rate[1, y, d, h] 
              + bin_heat_rate[2, y, d, h]
              + bin_heat_rate[3, y, d, h] <=
              2)
             for y in range(self.years)
             for d in range(self.days)
             for h in range(self.hours)
             ),
            "Binary cap 1"
        )
        
        m.addConstrs(
            ((bin_heat_rate[0, y, d, h] 
              + bin_heat_rate[1, y, d, h]
              + bin_heat_rate[4, y, d, h] ==
              2)
             for y in range(self.years)
             for d in range(self.days)
             for h in range(self.hours)
             ),
            "Binary cap 2"
        )
        
        #----------------------------------------------------------------------#
        # Battery Operation                                                    #
        #----------------------------------------------------------------------#

        m.addConstrs(
            (
                (soc[y, d, h] == soc[y, d, h - 1]
                 + self.bat_eff * b_in[y, d, h]
                 - b_out[y, d, h] / self.bat_eff)
                for y in range(self.years)
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
                for y in range(self.years)
                for d in range(self.days)
            ),
            ' SoC of hour 0'
        )
        m.addConstrs(
            (
                (self.min_soc 
                 * 4 * inst_cap['Owned Batteries', y] <=
                 soc[y, d, h])
                for y in range(self.years)
                for d in range(self.days)
                for h in range(self.hours)
            ),
            'SoC capacity 1'
        )
        m.addConstrs(
            (
                (4 * inst_cap['Owned Batteries', y] >=
                 soc[y, d, h])
                for y in range(self.years)
                for d in range(self.days)
                for h in range(self.hours)
            ),
            'SoC capacity 2'
        )
        
        '''
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

        '''
        
        '''
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
        '''
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

        ret = np.ones((len(self.techs), self.years)) # retired capacity
        inst = np.zeros((len(self.techs), self.years)) # installed capacity
        added = np.zeros((len(self.techs), self.years)) # added capacity
        disp_gen = np.zeros((self.days, self.hours))
        unmetD = np.zeros((self.days, self.hours))
        bat_in = np.zeros((self.days, self.hours))
        bat_out = np.zeros((self.days, self.hours))
        num_households = np.ones((len(self.house), self.years))
        feed_in_energy = np.zeros((self.days, self.hours))

        for y in range(self.years):
            for g in self.techs:
                ret[self.techs.tolist().index(g)][y] = ret_cap[g, y].X
                inst[self.techs.tolist().index(g)][y] = inst_cap[g, y].X
                added[self.techs.tolist().index(g)][y] = added_cap[g, y].X

        for d in range(self.days):
            for h in range(self.hours):
                disp_gen[d][h] = disp['Diesel Generator', 1, d, h].X
                unmetD[d][h] = ud[12, d, h].X
                bat_in[d][h] = b_in[12, d, h].X
                bat_out[d][h] = b_out[12, d, h].X
                #feed_in_energy[d][h] = disp['Feed In Prosumers', 12, d, h].X

        for house in self.house:
            for y in range(self.years):
                num_households[self.house.tolist().index(house)][y] = np.abs(h_weight[house, y].X)


        total_demand = np.zeros((self.days, self.hours))

        '''
        for house in self.house:
            for d in range(self.days):
                for h in range(self.hours):
                    total_demand[d][h] += self.res_demand[house][d][h] * num_households[self.house.tolist().index(house)][12]
        '''

        return_array = [ret, inst, added, disp_gen, unmetD, bat_in, bat_out, num_households] # no feed_in_energy, total_demand

        return return_array
