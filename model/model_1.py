# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 20:28:54 2025

@author: Elsa
"""

import numpy as np
import pandas as pd
from gurobipy import quicksum
from gurobipy import GRB
from gurobipy import Model

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

class Model_1:

    def __init__(self, _file_name):
        self._file_name = _file_name

    #loading model parameters
    def load_data(self):
        'read the excel file'

        self.data = pd.read_excel(self._file_name, decimal=',', sheet_name=None)
        self.tech_df = self.data['tech'].set_index('Unnamed: 0')
        self.capex_df = self.data['capex'].set_index('Unnamed: 0')
        
        #----------------------------------------------------------------------#
        # General Parameters                                                   #
        #----------------------------------------------------------------------#
        
        self.RE_max = int(self.data['parameters']['Allow RE'][0])

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
        self.init_cap = self.tech_df['Initial capacity'][:-1].to_dict()

        #Household capacitiesm
        self.max_house = self.data['rent_cap'].loc[0].iloc[1::].to_numpy()
        self.avg_pv_cap = self.data['rent_cap'].loc[1].iloc[1::].to_numpy()
        self.cap_fact = self.data['cap_factors'].iloc[:, 1:].to_numpy()

        #Capacities accessible via strings
        self.house = self.data['rent_cap'].columns.to_numpy()[1::]
        
        self.max_house_str = {f'Type {i+1}' : self.max_house[i]
                              for i in range(len(self.house))}

        self.avg_pv_cap_str = {f'Type {i+1}' : self.avg_pv_cap[i]
                              for i in range(len(self.house))}

        #----------------------------------------------------------------------#
        # Lifetime                                                             #
        #----------------------------------------------------------------------#

        #Remaining lifetime
        self.life_0 = self.tech_df['Remaining lifetime'][:-1].to_dict()
        self.life_0 = {k: int(v) for k, v in self.life_0.items()}

        #Technology lifetime
        self.life = self.tech_df['Lifetime'][:-1].to_dict()
        self.life = {k: int(v) for k, v in self.life.items()}


        #----------------------------------------------------------------------#
        # Costs                                                                #
        #----------------------------------------------------------------------#

        #Technology costs
        self.ucc = {col: self.capex_df[col].to_numpy() 
                    for col in self.capex_df.columns}
        self.uofc = self.tech_df['UOFC'][:-1].to_dict()
        self.uovc = self.tech_df['UOVC'][:-1].to_dict()

        #heat rate curve
        self.heat_r_k = self.data['heat_rate']['HR'].to_numpy()
        
        self.diesel_p = self.tech_df.loc['Diesel'][-1]

        #----------------------------------------------------------------------#
        # Electricity Demand                                                   #
        #----------------------------------------------------------------------#

        # Demand
        self.demand = {f'Type {i+1}': 
                       self.data[f'elec_demand ({i+1})'].iloc[:, 1:].to_numpy()
                       for i in range(len(self.house))}
        
        
        # Surplus
        self.surplus = self.demand.copy()
        
        self.aux_min = {f'Type {i+1}': 
                       np.zeros((self.days, self.hours))
                       for i in range(len(self.house))}
        
        self.aux_max = {f'Type {i+1}': 
                       np.zeros((self.days, self.hours))
                       for i in range(len(self.house))}
        
        # Positive surplus can be fed-in, negative surplus is additional demand     
        for h in self.surplus: # house type
            for i in range(len(self.surplus[h])): # days
                for j in range(len(self.surplus[h][i])): #hours
                    self.surplus[h][i][j] = (self.cap_fact[i][j] 
                                             * self.avg_pv_cap_str[h]
                                             - self.surplus[h][i][j])
                    
                    if self.surplus[h][i][j] > 0:
                        self.aux_max[h][i][j] = self.surplus[h][i][j]
                        
                    elif self.surplus[h][i][j] < 0:
                        self.aux_min[h][i][j] = self.surplus[h][i][j]

        #----------------------------------------------------------------------#
        # Battery and other Parameters                                         #
        #----------------------------------------------------------------------#

        self.min_soc = self.data['parameters']['min SoC'][0]
        self.bat_eff = self.data['parameters']['Battery Eff'][0]
        self.y_DG_cap = self.data['parameters']['DG annual capacity'][0]
        
        #----------------------------------------------------------------------#
        # Sets                                                                 #
        #----------------------------------------------------------------------#

        # All technologies (['Disel Generator', 'Owned PV', 'Owned Batteries'])
        self.techs = self.data['tech'].iloc[:, 0].to_numpy()[:3] 
        # Generation technologies (['Diesel Generator', 'Owned PV'])
        self.techs_g = self.techs[:2] 



    def solve(self, fit, elec_price, md_level, ud_penalty, re_level=0, 
              voll=0.7, total_budget=np.inf, interest=0.1, interest_re=0.04,
              re_start=0, pros_perc=None):
        'Create and solve the model'

        self.fit = fit
        self.elec_price = elec_price
        self.md_level = md_level
        self.ud_penalty = ud_penalty
        self.re_level = re_level
        self.voll = voll
        self.total_budget = total_budget
        self.i = interest
        self.i_re = interest_re
        self.re_start = re_start
        
        current_perc = (sum(self.max_house[self.avg_pv_cap > 0]) 
                        / sum(self.max_house))
        
        if pros_perc == None or pros_perc == current_perc:
            self.pros_perc = current_perc
        else:
            change = (pros_perc - current_perc) * sum(self.max_house)
            new_pros = self.max_house[1] + change
            new_cons = self.max_house[0] - change
            new_max_house = np.array([round(new_cons), round(new_pros)])
            
            self.max_house = new_max_house
            self.max_house_str = {f'Type {i+1}' : new_max_house[i]
                                  for i in range(len(self.house))}
            
            self.pros_perc = pros_perc
            
        m = Model('Model_1')

        #----------------------------------------------------------------------#
        #                                                                      #
        # Decision Variables                                                   #
        #                                                                      #
        #----------------------------------------------------------------------#

        added_cap = m.addVars(self.techs, self.years, name='addedCap', lb = 0)

        inst_cap = m.addVars(self.techs, self.years, name='instCap', lb=0)

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

        soc = m.addVars(self.years, self.days, self.hours, 
                        name='SoC', lb = 0)
        
        soc_0 = m.addVars(self.years, self.days, name='initSoC', lb = 0)
        
        ud = m.addVars(self.years, self.days, self.hours,
                       name="UnmetDemand", lb=0)
        
        # Auxiliary variables for heat rate
        bin_heat_rate = m.addVars(range(2), self.years,
                      self.days, self.hours,
                      vtype=GRB.BINARY, name='binHeatRate')
        
        d_cons = m.addVars(self.years, self.days, self.hours,
                           name='dieselCons')
        
    
        #Intermediate variables
        tr = m.addVars(self.years, name='total revenue')
        tcc = m.addVars(self.years, name='total capital cost')
        tovc = m.addVars(self.years, name='total operation variable cost')
        tofc = m.addVars(self.years, name='total operation fixed cost')
        tudc = m.addVars(self.years, name='total unmet demand cost')
        tp = m.addVars(self.years, name='total profits', lb=-GRB.INFINITY)
        salvage = m.addVars(self.techs, name='salvage value')
        
        
        #----------------------------------------------------------------------#
        #                                                                      #
        # Objective function                                                   #
        #                                                                      #
        #----------------------------------------------------------------------#
        
        m.setObjective(quicksum(tp[y] * (1 / ((1 + self.i) ** y))
                                for y in range(self.years))
                       + quicksum(salvage[g]
                                  for g in self.techs)
                       * (1 / ((1 + self.i)) ** self.years), 
                       GRB.MAXIMIZE)

        #----------------------------------------------------------------------#
        #                                                                      #
        # Constraints                                                          #
        #                                                                      #
        #----------------------------------------------------------------------#
        
        #----------------------------------------------------------------------#
        # Total Profits                                                        #
        #----------------------------------------------------------------------#
        m.addConstrs((tr[y] ==
                       self.elec_price
                       * (
                           quicksum(disp[g, y, d, h] * self.d_weights[d]
                                    for g in self.techs_g
                                    for d in range(self.days)
                                    for h in range(self.hours))
                           + quicksum((b_out[y, d, h] - b_in[y, d, h])
                                      * self.d_weights[d]
                                      for d in range(self.days)
                                      for h in range(self.hours))
                           + quicksum(feed_in[i, y, d, h]
                                      * self.d_weights[d]
                                      for i in self.house
                                      for d in range(self.days)
                                      for h in range(self.hours))
                           )
                       for y in range(self.years)
                       ),
                     name='yearly total revenues')

        m.addConstrs(((tcc[y] ==
                       quicksum(added_cap[g, y] * self.ucc[g][y]
                                for g in self.techs)) 
                       for y in range(self.years)
                       ),
                     name='yearly total capital costs'
                     )
        
        m.addConstr(((quicksum(quicksum(added_cap[g, y] * self.ucc[g][y]
                            for g in self.techs)
                               * (1 / ((1 + self.i) ** y))
                               for y in range(self.years))
                  <= self.total_budget)
                     ),
                    name='yearly budget CAPEX constraint'
                    )
        
        m.addConstrs(((tovc[y] ==
                       quicksum((self.uovc[g] * disp[g, y, d, h] 
                                 * self.d_weights[d])
                                for g in self.techs_g
                                for d in range(self.days)
                                for h in range(self.hours))
                       + quicksum((self.fit * feed_in[i, y, d, h]
                                   * self.d_weights[d])
                                  for i in self.house
                                  for d in range(self.days)
                                  for h in range(self.hours))
                       + quicksum((self.diesel_p * d_cons[y, d, h]
                                   + self.uovc['Owned Batteries'] 
                                   * b_in[y, d, h])
                                  * self.d_weights[d]
                                  for d in range(self.days)
                                  for h in range(self.hours)))
                      for y in range(self.years)
                      ),
                     name='yearly total operation variable costs')
        
        m.addConstrs(((tofc[y] == 
                       quicksum((inst_cap[g, y] * self.uofc[g]) 
                                for g in self.techs)) 
                      for y in range(self.years)
                      ),
                     name='yearly total operation fixed costs'
                     )
        
        m.addConstrs(((tudc[y] ==
                       quicksum((ud[y, d, h] 
                                 * self.ud_penalty
                                 * self.d_weights[d])
                                for d in range(self.days)
                                for h in range(self.hours)))
                      for y in range(self.years)
                      ),
                     name='yearly total unmet demand costs')
        
        m.addConstrs(((tp[y] == 
                       (tr[y] - tcc[y] - tofc[y] - tovc[y] - tudc[y]))
                       for y in range(self.years)
                       ),
                     name='yearly total profits'
                     )

        m.addConstrs(((salvage[g] ==
                       quicksum(added_cap[g, y] 
                       * self.ucc[g][self.years - 1]
                       *(1 - (self.years - y) / self.years)
                       for y in range(self.years - self.life[g]))
                       for g in self.techs)
                      ), name='Salvage value'
                     )        
        
        #----------------------------------------------------------------------#
        # Demand-Supply Balance                                                #
        #----------------------------------------------------------------------#
        
        m.addConstrs(((b_out[y, d, h] 
                        + quicksum(disp[g, y, d, h] for g in self.techs_g) 
                        + quicksum(feed_in[i, y, d, h] for i in self.house) 
                        + ud[y, d, h] == 
                        b_in[y, d, h]
                        - sum(self.aux_min[i][d][h] * self.max_house_str[i]
                              for i in self.house))
                      for h in range(self.hours)
                      for d in range(self.days)
                      for y in range(self.years)
                      ),
                     "Supply-demand balance"
                     )
        
        m.addConstrs(((quicksum(ud[y, d, h] * self.d_weights[d]
                                for d in range(self.days)
                                for h in range(self.hours)) 
                       <= (1 - self.md_level) 
                       * ( - sum(self.aux_min[i][d][h] 
                                 * self.max_house_str[i] 
                                 * self.d_weights[d]
                                      for i in self.house
                                      for d in range(self.days)
                                      for h in range(self.hours))))
                      for y in range(self.years)
                      ),
                     "maximum yearly unmet demand"
                     )
        
        # Feed-in capacity constraints
        m.addConstrs(((feed_in[i, y, d, h] <=
                       self.RE_max 
                       * self.aux_max[i][d][h] 
                       * self.max_house_str[i])
                      for i in self.house
                      for h in range(self.hours)
                      for d in range(self.days)
                      for y in range(self.years)
                      ),
                     "Feed in cap"
            )
        
        m.addConstrs(((quicksum(feed_in[i, y, d, h] * self.d_weights[d]
                                for i in self.house
                                for d in range(self.days)
                                for h in range(self.hours))
                       + quicksum(disp['Owned PV', y, d, h] * self.d_weights[d]
                                  for d in range(self.days)
                                  for h in range(self.hours))
                       >= (quicksum(disp[g, y, d, h] * self.d_weights[d]
                                    for g in self.techs_g
                                    for d in range(self.days)
                                    for h in range(self.hours))
                           + quicksum(feed_in[i, y, d, h] * self.d_weights[d]
                                for i in self.house
                                for d in range(self.days)
                                for h in range(self.hours)))
                       * self.re_level * self.RE_max)
                      for y in range(re_start, self.years)
                      ),
                     name='Min Renewable Energy'
                     )
    
        
        m.addConstrs(((ud[y, d, h] <= 
                       - sum(self.aux_min[i][d][h] * self.max_house_str[i]
                             for i in self.house))
                      for y in range(self.years)
                      for d in range(self.days)
                      for h in range(self.hours)
                      ),
                     'Max unmet demand'
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
        
        
        M = - (np.min(self.demand[min(self.demand)]) 
               * max(self.max_house)
               / np.min(self.cap_fact[self.cap_fact != 0]))
        
        m.addConstrs(((inst_cap['Owned PV', y] <= self.RE_max * M * 100)
                      for y in range(self.years)
                      ),
                     name = 'max PV'
                     )
        
        m.addConstrs(((inst_cap['Owned Batteries', y] <= self.RE_max * M * 100)
                      for y in range(self.years)
                      ),
                     name = 'max Bat'
                     )
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
                for y in range(self.life_0[g])   
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
                (quicksum(disp['Diesel Generator', y, d, h]
                          * self.d_weights[d]
                          for h in range(self.hours)
                          for d in range(self.days)) <=
                 inst_cap[('Diesel Generator', y)]
                 * self.y_DG_cap
                 * sum(self.d_weights)
                 * self.hours
                 )
                for y in range(self.years)
            ),
            "Maximum annual DG dispatch"
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
        M = - (np.min(self.demand[min(self.demand)]) 
               * sum(self.max_house) 
               * max(self.heat_r_k) * 100)
        e = 0.01
        
        '''
        m.addConstrs(((d_cons[y, d, h] ==
                       disp['Diesel Generator', y, d, h]
                       * self.heat_r_k[1])
                      for y in range(self.years)
                      for d in range(self.days)
                      for h in range(self.hours)
                      ),
                     name='test')
        '''
        m.addConstrs(((disp['Diesel Generator', y, d, h] <=
                       0.3 * inst_cap['Diesel Generator', y]
                       + (1 - bin_heat_rate[0, y, d, h]) * M)
                      for y in range(self.years)
                      for d in range(self.days)
                      for h in range(self.hours)
                      ),
                     name="bound 1")
        
        m.addConstrs(((d_cons[y, d, h] <=
                       self.heat_r_k[0] * disp['Diesel Generator', y, d, h]
                       + (1 - bin_heat_rate[0, y, d, h]) * M)
                      for y in range(self.years)
                      for d in range(self.days)
                      for h in range(self.hours)
                      ),
                     name="case 1.1")
        
        m.addConstrs(((d_cons[y, d, h] >=
                       self.heat_r_k[0] * disp['Diesel Generator', y, d, h]
                       - (1 - bin_heat_rate[0, y, d, h]) * M)
                      for y in range(self.years)
                      for d in range(self.days)
                      for h in range(self.hours)
                      ),
                     name="case 1.2")
        
        m.addConstrs(((disp['Diesel Generator', y, d, h] >=
                       0.3 * inst_cap['Diesel Generator', y]
                       - (1 - bin_heat_rate[1, y, d, h]) * M + e)
                      for y in range(self.years)
                      for d in range(self.days)
                      for h in range(self.hours)
                      ),
                     name="bound 2.1")
        
        m.addConstrs(((disp['Diesel Generator', y, d, h] <=
                       0.6 * inst_cap['Diesel Generator', y]
                       + (1 - bin_heat_rate[1, y, d, h]) * M)
                      for y in range(self.years)
                      for d in range(self.days)
                      for h in range(self.hours)
                      ),
                     name="bound 2.2")
        
        m.addConstrs(((d_cons[y, d, h] <=
                       self.heat_r_k[1] * disp['Diesel Generator', y, d, h]
                       + (1 - bin_heat_rate[1, y, d, h]) * M)
                      for y in range(self.years)
                      for d in range(self.days)
                      for h in range(self.hours)
                      ),
                     name="case 2.1")
        
        m.addConstrs(((d_cons[y, d, h] >=
                       self.heat_r_k[1] * disp['Diesel Generator', y, d, h]
                       - (1 - bin_heat_rate[1, y, d, h]) * M)
                      for y in range(self.years)
                      for d in range(self.days)
                      for h in range(self.hours)
                      ),
                     name="case 2.2")
        
        m.addConstrs(((disp['Diesel Generator', y, d, h] >=
                       0.6 * inst_cap['Diesel Generator', y]
                       - (bin_heat_rate[0, y, d, h] + bin_heat_rate[1, y, d, h]) 
                       * M + e)
                      for y in range(self.years)
                      for d in range(self.days)
                      for h in range(self.hours)
                      ),
                     name="bound 3")
        
        m.addConstrs(((d_cons[y, d, h] <=
                       self.heat_r_k[2] * disp['Diesel Generator', y, d, h]
                       + (bin_heat_rate[0, y, d, h] + bin_heat_rate[1, y, d, h]) 
                       * M)
                      for y in range(self.years)
                      for d in range(self.days)
                      for h in range(self.hours)
                      ),
                     name="case 3.1")
        
        m.addConstrs(((d_cons[y, d, h] >=
                       self.heat_r_k[1] * disp['Diesel Generator', y, d, h]
                       - (bin_heat_rate[0, y, d, h] + bin_heat_rate[1, y, d, h]) 
                       * M)
                      for y in range(self.years)
                      for d in range(self.days)
                      for h in range(self.hours)
                      ),
                     name="case 3.2")
        
        #----------------------------------------------------------------------#
        # Battery Operation                                                    #
        #----------------------------------------------------------------------#

        m.addConstrs(
            (
                (soc[y, d, h] == soc[y, d, h - 1]
                 + b_in[y, d, h] * self.bat_eff
                 - b_out[y, d, h] / self.bat_eff)
                for y in range(self.years)
                for d in range(self.days)
                for h in range(1, self.hours)
            ),
            'SoC tracking'
        )
        
        m.addConstrs(((soc[y, d, 0] == soc[y, d, 23] # self.hours-1
                       + b_in[y, d, 0] * self.bat_eff
                       - b_out[y, d, 0] / self.bat_eff)
                      for y in range(self.years)
                      for d in range(self.days)), 
                     "Initial SoC"
                     )
        
        m.addConstrs(
            (
                (self.min_soc * 4 * inst_cap['Owned Batteries', y] <=
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
        m.addConstrs(
            (
                (self.min_soc * 4 * inst_cap['Owned Batteries', y] <=
                 soc_0[y, d])
                for y in range(self.years)
                for d in range(self.days)
            ),
            'SoC_0 capacity 1'
        )
        m.addConstrs(
            (
                (4 * inst_cap['Owned Batteries', y] >=
                 soc_0[y, d])
                for y in range(self.years)
                for d in range(self.days)
            ),
            'SoC_0 capacity 2'
        )
    
        
        #----------------------------------------------------------------------#
        # Optimization                                                         #
        #----------------------------------------------------------------------#
        
        m.setParam("Presolve", 1)
        m.optimize()

        #----------------------------------------------------------------------#
        #                                                                      #
        # Return Output                                                        #
        #                                                                      #
        #----------------------------------------------------------------------#
        
        self.m = m
        self.feed_in = feed_in
        self.soc = soc
        self.added_cap = added_cap
        self.inst_cap = inst_cap
        self.disp = disp
        self.b_in = b_in
        self.b_out = b_out
        self.ret_cap = ret_cap
        self.soc = soc
        self.d_cons = d_cons
        self.bin_heat_rate = bin_heat_rate
        self.ud = ud
        self.soc_0 = soc_0
        
        #Intermediate variables
        self.tr = tr
        self.tcc = tcc
        self.tovc = tovc
        self.tofc = tofc
        self.tp = tp

        variables = {var.VarName : var for var in m.getVars()}
        self.variables = variables
        self.obj = m.getObjective().getValue()
