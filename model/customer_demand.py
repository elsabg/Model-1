import numpy as np
from gurobipy import *

#--------------------------------------------------------------------------------#
#                                                                                #
# calculate the residual (without PV) demand and feed in of all households       #
#                                                                                #
#--------------------------------------------------------------------------------#

def calc_res_demand(self):
    """calculate customer demand subtracted by the PV generation"""
    for h_type in self.res_demand:
        for i in range(len(self.res_demand[h_type])):
            for j in range(len(self.res_demand[h_type][i])):
                self.res_demand[h_type][i][j] = max(0, (self.res_demand[h_type][i][j]
                                                   - self.cap_fact[i][j] * self.avg_pv_cap_str[h_type]))

def calc_pros_feedin(self):
    """calculate the feed in of prosumers"""
    for h_type in self.pros_feedin:
        for i in range(len(self.pros_feedin[h_type])):
            for j in range(len(self.pros_feedin[h_type][i])):
                self.pros_feedin[h_type][i][j] = max(0, (self.cap_fact[i][j] * self.avg_pv_cap_str[h_type]
                                                    - self.pros_feedin[h_type][i][j]))


#--------------------------------------------------------------------------------#
#                                                                                #
# hourly Microgrid demand (normal/demand elasticity)                             #
#                                                                                #
#--------------------------------------------------------------------------------#

def mc_demand(self, d, h):
    """returns hourly Microgrid demand of all household types"""
    hourly_demand = 0
    for i in self.house:
        hourly_demand += self.res_demand[i][d][h] * self.max_house_str[i]
    return hourly_demand

def elastic_mc_demand(self, bin_price_curve, y, d, h):
    """return hourly Microgrid demand of all household types depending on the electricity price"""
    hourly_demand = 0
    for i in self.house:
        hourly_demand += ((calc_elastic_monthdem_const(self, bin_price_curve, y, d) / self.hist_demand[d])
                          * self.res_demand[i][d][h] * self.max_house_str[i])
    return hourly_demand

#--------------------------------------------------------------------------------#
#                                                                                #
# other functions to calculate elastic demand                                    #
#                                                                                #
#--------------------------------------------------------------------------------#

def calc_elastic_mondemand(self, elec_price, d):
    """calculates the monthly demand depending on electricity price"""
    return (self.hist_demand[d] *
            (1 - self.elasticity + self.elasticity * (elec_price / self.hist_price)))

def calc_elastic_monthdem_const(self, bin_price_curve, y, d):
    """calculates the monthly demand depending on electricity price for the balance constraint"""
    return quicksum(self.disp_steps_month[self.steps - 1 - i][d] * bin_price_curve[i , y - 1] for i in range(self.steps))


def demand_sum_year(self, y, disp, ud, b_out, b_in):
    """calculates the annual demand in the grid"""
    dem_year = quicksum(
            quicksum(
        quicksum(disp[g, y, d, h] for g in self.techs_g) + ud[y, d, h] + b_out[y, d, h] - b_in[y, d, h]
        for h in range(self.hours))* self.d_weights[d]
        for d in range(self.days)
    )
    return dem_year

def calc_disp_price_steps(self):
    """calculate the steps of disp for demand elasticity"""
    demand_0 = sum(calc_elastic_mondemand(self, 0, d) for d in range(self.days))
    delta_q0 = demand_0 / self.steps
    price_0 = (- demand_0) * self.hist_price / (np.sum(self.hist_demand[d] for d in range(self.days)) * self.elasticity)
    delta_p0 = price_0 / self.steps
    disp_steps_year = np.zeros(self.steps)
    disp_steps_month = np.zeros((self.steps, self.days))
    price_steps = np.zeros(self.steps)
    for i in range(self.steps):
        disp_steps_year[i] = delta_q0 * (i + 1)
        for d in range(self.days):
            disp_steps_month[i][d] = (calc_elastic_mondemand(self, 0, d) / self.steps) * (i + 1)
        price_steps[i] = delta_p0 * (i + 1)
    return disp_steps_year, disp_steps_month, price_steps
