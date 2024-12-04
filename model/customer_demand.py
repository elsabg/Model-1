import numpy as np
from gurobipy import *

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

def calc_elastic_mondemand(self, elec_price, d):
    """calculates the monthly demand depending on electricity price"""
    return (self.hist_demand[d] *
            (1 - self.elasticity + self.elasticity * (elec_price / self.hist_price)))

def calc_elastic_monthdem_const(self, bin_price_curve, y, d):
    """calculates the monthly demand depending on electricity price"""
    return quicksum(self.disp_steps_month[self.p_steps - 1 - i][d] * bin_price_curve[i] for i in range(self.p_steps))

'''
def calc_elastic_monthdem_const(self, bin_price_curve, y, d):
    """calculates the monthly demand depending on electricity price"""
    return (self.hist_demand[d] *
            (1 - self.elasticity + self.elasticity * (quicksum(bin_price_curve[i] * self.price_steps[i] for i in range(self.p_steps))
                                                      / self.hist_price)))
'''

def disp_sum_year(self, year, disp, b_out, b_in):
    """calculate the annual dispatched generation"""
    disp_year = quicksum(
        (quicksum(disp[g, 1, d, h] for g in self.techs_g) + b_out[1, d, h] - b_in[1, d, h]) * self.d_weights[d]
        for d in range(self.days)
        for h in range(self.hours)
    )
    return disp_year

def calc_disp_steps(self):
    """calculate the steps of disp for demand elasticity"""
    demand_0 = sum(calc_elastic_mondemand(self, 0, d) for d in range(self.days))
    delta_q0 = demand_0 / self.p_steps
    price_0 = (- demand_0) * self.hist_price / (np.sum(self.hist_demand[d] for d in range(self.days)) * self.elasticity)
    delta_p0 = price_0 / self.p_steps
    disp_steps_year = np.zeros(self.p_steps)
    disp_steps_month = np.zeros((self.p_steps, self.days))
    price_steps = np.zeros(self.p_steps)
    for i in range(self.p_steps):
        disp_steps_year[i] = delta_q0 * (i + 1)
        for d in range(self.days):
            disp_steps_month[i][d] = (calc_elastic_mondemand(self, 0, d) / self.p_steps) * (i + 1)
        price_steps[i] = delta_p0 * (i + 1)
    return disp_steps_year, disp_steps_month, price_steps


'''
def mc_price_curve(self, year, disp):
    """returns the electricity price for the current Microgrid Demand"""
    price = (disp_sum_year(self, year, disp)
             / calc_res_demand_year(self) + self.elasticity - 1) * (self.hist_price / self.elasticity)
    return price
    
def calc_res_demand_year(self):
    """calculate the annual residential demand"""
    res_demand_year = 0
    for d in self.days:
        for h in self.hours:
            for i in self.house:
                res_demand_year += self.res_demand[i][d][h] * self.max_house_str[i] * self.d_weights[d]
    return res_demand_year
'''