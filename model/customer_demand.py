import numpy as np

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

def calc_elastic_mondemand(self, d):
    """calculates the monthly demand depending on electricity price"""
    self.elastic_mondemand = (self.hist_demand[d] *
                            (1 - self.elasticity + self.elasticity * (self.elec_price / self.hist_price[d])))

def mc_demand(self, d, h):
    """returns hourly Microgrid demand of all household types"""
    hourly_demand = 0
    for i in self.house:
        hourly_demand += self.res_demand[i][d][h] * self.max_house_str[i]
    return hourly_demand

def elastic_mc_demand(self, d, h):
    """return hourly Microgrid demand of all household types depending on the electricity price"""
    hourly_demand = 0
    for i in self.house:
        hourly_demand += ((self.elastic_mondemand[d] / self.hist_demand[d])
                          * self.res_demand[i][d][h] * self.max_house_str[i])
    return hourly_demand

def mc_price_curve(self, disp):
    """returns the electricity price for the current Microgrid Demand"""
    el_price = 0
    for d in range(self.days):
        el_price += ((((self.elastic_mondemand[d] / self.hist_demand[d]) - 1 + self.elasticity)
                     / self.elasticity) * self.hist_price) * (self.d_weights[d] / sum(self.d_weights))
    return el_price