import numpy as np
from gurobipy import *
from matplotlib import pyplot as plt


from matplotlib.ticker import FixedLocator, FuncFormatter

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
#--------------------------------------------------------------------------------#
#                                                                                #
# calculate the residual (without PV) demand and feed in of prosumers            #
#                                                                                #
#--------------------------------------------------------------------------------#

def calc_pros_demand_feedin(self):
    """calculates Prosumer demand and feedin electricity"""
    for h_type in self.res_demand:
        if h_type != 'Type 1':
            for i in range(len(self.res_demand[h_type])):
                res_demand, feedin = pros_behavoir(self, h_type, i)
                for h in range(self.hours):
                    self.res_demand[h_type][i][h] = res_demand[h]
                    self.pros_feedin[h_type][i][h] = feedin[h]

def pros_behavoir(self, h_type, day):
    """returns the prosumer demand and feedin with an energy storage"""
    pv_generation = np.zeros(self.hours)
    res_demand = np.zeros(self.hours)
    feed_in = np.zeros(self.hours)
    soc = np.zeros(self.hours)
    for i in range(2): # run twice to get soc from hour 24 for hour 0
        for h in range(self.hours):
            pv_generation[h] = self.cap_fact[day][h] * self.avg_pv_cap_str[h_type]
            res_demand[h] = self.res_demand[h_type][day][h] - pv_generation[h]
            if h != 0:
                soc[h] = soc[h - 1]
            else:
                soc[h] = soc[23]
            if res_demand[h] < 0:
                if soc[h] - res_demand[h] < self.pros_soc_max:
                    soc[h] -= res_demand[h]
                else:
                    feed_in[h] = (-res_demand[h]) - (self.pros_soc_max - soc[h])
                    soc[h] = self.pros_soc_max
                res_demand[h] = 0
            else:
                bat_out = res_demand[h]
                res_demand[h] = res_demand[h] - min(soc[h] - self.pros_soc_min, res_demand[h])
                soc[h] -= min(soc[h] - self.pros_soc_min, bat_out)

    return res_demand, feed_in

#--------------------------------------------------------------------------------#
#                                                                                #
# hourly Microgrid demand (normal/demand elasticity)                             #
#                                                                                #
#--------------------------------------------------------------------------------#

def mc_demand(self, num_house, y, d, h):
    """returns hourly Microgrid demand of all household types"""
    hourly_demand = 0
    if y == 0:
        for i in self.house:
            hourly_demand += self.res_demand[i][d][h] * num_house[i]
    else:
        for i in self.house:
            hourly_demand += self.res_demand[i][d][h] * num_house[i, y - 1]
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
    return quicksum(self.disp_steps_month[self.steps - 1 - i][d] * bin_price_curve[i] for i in range(self.steps))


def demand_sum_year(self, year, disp, ud, b_out, b_in):
    """calculates the annual demand in the grid"""
    dem_year = quicksum(
            quicksum(
        quicksum(disp[g, year, d, h] for g in self.techs_g) + ud[year, d, h] + b_out[year, d, h] - b_in[year, d, h]
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

#--------------------------------------------------------------------------------#
#                                                                                #
# plot functions                                                                 #
#                                                                                #
#--------------------------------------------------------------------------------#

def plot_demand(self):
    '''plot the microgrid demand of the consumer and prosumer households'''
    hours = np.arange(24)
    fig1, axs1 = plt.subplots(figsize=(10, 8))
    axs1.plot(hours, self.res_demand['Type 1'][0], label='Summer', color='orange',marker='o', linewidth=4)
    axs1.plot(hours, self.res_demand['Type 1'][1], label='Spring/Autumn', color='darkred', marker='o', linewidth=4)
    axs1.plot(hours, self.res_demand['Type 1'][2], label='Winter', marker='o', linewidth=4)
    fig1.suptitle("Consumer", fontsize=35)
    #axs1.legend(loc='lower right', fontsize=30)

    fig2, axs2 = plt.subplots(figsize=(10, 8))
    axs2.plot(hours, self.res_demand['Type 2'][0], label='Summer', color='orange',marker='o', linewidth=4)
    axs2.plot(hours, self.res_demand['Type 2'][1],  label='Spring/Autumn',color='darkred', marker='o', linewidth=4)
    axs2.plot(hours, self.res_demand['Type 2'][2], label='Winter', marker='o', linewidth=4)
    fig2.suptitle("Prosumer", fontsize=35)
    axs2.legend(loc='upper center', fontsize=30)

    for ax in [axs1, axs2]:
        ax.set_ylabel('Grid Demand (kWh)', fontsize=30)
        ax.set_xlabel('Hour of Day (h)', fontsize=30)
        ax.set_ylim(bottom=0, top = 1.1 * max(self.res_demand['Type 1'][0]))
        ax.set_xticks(hours[::4])
        ax.set_xticklabels(hours[::4], fontsize=30)
        ax.tick_params(axis='y', labelsize=30)  # Set y-tick label size
        ax.grid(which="major", axis="y", color="#758D99", alpha=0.4, zorder=1)

    fig1.savefig('plots/demandCons')
    fig2.savefig('plots/demandPros')
    fig1.tight_layout()
    fig2.tight_layout()

    plt.show()

def plot_pros_demand_ud(self):
    '''plot the distribution of unmet demand for a prosumer household'''
    hours = np.arange(24)
    fig2, axs2 = plt.subplots(figsize=(10, 6))
    axs2.plot(hours, self.res_demand['Type 2'][0], label='Grid Demand', color='black',marker='o', linewidth=4)
    axs2.bar(hours, self.ud_pros[0], label='Unmet Demand', color='deepskyblue')
    #fig2.suptitle("Prosumer", fontsize=35)
    axs2.legend(loc='upper center', fontsize=30)

    for ax in [axs2]:
        ax.set_ylabel('Energy per Household (kWh)', fontsize=30)
        ax.set_xlabel('Hour of Day (h)', fontsize=30)
        ax.set_ylim(bottom=0, top = 1.1 * max(self.res_demand['Type 2'][0]))
        ax.set_xticks(hours[::4])
        ax.set_xticklabels(hours[::4], fontsize=30)
        ax.tick_params(axis='y', labelsize=30)  # Set y-tick label size

        ax.grid(which="major", axis="y", color="#758D99", alpha=0.4, zorder=1)

    fig2.tight_layout()
    fig2.savefig('plots/demand_unmet_Pros')

    plt.show()

def fill_pros_demandarray(self, unmetD, disp_feedin, num_households):
    """retruns Array with prosumer demand data"""
    ret = np.zeros((2,3))
    ud_years = np.zeros((self.years))
    fi_years = np.zeros((self.years))
    for d in range(self.days):
        ret[0][1] += np.sum(self.res_demand['Type 2'][d]) * self.d_weights[d]
        ret[0][2] += np.sum(self.pros_feedin['Type 2'][d]) * self.d_weights[d]
        for i in range(2):
            ret[i][0] += np.sum(self.demand_2[d]) * self.d_weights[d]
    for y in range(self.years):
        for d in range(self.days):
            ud_years[y] += np.sum(unmetD[y][d]) * self.d_weights[d]
            fi_years[y] += np.sum(disp_feedin[y][d]) * self.d_weights[d]
        ud_years[y] = ud_years[y] / num_households[1][y]
        fi_years[y] = fi_years[y] / num_households[1][y]
    ret[1][1] = np.mean(ud_years)
    ret[1][2] = ret[0][2] - np.mean(fi_years)

    return ret