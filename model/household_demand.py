import numpy as np
import pandas as pd



soc = np.zeros(24)
power = 1
soc_max = 6
soc_min = 6 * 0.2
max_pvcap = 2.7

demand = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1, 1, 1, 1, 1, 1, 1, 1, 1])
feed_in = np.zeros(24)
res_demand = np.zeros(24)
pv_production = np.zeros(24)
pv_cap_fact = np.array([0,0,0,0,0,0,0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1,1,1,1,1,1,0,0,0,0,0,0])

for n in range(2):
    for h in range(24):
        pv_production[h] = max_pvcap * pv_cap_fact[h]
        res_demand[h] = demand[h] - pv_production[h]
        if h != 0:
            soc[h] = soc[h - 1]
        else:
            soc[h] = soc[23]
        if res_demand[h] < 0:
            if soc[h] - res_demand[h] < soc_max:
                soc[h] -= res_demand[h]
            else:
                feed_in[h] = (-res_demand[h]) - (soc_max - soc[h])
                soc[h] = soc_max
            res_demand[h] = 0


        else:
            bat_out = res_demand[h]
            res_demand[h] = res_demand[h] - min(soc[h] - soc_min, res_demand[h])
            soc[h] -= min(soc[h] - soc_min, bat_out)


print('DEMAND:')
print(demand)
print('PV_PRODUCTION:')
print(pv_production)
print('RES_DEMAND:')
print(res_demand)
print('SOC:')
print(soc)
print('FEEDIN:')
print(feed_in)


