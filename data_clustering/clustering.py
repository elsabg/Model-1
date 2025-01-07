import numpy as np

import pandas as pd



months = ['Aug', 'Sep']
number_households = 4
data = [[None, None, None, None],
        [None, None, None, None]]

#----------------------------------------------------------------------#
# Read demand data                                                     #
#----------------------------------------------------------------------#

for m in range(len(months)):
    for i in range(number_households):
        data[m][i] = {sheet: df.fillna(0) for sheet, df in
                        pd.read_excel(months[m] + '- Household' + str(i + 1) + '.xlsx', decimal=',',
                                    sheet_name=None).items()}

#----------------------------------------------------------------------#
# Read PV data                                                         #
#----------------------------------------------------------------------#

pv_data = pd.read_excel('pv_production.xlsx', decimal=',', sheet_name=None)

#----------------------------------------------------------------------#
# Calculate average daily demand for each household (per month)        #
#----------------------------------------------------------------------#

len_aug = min(len(data[0][i]) for i in range(number_households))
len_sept = min(len(data[1][i]) for i in range(number_households))

data_aug = [None] * len_aug
data_sept = [None] * len_sept
avg_days_aug = np.zeros((len_aug, 24))
avg_days_sept = np.zeros((len_sept, 24))
avg_aug= np.zeros((number_households + 1, 24))
avg_sept = np.zeros((number_households + 1, 24))

for house in range(number_households):
    for d in range(len_aug):
        data_aug[d] = data[0][house][str(d+1)+'-8-2024'].to_numpy()[:,1]
        for i in range(0, 1440, 60):
            avg_days_aug[d][i // 60] = np.mean(data_aug[d][i:i + 60])
    for d in range(len_sept):
        data_sept[d] = data[1][house][str(d + 1) + '-9-2024'].to_numpy()[:,1]
        for i in range(0, 1440, 60):
            avg_days_sept[d][i // 60] = np.mean(data_sept[d][i:i + 60])
    for h in range(24):
        avg_aug[house][h] = np.mean(avg_days_aug[:, h])
        avg_sept[house][h] = np.mean(avg_days_sept[:, h])



avg_aug = avg_aug / 1000
avg_sept = avg_sept / 1000

for h in range(24):
    avg_aug[4][h] = np.round(np.mean(avg_aug[:-1, h]), 2)
    avg_sept[4][h] = np.round(np.mean(avg_sept[:-1, h]), 2)

#----------------------------------------------------------------------#
# Calculate average daily pv capacity factors (per month)              #
#----------------------------------------------------------------------#

avg_pv_summer = np.zeros(24)
avg_pv_spraut = np.zeros(24)
avg_pv_winter = np.zeros(24)

for h in range(24):
    avg_pv_summer[h] = np.mean(pv_data['Summer'].to_numpy()[h::24])
    avg_pv_spraut[h] = np.mean(pv_data['SprAut'].to_numpy()[h::24])
    avg_pv_winter[h] = np.mean(pv_data['Winter'].to_numpy()[h::24])

avg_pv_summer = np.round(avg_pv_summer, 2)
avg_pv_spraut = np.round(avg_pv_spraut, 2)
avg_pv_winter = np.round(avg_pv_winter, 2)

#----------------------------------------------------------------------#
# Save results to excel                                                #
#----------------------------------------------------------------------#

avg_aug = pd.DataFrame(avg_aug)
avg_sept = pd.DataFrame(avg_sept)

avg_pv_summer = pd.DataFrame(avg_pv_summer)
avg_pv_spraut = pd.DataFrame(avg_pv_spraut)
avg_pv_winter = pd.DataFrame(avg_pv_winter)

with pd.ExcelWriter('results.xlsx', engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    avg_aug.to_excel(writer, sheet_name='August', index=False)
    avg_sept.to_excel(writer, sheet_name='September', index=False)
    avg_pv_summer.to_excel(writer, sheet_name='PV Summer', index=False)
    avg_pv_spraut.to_excel(writer, sheet_name='PV SpringAutumn', index=False)
    avg_pv_winter.to_excel(writer, sheet_name='PV Winter', index=False)



