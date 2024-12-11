import numpy as np

import pandas as pd


def read_data():
    months = ['Aug', 'Sep']

    data = [[None, None, None, None],
            [None, None, None, None]]

    for m in range(2):
        for i in range(4):
            data[m][i] = {sheet: df.fillna(0) for sheet, df in
                          pd.read_excel(months[m] + '- Household' + str(i + 1) + '.xlsx', decimal=',',
                                        sheet_name=None).items()}
            #data[m][i] = pd.read_excel(months[m] + '- Household' + str(i + 1) + '.xlsx', decimal=',', sheet_name=None).fillna(0)
    return data



data = read_data()
data_aug = [None] * 31
data_sept = [None] * 17
avg_days_aug = np.zeros((31, 24))
avg_days_sept = np.zeros((17, 24))
avg_aug= np.zeros((5, 24))
avg_sept = np.zeros((5, 24))

for house in range(4):
    for d in range(31):
        data_aug[d] = data[0][house][str(d+1)+'-8-2024'].to_numpy()[:,1]
        for i in range(0, 1440, 60):
            avg_days_aug[d][i // 60] = np.mean(data_aug[d][i:i + 60])
    for d in range(17):
        data_sept[d] = data[1][house][str(d + 1) + '-9-2024'].to_numpy()[:,1]
        for i in range(0, 1440, 60):
            avg_days_sept[d][i // 60] = np.mean(data_sept[d][i:i + 60])
    for h in range(24):
        avg_aug[house][h] = np.mean(avg_days_aug[:, h])
        avg_sept[house][h] = np.mean(avg_days_sept[:, h])

pv_data = pd.read_excel('pv_production.xlsx', decimal=',', sheet_name=None)

avg_pv_aug = np.zeros((24))
avg_pv_sept = np.zeros((24))
avg_pv_jan = np.zeros((24))

for h in range(24):
    avg_pv_aug[h] = np.mean(pv_data['august'].to_numpy()[h::24])
    avg_pv_sept[h] = np.mean(pv_data['september'].to_numpy()[h::24])
    avg_pv_jan[h] = np.mean(pv_data['jannuary'].to_numpy()[h::24])



avg_aug = np.round(avg_aug / 1000, 2)
avg_sept = np.round(avg_sept / 1000, 2)

avg_pv_aug = np.round(avg_pv_aug, 2)
avg_pv_sept = np.round(avg_pv_sept, 2)
avg_pv_jan = np.round(avg_pv_jan, 2)


for h in range(24):
    avg_aug[4][h] = np.mean(avg_aug[:-1, h])
    avg_sept[4][h] = np.mean(avg_sept[:-1, h])

avg_aug = pd.DataFrame(avg_aug)
avg_sept = pd.DataFrame(avg_sept)

avg_pv_aug = pd.DataFrame(avg_pv_aug.T)
avg_pv_sept = pd.DataFrame(avg_pv_sept.T)
avg_pv_jan = pd.DataFrame(avg_pv_jan.T)

with pd.ExcelWriter('results.xlsx', engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    avg_aug.to_excel(writer, sheet_name='August', index=False)
    avg_sept.to_excel(writer, sheet_name='September', index=False)
    avg_pv_aug.to_excel(writer, sheet_name='PV August', index=False)
    avg_pv_sept.to_excel(writer, sheet_name='PV September', index=False)
    avg_pv_jan.to_excel(writer, sheet_name='PV January', index=False)



