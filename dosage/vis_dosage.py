import pandas as pd
import pdb
import matplotlib.pyplot as plt
import numpy as np
import datetime
import dateutil.relativedelta
import seaborn as sns

prediction_win_size = 6
dosage_data = pd.read_csv('final_predictions_mupod_dosage.csv', header=None)
dosage_data.columns = ['ENROLID', 'FILLDATE', 'DOSAGE']
dosage_data['FILLDATE'] = dosage_data['FILLDATE'].str.replace('-','')
dosage_data = dosage_data.astype(int)

test_set_predictions = pd.read_csv('final_predictions_mupod.csv')
test_set_predictions.columns =['ENROLID', 'TP', 'TN', 'FP', 'FN']
test_set_predictions['dosage_avg'] = 0
test_set_predictions['dosage_std'] = 0
test_set_predictions['dosage_avg_6month'] = 0
test_set_predictions['dosage_std_6month'] = 0

test_set_predictions['dosage_1_month_before'] = 0
test_set_predictions['dosage_2_month_before'] = 0
test_set_predictions['dosage_3_month_before'] = 0
test_set_predictions['dosage_4_month_before'] = 0
test_set_predictions['dosage_5_month_before'] = 0
test_set_predictions['dosage_6_month_before'] = 0


test_demographics = pd.read_csv('test_demographics.csv')


dosage_data_grouped = dosage_data.groupby(by='ENROLID')

i=0
for key, group in dosage_data_grouped:
    # pdb.set_trace()
    if i%10000 ==0:
        print(i)
    i+=1
    current_sample_demog = test_demographics[test_demographics['ENROLID'] == key]
    current_diagnoses_date = current_sample_demog['DIAGNOSES_DATE_OR_last_record_Date']
    
    # Average and STD of all dosage from the first month to the diagnoses/last record date.
    test_set_predictions.loc[test_set_predictions['ENROLID']==key , 'dosage_avg'] = group[group['FILLDATE'] <= int(current_diagnoses_date)]['DOSAGE'].mean()
    test_set_predictions.loc[test_set_predictions['ENROLID']==key , 'dosage_std'] = group[group['FILLDATE'] <= int(current_diagnoses_date)]['DOSAGE'].std()

    # Average and STD of the dosage within the 6-month prediction window
    start_pred_wind = datetime.datetime.strptime(str(int(current_diagnoses_date)), '%Y%m') - dateutil.relativedelta.relativedelta(months=6)
    end_pred_wind = datetime.datetime.strptime(str(int(current_diagnoses_date)), '%Y%m') - dateutil.relativedelta.relativedelta(months=1)    
    withing_prediction_win_records = group[(group['FILLDATE'] <= int(end_pred_wind.strftime('%Y%m'))) & (group['FILLDATE'] >= int(start_pred_wind.strftime('%Y%m')))]
    
    if len(withing_prediction_win_records)< prediction_win_size:
        test_set_predictions.loc[test_set_predictions['ENROLID']==key , 'dosage_avg_6month'] = withing_prediction_win_records['DOSAGE'].append(pd.Series([0]*(prediction_win_size-withing_prediction_win_records.shape[0]))).mean()
        test_set_predictions.loc[test_set_predictions['ENROLID']==key , 'dosage_std_6month'] = withing_prediction_win_records['DOSAGE'].append(pd.Series([0]*(prediction_win_size-withing_prediction_win_records.shape[0]))).std()
    else:    
        test_set_predictions.loc[test_set_predictions['ENROLID']==key , 'dosage_avg_6month'] = withing_prediction_win_records['DOSAGE'].mean()
        test_set_predictions.loc[test_set_predictions['ENROLID']==key , 'dosage_std_6month'] = withing_prediction_win_records['DOSAGE'].std()

    # Dosage for the 6-month prediction window seperately for each month
    month_before_temp = datetime.datetime.strptime(str(int(current_diagnoses_date)), '%Y%m') - dateutil.relativedelta.relativedelta(months=6)    
    temp_records = group[group['FILLDATE'] == int(month_before_temp.strftime('%Y%m'))]
    if len(temp_records) == 1:
        test_set_predictions.loc[test_set_predictions['ENROLID']==key , 'dosage_6_month_before'] = temp_records['DOSAGE'].values
    elif len(temp_records) < 1:    
        test_set_predictions.loc[test_set_predictions['ENROLID']==key , 'dosage_6_month_before'] = 0
    elif len(temp_records) > 1:    
        pdb.set_trace()
        print('Warning')

    month_before_temp = datetime.datetime.strptime(str(int(current_diagnoses_date)), '%Y%m') - dateutil.relativedelta.relativedelta(months=5)    
    temp_records = group[group['FILLDATE'] == int(month_before_temp.strftime('%Y%m'))]
    if len(temp_records) == 1:
        test_set_predictions.loc[test_set_predictions['ENROLID']==key , 'dosage_5_month_before'] = temp_records['DOSAGE'].values
    elif len(temp_records) < 1:    
        test_set_predictions.loc[test_set_predictions['ENROLID']==key , 'dosage_5_month_before'] = 0
    elif len(temp_records) > 1:    
        pdb.set_trace()
        print('Warning')

    month_before_temp = datetime.datetime.strptime(str(int(current_diagnoses_date)), '%Y%m') - dateutil.relativedelta.relativedelta(months=4)    
    temp_records = group[group['FILLDATE'] == int(month_before_temp.strftime('%Y%m'))]
    if len(temp_records) == 1:
        test_set_predictions.loc[test_set_predictions['ENROLID']==key , 'dosage_4_month_before'] = temp_records['DOSAGE'].values
    elif len(temp_records) < 1:    
        test_set_predictions.loc[test_set_predictions['ENROLID']==key , 'dosage_4_month_before'] = 0
    elif len(temp_records) > 1:    
        pdb.set_trace()
        print('Warning')


    month_before_temp = datetime.datetime.strptime(str(int(current_diagnoses_date)), '%Y%m') - dateutil.relativedelta.relativedelta(months=3)    
    temp_records = group[group['FILLDATE'] == int(month_before_temp.strftime('%Y%m'))]
    if len(temp_records) == 1:
        test_set_predictions.loc[test_set_predictions['ENROLID']==key , 'dosage_3_month_before'] = temp_records['DOSAGE'].values
    elif len(temp_records) < 1:    
        test_set_predictions.loc[test_set_predictions['ENROLID']==key , 'dosage_3_month_before'] = 0
    elif len(temp_records) > 1:    
        pdb.set_trace()
        print('Warning')

    month_before_temp = datetime.datetime.strptime(str(int(current_diagnoses_date)), '%Y%m') - dateutil.relativedelta.relativedelta(months=2)    
    temp_records = group[group['FILLDATE'] == int(month_before_temp.strftime('%Y%m'))]
    if len(temp_records) == 1:
        test_set_predictions.loc[test_set_predictions['ENROLID']==key , 'dosage_2_month_before'] = temp_records['DOSAGE'].values
    elif len(temp_records) < 1:    
        test_set_predictions.loc[test_set_predictions['ENROLID']==key , 'dosage_2_month_before'] = 0
    elif len(temp_records) > 1:    
        pdb.set_trace()
        print('Warning')


    month_before_temp = datetime.datetime.strptime(str(int(current_diagnoses_date)), '%Y%m') - dateutil.relativedelta.relativedelta(months=1)    
    temp_records = group[group['FILLDATE'] == int(month_before_temp.strftime('%Y%m'))]
    if len(temp_records) == 1:
        test_set_predictions.loc[test_set_predictions['ENROLID']==key , 'dosage_1_month_before'] = temp_records['DOSAGE'].values
    elif len(temp_records) < 1:    
        test_set_predictions.loc[test_set_predictions['ENROLID']==key , 'dosage_1_month_before'] = 0
    elif len(temp_records) > 1:    
        pdb.set_trace()
        print('Warning')



test_set_predictions_tp = test_set_predictions[test_set_predictions['TP']==1]
test_set_predictions_tn = test_set_predictions[test_set_predictions['TN']==1]

# Stat of MME across all records
avg_mme_tp = test_set_predictions_tp['dosage_avg'].mean()
std_mme_tp = test_set_predictions_tp['dosage_avg'].std()
perc_25_tp = np.percentile(test_set_predictions_tp['dosage_avg'], 25)
perc_50_tp = np.percentile(test_set_predictions_tp['dosage_avg'], 50)
perc_75_tp = np.percentile(test_set_predictions_tp['dosage_avg'], 75)

avg_mme_tn = test_set_predictions_tn['dosage_avg'].mean()
std_mme_tn = test_set_predictions_tn['dosage_avg'].std()
perc_25_tn = np.percentile(test_set_predictions_tn['dosage_avg'], 25)
perc_50_tn = np.percentile(test_set_predictions_tn['dosage_avg'], 50)
perc_75_tn = np.percentile(test_set_predictions_tn['dosage_avg'], 75)


# Stat of MME within the predic tion window
avg_within_win_mme_tp = test_set_predictions_tp['dosage_avg_6month'].mean()
std_within_win_mme_tp = test_set_predictions_tp['dosage_avg_6month'].std()
perc_within_win_25_tp = np.percentile(test_set_predictions_tp['dosage_avg_6month'], 25)
perc_within_win_50_tp = np.percentile(test_set_predictions_tp['dosage_avg_6month'], 50)
perc_within_win_75_tp = np.percentile(test_set_predictions_tp['dosage_avg_6month'], 75)

avg_within_win_mme_tn = test_set_predictions_tn['dosage_avg_6month'].mean()
std_within_win_mme_tn = test_set_predictions_tn['dosage_avg_6month'].std()
perc_within_win_25_tn = np.percentile(test_set_predictions_tn['dosage_avg_6month'], 25)
perc_within_win_50_tn = np.percentile(test_set_predictions_tn['dosage_avg_6month'], 50)
perc_within_win_75_tn = np.percentile(test_set_predictions_tn['dosage_avg_6month'], 75)


# Stat of MME 1 month before
avg_1_month_before_mme_tp = test_set_predictions_tp['dosage_1_month_before'].mean()
std_1_month_before_mme_tp = test_set_predictions_tp['dosage_1_month_before'].std()
perc_1_month_before_25_tp = np.percentile(test_set_predictions_tp['dosage_1_month_before'], 25)
perc_1_month_before_50_tp = np.percentile(test_set_predictions_tp['dosage_1_month_before'], 50)
perc_1_month_before_75_tp = np.percentile(test_set_predictions_tp['dosage_1_month_before'], 75)

avg_1_month_before_mme_tn = test_set_predictions_tn['dosage_1_month_before'].mean()
std_1_month_before_mme_tn = test_set_predictions_tn['dosage_1_month_before'].std()
perc_1_month_before_25_tn = np.percentile(test_set_predictions_tn['dosage_1_month_before'], 25)
perc_1_month_before_50_tn = np.percentile(test_set_predictions_tn['dosage_1_month_before'], 50)
perc_1_month_before_75_tn = np.percentile(test_set_predictions_tn['dosage_1_month_before'], 75)


# Stat of MME 2 month before
avg_2_month_before_mme_tp = test_set_predictions_tp['dosage_2_month_before'].mean()
std_2_month_before_mme_tp = test_set_predictions_tp['dosage_2_month_before'].std()
perc_2_month_before_25_tp = np.percentile(test_set_predictions_tp['dosage_2_month_before'], 25)
perc_2_month_before_50_tp = np.percentile(test_set_predictions_tp['dosage_2_month_before'], 50)
perc_2_month_before_75_tp = np.percentile(test_set_predictions_tp['dosage_2_month_before'], 75)

avg_2_month_before_mme_tn = test_set_predictions_tn['dosage_2_month_before'].mean()
std_2_month_before_mme_tn = test_set_predictions_tn['dosage_2_month_before'].std()
perc_2_month_before_25_tn = np.percentile(test_set_predictions_tn['dosage_2_month_before'], 25)
perc_2_month_before_50_tn = np.percentile(test_set_predictions_tn['dosage_2_month_before'], 50)
perc_2_month_before_75_tn = np.percentile(test_set_predictions_tn['dosage_2_month_before'], 75)


# Stat of MME 3 month before
avg_3_month_before_mme_tp = test_set_predictions_tp['dosage_3_month_before'].mean()
std_3_month_before_mme_tp = test_set_predictions_tp['dosage_3_month_before'].std()
perc_3_month_before_25_tp = np.percentile(test_set_predictions_tp['dosage_3_month_before'], 25)
perc_3_month_before_50_tp = np.percentile(test_set_predictions_tp['dosage_3_month_before'], 50)
perc_3_month_before_75_tp = np.percentile(test_set_predictions_tp['dosage_3_month_before'], 75)

avg_3_month_before_mme_tn = test_set_predictions_tn['dosage_3_month_before'].mean()
std_3_month_before_mme_tn = test_set_predictions_tn['dosage_3_month_before'].std()
perc_3_month_before_25_tn = np.percentile(test_set_predictions_tn['dosage_3_month_before'], 25)
perc_3_month_before_50_tn = np.percentile(test_set_predictions_tn['dosage_3_month_before'], 50)
perc_3_month_before_75_tn = np.percentile(test_set_predictions_tn['dosage_3_month_before'], 75)


# Stat of MME 4 month before
avg_4_month_before_mme_tp = test_set_predictions_tp['dosage_4_month_before'].mean()
std_4_month_before_mme_tp = test_set_predictions_tp['dosage_4_month_before'].std()
perc_4_month_before_25_tp = np.percentile(test_set_predictions_tp['dosage_4_month_before'], 25)
perc_4_month_before_50_tp = np.percentile(test_set_predictions_tp['dosage_4_month_before'], 50)
perc_4_month_before_75_tp = np.percentile(test_set_predictions_tp['dosage_4_month_before'], 75)

avg_4_month_before_mme_tn = test_set_predictions_tn['dosage_4_month_before'].mean()
std_4_month_before_mme_tn = test_set_predictions_tn['dosage_4_month_before'].std()
perc_4_month_before_25_tn = np.percentile(test_set_predictions_tn['dosage_4_month_before'], 25)
perc_4_month_before_50_tn = np.percentile(test_set_predictions_tn['dosage_4_month_before'], 50)
perc_4_month_before_75_tn = np.percentile(test_set_predictions_tn['dosage_4_month_before'], 75)


# Stat of MME 5 month before
avg_5_month_before_mme_tp = test_set_predictions_tp['dosage_5_month_before'].mean()
std_5_month_before_mme_tp = test_set_predictions_tp['dosage_5_month_before'].std()
perc_5_month_before_25_tp = np.percentile(test_set_predictions_tp['dosage_5_month_before'], 25)
perc_5_month_before_50_tp = np.percentile(test_set_predictions_tp['dosage_5_month_before'], 50)
perc_5_month_before_75_tp = np.percentile(test_set_predictions_tp['dosage_5_month_before'], 75)

avg_5_month_before_mme_tn = test_set_predictions_tn['dosage_5_month_before'].mean()
std_5_month_before_mme_tn = test_set_predictions_tn['dosage_5_month_before'].std()
perc_5_month_before_25_tn = np.percentile(test_set_predictions_tn['dosage_5_month_before'], 25)
perc_5_month_before_50_tn = np.percentile(test_set_predictions_tn['dosage_5_month_before'], 50)
perc_5_month_before_75_tn = np.percentile(test_set_predictions_tn['dosage_5_month_before'], 75)


# Stat of MME 6 month before
avg_6_month_before_mme_tp = test_set_predictions_tp['dosage_6_month_before'].mean()
std_6_month_before_mme_tp = test_set_predictions_tp['dosage_6_month_before'].std()
perc_6_month_before_25_tp = np.percentile(test_set_predictions_tp['dosage_6_month_before'], 25)
perc_6_month_before_50_tp = np.percentile(test_set_predictions_tp['dosage_6_month_before'], 50)
perc_6_month_before_75_tp = np.percentile(test_set_predictions_tp['dosage_6_month_before'], 75)

avg_6_month_before_mme_tn = test_set_predictions_tn['dosage_6_month_before'].mean()
std_6_month_before_mme_tn = test_set_predictions_tn['dosage_6_month_before'].std()
perc_6_month_before_25_tn = np.percentile(test_set_predictions_tn['dosage_6_month_before'], 25)
perc_6_month_before_50_tn = np.percentile(test_set_predictions_tn['dosage_6_month_before'], 50)
perc_6_month_before_75_tn = np.percentile(test_set_predictions_tn['dosage_6_month_before'], 75)

pdb.set_trace()

# Plot density functions
# sns.kdeplot(test_set_predictions_tp['dosage_avg_6month'], color='red')
# sns.kdeplot(test_set_predictions_tn['dosage_avg_6month'], color='blue')
# plt.savefig('dens_functions_dosage_avg_6month.png', dpi=600)
# plt.close()

# # Plot violin plots
# sns.violinplot(test_set_predictions_tp['dosage_avg_6month'], color='red')
# sns.violinplot(test_set_predictions_tn['dosage_avg_6month'], color='blue')
# plt.savefig('violins_dosage_avg_6month.png', dpi=600)
# plt.close()

# bins = 30
# plt.hist(test_set_predictions_tp['dosage_avg_6month'], color='red')
# plt.legend()
# plt.xlabel('Opioid medications', fontsize=12)
# plt.ylabel('Frequency', fontsize=12)
# plt.savefig("histogram.png", dpi=600)
# print('Test')