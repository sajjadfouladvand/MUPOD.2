import pdb
import pandas as pd
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind
import math
import numpy as np


# ==== Reading the data
# pdb.set_trace()

# train_meds_filename = 'outputs/train_medications.csv'
# train_diags_filename = 'outputs/train_diagnoses.csv'
# train_procs_filename = 'outputs/train_procedures.csv'
# train_demogs_filename = 'outputs/train_demographics.csv'
# train_labels_filename = 'outputs/train_labels.csv'

# validation_meds_filename = 'outputs/validation_medications.csv'
# validation_diags_filename = 'outputs/validation_diagnoses.csv'
# validation_procs_filename = 'outputs/validation_procedures.csv'
# validation_demogs_filename = 'outputs/validation_demographics.csv'
# validation_labels_filename = 'outputs/validation_labels.csv'

# test_meds_filename = 'outputs/validation_medications.csv'
# test_diags_filename = 'outputs/validation_diagnoses.csv'
# test_procs_filename = 'outputs/validation_procedures.csv'
# test_demogs_filename = 'outputs/validation_demographics.csv'
# test_labels_filename = 'outputs/validation_labels.csv'



train_stationary_filename = 'outputs/train_stationary.csv'#_normalized_features_filtered.csv'
validation_stationary_filename = 'outputs/validation_stationary.csv'#_stationary_normalized_features_filtered.csv'
test_stationary_filename = 'outputs/test_stationary.csv'#_stationary_normalized_features_filtered.csv'

train_meta_filename = 'outputs/train_demographics_shuffled.csv'
validation_meta_filename = 'outputs/validation_demographics_shuffled.csv'
test_meta_filename = 'outputs/test_demographics_shuffled.csv'

# Stats on the medications for oud-no
train_data = pd.read_csv(train_stationary_filename)
validation_data = pd.read_csv(validation_stationary_filename)
test_data = pd.read_csv(test_stationary_filename)
data_all = pd.concat([train_data, validation_data, test_data])

data_all_oud_yes = data_all[data_all['Label']==1]
data_all_oud_no = data_all[data_all['Label']==0]


train_metadata = pd.read_csv(train_meta_filename)
validation_metadata = pd.read_csv(validation_meta_filename)
test_metadata = pd.read_csv(test_meta_filename)
data_all_meta = pd.concat([train_metadata, validation_metadata, test_metadata])
data_all_meta_oud_yes = data_all_meta[data_all_meta[' Label']==1]
data_all_meta_oud_no = data_all_meta[data_all_meta[' Label']==0]


# Calculate demographics
dob_oud_yes = data_all_meta_oud_yes['DOB']
age_oud_yes = 2020 - dob_oud_yes

dob_oud_no = data_all_meta_oud_no['DOB']
age_oud_no = 2020 - dob_oud_no


dob_all = data_all_meta['DOB']
age_all = 2020 - dob_all


num_oud_yes_sex_1 = data_all_meta_oud_yes[data_all_meta_oud_yes['SEX']==1].shape[0]
num_oud_yes_sex_2 = data_all_meta_oud_yes[data_all_meta_oud_yes['SEX']==2].shape[0]

num_oud_no_sex_1 = data_all_meta_oud_no[data_all_meta_oud_no['SEX']==1].shape[0]
num_oud_no_sex_2 = data_all_meta_oud_no[data_all_meta_oud_no['SEX']==2].shape[0]

num_oud_yes_patients = data_all_oud_yes.shape[0]
num_oud_no_patients = data_all_oud_no.shape[0]


with open('results/visualization_results/stat_file_for_Table1.csv','w') as stat_file:        
    
    # Average and std of age in case and cohort
    stat_file.write('Average age in oud_yes is:\n')
    stat_file.write(str(age_oud_yes.mean()))
    stat_file.write('\n')
    stat_file.write('Standart deviation of age in oud_yes is:\n')
    stat_file.write(str(age_oud_yes.std()))        
    stat_file.write('\n')
    stat_file.write('25 percentile of age in oud_yes is:\n')
    stat_file.write(str(np.percentile(age_oud_yes, 25)))        
    stat_file.write('\n')
    stat_file.write('50 percentile of age in oud_yes is:\n')
    stat_file.write(str(np.percentile(age_oud_yes, 50)))        
    stat_file.write('\n')
    stat_file.write('75 percentile of age in oud_yes is:\n')
    stat_file.write(str(np.percentile(age_oud_yes, 75)))        
    stat_file.write('\n')    


    stat_file.write('Average age in oud_no is:\n')
    stat_file.write(str(age_oud_no.mean()))
    stat_file.write('\n')
    stat_file.write('Standart deviation of age in oud_no is:\n')
    stat_file.write(str(age_oud_no.std()))                
    stat_file.write('\n')
    stat_file.write('25 percentile of age in oud_no is:\n')
    stat_file.write(str(np.percentile(age_oud_no, 25)))        
    stat_file.write('\n')
    stat_file.write('50 percentile of age in oud_no is:\n')
    stat_file.write(str(np.percentile(age_oud_no, 50)))        
    stat_file.write('\n')
    stat_file.write('75 percentile of age in oud_no is:\n')
    stat_file.write(str(np.percentile(age_oud_no, 75)))        
    stat_file.write('\n') 

    stat_file.write('Average age in all patients:\n')
    stat_file.write(str(age_all.mean()))
    stat_file.write('\n')
    stat_file.write('Standart deviation of age in all patients:\n')
    stat_file.write(str(age_all.std()))                
    stat_file.write('\n')
    stat_file.write('25 percentile of age in all patients:\n')
    stat_file.write(str(np.percentile(age_all, 25)))        
    stat_file.write('\n')
    stat_file.write('50 percentile of age in all patients:\n')
    stat_file.write(str(np.percentile(age_all, 50)))        
    stat_file.write('\n')
    stat_file.write('75 percentile of age in all patients:\n')
    stat_file.write(str(np.percentile(age_all, 75)))        
    stat_file.write('\n') 

    # Average and std of the number of month patinets hasve been on opioid medications in case and cohort       
    stat_file.write('Average number of month on opioid in oud_yes is:\n')
    stat_file.write(str(data_all_meta_oud_yes['NUM_MONTHLY_OPIOID_PRESCS'].mean()))
    stat_file.write('\n')
    stat_file.write('Standart deviation of month on opioid in oud_yes is:\n')
    stat_file.write(str(data_all_meta_oud_yes['NUM_MONTHLY_OPIOID_PRESCS'].std()))        
    stat_file.write('\n')

    stat_file.write('Average number of month on opioid in oud_no is:\n')
    stat_file.write(str(data_all_meta_oud_no['NUM_MONTHLY_OPIOID_PRESCS'].mean()))
    stat_file.write('\n')
    stat_file.write('Standart deviation of month on opioid in oud_no is:\n')
    stat_file.write(str(data_all_meta_oud_no['NUM_MONTHLY_OPIOID_PRESCS'].std()))        
    stat_file.write('\n')


    stat_file.write('Average number of month on opioid in all data is:\n')
    stat_file.write(str(data_all_meta['NUM_MONTHLY_OPIOID_PRESCS'].mean()))
    stat_file.write('\n')
    stat_file.write('Standart deviation of month on opioid in all data is:\n')
    stat_file.write(str(data_all_meta['NUM_MONTHLY_OPIOID_PRESCS'].std()))        
    stat_file.write('\n')



    # Percentiles of on OUD
    stat_file.write('25 percentile of month on opioid in oud_yes is:\n')
    stat_file.write(str(np.percentile(data_all_meta_oud_yes['NUM_MONTHLY_OPIOID_PRESCS'], 25)))   
    stat_file.write('\n')
    stat_file.write('50 percentile of month on opioid in oud_yes is:\n')
    stat_file.write(str(np.percentile(data_all_meta_oud_yes['NUM_MONTHLY_OPIOID_PRESCS'], 50)))   
    stat_file.write('\n')
    stat_file.write('75 percentile of month on opioid in oud_yes is:\n')
    stat_file.write(str(np.percentile(data_all_meta_oud_yes['NUM_MONTHLY_OPIOID_PRESCS'], 75)))   
    stat_file.write('\n')    

    stat_file.write('25 percentile of month on opioid in oud_no is:\n')
    stat_file.write(str(np.percentile(data_all_meta_oud_no['NUM_MONTHLY_OPIOID_PRESCS'], 25)))   
    stat_file.write('\n')
    stat_file.write('50 percentile of month on opioid in oud_no is:\n')
    stat_file.write(str(np.percentile(data_all_meta_oud_no['NUM_MONTHLY_OPIOID_PRESCS'], 50)))   
    stat_file.write('\n')
    stat_file.write('75 percentile of month on opioid in oud_no is:\n')
    stat_file.write(str(np.percentile(data_all_meta_oud_no['NUM_MONTHLY_OPIOID_PRESCS'], 75)))   
    stat_file.write('\n')    

    stat_file.write('25 percentile of month on opioid in all data is:\n')
    stat_file.write(str(np.percentile(data_all_meta['NUM_MONTHLY_OPIOID_PRESCS'], 25)))   
    stat_file.write('\n')
    stat_file.write('50 percentile of month on opioid in all data is:\n')
    stat_file.write(str(np.percentile(data_all_meta['NUM_MONTHLY_OPIOID_PRESCS'], 50)))   
    stat_file.write('\n')
    stat_file.write('75 percentile of month on opioid in all data is:\n')
    stat_file.write(str(np.percentile(data_all_meta['NUM_MONTHLY_OPIOID_PRESCS'], 75)))   
    stat_file.write('\n') 

    # Average and std of data availibility in case and cohort
    stat_file.write('Average data availibility(month) in oud_yes is:\n')
    stat_file.write(str(data_all_meta_oud_yes['NUM_MONTHS_IN_DATA'].mean()))
    stat_file.write('\n')
    stat_file.write('Standart deviation of data availibility(month) in oud_yes is:\n')
    stat_file.write(str(data_all_meta_oud_yes['NUM_MONTHS_IN_DATA'].std()))        
    stat_file.write('\n')

    stat_file.write('Average data availibility(month) in oud_no is:\n')
    stat_file.write(str(data_all_meta_oud_no['NUM_MONTHS_IN_DATA'].mean()))
    stat_file.write('\n')
    stat_file.write('Standart deviation of data availibility(month) in oud_no is:\n')
    stat_file.write(str(data_all_meta_oud_no['NUM_MONTHS_IN_DATA'].std()))        
    stat_file.write('\n')

    stat_file.write('Average data availibility(month) in all data is:\n')
    stat_file.write(str(data_all_meta['NUM_MONTHS_IN_DATA'].mean()))
    stat_file.write('\n')
    stat_file.write('Standart deviation of data availibility(month) in all data is:\n')
    stat_file.write(str(data_all_meta['NUM_MONTHS_IN_DATA'].std()))        
    stat_file.write('\n')

    # Percentiles of data availability
    stat_file.write('25 percentile of data availibility(month) in oud_yes is:\n')
    stat_file.write(str(np.percentile(data_all_meta_oud_yes['NUM_MONTHS_IN_DATA'], 25)))   
    stat_file.write('\n')
    stat_file.write('50 percentile of data availibility(month) in oud_yes is:\n')
    stat_file.write(str(np.percentile(data_all_meta_oud_yes['NUM_MONTHS_IN_DATA'], 50)))   
    stat_file.write('\n')
    stat_file.write('75 percentile of data availibility(month) in oud_yes is:\n')
    stat_file.write(str(np.percentile(data_all_meta_oud_yes['NUM_MONTHS_IN_DATA'], 75)))   
    stat_file.write('\n')    

    stat_file.write('25 percentile of data availibility(month) in oud_no is:\n')
    stat_file.write(str(np.percentile(data_all_meta_oud_no['NUM_MONTHS_IN_DATA'], 25)))   
    stat_file.write('\n')
    stat_file.write('50 percentile of data availibility(month) in oud_no is:\n')
    stat_file.write(str(np.percentile(data_all_meta_oud_no['NUM_MONTHS_IN_DATA'], 50)))   
    stat_file.write('\n')
    stat_file.write('75 percentile of data availibility(month) in oud_no is:\n')
    stat_file.write(str(np.percentile(data_all_meta_oud_no['NUM_MONTHS_IN_DATA'], 75)))   
    stat_file.write('\n')    

    stat_file.write('25 percentile of data availibility(month) in all data is:\n')
    stat_file.write(str(np.percentile(data_all_meta['NUM_MONTHS_IN_DATA'], 25)))   
    stat_file.write('\n')
    stat_file.write('50 percentile of data availibility(month)d in all data is:\n')
    stat_file.write(str(np.percentile(data_all_meta['NUM_MONTHS_IN_DATA'], 50)))   
    stat_file.write('\n')
    stat_file.write('75 percentile of data availibility(month) in all data is:\n')
    stat_file.write(str(np.percentile(data_all_meta['NUM_MONTHS_IN_DATA'], 75)))   
    stat_file.write('\n') 
    
 
    
    # Sex 
    stat_file.write('Number of people with SEX=1 in oud_yes cohort is:\n')
    stat_file.write(str(num_oud_yes_sex_1))
    stat_file.write('\n')
    stat_file.write('Number of people with SEX=1 in oud_no cohort is:\n')
    stat_file.write(str(num_oud_no_sex_1))
    stat_file.write('\n')

    stat_file.write('Number of people with SEX=2 in oud_yes cohort is:\n')
    stat_file.write(str(num_oud_yes_sex_2))
    stat_file.write('\n')       
    stat_file.write('Number of people with SEX=2 in oud_no cohort is:\n')
    stat_file.write(str(num_oud_no_sex_2))
    stat_file.write('\n')

    stat_file.write('Number of people with SEX=1 in all data is:\n')
    stat_file.write(str(num_oud_yes_sex_1+num_oud_no_sex_1))
    stat_file.write('\n')       
    stat_file.write('Number of people with SEX=2 in all data is:\n')
    stat_file.write(str(num_oud_yes_sex_2+num_oud_no_sex_2))
    stat_file.write('\n')

month_on_op_ustat, month_on_op_pvalue = mannwhitneyu(data_all_meta_oud_yes['NUM_MONTHLY_OPIOID_PRESCS'], data_all_meta_oud_no['NUM_MONTHLY_OPIOID_PRESCS'])
month_in_data_ustat, month_in_data_pvalue = mannwhitneyu(data_all_meta_oud_yes['NUM_MONTHS_IN_DATA'], data_all_meta_oud_no['NUM_MONTHS_IN_DATA'])

with open('results/visualization_results/p_values_u_stats_forTable1.csv','w') as stat_file:  
    stat_file.write('Feature name, Num patients in oud_yes, percentage in oud_yes, number of patinets in our_no, percentage in oud_no, P-value, u-statistic, number of patinets in all data, perc in all data \n')
    
    # Month on opioid        
    stat_file.write('Months on opioid')
    stat_file.write(',')
    stat_file.write('-')
    stat_file.write(',')
    stat_file.write('-')    
    stat_file.write(',')
    stat_file.write('_')
    stat_file.write(',')
    stat_file.write('-')    
    stat_file.write(',')
    stat_file.write(str(month_on_op_pvalue))
    stat_file.write(',')
    stat_file.write(str(month_on_op_ustat))
    stat_file.write(',')
    stat_file.write('-')
    stat_file.write(',')
    stat_file.write('-')
    stat_file.write('\n')

    stat_file.write('Data availibility(months)')
    stat_file.write(',')
    stat_file.write('-')
    stat_file.write(',')
    stat_file.write('-')    
    stat_file.write(',')
    stat_file.write('_')
    stat_file.write(',')
    stat_file.write('-')    
    stat_file.write(',')
    stat_file.write(str(month_on_op_pvalue))
    stat_file.write(',')
    stat_file.write(str(month_on_op_ustat))
    stat_file.write(',')
    stat_file.write('-')
    stat_file.write(',')
    stat_file.write('-')
    stat_file.write('\n')

    # All other medications, diagnoses and procedures features
    for i in range(1, len(data_all.columns)-1):
        current_feature = data_all.columns[i]  
        # if current_feature == '65_tcgp_2digit':
        #     pdb.set_trace()
        # print(current_feature)
        if sum(data_all_oud_yes[current_feature].values == data_all_oud_no[current_feature].values) == len(data_all_oud_yes[current_feature].values):
            stat_file.write(current_feature)
            stat_file.write(',')
            stat_file.write(str(sum(data_all_oud_yes[current_feature]>0)))
            stat_file.write(',')
            stat_file.write(str(sum(data_all_oud_yes[current_feature]>0)/num_oud_yes_patients))
            stat_file.write(',')
            stat_file.write(str(sum(data_all_oud_no[current_feature]>0)))
            stat_file.write(',')
            stat_file.write(str(sum(data_all_oud_no[current_feature]>0)/num_oud_no_patients))
            stat_file.write(',')            
            stat_file.write('identical vars')
            stat_file.write(',')
            stat_file.write('identical vars')
            stat_file.write(',')
            stat_file.write(str(sum(data_all[current_feature]>0)))
            stat_file.write(',')
            stat_file.write(str(sum(data_all[current_feature]>0)/(num_oud_no_patients+num_oud_yes_patients)))
            stat_file.write('\n')
        else:        
            temp_u_statistic, temp_p_value = mannwhitneyu(data_all_oud_yes[current_feature], data_all_oud_no[current_feature])   
            stat_file.write(current_feature)
            stat_file.write(',')
            stat_file.write(str(sum(data_all_oud_yes[current_feature]>0)))
            stat_file.write(',')
            stat_file.write(str(sum(data_all_oud_yes[current_feature]>0)/num_oud_yes_patients))
            stat_file.write(',')
            stat_file.write(str(sum(data_all_oud_no[current_feature]>0)))
            stat_file.write(',')
            stat_file.write(str(sum(data_all_oud_no[current_feature]>0)/num_oud_no_patients))
            stat_file.write(',')               
            stat_file.write(str(temp_p_value))
            stat_file.write(',')
            stat_file.write(str(temp_u_statistic))
            stat_file.write(',')
            stat_file.write(str(sum(data_all[current_feature]>0)))
            stat_file.write(',')
            stat_file.write(str(sum(data_all[current_feature]>0)/(num_oud_no_patients+num_oud_yes_patients)))

            stat_file.write('\n')


# def compute_stats(train_stationary_filename 
#                                 , validation_stationary_filename     
#                                 , test_stationary_filename 
#                                 , train_stationary_filename_raw 
#                                 , validation_stationary_filename_raw     
#                                 , test_stationary_filename_raw     
#                                 , train_demographics_filename
#                                 , validation_demographics_filename
#                                 , test_demographics_filename                                 
#                              ):
#     # Reading demographics
#     train_demogs = pd.read_csv(train_demographics_filename)
#     validation_demogs = pd.read_csv(validation_demographics_filename)
#     test_demogs = pd.read_csv(test_demographics_filename)
#     demogs_all = pd.concat([train_demogs, validation_demogs, test_demogs])
#     demogs_all_pos = demogs_all[demogs_all[' Label']==1]
#     demogs_all_neg = demogs_all[demogs_all[' Label']==0]
#     month_on_op_ustat, month_on_op_pvalue = mannwhitneyu(demogs_all_pos['NUM_MONTHLY_OPIOID_PRESCS'], demogs_all_neg['NUM_MONTHLY_OPIOID_PRESCS'])
#     month_in_data_ustat, month_in_data_pvalue = mannwhitneyu(demogs_all_pos['NUM_MONTHS_IN_DATA'], demogs_all_neg['NUM_MONTHS_IN_DATA'])

#     # Reading the data before normalization withactual age values
#     train_data_not_norm=pd.read_csv(train_stationary_filename_raw, index_col='ENROLID')
#     validation_data_not_norm=pd.read_csv(validation_stationary_filename_raw, index_col='ENROLID')
#     test_data_not_norm=pd.read_csv(test_stationary_filename_raw, index_col='ENROLID')    
#     data_all_not_norm = pd.concat([train_data_not_norm, test_data_not_norm,validation_data_not_norm])
#     data_all_pos_not_norm = data_all_not_norm[data_all_not_norm['Label']==1] 
#     data_all_neg_not_norm = data_all_not_norm[data_all_not_norm['Label']==0] 

#     # Reading the data after normalization
#     train_data=pd.read_csv(train_stationary_filename, index_col='ENROLID')
#     validation_data=pd.read_csv(validation_stationary_filename, index_col='ENROLID')
#     test_data=pd.read_csv(test_stationary_filename, index_col='ENROLID')    
#     data_all = pd.concat([train_data, test_data,validation_data])

#     data_all_pos = data_all[data_all['Label']==1] 
#     data_all_neg = data_all[data_all['Label']==0] 

#     age_u_statistic, age_p_value = mannwhitneyu(data_all_pos['Age'], data_all_neg['Age'])
    
#     # t-test
#     #ttest_ind(data_all_pos['Age'], data_all_neg['Age'])
    
#     num_sex2_pos = len(data_all_pos[data_all_pos['Sex']==1])
#     num_sex1_pos = len(data_all_pos[data_all_pos['Sex']==0])

#     num_sex2_neg = len(data_all_neg[data_all_neg['Sex']==1])
#     num_sex1_neg = len(data_all_neg[data_all_neg['Sex']==0])
#     # pdb.set_trace() 
#     with open('results/visualization_results/stat_file.csv','w') as stat_file:        
        
#         # Average and std of age in case and cohort
#         stat_file.write('Average age in oud_yes is:\n')
#         stat_file.write(str(data_all_pos_not_norm['Age'].mean()))
#         stat_file.write('\n')
#         stat_file.write('Standart deviation of age in oud_yes is:\n')
#         stat_file.write(str(data_all_pos_not_norm['Age'].std()))        
#         stat_file.write('\n')

#         stat_file.write('Average age in oud_no is:\n')
#         stat_file.write(str(data_all_neg_not_norm['Age'].mean()))
#         stat_file.write('\n')
#         stat_file.write('Standart deviation of age in oud_no is:\n')
#         stat_file.write(str(data_all_neg_not_norm['Age'].std()))                
#         stat_file.write('\n')

#         # Average and std of the number of month patinets hasve been on opioid medications in case and cohort       
#         stat_file.write('Average number of month on opioid in oud_yes is:\n')
#         stat_file.write(str(demogs_all_pos['NUM_MONTHLY_OPIOID_PRESCS'].mean()))
#         stat_file.write('\n')
#         stat_file.write('Standart deviation of month on opioid in oud_yes is:\n')
#         stat_file.write(str(demogs_all_pos['NUM_MONTHLY_OPIOID_PRESCS'].std()))        
#         stat_file.write('\n')

#         stat_file.write('Average number of month on opioid in oud_no is:\n')
#         stat_file.write(str(demogs_all_neg['NUM_MONTHLY_OPIOID_PRESCS'].mean()))
#         stat_file.write('\n')
#         stat_file.write('Standart deviation of month on opioid in oud_no is:\n')
#         stat_file.write(str(demogs_all_neg['NUM_MONTHLY_OPIOID_PRESCS'].std()))        
#         stat_file.write('\n')

#         # Average and std of data availibility in case and cohort
#         stat_file.write('Average data availibility(month) in oud_yes is:\n')
#         stat_file.write(str(demogs_all_pos['NUM_MONTHS_IN_DATA'].mean()))
#         stat_file.write('\n')
#         stat_file.write('Standart deviation of data availibility(month) in oud_yes is:\n')
#         stat_file.write(str(demogs_all_pos['NUM_MONTHS_IN_DATA'].std()))        
#         stat_file.write('\n')

#         stat_file.write('Average data availibility(month) in oud_no is:\n')
#         stat_file.write(str(demogs_all_neg['NUM_MONTHS_IN_DATA'].mean()))
#         stat_file.write('\n')
#         stat_file.write('Standart deviation of data availibility(month) in oud_no is:\n')
#         stat_file.write(str(demogs_all_neg['NUM_MONTHS_IN_DATA'].std()))        
#         stat_file.write('\n')

#         # Sex 
#         stat_file.write('Number of people with SEX=1 in oud_yes cohort is:\n')
#         stat_file.write(str(num_sex1_pos))
#         stat_file.write('\n')
#         stat_file.write('Number of people with SEX=1 in oud_no cohort is:\n')
#         stat_file.write(str(num_sex1_neg))
#         stat_file.write('\n')

#         stat_file.write('Number of people with SEX=2 in oud_yes cohort is:\n')
#         stat_file.write(str(num_sex2_pos))
#         stat_file.write('\n')       
#         stat_file.write('Number of people with SEX=2 in oud_no cohort is:\n')
#         stat_file.write(str(num_sex2_neg))
#         stat_file.write('\n')

  
#     # pdb.set_trace()     
#     with open('results/visualization_results/p_values_u_stats.csv','w') as stat_file:  
#         stat_file.write('Feature name, Num patients in oud_yes, number of patinets in our_no, P-value, u-statistic\n')
        
#         # Month on opioid        
#         stat_file.write('Months on opioid')
#         stat_file.write(',')
#         stat_file.write('-')
#         stat_file.write(',')
#         stat_file.write('_')
#         stat_file.write(',')
#         stat_file.write(str(month_on_op_pvalue))
#         stat_file.write(',')
#         stat_file.write(str(month_on_op_ustat))
#         stat_file.write('\n')

#         stat_file.write('Data availibility(months)')
#         stat_file.write(',')
#         stat_file.write('-')
#         stat_file.write(',')
#         stat_file.write('_')
#         stat_file.write(',')
#         stat_file.write(str(month_in_data_pvalue))
#         stat_file.write(',')
#         stat_file.write(str(month_in_data_ustat))
#         stat_file.write('\n')

#         # All other medications, diagnoses and procedures features
#         for i in range(len(data_all.columns)-1):
#             current_feature = data_all.columns[i]  
#             # print(current_feature)
#             if sum(data_all_pos[current_feature].values == data_all_neg[current_feature].values) == len(data_all_pos[current_feature].values):
#                 stat_file.write(current_feature)
#                 stat_file.write(',')
#                 stat_file.write(str(sum(data_all_pos[current_feature]>0)))
#                 stat_file.write(',')
#                 stat_file.write(str(sum(data_all_neg[current_feature]>0)))
#                 stat_file.write(',')
#                 stat_file.write('identical vars')
#                 stat_file.write(',')
#                 stat_file.write('identical vars')
#                 stat_file.write('\n')
#             else:        
#                 temp_u_statistic, temp_p_value = mannwhitneyu(data_all_pos[current_feature], data_all_neg[current_feature])   
#                 stat_file.write(current_feature)
#                 stat_file.write(',')
#                 stat_file.write(str(sum(data_all_pos[current_feature]>0)))
#                 stat_file.write(',')
#                 stat_file.write(str(sum(data_all_neg[current_feature]>0)))
#                 stat_file.write(',')                
#                 stat_file.write(str(temp_p_value))
#                 stat_file.write(',')
#                 stat_file.write(str(temp_u_statistic))
#                 stat_file.write('\n')

