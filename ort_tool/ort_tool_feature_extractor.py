import operator
import os
import pandas as pd
import pdb
import numpy as np
import itertools
import math 
from sklearn import metrics
import matplotlib.pyplot as plt

# pdb.set_trace()
icd10_ind=0
icd9_ind=1




data_all = pd.ExcelFile('ort_tool/icd_codes_for_ort.xlsx')

fam_hist_alcohol = pd.read_excel(data_all, 'fam_hist_alcohol')
fam_hist_alcohol = fam_hist_alcohol.apply(lambda s:s.str.replace("'", ""))
fam_hist_alcohol.columns = fam_hist_alcohol.columns.str.replace("'",'')

fam_hist_illeg_drug = pd.read_excel(data_all, 'fam_hist_illeg_drug')
fam_hist_illeg_drug.columns = fam_hist_illeg_drug.columns.str.replace("'",'')
fam_hist_illeg_drug = fam_hist_illeg_drug.apply(lambda s:s.str.replace("'", ""))

fam_hist_rx_drug = pd.read_excel(data_all, 'fam_hist_rx_drug')
fam_hist_rx_drug.columns = fam_hist_rx_drug.columns.str.replace("'",'')
fam_hist_rx_drug = fam_hist_rx_drug.apply(lambda s:s.str.replace("'", ""))

pers_hist_alcohol = pd.read_excel(data_all, 'pers_hist_alcohol')
pers_hist_alcohol.columns = pers_hist_alcohol.columns.str.replace("'",'')
pers_hist_alcohol = pers_hist_alcohol.apply(lambda s:s.str.replace("'", ""))

pers_hist_illeg_drug = pd.read_excel(data_all, 'pers_hist_illeg_drug')
pers_hist_illeg_drug.columns = pers_hist_illeg_drug.columns.str.replace("'",'')
pers_hist_illeg_drug = pers_hist_illeg_drug.apply(lambda s:s.str.replace("'", ""))

pers_hist_rx_drug = pd.read_excel(data_all, 'pers_hist_rx_drug')
pers_hist_rx_drug.columns = pers_hist_rx_drug.columns.str.replace("'",'')
pers_hist_rx_drug = pers_hist_rx_drug.apply(lambda s:s.str.replace("'", ""))

pers_hist_add = pd.read_excel(data_all, 'pers_hist_add')
pers_hist_add.columns = pers_hist_add.columns.str.replace("'",'')
pers_hist_add = pers_hist_add.apply(lambda s:s.str.replace("'", ""))

pers_hist_schi = pd.read_excel(data_all, 'pers_hist_schi')
pers_hist_schi.columns = pers_hist_schi.columns.str.replace("'",'')
pers_hist_schi = pers_hist_schi.apply(lambda s:s.str.replace("'", ""))

pers_hist_ocd = pd.read_excel(data_all, 'pers_hist_ocd')
pers_hist_ocd.columns = pers_hist_ocd.columns.str.replace("'",'')
pers_hist_ocd = pers_hist_ocd.apply(lambda s:s.str.replace("'", ""))

pers_hist_bipolar = pd.read_excel(data_all, 'pers_hist_bipolar')
pers_hist_bipolar.columns = pers_hist_bipolar.columns.str.replace("'",'')
pers_hist_bipolar = pers_hist_bipolar.apply(lambda s:s.str.replace("'", ""))

pers_hist_depp = pd.read_excel(data_all, 'pers_hist_depp')
pers_hist_depp.columns = pers_hist_depp.columns.str.replace("'",'')
pers_hist_depp = pers_hist_depp.apply(lambda s:s.str.replace("'", ""))


# Create the ICD10 to ICD9 convertor
ICD10_to_ICD9_list=[]
ICD10_to_ICD9_dict={}
temp_icds=[]
#pdb.set_trace()
with open('ort_tool/ICD10_to_ICD9_all.csv') as ICD_file:
    next(ICD_file)
    for line in ICD_file:
        line_ICDs=line.split(',')
        #line_ICDs = [int(i) for i in line_ICDs]
        ICD10_to_ICD9_list.append(line_ICDs)
ICD10_to_ICD9_ar= np.array(ICD10_to_ICD9_list)
unique_ICD10s= np.unique(ICD10_to_ICD9_ar[:,0])
# pdb.set_trace()
for i in range(len(unique_ICD10s)):
    ICD10_to_ICD9_dict[unique_ICD10s[i]]=[0]
# pdb.set_trace()
i=0
ICD10_to_ICD9_list_sorted=sorted(ICD10_to_ICD9_list, key=operator.itemgetter(0))
while i<len(ICD10_to_ICD9_list_sorted):
    current_icd= ICD10_to_ICD9_list_sorted[i][icd10_ind]
    #if i==737:
        #pdb.set_trace()
    while current_icd == ICD10_to_ICD9_list_sorted[i][icd10_ind]:
          temp_icds.append(ICD10_to_ICD9_list_sorted[i][icd9_ind])
          i=i+1
          if i>= len(ICD10_to_ICD9_list_sorted):
            #pdb.set_trace()   
            break
    # pdb.set_trace()
    ICD10_to_ICD9_dict[current_icd]=temp_icds   
    temp_icds = []  

# pdb.set_trace()
# Extract ICD9 codes
fam_hist_alcohol_icd9 = []
for i in range(len(fam_hist_alcohol['ICD-10-CM CODE'])):
    current_icd = fam_hist_alcohol['ICD-10-CM CODE'][i]
    if  current_icd in ICD10_to_ICD9_dict.keys():
       fam_hist_alcohol_icd9.extend(ICD10_to_ICD9_dict[current_icd]) 

# pdb.set_trace()
fam_hist_illeg_drug_icd9 = []
for i in range(len(fam_hist_illeg_drug['ICD-10-CM CODE'])):
    current_icd = fam_hist_illeg_drug['ICD-10-CM CODE'][i]
    if  current_icd in ICD10_to_ICD9_dict.keys():
       fam_hist_illeg_drug_icd9.extend(ICD10_to_ICD9_dict[current_icd]) 

# pdb.set_trace()
fam_hist_rx_drug_icd9 = []
for i in range(len(fam_hist_rx_drug['ICD-10-CM CODE'])):
    current_icd = fam_hist_rx_drug['ICD-10-CM CODE'][i]
    if  current_icd in ICD10_to_ICD9_dict.keys():
       fam_hist_rx_drug_icd9.extend(ICD10_to_ICD9_dict[current_icd]) 
# pdb.set_trace()
pers_hist_alcohol_icd9 = []
for i in range(len(pers_hist_alcohol['ICD-10-CM CODE'])):
    current_icd = pers_hist_alcohol['ICD-10-CM CODE'][i]
    if  current_icd in ICD10_to_ICD9_dict.keys():
       pers_hist_alcohol_icd9.extend(ICD10_to_ICD9_dict[current_icd]) 
# pdb.set_trace()
pers_hist_illeg_drug_icd9 = []
for i in range(len(pers_hist_illeg_drug['ICD-10-CM CODE'])):
    current_icd = pers_hist_illeg_drug['ICD-10-CM CODE'][i]
    if  current_icd in ICD10_to_ICD9_dict.keys():
       pers_hist_illeg_drug_icd9.extend(ICD10_to_ICD9_dict[current_icd]) 

# pdb.set_trace()
pers_hist_rx_drug_icd9 = []
for i in range(len(pers_hist_rx_drug['ICD-10-CM CODE'])):
    current_icd = pers_hist_rx_drug['ICD-10-CM CODE'][i]
    if  current_icd in ICD10_to_ICD9_dict.keys():
       pers_hist_rx_drug_icd9.extend(ICD10_to_ICD9_dict[current_icd]) 

# pdb.set_trace()
pers_hist_add_icd9 = []
for i in range(len(pers_hist_add['ICD-10-CM CODE'])):
    current_icd = pers_hist_add['ICD-10-CM CODE'][i]
    if  current_icd in ICD10_to_ICD9_dict.keys():
       pers_hist_add_icd9.extend(ICD10_to_ICD9_dict[current_icd]) 
# pdb.set_trace()

pers_hist_schi_icd9 = []
for i in range(len(pers_hist_schi['ICD-10-CM CODE'])):
    current_icd = pers_hist_schi['ICD-10-CM CODE'][i]
    if  current_icd in ICD10_to_ICD9_dict.keys():
       pers_hist_schi_icd9.extend(ICD10_to_ICD9_dict[current_icd]) 

# pdb.set_trace()
pers_hist_ocd_icd9 = []
for i in range(len(pers_hist_ocd['ICD-10-CM CODE'])):
    current_icd = pers_hist_ocd['ICD-10-CM CODE'][i]
    if  current_icd in ICD10_to_ICD9_dict.keys():
       pers_hist_ocd_icd9.extend(ICD10_to_ICD9_dict[current_icd]) 

# pdb.set_trace()
pers_hist_bipolar_icd9 = []
for i in range(len(pers_hist_bipolar['ICD-10-CM CODE'])):
    current_icd = pers_hist_bipolar['ICD-10-CM CODE'][i]
    if  current_icd in ICD10_to_ICD9_dict.keys():
       pers_hist_bipolar_icd9.extend(ICD10_to_ICD9_dict[current_icd]) 

# pdb.set_trace()
pers_hist_depp_icd9 = []
for i in range(len(pers_hist_depp['ICD-10-CM CODE'])):
    current_icd = pers_hist_depp['ICD-10-CM CODE'][i]
    if  current_icd in ICD10_to_ICD9_dict.keys():
       pers_hist_depp_icd9.extend(ICD10_to_ICD9_dict[current_icd]) 
# pdb.set_trace()
test_set_ort_scores = pd.DataFrame(columns=['ENROLID', 'fam_hist_alcohol', 'fam_hist_illeg_drug', 'fam_hist_rx_drug', 'pers_hist_alcohol', 'pers_hist_illeg_drug', 'pers_hist_rx_drug', 'pers_hist_add', 'pers_hist_schi', 'pers_hist_ocd', 'pers_hist_bipolar', 'depression','age_between_16_45', 'ort_score', 'ort_score_sigmoid','grand_truth_label', 'predicted_label'])
# Read test patients medications, diagnoses, procedures and demographics

test_metadata = pd.read_csv('outputs/test_demographics.csv')
with open('outputs/test_medications_shuffled.csv') as test_meds_file, open('outputs/test_diagnoses_shuffled.csv') as test_diags_file, open('outputs/test_procedures_shuffled.csv') as test_procs_file:
    line_counter=-1
    for line_diag in test_diags_file:
        line_counter+=1
        # if line_counter == 1000:
        #     break
        if line_counter % 1000 ==0:
            print(line_counter)
        # print(line_counter)
        line_diag = line_diag.split(',')
        line_diag_splitted = [list(y) for x, y in itertools.groupby(line_diag[1:], lambda z: z == 'EOV') if not x]            
        if len(line_diag_splitted) == 0:
            line_counter-=1
            continue
        current_enrolid = line_diag[0]
        test_set_ort_scores.loc[len(test_set_ort_scores)] = 0

        test_set_ort_scores['ENROLID'][line_counter] = current_enrolid

        current_age = int(line_diag_splitted[-1][0])//100 - test_metadata[test_metadata['ENROLID'] == float(current_enrolid)]['DOB'].values[0]
        if current_age>= 16 and current_age <= 45:
            test_set_ort_scores['age_between_16_45'][line_counter] = 1

        for i in range(len(line_diag_splitted)):
            # print('i is {}'.format(i))
            for j in range(1, len(line_diag_splitted[i])): 
                # print('j is {}'.format(j))
                current_diag = line_diag_splitted[i][j]    
                if (current_diag in fam_hist_alcohol['ICD-10-CM CODE'].values) or (current_diag in fam_hist_alcohol_icd9):
                    test_set_ort_scores['fam_hist_alcohol'][line_counter] = 1
                elif (current_diag in fam_hist_illeg_drug['ICD-10-CM CODE'].values) or (current_diag in fam_hist_illeg_drug_icd9): 
                    test_set_ort_scores['fam_hist_illeg_drug'][line_counter] = 1
                elif (current_diag in fam_hist_rx_drug['ICD-10-CM CODE'].values) or (current_diag in fam_hist_rx_drug_icd9):
                    test_set_ort_scores['fam_hist_rx_drug'][line_counter] = 1
                elif (current_diag in pers_hist_alcohol['ICD-10-CM CODE'].values) or (current_diag in pers_hist_alcohol_icd9):
                    test_set_ort_scores['pers_hist_alcohol'][line_counter] = 1
                elif (current_diag in pers_hist_illeg_drug['ICD-10-CM CODE'].values) or (current_diag in pers_hist_illeg_drug_icd9):
                    test_set_ort_scores['pers_hist_illeg_drug'][line_counter] = 1
                elif (current_diag in pers_hist_rx_drug['ICD-10-CM CODE'].values) or (current_diag in pers_hist_rx_drug_icd9):
                    test_set_ort_scores['pers_hist_rx_drug'][line_counter] = 1
                elif (current_diag in pers_hist_add['ICD-10-CM CODE'].values) or (current_diag in pers_hist_add_icd9):                                                    
                    test_set_ort_scores['pers_hist_add'][line_counter] = 1
                elif (current_diag in pers_hist_schi['ICD-10-CM CODE'].values) or (current_diag in pers_hist_schi_icd9):                                                    
                    test_set_ort_scores['pers_hist_schi'][line_counter] = 1
                elif (current_diag in pers_hist_ocd['ICD-10-CM CODE'].values) or (current_diag in pers_hist_ocd_icd9):                                                    
                    test_set_ort_scores['pers_hist_ocd'][line_counter] = 1
                elif (current_diag in pers_hist_bipolar['ICD-10-CM CODE'].values) or (current_diag in pers_hist_bipolar_icd9):                                                    
                    test_set_ort_scores['pers_hist_bipolar'][line_counter] = 1
                elif (current_diag in pers_hist_depp['ICD-10-CM CODE'].values) or (current_diag in pers_hist_depp_icd9):                                                    
                    test_set_ort_scores['depression'][line_counter] = 1
        # if  test_set_ort_scores['ENROLID'][line_counter] == 0:
        #     pdb.set_trace()
        #     print('test')           

# pdb.set_trace()
# Calculating the ORT scores


score_term_1 = test_set_ort_scores[['fam_hist_alcohol', 'fam_hist_illeg_drug', 'fam_hist_rx_drug', 'pers_hist_alcohol', 'pers_hist_illeg_drug', 'pers_hist_rx_drug']].sum(axis=1)

score_term_2 = test_set_ort_scores[['pers_hist_add', 'pers_hist_schi', 'pers_hist_ocd', 'pers_hist_bipolar']].any(axis=1).astype(int)

score_term_3 = test_set_ort_scores['depression']

score_term_4 = test_set_ort_scores['age_between_16_45']

test_set_ort_scores['ort_score'] = score_term_1 + score_term_2 + score_term_3 + score_term_4

test_set_ort_scores.loc[test_set_ort_scores['ort_score'] >= 3, 'predicted_label'] = 1

test_set_ort_scores.to_csv('ort_tool/test_set_ort_scores.csv', index=False)
# pdb.set_trace()
# Calculating the performance
tp=0
tn=0
fp=0
fn=0
for i in range(len(test_set_ort_scores)):
    current_enrolid = float(test_set_ort_scores.iloc[i, 0])
    current_label = test_metadata[test_metadata['ENROLID'] == current_enrolid][' Label'].values[0]

    test_set_ort_scores['grand_truth_label'][i] = current_label

    if(test_set_ort_scores['predicted_label'][i] ==0 and current_label == 0):
        tn=tn+1
    elif(test_set_ort_scores['predicted_label'][i] == 1 and current_label == 1):
        tp=tp+1
    elif(test_set_ort_scores['predicted_label'][i] == 1 and current_label == 0):
        fp=fp+1
    elif(test_set_ort_scores['predicted_label'][i] == 0 and current_label == 1):
        fn=fn+1          
if( (tp+fp)==0):
    precision=0
else:
    precision=tp/(tp+fp)

if (tp+fn) == 0:
    recall = 0
else:    
    recall=tp/(tp+fn)
if (tp+fn)==0:
    sensitivity=0
else:    
    sensitivity=tp/(tp+fn)
if (tn+fp)==0:
    specificity=0
else:
    specificity=tn/(tn+fp)
accuracy = (tp+tn)/(tp+tn+fp+fn)

F1_score = (2*precision*recall)/(precision+recall)

# pdb.set_trace()
              
# Sigmoid
test_set_ort_scores['ort_score_sigmoid'] = test_set_ort_scores['ort_score'].apply(lambda x : 1 / (1 + math.exp(-x)))

# test_set_ort_scores_real_labels = []
# for i in range(len(test_set_ort_scores)):
#     current_enrolid = float(test_set_ort_scores['ENROLID'][i])
#     current_label = test_metadata[test_metadata['ENROLID'] == current_enrolid][' Label'].values[0]    
#     test_set_ort_scores_real_labels.append([current_enrolid, current_label])

# pdb.set_trace()



fpr_ort, tpr_ort, threshold_ort = metrics.roc_curve(y_true = test_set_ort_scores['grand_truth_label'].values.tolist(), y_score = test_set_ort_scores['ort_score_sigmoid'], pos_label=1)
roc_auc_ort = metrics.auc(fpr_ort, tpr_ort)
test_auc_ort=metrics.roc_auc_score(test_set_ort_scores['grand_truth_label'].values.tolist(), test_set_ort_scores['ort_score_sigmoid'])   

with open('ort_tool/ort_tool_test_results.csv','w') as ort_res_file:
    ort_res_file.write('Accuract, Precision, Recall, F1, AUC, TP, TN, FP, FN\n')
    ort_res_file.write(str(accuracy))
    ort_res_file.write(',')
    ort_res_file.write(str(precision))
    ort_res_file.write(',')    
    ort_res_file.write(str(recall))
    ort_res_file.write(',')    
    ort_res_file.write(str(F1_score))
    ort_res_file.write(',')    
    ort_res_file.write(str(test_auc_ort))
    ort_res_file.write(',')  
    ort_res_file.write(str(tp))
    ort_res_file.write(',')  
    ort_res_file.write(str(tn))
    ort_res_file.write(',')  
    ort_res_file.write(str(fp))
    ort_res_file.write(',')  
    ort_res_file.write(str(fn))
    ort_res_file.write('\n')  


softmax_mupod = pd.read_csv('results/MUPOD/restoring_best_model__softmax_test_123456.csv', header=None)
test_mupod_meds = pd.read_csv('outputs/test_meds_represented.csv', header=None)
test_labels_mupod = test_mupod_meds.iloc[:,-5:-3]
fpr_mupod, tpr_mupod, threshold_mupod = metrics.roc_curve(test_labels_mupod.iloc[:,0], softmax_mupod.iloc[:,0])
roc_auc_mupod = metrics.auc(fpr_mupod, tpr_mupod)
# roc_auc_mupod=metrics.roc_auc_score(test_labels_mupod.iloc[:,0], softmax_mupod.iloc[:,0])   



# Single stream transformer
test_data_single_trans = pd.read_csv('outputs/test_meds_diags_procs_demogs_represented.csv', header=None)
test_labels_single_str= test_data_single_trans.iloc[:,-5:-3]
softmax_single_str = pd.read_csv('results/single_stream_transformer/restoring_best_signlestreamT_model_predictions_testing_soft_3000.csv', header=None, delimiter=' ')
fpr_single_str, tpr_single_str, threshold_single_str = metrics.roc_curve(test_labels_single_str.iloc[:,0], softmax_single_str.iloc[:,0])
roc_auc_single_str = metrics.auc(fpr_single_str, tpr_single_str)
# test_auc_single_str=metrics.roc_auc_score(test_labels_single_str.iloc[:,0], softmax_single_str.iloc[:,0])   

# Single stream LSTM
test_data_single_lstm = pd.read_csv('outputs/test_demographics_shuffled.csv')
softmax_single_lstm = pd.read_csv('results/LSTM_single_stream/softmax_predictions_lstm_single_stream_using_saved_model.csv', header=None)
fpr_single_lstm, tpr_single_lstm, threshold_single_lstm = metrics.roc_curve(test_data_single_lstm.iloc[:,-1], softmax_single_lstm.iloc[:,0])
roc_auc_single_lstm = metrics.auc(fpr_single_lstm, tpr_single_lstm)
# test_auc_single_lstm=metrics.roc_auc_score(test_data_single_lstm.iloc[:,-1], softmax_single_lstm.iloc[:,0])   


plt.title('Receiver Operating Characteristic')
plt.plot(fpr_mupod, tpr_mupod, 'green', label = 'MUPOD AUC = %0.2f' % roc_auc_mupod)
plt.plot(fpr_single_str, tpr_single_str, 'magenta', label = 'Transformer AUC = %0.2f' % roc_auc_single_str)
plt.plot(fpr_single_lstm, tpr_single_lstm, 'orange', label = 'LSTM AUC = %0.2f' % roc_auc_single_lstm)
plt.plot(fpr_ort, tpr_ort, 'blue', label = 'ORT AUC = %0.2f' % roc_auc_ort)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('ort_tool/roc_ort.png', dpi=300)
print('test')



