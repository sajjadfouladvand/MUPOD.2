import operator
import os
import pandas as pandas
import pdb
import numpy as np

pdb.set_trace()
icd10_ind=0
icd9_ind=1
# ICD codes in ORT
# I removed 'F1011' because it's Alcohol abuse, in remission
alcohol_icd10 = ['F1010', 'F10120', 'F10121', 'F10129', 'F10130', 'F10131', 'F10132', 
'F10139', 'F1014', 'F10150', 'F10151', 'F10159', 'F10180', 'F10181', 'F10182', 'F10188', 'F1019', 'F1020', 'F1021', 'F10220', 'F10221',
'F10229', 'F10230', 'F10231', 'F10232', 'F10239', 'F1024', 'F10250', 'F10251', 'F10259', 'F1026', 'F1027', 'F10280', 'F10281', 'F10282', 'F10288', 'F1029', 'F10920', 'F10921', 'F10929', 'F10930', 'F10931', 
'F10932', 'F10939', 'F1094', 'F10950', 'F10951', 'F10959', 'F1096', 'F1097', 'F10980', 'F10981', 'F10982','F10988','F1099']


# rx_drugs_icds = []
# add_icds = []
# ocd_icds = 
# bipolar_icds = 
# schizophrenia_icds = 
# depression_icds = 


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
pdb.set_trace()
for i in range(len(unique_ICD10s)):
    ICD10_to_ICD9_dict[unique_ICD10s[i]]=[0]
pdb.set_trace()
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
pdb.set_trace()
print('Test')
# Read test patients medications, diagnoses, procedures and demographics


