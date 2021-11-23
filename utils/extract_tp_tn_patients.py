import os
import pdb
import pandas as pd
import numpy as np
from sklearn import metrics

class ReadingData(object):
    def __init__(self, path_t="", path_l=""):
        """
        Reading the file in path_t. Path_l doesn't do anything and will be removed from next versions.
        """
        self.data = []
        self.labels = []
        self.seqlen = []
        s=[]
        temp=[]
        with open(path_t) as f:
              # temp_counter = 0
              for line in f:                   
                  # temp_counter += 1
                  # if temp_counter >100:
                  #   break
                  d_temp=line.split(',')
                  d_temp=[float(x) for x in d_temp]
                  self.data.append(d_temp)
                  d_temp=[]
                  s=[]
        self.batch_id = 0
    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        if len(batch_data) < batch_size:
            batch_data = batch_data + (self.data[0:(batch_size - len(batch_data))])       
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data

F1_idx = 10
one=1
zero=0
num_time_steps = 138
num_classes=2
one=1
two=2
d_demogs=2

#================= Read test performances and find the best model =================
validation_res=[]
with open("results/MUPOD/validation_results_twostreamT_thresholding.csv") as validation_results:
  next(validation_results)
  for line in validation_results:
      line_perf=line.replace(',\n','').split(',')
      line_perf = [float(i) for i in line_perf]  
      validation_res.append(line_perf)
# pdb.set_trace()
validation_res_ar=np.array(validation_res)
best_validation_res_ar=validation_res_ar[np.argmax(validation_res_ar[:,F1_idx]),:]
best_model_number= int(best_validation_res_ar[0])
learning_rate= best_validation_res_ar[1]
num_layers=int(best_validation_res_ar[2])
num_heads = int(best_validation_res_ar[3])
BATCH_SIZE=int(best_validation_res_ar[4])
dropout_rate=best_validation_res_ar[5]
EPOCHS=int(best_validation_res_ar[6])
optimum_threshold=best_validation_res_ar[17]
regu_factor = best_validation_res_ar[18]
#===================================================


#====================================================
test_filename_meds='outputs/test_meds_represented.csv'
test_filename_diags='outputs/test_diags_represented.csv'
test_filename_procs='outputs/test_procs_represented.csv'
test_filename_demogs='outputs/test_demographics_multihot.csv'

testset_meds = ReadingData(path_t=test_filename_meds)
# testset_diags = ReadingData(path_t=test_filename_diags)
# testset_procs = ReadingData(path_t=test_filename_procs)
test_demogs = ReadingData(path_t=test_filename_demogs)
#===== test meds
test_data = testset_meds.data
test_data_ar = np.array(test_data)
# test_meds_enrolids= test_data_ar[:,0]
# pdb.set_trace()
# test_data_ar_reshaped = np.reshape(test_data_ar[:,one:-5],(len(test_data_ar), num_time_steps, d_meds))   
testset_labels = test_data_ar[:,-5:-3]
# medications_test = tf.convert_to_tensor(test_data_ar_reshaped, np.float32)
# test_meds_enrolids = test_data_ar[:,0]

# test_split_k_all = find_divisables(len(test_data_ar))
# test_split_k = int(len(test_data_ar)/test_split_k_all[1])

# #==== test diags
# test_data = testset_diags.data 
# test_data_ar=np.array(test_data)
# test_diags_enrolids= test_data_ar[:,0]
# test_data_ar_reshaped=np.reshape(test_data_ar[:,one:-5],(len(test_data_ar), num_time_steps, d_diags ))   
# diagnoses_test = tf.convert_to_tensor(test_data_ar_reshaped, np.float32)
# test_diags_enrolids = test_data_ar[:,0]

# #==== test procs
# test_data = testset_procs.data 
# test_data_ar=np.array(test_data)
# test_procs_enrolids= test_data_ar[:,0]
# test_data_ar_reshaped=np.reshape(test_data_ar[:,one:-5],(len(test_data_ar), num_time_steps, d_procs ))   
# procedures_test = tf.convert_to_tensor(test_data_ar_reshaped, np.float32)
# test_procs_enrolids = test_data_ar[:,0]

#==== test demogs
test_data = test_demogs.data
test_data_ar=np.array(test_data)
test_demogs_enrolids= test_data_ar[:,0]
batch_x_ar_reshaped=np.reshape(test_data_ar[:,one:],(len(test_data_ar), num_time_steps, d_demogs+1))   
# test_demog_info=tf.convert_to_tensor(batch_x_ar_reshaped[:,:,one:], np.float32)

# pdb.set_trace()
# if (sum( test_meds_enrolids != test_diags_enrolids ) != 0) or (sum( test_meds_enrolids != test_procs_enrolids ) != 0) or  (sum(test_diags_enrolids != test_demogs_enrolids) != 0):
#   print("Error: enrolids don't match")
#   pdb.set_trace()
#====================================================
['TP', 'TN', 'FP', 'FN']
test_softmax = pd.read_csv('results/MUPOD/restoring_best_model__softmax_test_123456.csv', header=None)
probabilities_test_pos=test_softmax.iloc[:,0]
test_auc = metrics.roc_auc_score(testset_labels[:,0], probabilities_test_pos)
# test_auc = metrics.auc(fpr, tpr)
final_predictions = []
tp_test=0
tn_test=0
fp_test=0
fn_test=0
for i in range(len(probabilities_test_pos)):  
    current_enrolid = test_data_ar[i,0]
    if(probabilities_test_pos[i]<optimum_threshold and testset_labels[i,0]==0):
        tn_test=tn_test+1
        final_predictions.append([current_enrolid, 0, 1, 0, 0])
    elif(probabilities_test_pos[i]>=optimum_threshold and testset_labels[i,0]==1):
        tp_test=tp_test+1
        final_predictions.append([current_enrolid, 1, 0, 0, 0])
    elif(probabilities_test_pos[i]>=optimum_threshold and testset_labels[i,0]==0):
        fp_test=fp_test+1
        final_predictions.append([current_enrolid, 0, 0, 1, 0])
    elif(probabilities_test_pos[i]<optimum_threshold and testset_labels[i,0]==1):
        fn_test=fn_test+1
        final_predictions.append([current_enrolid, 0, 0, 0, 1])
if((tp_test+fp_test)==0):
    precision_test=0
else:
    precision_test=tp_test/(tp_test+fp_test)
recall_test=tp_test/(tp_test+fn_test)
sensitivity_test=tp_test/(tp_test+fn_test)
specificity_test=tn_test/(tn_test+fp_test)    
if (precision_test+recall_test) !=0:
    F1Score_test=(2*precision_test*recall_test)/(precision_test+recall_test)      
else:
    F1Score_test=0        
accuracy_test= (tp_test+tn_test)/(tp_test+tn_test+fp_test+fn_test)
np.savetxt('results/MUPOD/final_predictions_mupod.csv', final_predictions, header='ENROLID, TP, TN, FP, FN', delimiter=',')
# pdb.set_trace()
print('End')