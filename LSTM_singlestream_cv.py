import  models.LSTM_model_singlestream as lsad
import os
import random as rnd
import pdb
import numpy as np
import random
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

def concat_csv_files(train_data_filename, validation_data_filename, stream_name):
    # pdb.set_trace()
    # remove the file if exist
    if os.path.exists('outputs/train_and_validation_'+stream_name+'_multihot.csv'): 
        os.remove('outputs/train_and_validation_'+stream_name+'_multihot.csv')
    
    # Write train data
    with open(train_data_filename) as train_data_file, open('outputs/train_and_validation_'+stream_name+'_multihot.csv', 'a') as train_val_file:
        for line in train_data_file:
            train_val_file.write(line)
    # pdb.set_trace()
    # Write validation data
    with open(validation_data_filename) as validation_data_file, open('outputs/train_and_validation_'+stream_name+'_multihot.csv', 'a') as train_val_file:
        for line in validation_data_file:
            if line.split(',')[0] == 'ENROLID':
                continue
            train_val_file.write(line)
        
model_number = 1000
dropout_r = 0.4
representing = True
num_epochs = 10
regu_fact = 0.000001
learning_r = 0.01
n_hid = 48
batch_sz = 256

# pdb.set_trace()
start_time = time.time()
train_meds_filename = 'outputs/train_medications_multihot.csv'
train_diags_filename = 'outputs/train_diagnoses_multihot.csv'
train_procs_filename = 'outputs/train_procedures_multihot.csv'
train_demogs_filename = 'outputs/train_demographics_multihot.csv'
train_metadata_filename = 'outputs/train_demographics_shuffled.csv'


validation_meds_filename = 'outputs/validation_medications_multihot.csv'
validation_diags_filename = 'outputs/validation_diagnoses_multihot.csv'
validation_procs_filename = 'outputs/validation_procedures_multihot.csv'
validation_demogs_filename = 'outputs/validation_demographics_multihot.csv'
validation_metadata_filename = 'outputs/validation_demographics_shuffled.csv'

print('Concatenating train and validation files ...')

concat_csv_files(train_meds_filename, validation_meds_filename, 'medications')
concat_csv_files(train_diags_filename, validation_diags_filename, 'diagnoses')
concat_csv_files(train_procs_filename, validation_procs_filename, 'procedures')
concat_csv_files(train_demogs_filename, validation_demogs_filename, 'demographics')
concat_csv_files(train_metadata_filename, validation_metadata_filename, 'metadata')


# pdb.set_trace()
# data_all_train = pd.read_csv('outputs/train_procedures_multihot.csv', skiprows=[0])

train_meds_filename = 'outputs/train_and_validation_medications_multihot.csv'
train_diags_filename = 'outputs/train_and_validation_diagnoses_multihot.csv'
train_procs_filename = 'outputs/train_and_validation_procedures_multihot.csv'
train_demogs_filename = 'outputs/train_and_validation_demographics_multihot.csv'
train_metadata_filename = 'outputs/train_and_validation_metadata_multihot.csv'


test_meds_filename = 'outputs/test_medications_multihot.csv'
test_diags_filename = 'outputs/test_diagnoses_multihot.csv'
test_procs_filename = 'outputs/test_procedures_multihot.csv'
test_demogs_filename = 'outputs/test_demographics_multihot.csv'
test_metadata_filename = 'outputs/test_demographics_shuffled.csv'

print('========================')
print('Train file names are:')
print(train_meds_filename)
print(train_diags_filename)
print(train_procs_filename)
print(train_demogs_filename)
print(train_metadata_filename)
print('========================')
# pdb.set_trace()
# softmax_predictions_temp, accuracy_temp, precision, recall, sensitivity, specificity, tp, tn, fp, fn, test_auc =lsad.main(1000, validation_results[max_f1_index,15], False, validation_results[max_f1_index,12], validation_results[max_f1_index,14], validation_results[max_f1_index,0],int(validation_results[max_f1_index,1]), int(validation_results[max_f1_index,2]), train_meds_filename, train_diags_filename, train_procs_filename, train_demogs_filename, train_metadata_filename, validation_meds_filename, validation_diags_filename, validation_procs_filename, validation_demogs_filename, validation_metadata_filename)
softmax_predictions_temp, accuracy_temp, precision, recall, sensitivity, specificity, tp, tn, fp, fn, test_auc =lsad.main(model_number, dropout_r, representing, num_epochs, regu_fact, learning_r, n_hid , batch_sz, train_meds_filename, train_diags_filename, train_procs_filename, train_demogs_filename, train_metadata_filename, validation_meds_filename, validation_diags_filename, validation_procs_filename, validation_demogs_filename, validation_metadata_filename)
np.savetxt('results/LSTM_single_stream/softmax_predictions_lstm_single_stream.csv', softmax_predictions_temp, delimiter=',')
header_results_filename= "Learning Rate, Number of hidden neurons, Batch Size, accuracy, Precision, Recall, F1_score, specificity, TP, TN, FP,FN, Num Iterations, AUC, Regularization coefficient\n"
with open('results/LSTM_single_stream/Results_test_lstm_single_stream.csv', 'w') as res_f:
        res_f.write("".join(["".join(x) for x in header_results_filename]))      
        res_f.write(str(learning_r))#str(validation_results[max_f1_index,0]))
        res_f.write(", ")
        res_f.write(str(n_hid))#str(validation_results[max_f1_index,1]))
        res_f.write(", ")
        res_f.write(str(batch_sz))#str(validation_results[max_f1_index,2]))
        res_f.write(", ")
        res_f.write(str(accuracy_temp))
        res_f.write(", ")
        res_f.write(str(precision))
        res_f.write(", ")
        res_f.write(str(recall))
        res_f.write(", ")
        if (precision + recall) == 0:
            F1=0
        else:    
            F1=(2*precision*recall)/(precision+recall)
        res_f.write(str(F1))
        res_f.write(", ")
        res_f.write(str(specificity))
        res_f.write(", ")
        res_f.write(str(tp))
        res_f.write(", ")
        res_f.write(str(tn))
        res_f.write(", ")
        res_f.write(str(fp))
        res_f.write(", ")
        res_f.write(str(fn))
        res_f.write(",")
        res_f.write(str(num_epochs))#str(validation_results[max_f1_index, 12]))
        res_f.write(",")  
        res_f.write(str(test_auc))
        res_f.write(",")  
        res_f.write(str(regu_fact))#str(validation_results[max_f1_index,14]))
        res_f.write("\n")                        
                     
