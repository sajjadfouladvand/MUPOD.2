import  models.LSTM_model_singlestream_test as lsad
import os
import random as rnd
import pdb
import numpy as np
import random
import time

model_number = 1000
dropout_r = 0.4
representing = False
num_epochs = 10
regu_fact = 0.000001
learning_r = 0.01
n_hid = 48
batch_sz = 256
trained_model_path = 'saved_models/lstm_single_stream/'

test_meds_filename = 'outputs/test_medications_multihot.csv'
test_diags_filename = 'outputs/test_diagnoses_multihot.csv'
test_procs_filename = 'outputs/test_procedures_multihot.csv'
test_demogs_filename = 'outputs/test_demographics_multihot.csv'
test_metadata_filename = 'outputs/test_demographics_shuffled.csv'

print('Testing the model using the trained model ... ')

softmax_predictions_temp, accuracy_temp, precision, recall, sensitivity, specificity, tp, tn, fp, fn, test_auc =lsad.main(model_number, dropout_r, representing, num_epochs, regu_fact, learning_r, n_hid , batch_sz, trained_model_path, test_meds_filename, test_diags_filename, test_procs_filename, test_demogs_filename, test_metadata_filename)
np.savetxt('results/LSTM_single_stream/softmax_predictions_lstm_single_stream_using_saved_model.csv', softmax_predictions_temp, delimiter=',')
header_results_filename= "Learning Rate, Number of hidden neurons, Batch Size, accuracy, Precision, Recall, F1_score, specificity, TP, TN, FP,FN, Num Iterations, AUC, Regularization coefficient\n"
with open('results/LSTM_single_stream/Results_test_lstm_single_stream_using_saved_model.csv', 'w') as res_f:
        res_f.write("".join(["".join(x) for x in header_results_filename]))      
        res_f.write(str(learning_r))
        res_f.write(", ")
        res_f.write(str(n_hid))
        res_f.write(", ")
        res_f.write(str(batch_sz))
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
        res_f.write(str(num_epochs))
        res_f.write(",")  
        res_f.write(str(test_auc))
        res_f.write(",")  
        res_f.write(str(regu_fact))
        res_f.write("\n")                        
                     
