import  models.LSTM_model_singlestream as lsad
import os
import random as rnd
import pdb
import numpy as np
import random
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
main_counter=0
num_random_searches=1

# pdb.set_trace()
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

    
header_results_filename= "Learning Rate, Number of hidden neurons, Batch Size, accuracy, Precision, Recall, F1_score, specificity, TP, TN, FP,FN, Num Iterations, AUC, Regularization coefficient\n"
validation_results=np.zeros((num_random_searches, 15))
with open('results/LSTM_single_stream/Results_validation_single_stream_lstm.csv', 'w') as res_f:  
    res_f.write("".join(["".join(x) for x in header_results_filename]))      
    learning_rate_ar=np.random.uniform(0.000001,0.1,num_random_searches)
    # reg_coeff_ar=np.random.uniform(0.000001,0.1,num_random_searches)
    num_hidden_array=np.random.randint(10,200,num_random_searches)
    batch_size_array=2** np.random.randint(5,9,num_random_searches)
    # iterations_array=np.random.randint(10000,2000000,num_random_searches)
    while (main_counter < num_random_searches):
            #print("WE ARE WITHIN WHILE")
        print("====================================")    
        print("Main counter is:")
        print(main_counter)
        print("====================================")
        learning_rt= 0.01#random.choice([0.01, 0.001, 0.0001])#learning_rate_ar[main_counter]
        n_hid=64#random.choice([8, 16, 32, 64, 128, 256])#  num_hidden_array[main_counter]
        batch_sz=64#random.choice([64, 256, 512])
        training_iters_up=1#1000#random.choice([10000, 50000, 100000,200000])#, 800000, 1000000])
        reg_coeff =0.000001#random.choice([0.000001, 0.00001, 0.0001])#, 0.001, 0.01, 0.1])
        #pdb.set_trace()
        j=0
        while(j<1):
            j=j+1
            # pdb.set_trace()
            print(learning_rt)
            print(n_hid)
            print(batch_sz)
           
            _, accuracy_temp, precision, recall, sensitivity, specificity, tp, tn, fp, fn, validation_auc=lsad.main(main_counter, False, training_iters_up, reg_coeff, learning_rt,n_hid, batch_sz,train_meds_filename, train_diags_filename, train_procs_filename, train_demogs_filename, train_metadata_filename, validation_meds_filename, validation_diags_filename, validation_procs_filename, validation_demogs_filename, validation_metadata_filename)
            res_f.write(str(learning_rt))
            validation_results[main_counter, 0]=learning_rt
            res_f.write(", ")
            res_f.write(str(n_hid))
            validation_results[main_counter, 1]=n_hid
            res_f.write(", ")
            res_f.write(str(batch_sz))
            validation_results[main_counter, 2]=batch_sz
            res_f.write(", ")
            res_f.write(str(accuracy_temp))
            validation_results[main_counter, 3]=accuracy_temp
            res_f.write(", ")
            res_f.write(str(precision))
            validation_results[main_counter, 4]=precision
            res_f.write(", ")
            res_f.write(str(recall))
            validation_results[main_counter, 5]=recall
            res_f.write(", ")
            if(precision+recall ==0):
                F1=0
            else:    
                F1=(2*precision*recall)/(precision+recall)
            res_f.write(str(F1))
            validation_results[main_counter, 6]=F1
            res_f.write(", ")
            res_f.write(str(specificity))
            validation_results[main_counter, 7]=specificity
            res_f.write(", ")
            res_f.write(str(tp))
            validation_results[main_counter, 8]=tp
            res_f.write(", ")
            res_f.write(str(tn))
            validation_results[main_counter, 9]=tn
            res_f.write(", ")
            res_f.write(str(fp))
            validation_results[main_counter, 10]=fp
            res_f.write(", ")
            res_f.write(str(fn))
            validation_results[main_counter, 11]=fn
            res_f.write(", ")
            res_f.write(str(training_iters_up))
            validation_results[main_counter, 12]=training_iters_up
            res_f.write(",") 
            res_f.write(str(validation_auc))
            validation_results[main_counter, 13]=validation_auc
            res_f.write(",") 
            res_f.write(str(reg_coeff))
            validation_results[main_counter, 14] = reg_coeff
            res_f.write("\n")                                             
        # res_f.write("\n")         
        main_counter=main_counter+1       
pdb.set_trace()

print('Performing testing ...')
max_f1_index=np.argmax(validation_results, axis=0)[6]

train_meds_filename = 'outputs/train_medications_multihot.csv'
train_diags_filename = 'outputs/train_diagnoses_multihot.csv'
train_procs_filename = 'outputs/train_procedures_multihot.csv'
train_demogs_filename = 'outputs/train_demographics_multihot.csv'
train_metadata_filename = 'outputs/train_demographics_shuffled.csv'


test_meds_filename = 'outputs/test_medications_multihot.csv'
test_diags_filename = 'outputs/test_diagnoses_multihot.csv'
test_procs_filename = 'outputs/test_procedures_multihot.csv'
test_demogs_filename = 'outputs/test_demographics_multihot.csv'
test_metadata_filename = 'outputs/test_demographics_shuffled.csv'

# pdb.set_trace()
softmax_predictions_temp, accuracy_temp, precision, recall, sensitivity, specificity, tp, tn, fp, fn, test_auc =lsad.main(1000, False, validation_results[max_f1_index,12], validation_results[max_f1_index,14], validation_results[max_f1_index,0],int(validation_results[max_f1_index,1]), int(validation_results[max_f1_index,2]), train_meds_filename, train_diags_filename, train_procs_filename, train_demogs_filename, train_metadata_filename, validation_meds_filename, validation_diags_filename, validation_procs_filename, validation_demogs_filename, validation_metadata_filename)
np.savetxt('results/LSTM_single_stream/softmax_predictions_lstm_single_stream.csv', softmax_predictions_temp, delimiter=',')
with open('results/LSTM_single_stream/Results_test_lstm_single_stream.csv', 'w') as res_f:
        res_f.write("".join(["".join(x) for x in header_results_filename]))      
        res_f.write(str(validation_results[max_f1_index,0]))
        res_f.write(", ")
        res_f.write(str(validation_results[max_f1_index,1]))
        res_f.write(", ")
        res_f.write(str(validation_results[max_f1_index,2]))
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
        res_f.write(str(validation_results[max_f1_index, 12]))
        res_f.write(",")  
        res_f.write(str(test_auc))
        res_f.write(",")  
        res_f.write(str(validation_results[max_f1_index,14]))
        res_f.write("\n")                        
