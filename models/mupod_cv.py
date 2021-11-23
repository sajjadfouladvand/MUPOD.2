import os
import multi_stream_transformer_KFold_test as mupod_mdl
import numpy as np

imb_ratio = 10
min_epoch = 10
F1_idx = 10
#================= Read test performances and find the best model =================
validation_res=[]
with open("results/MUPOD/validation_results_twostreamT_thresholding.csv") as validation_results:
  next(validation_results)
  for line in validation_results:
      line_perf=line.replace(',\n','').split(',')
      line_perf = [float(i) for i in line_perf]  
      validation_res.append(line_perf)
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

header_results_filename= "Model Number (TWO STREAM-KFold), Learning Rate, Number of Layeres, Batch Size, Dropout Rate, Number of EPOCHS, test Accuracy, test Precision, test Recall, test F1-Score, test Specificity, test TP, test TN , test FP, test FN, test auc, test optimum threshold, regularization factor \n"
#pdb.set_trace()
# with open("results/MUPOD/restoring_best_model_test_results_twostreamT_thresholdin_KFold.csv", 'w') as results_file:
with open('results/MUPOD/restoring_best_model_test_results_twostreamT_thresholdin_1_to_'+str(imb_ratio)+'_KFold.csv', 'w') as results_file:    
    results_file.write("".join(["".join(x) for x in header_results_filename]))  

for i in range(100):
    if i % 10 ==0:
        print(i)
    # test_filename_meds='outputs/test_meds_represented_fold'+str(i)+'.csv'
    # test_filename_diags='outputs/test_diags_represented_fold'+str(i)+'.csv'
    # test_filename_procs='outputs/test_procs_represented_fold'+str(i)+'.csv'
    # test_filename_demogs='outputs/test_demographics_multihot_fold'+str(i)+'.csv'

    test_filename_meds='outputs/test_meds_represented_1_to_'+str(imb_ratio)+'_fold'+str(i)+'.csv'
    test_filename_diags='outputs/test_diags_represented_1_to_'+str(imb_ratio)+'_fold'+str(i)+'.csv'
    test_filename_procs='outputs/test_procs_represented_1_to_'+str(imb_ratio)+'_fold'+str(i)+'.csv'
    test_filename_demogs='outputs/test_demographics_multihot_1_to_'+str(imb_ratio)+'_fold'+str(i)+'.csv'

    checkpoint_path =  "saved_models/MUPOD/checkpoints/trained_model_" +str(best_model_number)
    accuracy_test, precision_test, recall_test, F1Score_test, specificity_test, tp_test, tn_test, fp_test, fn_test, test_auc = mupod_mdl.main(i,test_filename_meds, test_filename_diags, test_filename_procs, test_filename_demogs, checkpoint_path, best_model_number, learning_rate, num_layers, num_heads, BATCH_SIZE, dropout_rate, EPOCHS, optimum_threshold, regu_factor) 

    # with open('results/MUPOD/restoring_best_model_test_results_twostreamT_thresholdin_KFold.csv', 'a') as results_file:
    with open('results/MUPOD/restoring_best_model_test_results_twostreamT_thresholdin_1_to_'+str(imb_ratio)+'_KFold.csv', 'a') as results_file:
        results_file.write(str(best_model_number))
        results_file.write(",")
        results_file.write(str(learning_rate))
        results_file.write(",")
        results_file.write(str(num_layers))
        results_file.write(",")
        results_file.write(str(BATCH_SIZE))
        results_file.write(",")
        results_file.write(str(dropout_rate))
        results_file.write(",")
        results_file.write(str(EPOCHS))
        results_file.write(",")
        results_file.write(str(accuracy_test))
        results_file.write(", ")
        results_file.write(str(precision_test))
        results_file.write(", ")
        results_file.write(str(recall_test))
        results_file.write(", ")
        results_file.write(str(F1Score_test))
        results_file.write(", ")
        results_file.write(str(specificity_test))
        results_file.write(", ")                     
        results_file.write(str(tp_test))
        results_file.write(", ")
        results_file.write(str(tn_test))
        results_file.write(", ")
        results_file.write(str(fp_test))
        results_file.write(", ")
        results_file.write(str(fn_test))
        results_file.write(",")
        results_file.write(str(test_auc))
        results_file.write(",")
        results_file.write(str(optimum_threshold))
        results_file.write(",")
        results_file.write(str(regu_factor))
        results_file.write("\n")  