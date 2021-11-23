import os
import transformer_singlestream_KFold_test as mupod_mdl
import numpy as np

#================= Read test performances and find the best model =================
#================= Read test performances and find the best model =================
# pdb.set_trace()
imb_ratio = 10

F1_idx = 10
validation_res=[]
with open("results/single_stream_transformer/validation_results_single_stream_tranformer.csv") as validation_results:
  next(validation_results)
  for line in validation_results:
      line_perf=line.replace(',\n','').split(',')
      #pdb.set_trace()
      # line_perf[17].replace('\n','')
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
#===================================================
#===================================================

header_results_filename= "Model Number (SINGLE STREAM), Learning Rate, Number of Layeres, Batch Size, Dropout Rate, Number of EPOCHS, test Accuracy, test Precision, test Recall, test F1-Score, test Specificity, test TP, test TN , test FP, test FN, test auc, test optimum threshold \n"
with open('results/single_stream_transformer/restoring_best_singlestreamT_model_test_results__thresholdin_1_to_'+str(imb_ratio)+'_KFold.csv', 'w') as results_file:
    results_file.write("".join(["".join(x) for x in header_results_filename]))  

for i in range(100):
    if i % 10 ==0:
        print(i)
    # testing_filename_meds_diags_procs_demogs='outputs/test_meds_diags_procs_demogs_represented_fold'+str(i)+'.csv'
    testing_filename_meds_diags_procs_demogs='outputs/test_meds_diags_procs_demogs_represented_1_to_'+str(imb_ratio)+'_fold'+str(i)+'.csv'

    checkpoint_path =  "saved_models/checkpoints_single_stream/trained_model_" +str(best_model_number) 

    accuracy_test, precision_test, recall_test, F1Score_test, specificity_test, tp_test, tn_test, fp_test, fn_test, test_auc = mupod_mdl.main(i,testing_filename_meds_diags_procs_demogs, checkpoint_path, best_model_number, learning_rate, num_layers, num_heads, BATCH_SIZE, dropout_rate, EPOCHS, optimum_threshold) 

    with open('results/single_stream_transformer/restoring_best_singlestreamT_model_test_results__thresholdin_1_to_'+str(imb_ratio)+'_KFold.csv', 'a') as results_file:
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
        # results_file.write(",")
        # results_file.write(str(optimum_epoch))
        # results_file.write(",")  
        # results_file.write(str(regu_factor))
        results_file.write("\n")  