import pdb
import argparse
import sys
import os
import models.classical_ml_models as cl_ml


sys.path.append(os.getcwd())
parser = argparse.ArgumentParser()  


# === train, validation and test stationary data
parser.add_argument("--train_data_path", type=str, default='outputs/train_stationary_normalized.csv')    
parser.add_argument("--validation_data_path", type=str, default='outputs/validation_stationary_normalized.csv')
parser.add_argument("--test_data_path", type=str, default='outputs/test_stationary_normalized.csv')    

parser.add_argument("--train_data_path_features_filtered", type=str, default='outputs/train_stationary_normalized_features_filtered.csv')    
parser.add_argument("--validation_data_path_features_filtered", type=str, default='outputs/validation_stationary_normalized_features_filtered.csv')
parser.add_argument("--test_data_path_features_filtered", type=str, default='outputs/test_stationary_normalized_features_filtered.csv')    

parser.add_argument("--feature_selection", type=int, default=1, choices = [0,1])    
print('Warning: you are using the feature selection by default.')


parser.add_argument("--ml_model", type=str, default='none') 
parser.add_argument("--test_imb", type=int, default=0, choices=[0, 1]) 
parser.add_argument("--test_imb_ratio", type=int, default=2, choices=[2, 5, 10]) 
parser.add_argument("--cv_test", type=int, default=0, choices=[0, 1]) 

parser.add_argument("--trained_rf_path", type=str, default='saved_classical_ml_models/rf_model.pkl') 
parser.add_argument("--trained_lr_path", type=str, default='saved_classical_ml_models/lr_model.pkl') 


args = parser.parse_args()
if  args.ml_model == 'rf' and args.feature_selection == 0:
    print('Starting to train a random forest model using:\n')
    cl_ml.random_forest_model(args.train_data_path
                            , args.validation_data_path
                            , args.test_data_path
                            )
elif args.ml_model == 'lr' and args.feature_selection == 0:
    print('Starting to train a logistic regression model using:\n')   
    cl_ml.logistic_regression(args.train_data_path
                            , args.validation_data_path
                            , args.test_data_path
                            )   
elif args.ml_model == 'svm' and args.feature_selection == 0:
    print('Starting to train a SVM model using:\n')    
    cl_ml.support_vector_machine(args.train_data_path
                            , args.validation_data_path
                            , args.test_data_path
                            )                                
elif  args.ml_model == 'rf' and args.feature_selection == 1:
    print('Starting to train a random forest model using feature selection and:\n')
    cl_ml.random_forest_model(args.train_data_path_features_filtered
                            , args.validation_data_path_features_filtered
                            , args.test_data_path_features_filtered
                            )
elif args.ml_model == 'lr' and args.feature_selection == 1:
    print('Starting to train a logistic regression model using feature selection and:\n')   
    cl_ml.logistic_regression(args.train_data_path_features_filtered
                            , args.validation_data_path_features_filtered
                            , args.test_data_path_features_filtered
                            )   
elif args.ml_model == 'xgb' and args.feature_selection == 1:
    print('Starting to train a XGBoost  model using feature selection and:\n')   
    cl_ml.xgboos_model(args.train_data_path_features_filtered
                            , args.validation_data_path_features_filtered
                            , args.test_data_path_features_filtered
                            )   
elif args.ml_model == 'svm' and args.feature_selection == 1:
    print('Starting to train a SVM model using feature selection and:\n')    
    cl_ml.support_vector_machine(args.train_data_path_features_filtered
                            , args.validation_data_path_features_filtered
                            , args.test_data_path_features_filtered
                            )                                
elif args.ml_model == 'none':
    print('Warning: you have not selected any method to train.')


if args.test_imb == 1 :
    cl_ml.test_with_imb_data(args.trained_rf_path
                        , args.trained_lr_path
                        , args.test_imb_ratio
                        )
else:
    print('You have choosen not to test the models with imbalance test set.')    

if args.cv_test == 1 :
    header_results_filename= "Accuracy, Precision, Recall, F1, AUC, TP, TN , FP, FN \n"
    imb_ratio = 10
    #pdb.set_trace()
    # with open('results/classical_ml_models/RF_prediction_performance_KFold.csv', 'w') as results_file:
    # with open('results/classical_ml_models/LR_prediction_performance_KFold.csv', 'w') as results_file:    
    # with open('results/classical_ml_models/LR_prediction_performance_KFold_1_to_'+str(imb_ratio)+'.csv', 'w') as results_file:            
    with open('results/classical_ml_models/RF_prediction_performance_KFold_1_to_'+str(imb_ratio)+'.csv', 'w') as results_file:                    
        results_file.write("".join(["".join(x) for x in header_results_filename]))  

    model_math = 'saved_classical_ml_models/rf_model.pkl'
    # model_math = 'saved_classical_ml_models/lr_model.pkl'
    for i in range(100):
        print('Testing the stationary model using fold {}.'.format(i))
        # test_filename = 'outputs/test_stationary_normalized_features_filtered_fold'+str(i)+'.csv'
        test_filename = 'outputs/test_stationary_normalized_features_filtered_1_to_'+str(imb_ratio)+'_fold'+str(i)+'.csv'
        accuracy, precision, recall, specificity, F1, test_auc, tp, tn, fp, fn = cl_ml.test_with_cv(model_math
                                                                                                    , test_filename
                                                                                                    )
        # with open('results/classical_ml_models/RF_prediction_performance_KFold.csv', 'a') as results_file:
        # with open('results/classical_ml_models/LR_prediction_performance_KFold_1_to_'+str(imb_ratio)+'.csv', 'a') as results_file:    
        with open('results/classical_ml_models/RF_prediction_performance_KFold_1_to_'+str(imb_ratio)+'.csv', 'a') as results_file:        
            results_file.write(str(accuracy))
            results_file.write(",") 

            results_file.write(str(precision))
            results_file.write(",")
            
            results_file.write(str(recall))
            results_file.write(",")
            
            results_file.write(str(F1))
            results_file.write(",")

            results_file.write(str(test_auc))
            results_file.write(",")

            results_file.write(str(tp))
            results_file.write(",")

            results_file.write(str(tn))
            results_file.write(",")

            results_file.write(str(fp))
            results_file.write(",")

            results_file.write(str(fn))
            results_file.write("\n")        
else:
    print('You have choosen not to test the models with imbalance test set.')  


