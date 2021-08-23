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


parser.add_argument("--ml_model", type=str, default='rf') 

args = parser.parse_args()
if  args.ml_model == 'rf':
    print('Starting to train a random forest model using:\n')
    cl_ml.random_forest_model(args.train_data_path
                            , args.validation_data_path
                            , args.test_data_path
                            )
elif args.ml_model == 'lr':
    print('Starting to train a logistic regression model using:\n')   
    cl_ml.logistic_regression(args.train_data_path
                            , args.validation_data_path
                            , args.test_data_path
                            )   
elif args.ml_model == 'svm':
    print('Starting to train a SVM model using:\n')    
    cl_ml.support_vector_machine(args.train_data_path
                            , args.validation_data_path
                            , args.test_data_path
                            )                                