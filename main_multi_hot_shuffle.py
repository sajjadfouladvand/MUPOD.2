import pdb
import argparse
import sys
import os
import utils.deep_learning_preprocessing_functions as sta_funcs 

'''
This code reads medications, diagnoses, procedures and demographic data and 
convert their format to multi_hot format and then shuffle the data. After this, the data is ready to
be used to train deep learning models.
'''

sys.path.append(os.getcwd())
parser = argparse.ArgumentParser()  


# TRVNORM files
parser.add_argument("--dim_diags_file", type=str, default='trvnorm_data/dim_diagnoses_trvnorm.csv')    
parser.add_argument("--dim_procs_ccs_file", type=str, default='trvnorm_data/dim_proceduresCCS_trvnorm.csv')    

parser.add_argument("--distinct_diagcd_file", type=str, default='trvnorm_data/distinct_diagcd_trvnorm.csv')
parser.add_argument("--distinct_tcgpid_file", type=str, default='trvnorm_data/distinct_tcgpid_trvnorm.csv')    

parser.add_argument("--selected_features_meds", type=str, default='results/visualization_results/hist_stationary_meds_features_freq_features_filtered.csv')    
parser.add_argument("--selected_features_diags", type=str, default='results/visualization_results/hist_stationary_diags_features_freq_features_filtered.csv')    
parser.add_argument("--selected_features_procs", type=str, default='results/visualization_results/hist_stationary_procs_features_freq_features_filtered.csv')    



parser.add_argument("--tcgpi_num_digits", type=int, default=2)    
parser.add_argument("--fold_name", type=str, default='none', choices=["train", "validation", "test", "none"])    

parser.add_argument("--shuffle", type=int, default=0, choices=[0,1])    
parser.add_argument("--feature_selection", type=int, default=1, choices=[0,1])    
print('Wqarning: The feature selection is by defaul on. ')


if  parser.parse_args().fold_name == "train" and parser.parse_args().feature_selection==0:
       parser.add_argument("--meds_file", type=str, default='outputs/train_medications_shuffled.csv')    
       parser.add_argument("--diags_file", type=str, default='outputs/train_diagnoses_shuffled.csv')
       parser.add_argument("--procs_file", type=str, default='outputs/train_procedures_shuffled.csv')    
       parser.add_argument("--demogs_file", type=str, default='outputs/train_demographics_shuffled.csv') 
       args = parser.parse_args()
       sta_funcs.reformat_to_multihot(args.meds_file
                                   , args.diags_file
                                   , args.procs_file
                                   , args.demogs_file
                                   , args.dim_diags_file
                                   , args.dim_procs_ccs_file
                                   , args.distinct_tcgpid_file
                                   , args.tcgpi_num_digits
                                   , args.fold_name
                                   )
elif parser.parse_args().fold_name == "validation" and parser.parse_args().feature_selection==0:
       parser.add_argument("--meds_file", type=str, default='outputs/validation_medications_shuffled.csv')    
       parser.add_argument("--diags_file", type=str, default='outputs/validation_diagnoses_shuffled.csv')
       parser.add_argument("--procs_file", type=str, default='outputs/validation_procedures_shuffled.csv')    
       parser.add_argument("--demogs_file", type=str, default='outputs/validation_demographics_shuffled.csv') 
       args = parser.parse_args()
       sta_funcs.reformat_to_multihot(args.meds_file
                                   , args.diags_file
                                   , args.procs_file
                                   , args.demogs_file
                                   , args.dim_diags_file
                                   , args.dim_procs_ccs_file
                                   , args.distinct_tcgpid_file
                                   , args.tcgpi_num_digits
                                   , args.fold_name
                                   )
elif parser.parse_args().fold_name == "test" and parser.parse_args().feature_selection==0:
       parser.add_argument("--meds_file", type=str, default='outputs/test_medications_shuffled.csv')    
       parser.add_argument("--diags_file", type=str, default='outputs/test_diagnoses_shuffled.csv')
       parser.add_argument("--procs_file", type=str, default='outputs/test_procedures_shuffled.csv')    
       parser.add_argument("--demogs_file", type=str, default='outputs/test_demographics_shuffled.csv') 
       args = parser.parse_args()
       sta_funcs.reformat_to_multihot(args.meds_file
                                   , args.diags_file
                                   , args.procs_file
                                   , args.demogs_file
                                   , args.dim_diags_file
                                   , args.dim_procs_ccs_file
                                   , args.distinct_tcgpid_file
                                   , args.tcgpi_num_digits
                                   , args.fold_name
                                   )
elif  parser.parse_args().fold_name == "train" and parser.parse_args().feature_selection==1:
       parser.add_argument("--meds_file", type=str, default='outputs/train_medications_shuffled.csv')    
       parser.add_argument("--diags_file", type=str, default='outputs/train_diagnoses_shuffled.csv')
       parser.add_argument("--procs_file", type=str, default='outputs/train_procedures_shuffled.csv')    
       parser.add_argument("--demogs_file", type=str, default='outputs/train_demographics_shuffled.csv') 
       args = parser.parse_args()
       sta_funcs.reformat_to_multihot_with_feature_selection(args.meds_file
                                   , args.diags_file
                                   , args.procs_file
                                   , args.demogs_file
                                   , args.dim_diags_file
                                   , args.dim_procs_ccs_file
                                   , args.distinct_tcgpid_file
                                   , args.tcgpi_num_digits
                                   , args.fold_name
                                   , args.selected_features_meds
                                   , args.selected_features_diags
                                   , args.selected_features_procs
                                   )
elif parser.parse_args().fold_name == "validation" and parser.parse_args().feature_selection==1:
       parser.add_argument("--meds_file", type=str, default='outputs/validation_medications_shuffled.csv')    
       parser.add_argument("--diags_file", type=str, default='outputs/validation_diagnoses_shuffled.csv')
       parser.add_argument("--procs_file", type=str, default='outputs/validation_procedures_shuffled.csv')    
       parser.add_argument("--demogs_file", type=str, default='outputs/validation_demographics_shuffled.csv') 
       args = parser.parse_args()
       sta_funcs.reformat_to_multihot_with_feature_selection(args.meds_file
                                   , args.diags_file
                                   , args.procs_file
                                   , args.demogs_file
                                   , args.dim_diags_file
                                   , args.dim_procs_ccs_file
                                   , args.distinct_tcgpid_file
                                   , args.tcgpi_num_digits
                                   , args.fold_name
                                   , args.selected_features_meds
                                   , args.selected_features_diags
                                   , args.selected_features_procs                                   
                                   )
elif parser.parse_args().fold_name == "test" and parser.parse_args().feature_selection==1:
       parser.add_argument("--meds_file", type=str, default='outputs/test_medications_shuffled.csv')    
       parser.add_argument("--diags_file", type=str, default='outputs/test_diagnoses_shuffled.csv')
       parser.add_argument("--procs_file", type=str, default='outputs/test_procedures_shuffled.csv')    
       parser.add_argument("--demogs_file", type=str, default='outputs/test_demographics_shuffled.csv') 
       args = parser.parse_args()
       sta_funcs.reformat_to_multihot_with_feature_selection(args.meds_file
                                   , args.diags_file
                                   , args.procs_file
                                   , args.demogs_file
                                   , args.dim_diags_file
                                   , args.dim_procs_ccs_file
                                   , args.distinct_tcgpid_file
                                   , args.tcgpi_num_digits
                                   , args.fold_name
                                   , args.selected_features_meds
                                   , args.selected_features_diags
                                   , args.selected_features_procs                                   
                                   )
elif   parser.parse_args().fold_name == "none": 
       print('WARNING: None of the train, validation or test sets has been selected to convert to multihot.')        



if     parser.parse_args().shuffle == 1:
       parser.add_argument("--train_meds_file", type=str, default='outputs/train_medications.csv')    
       parser.add_argument("--train_diags_file", type=str, default='outputs/train_diagnoses.csv')
       parser.add_argument("--train_procs_file", type=str, default='outputs/train_procedures.csv')    
       parser.add_argument("--train_demogs_file", type=str, default='outputs/train_demographics.csv')        
       
       parser.add_argument("--validation_meds_file", type=str, default='outputs/validation_medications.csv')    
       parser.add_argument("--validation_diags_file", type=str, default='outputs/validation_diagnoses.csv')
       parser.add_argument("--validation_procs_file", type=str, default='outputs/validation_procedures.csv')    
       parser.add_argument("--validation_demogs_file", type=str, default='outputs/validation_demographics.csv')               

       parser.add_argument("--test_meds_file", type=str, default='outputs/test_medications.csv')    
       parser.add_argument("--test_diags_file", type=str, default='outputs/test_diagnoses.csv')
       parser.add_argument("--test_procs_file", type=str, default='outputs/test_procedures.csv')    
       parser.add_argument("--test_demogs_file", type=str, default='outputs/test_demographics.csv')               

       args = parser.parse_args()
       sta_funcs.shuffle_data(args.train_meds_file
                            , args.train_diags_file
                            , args.train_procs_file
                            , args.train_demogs_file
                            
                            , args.validation_meds_file
                            , args.validation_diags_file
                            , args.validation_procs_file
                            , args.validation_demogs_file                                   
                            
                            , args.test_meds_file
                            , args.test_diags_file
                            , args.test_procs_file
                            , args.test_demogs_file                                  
                            )   
elif   parser.parse_args().shuffle == 0:
       print('Wanring: you have chosen to not shuffle the data.')



