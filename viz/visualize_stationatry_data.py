import pdb
import argparse
import sys
import os
import visualization_functions as vis_tools


sys.path.append(os.getcwd())
parser = argparse.ArgumentParser()

parser.add_argument("--viz_method", type=str, default="none", choices = ["pca","tsne", "none"])    
parser.add_argument("--features_to_show", type=str, default="meds_diags_procs", choices = ["meds_diags_procs","all"])    
parser.add_argument("--sampled", type=int, default=1, choices = [0,1])    
parser.add_argument("--sample_size", type=int, default=2000)    
parser.add_argument("--sample_size_for_shap", type=float, default=0.05)  
parser.add_argument("--perplex", type=int, default=15)    
parser.add_argument("--num_it", type=int, default=2000)    
parser.add_argument("--lr_rate", type=int, default=200)    

parser.add_argument("--compute_stat", type=int, default=0, choices = [0,1])    
parser.add_argument("--feature_selection", type=int, default=0, choices = [0,1])    
parser.add_argument("--plot_shaps", type=int, default=0, choices = [0,1])    
parser.add_argument("--plot_shaps_from_saved_model", type=int, default=0, choices = [0,1])    

# parser.add_argument("--trained_model_path", type=str, default="results/visualization_results/shap_results_sep_12/xgb_model.pkl")    
parser.add_argument("--trained_model_path", type=str, default="saved_classical_ml_models/rf_model.pkl")    


parser.add_argument("--plot_feature_dist_flag", type=int, default=0, choices = [0,1])    
# parser.add_argument("--compute_shap", type=int, default=0, choices = [0,1])    
# parser.add_argument("--rf_model_path", type=str, default="results/classical_ml_models/rf_model.pkl")    
# parser.add_argument("--lr_model_path", type=str, default="results/classical_ml_models/lr_model.pkl")    

parser.add_argument("--train_stationary_filename", type=str, default="outputs/train_stationary_normalized.csv")    
parser.add_argument("--validation_stationary_filename", type=str, default="outputs/validation_stationary_normalized.csv")
parser.add_argument("--test_stationary_filename", type=str, default="outputs/test_stationary_normalized.csv")    

parser.add_argument("--train_stationary_filename_raw", type=str, default="outputs/train_stationary.csv")    
parser.add_argument("--validation_stationary_filename_raw", type=str, default="outputs/validation_stationary.csv")
parser.add_argument("--test_stationary_filename_raw", type=str, default="outputs/test_stationary.csv")    

parser.add_argument("--train_demographics_filename", type=str, default="outputs/train_demographics.csv")    
parser.add_argument("--validation_demographics_filename", type=str, default="outputs/validation_demographics.csv")
parser.add_argument("--test_demographics_filename", type=str, default="outputs/test_demographics.csv")    

parser.add_argument("--train_stationary_normalized_filtered_filename", type=str, default="outputs/train_stationary_normalized_features_filtered.csv")    
parser.add_argument("--validation_stationary_normalized_filtered_filename", type=str, default="outputs/validation_stationary_normalized_features_filtered.csv")
parser.add_argument("--test_stationary_normalized_filtered_filename", type=str, default="outputs/test_stationary_normalized_features_filtered.csv")    

parser.add_argument("--hist_meds_filepath", type=str, default="results/visualization_results/hist_stationary_meds_features_freq.csv")    
parser.add_argument("--hist_diags_filepath", type=str, default="results/visualization_results/hist_stationary_diags_features_freq.csv")
parser.add_argument("--hist_procs_filepath", type=str, default="results/visualization_results/hist_stationary_procs_features_freq.csv")    

if parser.parse_args().viz_method == "tsne":
    args = parser.parse_args()
    vis_tools.tSNE_visualization(args.train_stationary_filename 
                                , args.validation_stationary_filename     
                                , args.test_stationary_filename     
                                , args.sampled     
                                , args.sample_size
                                , args.features_to_show
                                , args.perplex
                                , args.num_it
                                , args.lr_rate)
elif parser.parse_args().viz_method == "pca":
    args = parser.parse_args()
    vis_tools.pca_visualization(args.train_stationary_filename 
                                , args.validation_stationary_filename     
                                , args.test_stationary_filename     
                                , args.sampled     
                                , args.sample_size
                                , args.features_to_show
                                )   
elif parser.parse_args().viz_method == "none":
    print("Warning: no visualization method has been selected.")


if parser.parse_args().compute_stat == 1:
    args = parser.parse_args()
    vis_tools.compute_stats(args.train_stationary_filename 
                                , args.validation_stationary_filename     
                                , args.test_stationary_filename 
                                , args.train_stationary_filename_raw 
                                , args.validation_stationary_filename_raw     
                                , args.test_stationary_filename_raw   
                                , args.train_demographics_filename
                                , args.validation_demographics_filename
                                , args.test_demographics_filename                              
                                )       
elif parser.parse_args().compute_stat == 0:    
    print('Warning: no method has been selected to compute statistics')



if parser.parse_args().plot_feature_dist_flag == 1:
    args = parser.parse_args()
    vis_tools.plot_feature_dist(args.train_stationary_filename 
                        , args.validation_stationary_filename     
                        , args.test_stationary_filename 
                        )
else:
    print('Warning: you have chosen to not compute feature distributions.')    

if parser.parse_args().feature_selection == 1:
    args = parser.parse_args()
    vis_tools.feature_selection_dists(args.hist_meds_filepath 
                        , args.hist_diags_filepath     
                        , args.hist_procs_filepath 
                        , args.train_stationary_filename 
                        , args.validation_stationary_filename     
                        , args.test_stationary_filename  
                        )
else:
    print('Warning: you have chosen to not perform feature selection.')    


if parser.parse_args().plot_shaps == 1:
    args = parser.parse_args()
    vis_tools.plot_shaps(args.train_stationary_normalized_filtered_filename
                        , args.validation_stationary_normalized_filtered_filename
                        , args.test_stationary_normalized_filtered_filename
                        , args.hist_meds_filepath 
                        , args.hist_diags_filepath     
                        , args.hist_procs_filepath     
                        , args.sample_size_for_shap                    
        )
else:
    print('Warning: you have chosen not to perform SHAP plots.')  

if parser.parse_args().plot_shaps_from_saved_model == 1:
    args = parser.parse_args()
    vis_tools.plot_shaps_from_saved_model(args.test_stationary_normalized_filtered_filename
                        , args.hist_meds_filepath 
                        , args.hist_diags_filepath     
                        , args.hist_procs_filepath     
                        , args.sample_size_for_shap    
                        , args.trained_model_path                
        )
else:
    print('Warning: you have chosen not to perform SHAP plots.')      
