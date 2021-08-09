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
parser.add_argument("--perplex", type=int, default=20)    
parser.add_argument("--num_it", type=int, default=2000)    
parser.add_argument("--lr_rate", type=int, default=200)    

parser.add_argument("--dist_diff", type=int, default=1, choices = [0,1])    
parser.add_argument("--dist_diff_method", type=str, default="none", choices = ["mann_witt","none"])    

parser.add_argument("--train_stationary_filename", type=str, default="outputs/train_stationary_normalized.csv")    
parser.add_argument("--validation_stationary_filename", type=str, default="outputs/validation_stationary_normalized.csv")
parser.add_argument("--test_stationary_filename", type=str, default="outputs/test_stationary_normalized.csv")    

parser.add_argument("--train_stationary_filename_raw", type=str, default="outputs/train_stationary.csv")    
parser.add_argument("--validation_stationary_filename_raw", type=str, default="outputs/validation_stationary.csv")
parser.add_argument("--test_stationary_filename_raw", type=str, default="outputs/test_stationary.csv")    


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


if parser.parse_args().dist_diff_method == "mann_witt":
    args = parser.parse_args()
    vis_tools.compute_stats(args.train_stationary_filename 
                                , args.validation_stationary_filename     
                                , args.test_stationary_filename 
                                , args.train_stationary_filename_raw 
                                , args.validation_stationary_filename_raw     
                                , args.test_stationary_filename_raw                                     
                                )       
elif parser.parse_args().dist_diff_method == "none":    
    print('Warning: no method has been selected to compare the distributions')




