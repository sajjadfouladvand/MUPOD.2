import pdb
import matplotlib as mpl 
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind
import math
import shap
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import csv
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import multiprocessing
import lightgbm as lgb

def performance_evaluation(rf_predictions
                        , test_data_for_eval
                        , best_model):

    # pdb.set_trace()
    labels = test_data_for_eval['Label'].values
    rf_test_auc=roc_auc_score(test_data_for_eval['Label'], best_model.predict_proba(test_data_for_eval.iloc[:,1:-1])[:,1])
    tp=0
    tn=0
    fn=0
    fp=0
    accuracy=0
    precision=0
    recall=0
    F1=0
    specificity=0
    for asses_ind in range(len(rf_predictions)):
        if(rf_predictions[asses_ind]==0 and labels[asses_ind]==0):
            tn=tn+1
        elif(rf_predictions[asses_ind]==0 and labels[asses_ind]==1):
            fn=fn+1
        elif(rf_predictions[asses_ind]==1 and labels[asses_ind]==1):
            tp=tp+1
        elif(rf_predictions[asses_ind]==1 and labels[asses_ind]==0):    
            fp=fp+1
    accuracy=(tn+tp)/(tn+tp+fn+fp)
    if(tp+fp == 0):
        precision=0
    else:
        precision=tp/(tp+fp)
    if(tp+fn==0):
        recall=0
    else:
        recall=tp/(tp+fn)
    if(precision==0 and recall==0):
        F1=0
    else:            
        F1=(2*precision*recall)/(precision+recall)
    if(tn+fp==0):
        specificity= 0
    else:
        specificity= tn/(tn+fp)    

    return tn, tp, fn, fp, accuracy, precision, recall, specificity, F1, rf_test_auc    

def write_results(tn
                , tp
                , fn
                , fp
                , accuracy
                , precision
                , recall
                , specificity
                , F1
                , rf_test_auc
                , model
                    ):
    # pdb.set_trace()
    with open('results/visualization_results/'+model+'_prediction_performance.csv', 'w') as f_results:
        f_results.write("Precision is: ")
        f_results.write(str(precision))
        f_results.write("\n")
        
        f_results.write("Recall is: ")
        f_results.write(str(recall))
        f_results.write("\n")
        
        f_results.write("Accuracy is: ")
        f_results.write(str(accuracy))
        f_results.write("\n") 

        f_results.write("F1 is: ")
        f_results.write(str(F1))
        f_results.write("\n")

        f_results.write("Specificity is: ")
        f_results.write(str(specificity))
        f_results.write("\n")

        f_results.write("AUC is: ")
        f_results.write(str(rf_test_auc))
        f_results.write("\n")

        f_results.write("TP is: ")
        f_results.write(str(tp))
        f_results.write("\n")

        f_results.write("TN is: ")
        f_results.write(str(tn))
        f_results.write("\n")

        f_results.write("FP is: ")
        f_results.write(str(fp))
        f_results.write("\n")

        f_results.write("FN is: ")
        f_results.write(str(fn))
        f_results.write("\n")

def compute_stats(train_stationary_filename 
                                , validation_stationary_filename     
                                , test_stationary_filename 
                                , train_stationary_filename_raw 
                                , validation_stationary_filename_raw     
                                , test_stationary_filename_raw     
                                , train_demographics_filename
                                , validation_demographics_filename
                                , test_demographics_filename                                 
                             ):
    # Reading demographics
    train_demogs = pd.read_csv(train_demographics_filename)
    validation_demogs = pd.read_csv(validation_demographics_filename)
    test_demogs = pd.read_csv(test_demographics_filename)
    demogs_all = pd.concat([train_demogs, validation_demogs, test_demogs])
    demogs_all_pos = demogs_all[demogs_all[' Label']==1]
    demogs_all_neg = demogs_all[demogs_all[' Label']==0]
    month_on_op_ustat, month_on_op_pvalue = mannwhitneyu(demogs_all_pos['NUM_MONTHLY_OPIOID_PRESCS'], demogs_all_neg['NUM_MONTHLY_OPIOID_PRESCS'])
    month_in_data_ustat, month_in_data_pvalue = mannwhitneyu(demogs_all_pos['NUM_MONTHS_IN_DATA'], demogs_all_neg['NUM_MONTHS_IN_DATA'])

    # Reading the data before normalization withactual age values
    train_data_not_norm=pd.read_csv(train_stationary_filename_raw, index_col='ENROLID')
    validation_data_not_norm=pd.read_csv(validation_stationary_filename_raw, index_col='ENROLID')
    test_data_not_norm=pd.read_csv(test_stationary_filename_raw, index_col='ENROLID')    
    data_all_not_norm = pd.concat([train_data_not_norm, test_data_not_norm,validation_data_not_norm])
    data_all_pos_not_norm = data_all_not_norm[data_all_not_norm['Label']==1] 
    data_all_neg_not_norm = data_all_not_norm[data_all_not_norm['Label']==0] 

    # Reading the data after normalization
    train_data=pd.read_csv(train_stationary_filename, index_col='ENROLID')
    validation_data=pd.read_csv(validation_stationary_filename, index_col='ENROLID')
    test_data=pd.read_csv(test_stationary_filename, index_col='ENROLID')    
    data_all = pd.concat([train_data, test_data,validation_data])

    data_all_pos = data_all[data_all['Label']==1] 
    data_all_neg = data_all[data_all['Label']==0] 

    age_u_statistic, age_p_value = mannwhitneyu(data_all_pos['Age'], data_all_neg['Age'])
    
    # t-test
    #ttest_ind(data_all_pos['Age'], data_all_neg['Age'])
    
    num_sex2_pos = len(data_all_pos[data_all_pos['Sex']==1])
    num_sex1_pos = len(data_all_pos[data_all_pos['Sex']==0])

    num_sex2_neg = len(data_all_neg[data_all_neg['Sex']==1])
    num_sex1_neg = len(data_all_neg[data_all_neg['Sex']==0])
    # pdb.set_trace() 
    with open('results/visualization_results/stat_file.csv','w') as stat_file:        
        
        # Average and std of age in case and cohort
        stat_file.write('Average age in oud_yes is:\n')
        stat_file.write(str(data_all_pos_not_norm['Age'].mean()))
        stat_file.write('\n')
        stat_file.write('Standart deviation of age in oud_yes is:\n')
        stat_file.write(str(data_all_pos_not_norm['Age'].std()))        
        stat_file.write('\n')

        stat_file.write('Average age in oud_no is:\n')
        stat_file.write(str(data_all_neg_not_norm['Age'].mean()))
        stat_file.write('\n')
        stat_file.write('Standart deviation of age in oud_no is:\n')
        stat_file.write(str(data_all_neg_not_norm['Age'].std()))                
        stat_file.write('\n')

        # Average and std of the number of month patinets hasve been on opioid medications in case and cohort       
        stat_file.write('Average number of month on opioid in oud_yes is:\n')
        stat_file.write(str(demogs_all_pos['NUM_MONTHLY_OPIOID_PRESCS'].mean()))
        stat_file.write('\n')
        stat_file.write('Standart deviation of month on opioid in oud_yes is:\n')
        stat_file.write(str(demogs_all_pos['NUM_MONTHLY_OPIOID_PRESCS'].std()))        
        stat_file.write('\n')

        stat_file.write('Average number of month on opioid in oud_no is:\n')
        stat_file.write(str(demogs_all_neg['NUM_MONTHLY_OPIOID_PRESCS'].mean()))
        stat_file.write('\n')
        stat_file.write('Standart deviation of month on opioid in oud_no is:\n')
        stat_file.write(str(demogs_all_neg['NUM_MONTHLY_OPIOID_PRESCS'].std()))        
        stat_file.write('\n')

        # Average and std of data availibility in case and cohort
        stat_file.write('Average data availibility(month) in oud_yes is:\n')
        stat_file.write(str(demogs_all_pos['NUM_MONTHS_IN_DATA'].mean()))
        stat_file.write('\n')
        stat_file.write('Standart deviation of data availibility(month) in oud_yes is:\n')
        stat_file.write(str(demogs_all_pos['NUM_MONTHS_IN_DATA'].std()))        
        stat_file.write('\n')

        stat_file.write('Average data availibility(month) in oud_no is:\n')
        stat_file.write(str(demogs_all_neg['NUM_MONTHS_IN_DATA'].mean()))
        stat_file.write('\n')
        stat_file.write('Standart deviation of data availibility(month) in oud_no is:\n')
        stat_file.write(str(demogs_all_neg['NUM_MONTHS_IN_DATA'].std()))        
        stat_file.write('\n')

        # Sex 
        stat_file.write('Number of people with SEX=1 in oud_yes cohort is:\n')
        stat_file.write(str(num_sex1_pos))
        stat_file.write('\n')
        stat_file.write('Number of people with SEX=1 in oud_no cohort is:\n')
        stat_file.write(str(num_sex1_neg))
        stat_file.write('\n')

        stat_file.write('Number of people with SEX=2 in oud_yes cohort is:\n')
        stat_file.write(str(num_sex2_pos))
        stat_file.write('\n')       
        stat_file.write('Number of people with SEX=2 in oud_no cohort is:\n')
        stat_file.write(str(num_sex2_neg))
        stat_file.write('\n')

  
    # pdb.set_trace()     
    with open('results/visualization_results/p_values_u_stats.csv','w') as stat_file:  
        stat_file.write('Feature name, Num patients in oud_yes, number of patinets in our_no, P-value, u-statistic\n')
        
        # Month on opioid        
        stat_file.write('Months on opioid')
        stat_file.write(',')
        stat_file.write('-')
        stat_file.write(',')
        stat_file.write('_')
        stat_file.write(',')
        stat_file.write(str(month_on_op_pvalue))
        stat_file.write(',')
        stat_file.write(str(month_on_op_ustat))
        stat_file.write('\n')

        stat_file.write('Data availibility(months)')
        stat_file.write(',')
        stat_file.write('-')
        stat_file.write(',')
        stat_file.write('_')
        stat_file.write(',')
        stat_file.write(str(month_in_data_pvalue))
        stat_file.write(',')
        stat_file.write(str(month_in_data_ustat))
        stat_file.write('\n')

        # All other medications, diagnoses and procedures features
        for i in range(len(data_all.columns)-1):
            current_feature = data_all.columns[i]  
            # print(current_feature)
            if sum(data_all_pos[current_feature].values == data_all_neg[current_feature].values) == len(data_all_pos[current_feature].values):
                stat_file.write(current_feature)
                stat_file.write(',')
                stat_file.write(str(sum(data_all_pos[current_feature]>0)))
                stat_file.write(',')
                stat_file.write(str(sum(data_all_neg[current_feature]>0)))
                stat_file.write(',')
                stat_file.write('identical vars')
                stat_file.write(',')
                stat_file.write('identical vars')
                stat_file.write('\n')
            else:        
                temp_u_statistic, temp_p_value = mannwhitneyu(data_all_pos[current_feature], data_all_neg[current_feature])   
                stat_file.write(current_feature)
                stat_file.write(',')
                stat_file.write(str(sum(data_all_pos[current_feature]>0)))
                stat_file.write(',')
                stat_file.write(str(sum(data_all_neg[current_feature]>0)))
                stat_file.write(',')                
                stat_file.write(str(temp_p_value))
                stat_file.write(',')
                stat_file.write(str(temp_u_statistic))
                stat_file.write('\n')



def tSNE_visualization(train_stationary_filename 
                                , validation_stationary_filename     
                                , test_stationary_filename     
                                , sampled     
                                , sample_size
                                , features_to_show
                                , perplex
                                , num_it
                                , lr_rate):

    train_data=pd.read_csv(train_stationary_filename, index_col='ENROLID')
    validation_data=pd.read_csv(validation_stationary_filename, index_col='ENROLID')
    test_data=pd.read_csv(test_stationary_filename, index_col='ENROLID')    
    data_all = pd.concat([train_data, test_data,validation_data])

    if sampled ==1:
        data_pos = data_all[data_all['Label'] == 1]
        data_neg = data_all[data_all['Label'] == 0]
        data_pos_sampled = data_pos.sample(n=int(sample_size/2), replace=False)
        data_neg_sampled = data_neg.sample(n=int(sample_size/2), replace=False)
        data = pd.concat([data_pos_sampled, data_neg_sampled])
        data=data.sample(frac=1)
        file_name = 'sampled_' + str(sample_size)         
    else:
        data=data_all.sample(frac=1)
        file_name = 'alldata'

    data['Label'] = data['Label'].map({1: 'OUD-positive', 0: 'OUD-negative'})

    if features_to_show == "meds_diags_procs":
        data_treatment_ready=data.iloc[:,:-3]
        features = 'meds_diags_procs'
    elif features_to_show == "all":    
        data_treatment_ready=data.iloc[:,:-1]
        features = 'allfeatures'

    tsne_model = TSNE(n_components=2, perplexity=perplex, n_iter=num_it)
    tsne_results = tsne_model.fit_transform(data_treatment_ready)
    print("kl divergence is: ",tsne_model.kl_divergence_)
    df_tsne_results = pd.DataFrame({'Enrolid': data.index, 'First dimension of tSNE': tsne_results[:,0], 'Second dimension of tSNE': tsne_results[:,1], 'Label': data['Label']})
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('tSNE 1', fontsize = 15)
    ax.set_ylabel('tSNE 2', fontsize = 15)
    targets = ['OUD-negative', "OUD-positive"]
    colors = ['b', 'r']
    for target, color in zip(targets,colors):
        indicesToKeep = df_tsne_results['Label'] == target
        ax.scatter(df_tsne_results.loc[indicesToKeep, 'First dimension of tSNE']
                   , df_tsne_results.loc[indicesToKeep, 'Second dimension of tSNE']
                   , c = color
                   , s = 50)
    ax.legend(targets)
    ax.grid()
    plt.savefig('results/visualization_results/tsne_'+file_name + '_' + features +'_'+str(perplex)+'_'+str(num_it)+'_'+str(lr_rate)+'_'+ str(tsne_model.kl_divergence_)[:5]+'.png', dpi=600)
    if sampled ==1:
        data.to_csv('results/visualization_results/sampled_data_forVis_'+ file_name + '_' + features +'_'+str(perplex)+'_'+str(num_it)+'_'+str(lr_rate)+'_'+ str(tsne_model.kl_divergence_)[:5]+'.csv')
    df_tsne_results.to_csv('results/visualization_results/tSNE_results_'+ file_name+ '_' + features +'_'+str(perplex)+'_'+str(num_it)+'_'+str(lr_rate)+'_'+ str(tsne_model.kl_divergence_)[:5]+'.csv')

def pca_visualization(train_stationary_filename 
                                , validation_stationary_filename     
                                , test_stationary_filename     
                                , sampled     
                                , sample_size
                                , features_to_show
                                ):
    train_data=pd.read_csv(train_stationary_filename, index_col='ENROLID')
    validation_data=pd.read_csv(validation_stationary_filename, index_col='ENROLID')
    test_data=pd.read_csv(test_stationary_filename, index_col='ENROLID')    
    data_all = pd.concat([train_data, test_data,validation_data])

    if sampled ==1:
        data_pos = data_all[data_all['Label'] == 1]
        data_neg = data_all[data_all['Label'] == 0]
        data_pos_sampled = data_pos.sample(n=int(sample_size/2), replace=False)
        data_neg_sampled = data_neg.sample(n=int(sample_size/2), replace=False)
        data = pd.concat([data_pos_sampled, data_neg_sampled])
        data=data.sample(frac=1)
        file_name = 'sampled_' + str(sample_size)         
    else:
        data=data_all.sample(frac=1)
        file_name = 'alldata'

    data['Label'] = data['Label'].map({1: 'OUD-positive', 0: 'OUD-negative'})

    if features_to_show == "meds_diags_procs":
        data_treatment_ready=data.iloc[:,:-3]
        features = 'meds_diags_procs'
    elif features_to_show == "all":    
        data_treatment_ready=data.iloc[:,:-1]
        features = 'allfeatures'
    pdb.set_trace()
    data_treatment_std = StandardScaler().fit_transform(data_treatment_ready)
    pca_model = PCA(n_components=2)
    data_pca = pca_model.fit_transform(data_treatment_std)

    pca_c = pca_model.components_
    pca_ev = pca_model.explained_variance_
    pca_evr = pca_model.explained_variance_ratio_
    with open('results/visualization_results/pca_stat.csv', 'w') as pca_file:
        pca_file.write('components_ are:')
        pca_file.write('\n')
        for i in range(2):
            pca_file.write(','.join(map(str, pca_c[i])))
            pca_file.write('\n')
        pca_file.write('explained_variance_ are:')
        pca_file.write('\n')
        pca_file.write(','.join(map(str, pca_ev)))
        pca_file.write('\n')        
        pca_file.write('explained_variance_ratio_ are:')
        pca_file.write('\n')
        pca_file.write(','.join(map(str, pca_evr)))
        pca_file.write('\n')        

    df_pca_final = pd.DataFrame({'Enrolid': data.index, 'principal component 1': data_pca[:,0], 'principal component 2': data_pca[:,1], 'Label': data['Label']})
    df_pca_final.to_csv('results/visualization_results/Final_PCA.csv')

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    targets = ['OUD-positive', 'OUD-negative']
    colors = ['r', 'b']
    for target, color in zip(targets,colors):
        indicesToKeep = df_pca_final['Label'] == target
        ax.scatter(df_pca_final.loc[indicesToKeep, 'principal component 1']
                   , df_pca_final.loc[indicesToKeep, 'principal component 2']
                   , c = color
                   , s = 50)
                   #,alpha=0.03)
    ax.legend(targets)
    ax.grid()
    plt.savefig('results/visualization_results/pca_'+file_name + '_' + features +'.png', dpi=600)

def plot_feature_dist(train_stationary_filename 
                        , validation_stationary_filename     
                        , test_stationary_filename 
                        ):
    # pdb.set_trace()
    meds_start_idx = 1
    meds_end_idx = 95

    diags_start_idx = 95
    diags_end_idx = 379

    procs_start_idx = 379
    procs_end_idx = -3

    demogs_start_idx = -3
    demogs_end_idx = -1

    labels_start_idx = -1    
    labels_end_idx = -1    

    print('Reading data:')
    print(train_stationary_filename)
    print(validation_stationary_filename)
    print(test_stationary_filename)

    train_data = pd.read_csv(train_stationary_filename)
    validation_data = pd.read_csv(validation_stationary_filename)
    test_data = pd.read_csv(test_stationary_filename)
    data_all = pd.concat([train_data, validation_data, test_data])

    features_freq_pd = pd.DataFrame(np.zeros((len(data_all.columns),2)), columns = ['Feature', 'Frequency'])
    features_freq = []
    for i in range(len(data_all.columns)):
        current_column = data_all.columns[i]
        features_freq_pd.iloc[i,0] = current_column
        features_freq_pd.iloc[i,1] = (data_all[[current_column]] > 0).sum().values[0]
         # ['Feature'].loc[i] = current_column
        # features_freq_pd['Frequency'].loc[i] = (data_all[[current_column]] > 0).sum().values[0]
        # [[current_column]] = (data_all[[current_column]] > 0).sum().values[0]
        # features_freq.append([current_column, (data_all[[current_column]] > 0).sum().values[0]])    
    # pdb.set_trace()    
    print('Ploting frequencies...')    
    feature_freq_meds = features_freq_pd.iloc[meds_start_idx:meds_end_idx,:].copy()
    feature_freq_meds.sort_values(by='Frequency').plot.bar(x='Feature', y='Frequency')
    plt.legend('',frameon=False)
    plt.xticks(fontsize=3, rotation=90)
    # plt.yticks(np.arange(0, feature_freq_meds['Frequency'].max()+1, 1000))
    plt.ylabel('Number of patients')
    plt.savefig('results/visualization_results/hist_stationary_meds_features.png', dpi=600)
    plt.close()

    feature_freq_diags = features_freq_pd.iloc[diags_start_idx:diags_end_idx,:].copy()
    feature_freq_diags.sort_values(by='Frequency').plot.bar(x='Feature', y='Frequency')
    plt.legend('',frameon=False)
    plt.ylabel('Number of patients')
    plt.xticks(fontsize=1, rotation=90)
    plt.savefig('results/visualization_results/hist_stationary_diags_features.png', dpi=600)
    plt.close()

    feature_freq_procs = features_freq_pd.iloc[procs_start_idx:procs_end_idx,:].copy()
    feature_freq_procs.sort_values(by='Frequency').plot.bar(x='Feature', y='Frequency')
    plt.legend('',frameon=False)
    plt.ylabel('Number of patients')    
    plt.xticks(fontsize=1, rotation=90)
    plt.savefig('results/visualization_results/hist_stationary_procs_features.png', dpi=600)
    plt.close()

    features_freq_pd.to_csv('results/visualization_results/hist_stationary_features_freq.csv', index=False)
    feature_freq_meds.to_csv('results/visualization_results/hist_stationary_meds_features_freq.csv', index=False)
    feature_freq_diags.to_csv('results/visualization_results/hist_stationary_diags_features_freq.csv', index=False)
    feature_freq_procs.to_csv('results/visualization_results/hist_stationary_procs_features_freq.csv', index=False)
    
    # pdb.set_trace()
    print('Successfully finished!')

def feature_selection_dists(hist_meds_filepath
                            , hist_diags_filepath
                            , hist_procs_filepath
                            , train_stationary_filename 
                            , validation_stationary_filename     
                            , test_stationary_filename  
                            ):
    # pdb.set_trace()
    # percent_th = 75
    # Read frequency files
    med_freq = pd.read_csv(hist_meds_filepath)
    diag_freq = pd.read_csv(hist_diags_filepath)
    proc_freq = pd.read_csv(hist_procs_filepath)

    mean = 1
    med_freq_sorted = med_freq.sort_values(by='Frequency', ascending=False)
    sum_squares = 0
    for i in range(len(med_freq)):
        sum_squares += med_freq_sorted['Frequency'].iloc[i] * (((i+1) - mean) ** 2)     
    std_med = np.sqrt(sum_squares/med_freq_sorted['Frequency'].sum())

    diag_freq_sorted = diag_freq.sort_values(by='Frequency', ascending=False)
    sum_squares = 0
    for i in range(len(diag_freq)):
        sum_squares += diag_freq_sorted['Frequency'].iloc[i] * (((i+1) - mean) ** 2)     
    std_diag = np.sqrt(sum_squares/diag_freq_sorted['Frequency'].sum())

    proc_freq_sorted = proc_freq.sort_values(by='Frequency', ascending=False)
    sum_squares = 0
    for i in range(len(proc_freq)):
        sum_squares += proc_freq_sorted['Frequency'].iloc[i] * (((i+1) - mean) ** 2)     
    std_proc = np.sqrt(sum_squares/proc_freq_sorted['Frequency'].sum())
    # pdb.set_trace()
    cut_off_th_med = mean + 2 * std_med
    cut_off_th_diag = mean + 2 * std_diag
    cut_off_th_proc = mean + 2 * std_proc

    med_freq_sorted.iloc[:int(cut_off_th_med)+1].to_csv(hist_meds_filepath[:-4]+'_features_filtered.csv', index=False)
    diag_freq_sorted.iloc[:int(cut_off_th_diag)+1].to_csv(hist_diags_filepath[:-4]+'_features_filtered.csv', index=False)
    proc_freq_sorted.iloc[:int(cut_off_th_proc)+1].to_csv(hist_procs_filepath[:-4]+'_features_filtered.csv', index=False)


    meds_features = med_freq_sorted.iloc[:int(cut_off_th_med)+1].Feature.tolist()
    diags_features = diag_freq_sorted.iloc[:int(cut_off_th_diag)+1].Feature.tolist()
    procs_features = proc_freq_sorted.iloc[:int(cut_off_th_proc)+1].Feature.tolist()
    feature_to_keep = ['ENROLID'] + meds_features + diags_features + procs_features + ['Age', 'Sex', 'Label']
    # pdb.set_trace()


    # cut_off_th_med = np.percentile(med_freq['Frequency'], percent_th)
    # med_freq[med_freq['Frequency'] >= cut_off_th_med].to_csv(hist_meds_filepath[:-4]+'_features_filtered.csv', index=False)

    # cut_off_th_diag = np.percentile(diag_freq['Frequency'], percent_th)
    # diag_freq[diag_freq['Frequency'] >= cut_off_th_diag].to_csv(hist_diags_filepath[:-4]+'_features_filtered.csv', index=False)

    # cut_off_th_proc = np.percentile(proc_freq['Frequency'], percent_th)
    # proc_freq[proc_freq['Frequency'] >= cut_off_th_proc].to_csv(hist_procs_filepath[:-4]+'_features_filtered.csv', index=False)


    with open('results/visualization_results/cut_off_values.csv', 'w') as cutt_of_vals_file:
        cutt_of_vals_file.write('The cut off value for medications frequencies is: \n')
        cutt_of_vals_file.write(str(cut_off_th_med))
        cutt_of_vals_file.write('\n')
        cutt_of_vals_file.write('The cut off value for diagnoses frequencies is: \n')
        cutt_of_vals_file.write(str(cut_off_th_diag))
        cutt_of_vals_file.write('\n')
        cutt_of_vals_file.write('The cut off value for procedures frequencies is: \n')
        cutt_of_vals_file.write(str(cut_off_th_proc))
    # pdb.set_trace()    
    print('Reading data:')
    print(train_stationary_filename)
    print(validation_stationary_filename)
    print(test_stationary_filename)

    train_data = pd.read_csv(train_stationary_filename)
    validation_data = pd.read_csv(validation_stationary_filename)
    test_data = pd.read_csv(test_stationary_filename)


    # meds_features = med_freq[med_freq['Frequency'] >= cut_off_th_med].Feature.tolist()
    # diags_features = diag_freq[diag_freq['Frequency'] >= cut_off_th_diag].Feature.tolist()
    # diags_features = [str(i) for i in diags_features]
    # procs_features = proc_freq[proc_freq['Frequency'] >= cut_off_th_proc].Feature.tolist()
    # pdb.set_trace()
    # feature_to_keep = ['ENROLID'] + meds_features + diags_features + procs_features + ['Age', 'Sex', 'Label']

    train_data_features_filtered =  train_data.loc[:, train_data.columns.isin(feature_to_keep)]
    train_data_features_filtered.to_csv(train_stationary_filename[:-4] + '_features_filtered.csv', index=False)
    validation_data_features_filtered =  validation_data.loc[:, validation_data.columns.isin(feature_to_keep)]
    validation_data_features_filtered.to_csv(validation_stationary_filename[:-4] + '_features_filtered.csv', index=False)
    test_data_features_filtered =  test_data.loc[:, test_data.columns.isin(feature_to_keep)]
    test_data_features_filtered.to_csv(test_stationary_filename[:-4] + '_features_filtered.csv', index=False)
    
    
    print('test')

def create_shap_and_plot(model, data, data_name):
    # shap_values = shap.TreeExplainer(model).shap_values(data_all_sampled.iloc[:,1:-1])
    plt.close()
    # pdb.set_trace()
    explainer = shap.Explainer(model, data)
   
    shap_values = explainer(data)

    shap.plots.scatter(shap_values[:,"65_tcgp_2digit"], color=shap_values)
    plt.savefig('results/visualization_results/scatter_65_tcgp_2digit_xgb_original_'+data_name+'.png', dpi=600)
    plt.close()

    shap.plots.scatter(shap_values[:,"Age"], color=shap_values)
    plt.savefig('results/visualization_results/scatter_Age_xgb_original_'+data_name+'.png', dpi=600)
    plt.close()

    shap.plots.scatter(shap_values[:,"Sex"], color=shap_values)
    plt.savefig('results/visualization_results/scatter_Sex_xgb_original_'+data_name+'.png', dpi=600)
    plt.close()

    
    shap.plots.beeswarm(shap_values)    
    plt.savefig('results/visualization_results/beeswarm_xgb_original_'+data_name+'.png', dpi=600)
    plt.close()

    shap.plots.beeswarm(shap_values)    
    fig = plt.gcf()
    fig.set_size_inches(18, 12)
    fig.savefig('results/visualization_results/beeswarm_xgb_'+data_name+'.png', dpi=600)
    plt.close(fig)

    shap.plots.waterfall(shap_values[0])
    plt.savefig('results/visualization_results/waterfall_0_xgb_original_'+data_name+'.png', dpi=600)
    plt.close()

    shap.plots.waterfall(shap_values[0])
    fig2 = plt.gcf()
    fig2.set_size_inches(18, 12)
    fig2.savefig('results/visualization_results/waterfall_0__xgb_'+data_name+'.png', dpi=600)
    plt.close(fig2)

    # visualize the first prediction's explanation with a force plot
    # shap.plots.force(shap_values[0])
    # plt.savefig('results/visualization_results/force_0_xgb_original.png')
    # plt.close()
    # fig = plt.gcf()
    # fig.set_size_inches(18, 12)
    # fig.savefig('results/visualization_results/force_0__xgb.png')
    # plt.close()

    # shap.plots.force(shap_values)
    # shap.force_plot(explainer.expected_value[0], shap_values[0])
    
    shap.plots.bar(shap_values)
    plt.savefig('results/visualization_results/bar_xgb_original_'+data_name+'.png', dpi=600)
    plt.close()

    fig3 = plt.gcf()
    fig3.set_size_inches(18, 12)
    fig3.savefig('results/visualization_results/bar_xgb_'+data_name+'.png', dpi=600)
    plt.close(fig3)
    return shap_values    

def plot_shaps (train_stationary_normalized_filtered_filename
                , validation_stationary_normalized_filtered_filename
                , test_stationary_normalized_filtered_filename
                , hist_meds_filepath
                , hist_diags_filepath
                , hist_procs_filepath    
                , sample_size_for_shap                        
            ):
    # pdb.set_trace()
    print('Reading the data ....')
    train_data = pd.read_csv(train_stationary_normalized_filtered_filename)
    validation_data = pd.read_csv(validation_stationary_normalized_filtered_filename)
    test_data = pd.read_csv(test_stationary_normalized_filtered_filename)

    med_freq = pd.read_csv(hist_meds_filepath)
    diag_freq = pd.read_csv(hist_diags_filepath)
    proc_freq = pd.read_csv(hist_procs_filepath)

    print('Filtering sparse features ....')
    mean = 1
    med_freq_sorted = med_freq.sort_values(by='Frequency', ascending=False)
    sum_squares = 0
    for i in range(len(med_freq)):
        sum_squares += med_freq_sorted['Frequency'].iloc[i] * (((i+1) - mean) ** 2)     
    std_med = np.sqrt(sum_squares/med_freq_sorted['Frequency'].sum())

    diag_freq_sorted = diag_freq.sort_values(by='Frequency', ascending=False)
    sum_squares = 0
    for i in range(len(diag_freq)):
        sum_squares += diag_freq_sorted['Frequency'].iloc[i] * (((i+1) - mean) ** 2)     
    std_diag = np.sqrt(sum_squares/diag_freq_sorted['Frequency'].sum())

    proc_freq_sorted = proc_freq.sort_values(by='Frequency', ascending=False)
    sum_squares = 0
    for i in range(len(proc_freq)):
        sum_squares += proc_freq_sorted['Frequency'].iloc[i] * (((i+1) - mean) ** 2)     
    std_proc = np.sqrt(sum_squares/proc_freq_sorted['Frequency'].sum())
    # pdb.set_trace()
    cut_off_th_med = mean + 1 * std_med
    cut_off_th_diag = mean + 1 * std_diag
    cut_off_th_proc = mean + 1 * std_proc
    
    meds_features = med_freq_sorted.iloc[:int(cut_off_th_med)+1].Feature.tolist()
    diags_features = diag_freq_sorted.iloc[:int(cut_off_th_diag)+1].Feature.tolist()
    procs_features = proc_freq_sorted.iloc[:int(cut_off_th_proc)+1].Feature.tolist()
    feature_to_keep = ['ENROLID'] + meds_features[:10] + diags_features[:10] + procs_features[:10] + ['Age', 'Sex', 'Label']
    if '-1000_ccs_proc' in feature_to_keep:
        feature_to_keep.remove('-1000_ccs_proc')
    if '-1000_ccs_diag' in  feature_to_keep:  
        feature_to_keep.remove('-1000_ccs_diag')
    # pdb.set_trace()
    train_data_features_filtered =  train_data.loc[:, train_data.columns.isin(feature_to_keep)]
    validation_data_features_filtered =  validation_data.loc[:, validation_data.columns.isin(feature_to_keep)]
    test_data_features_filtered =  test_data.loc[:, test_data.columns.isin(feature_to_keep)]
    
    data_all = pd.concat([train_data_features_filtered, validation_data_features_filtered, test_data_features_filtered])

    print('Sampling the data ....')
    data_pos = data_all[data_all['Label'] == 1]
    data_neg = data_all[data_all['Label'] == 0]
    data_pos_sampled = data_pos.sample(n=int(sample_size_for_shap * len(data_pos)), replace=False)
    data_neg_sampled = data_neg.sample(n=int(sample_size_for_shap * len(data_neg)), replace=False)
    data_all_sampled = pd.concat([data_pos_sampled, data_neg_sampled])
    data_all_sampled=data_all_sampled.sample(frac=1)

    data_all_sampled_train, data_all_sampled_test = train_test_split(data_all_sampled, test_size=0.1, random_state=1234)
    data_all_sampled_train.to_csv('results/visualization_results/data_all_sampled_train_for_shap.csv', index=False)
    data_all_sampled_test.to_csv('results/visualization_results/data_all_sampled_test_for_shap.csv', index=False)
    # pdb.set_trace()
    n_estimators = [100, 150, 200, 250, 300, 350]# [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Maximum number of levels in tree
    max_depth = [4, 8, 16]#, 32]#, 64, 128]# [int(x) for x in np.linspace(10, 110, num = 11)]
    gamma = [0.001, 0.01, 0.1, 1, 10]
    learning_rate = [0.0001, 0.001, 0.01, 0.1, 1]
    # Create the random grid
    hyperparameters = {'n_estimators': n_estimators
                   , 'max_depth': max_depth
                   , 'gamma': gamma
                   , 'learning_rate':learning_rate
                   }
    # pdb.set_trace()
    # params = {"max_bin": 512,"learning_rate": 0.05,"boosting_type": "gbdt","objective": "binary","metric": "binary_logloss","num_leaves": 10,"verbose": -1,"min_data": 100,"boost_from_average": True}
    # d_train = lgb.Dataset(data_all_sampled_train.iloc[:,1:-1], label=data_all_sampled_train['Label'])
    # d_test = lgb.Dataset(data_all_sampled_test.iloc[:,1:-1], label=data_all_sampled_test['Label'])
    
    # model = lgb.train(params, d_train, 10000, valid_sets=[d_test], early_stopping_rounds=50, verbose_eval=1000)
    # explainer = shap.TreeExplainer(model)
    # shap_values = explainer.shap_values(data_all_sampled.iloc[:,1:-1])
    # shap.summary_plot(shap_values, data_all_sampled.iloc[:,1:-1])


    randomCV = RandomizedSearchCV(estimator=xgb.XGBClassifier(booster='gbtree', verbosity=1, n_jobs=multiprocessing.cpu_count()-1, early_stopping_rounds=10, objective='binary:logistic', use_label_encoder=False), param_distributions=hyperparameters, n_iter=10, cv=3,scoring="f1")

    # model = xgb.XGBClassifier(objective='binary:logistic')
    # model.fit(data_all_sampled_train.iloc[:,1:-1], data_all_sampled_train['Label'])
    # pdb.set_trace()
    print('===========================')
    print('Tuning XGB model ....')
    randomCV.fit(data_all_sampled_train.iloc[:,1:-1], data_all_sampled_train['Label'])

    model = randomCV.best_estimator_
    (pd.DataFrame.from_dict(data=randomCV.best_params_, orient='index').to_csv('results/visualization_results/best_params_xgb_for_shap.csv', header=False))

    with open('results/visualization_results/xgb_model.pkl','wb') as f:
        pickle.dump(randomCV,f)
    # pdb.set_trace()
    print('===========================')
    print('Creating shap plots using test data ....')    
    shap_values = create_shap_and_plot(model, data_all_sampled_test.iloc[:,1:-1], 'test')

    with open('results/visualization_results/shap_values_uing_xgb_and_tes_data.pkl','wb') as f:
        pickle.dump(shap_values,f)

    print('Creating shap plots using all data ....')    
    shap_values = create_shap_and_plot(model, data_all_sampled.iloc[:,1:-1], 'alldata')

    with open('results/visualization_results/shap_values_uing_xgb_and_alldata_data.pkl','wb') as f:
        pickle.dump(shap_values,f)

    predictions = model.predict(data_all_sampled_test.iloc[:,1:-1])

    np.savetxt('results/visualization_results/predictions_xgb_for_shap.csv', predictions, delimiter=',')

    tn, tp, fn, fp, accuracy, precision, recall, specificity, F1, rf_test_auc = performance_evaluation(predictions
                                                                            , data_all_sampled_test
                                                                            , model
                                                                            )   
    write_results(tn, tp, fn, fp, 
                accuracy, precision, recall, specificity
                , F1, rf_test_auc
                , 'xgb_for_shap_')
    # pdb.set_trace()
    print('Test')


def plot_shaps_from_saved_model (test_stationary_normalized_filtered_filename
                , hist_meds_filepath
                , hist_diags_filepath
                , hist_procs_filepath    
                , sample_size_for_shap   
                , trained_model_path                     
            ):
    pdb.set_trace()
    print('Reading the test data ....')
    test_data = pd.read_csv(test_stationary_normalized_filtered_filename)
    test_data = test_data.sample(frac=1).reset_index(drop=True) 

    randomCV =  pickle.load(open(trained_model_path, 'rb'))
    model = randomCV.best_estimator_
    explainer = shap.Explainer(model, test_data.iloc[:,1:-1])
    shap_values = explainer(test_data.iloc[:,1:-1])
    shap.plots.beeswarm(shap_values, plot_size=[14,8])    



    med_freq = pd.read_csv(hist_meds_filepath)
    diag_freq = pd.read_csv(hist_diags_filepath)
    proc_freq = pd.read_csv(hist_procs_filepath)

    print('Filtering sparse features ....')
    mean = 1
    med_freq_sorted = med_freq.sort_values(by='Frequency', ascending=False)
    sum_squares = 0
    for i in range(len(med_freq)):
        sum_squares += med_freq_sorted['Frequency'].iloc[i] * (((i+1) - mean) ** 2)     
    std_med = np.sqrt(sum_squares/med_freq_sorted['Frequency'].sum())

    diag_freq_sorted = diag_freq.sort_values(by='Frequency', ascending=False)
    sum_squares = 0
    for i in range(len(diag_freq)):
        sum_squares += diag_freq_sorted['Frequency'].iloc[i] * (((i+1) - mean) ** 2)     
    std_diag = np.sqrt(sum_squares/diag_freq_sorted['Frequency'].sum())

    proc_freq_sorted = proc_freq.sort_values(by='Frequency', ascending=False)
    sum_squares = 0
    for i in range(len(proc_freq)):
        sum_squares += proc_freq_sorted['Frequency'].iloc[i] * (((i+1) - mean) ** 2)     
    std_proc = np.sqrt(sum_squares/proc_freq_sorted['Frequency'].sum())
    # pdb.set_trace()
    cut_off_th_med = mean + 1 * std_med
    cut_off_th_diag = mean + 1 * std_diag
    cut_off_th_proc = mean + 1 * std_proc
    
    meds_features = med_freq_sorted.iloc[:int(cut_off_th_med)+1].Feature.tolist()
    diags_features = diag_freq_sorted.iloc[:int(cut_off_th_diag)+1].Feature.tolist()
    procs_features = proc_freq_sorted.iloc[:int(cut_off_th_proc)+1].Feature.tolist()
    feature_to_keep = ['ENROLID'] + meds_features + diags_features + procs_features + ['Age', 'Sex', 'Label']

    # pdb.set_trace()
    test_data_features_filtered =  test_data.loc[:, test_data.columns.isin(feature_to_keep)]
    
    data_all = test_data_features_filtered

    print('Sampling the data ....')
    data_pos = data_all[data_all['Label'] == 1]
    data_neg = data_all[data_all['Label'] == 0]
    data_pos_sampled = data_pos.sample(n=int(sample_size_for_shap * len(data_pos)), replace=False)
    data_neg_sampled = data_neg.sample(n=int(sample_size_for_shap * len(data_neg)), replace=False)
    data_all_sampled = pd.concat([data_pos_sampled, data_neg_sampled])
    data_all_sampled=data_all_sampled.sample(frac=1)

    data_all_sampled.to_csv('results/visualization_results/data_all_sampled_using_trained_model.csv', index=False)

    print('===========================')
    print('Loading XGB model ....')
    # trained_model_path = 'results/visualization_results/shap_results_sep_12/xgb_model.pkl'
    randomCV =  pickle.load(open(trained_model_path, 'rb'))
    model = randomCV.best_estimator_

    print('===========================')
    print('Creating shap plots ....')    
    explainer = shap.Explainer(model, data_all_sampled.iloc[:,1:-1])
    shap_values = explainer(data_all_sampled.iloc[:,1:-1])
    shap.plots.beeswarm(shap_values, plot_size=[14,8])    
    # fig = plt.gcf()
    # fig.set_size_inches(18, 12)
    plt.savefig('results/visualization_results/beeswarm_xgb_using_trained_model.png', dpi=600)
    plt.close()


    predictions = model.predict(data_all_sampled.iloc[:,1:-1])

    np.savetxt('results/visualization_results/predictions_xgb_for_shap_using_trained_model.csv', predictions, delimiter=',')

    tn, tp, fn, fp, accuracy, precision, recall, specificity, F1, rf_test_auc = performance_evaluation(predictions
                                                                            , data_all_sampled
                                                                            , model
                                                                            )   
    write_results(tn, tp, fn, fp, 
                accuracy, precision, recall, specificity
                , F1, rf_test_auc
                , 'trained_xgb')
    # pdb.set_trace()
    print('Test')



