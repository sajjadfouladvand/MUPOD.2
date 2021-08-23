import pdb
import matplotlib as mpl 
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind
import math
import shap
import pickle
import numpy as np

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







