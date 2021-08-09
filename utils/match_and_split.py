import os
import numpy as np
import pdb
import random
import pandas as pd
import itertools
from datetime import datetime
from scipy.spatial.distance import cdist
import pickle
import time
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from yellowbrick.cluster import KElbowVisualizer

def normalize_min_max(oud_yes_df
                    , oud_no_df):
    # pdb.set_trace()
    epsil = 2.220446049250313e-16
    oud_yes_df_mins = oud_yes_df.min()
    oud_no_df_mins = oud_no_df.min()
    temp_mins = pd.concat([oud_yes_df_mins, oud_no_df_mins], axis=1, ignore_index=True)
    global_mins = temp_mins.min(axis=1)


    oud_yes_df_max = oud_yes_df.max()
    oud_no_df_max = oud_no_df.max()
    temp_max = pd.concat([oud_yes_df_max, oud_no_df_max], axis=1, ignore_index=True)
    global_max = temp_max.max(axis=1) 
    
    # global_max[global_max == global_mins]

    normalized_oud_yes=(oud_yes_df-global_mins)/((global_max-global_mins) + epsil)    
    normalized_oud_no=(oud_no_df-global_mins)/((global_max-global_mins) + epsil)    
    
    return normalized_oud_yes, normalized_oud_no

def find_closest(df, val, k):
    # The k closets are definately within closest+k and closets-k. Therefore, the k smallets values within this interval are the k-closets ones.
    # Note, although the values are sorted, but still all the k-nearest might be in closets-k to closets or all withing closet to closet to closest+k. So we need to look at the entire [closest-k, closest+k] interval
    return df.loc[df['AVG_DIST_TO_K'].sub(val).abs().nsmallest(k).index]

def find_closest_sorted(df, val, k):
    # searchsorted use binary search to find the insert location for val
    closest = df['AVG_DIST_TO_K'].searchsorted(val)
    if (closest - k) <0:
        return find_closest(df.iloc[:closest + k], val, k)    
    return find_closest(df.iloc[closest - k:closest + k], val, k)

def diff_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month

def remove_existing_train_val_test():
    if os.path.exists("outputs/train_medications.csv"):
        os.remove("outputs/train_medications.csv")
    if os.path.exists("outputs/train_diagnoses.csv"):    
        os.remove("outputs/train_diagnoses.csv")
    if os.path.exists("outputs/train_procedures.csv"):    
        os.remove("outputs/train_procedures.csv")
    if os.path.exists("outputs/train_labels.csv"):
        os.remove("outputs/train_labels.csv")
    if os.path.exists("outputs/train_demographics.csv"):
        os.remove("outputs/train_demographics.csv")    

    if os.path.exists("outputs/validation_medications.csv"):
        os.remove("outputs/validation_medications.csv")
    if os.path.exists("outputs/validation_diagnoses.csv"):
        os.remove("outputs/validation_diagnoses.csv")
    if os.path.exists("outputs/validation_procedures.csv"):
        os.remove("outputs/validation_procedures.csv")
    if os.path.exists("outputs/validation_labels.csv"):
        os.remove("outputs/validation_labels.csv")
    if os.path.exists("outputs/validation_demographics.csv"):
        os.remove("outputs/validation_demographics.csv")

    if os.path.exists("outputs/test_medications.csv"):
        os.remove("outputs/test_medications.csv")
    if os.path.exists("outputs/test_diagnoses.csv"):
        os.remove("outputs/test_diagnoses.csv")
    if os.path.exists("outputs/test_procedures.csv"):
        os.remove("outputs/test_procedures.csv")
    if os.path.exists("outputs/test_labels.csv"):
        os.remove("outputs/test_labels.csv")
    if os.path.exists("outputs/test_demographics.csv"):
        os.remove("outputs/test_demographics.csv")    

def sampling_oud_yes_cohort(demogs_oud_yes_path
                            , num_sample
                            ):
    # pdb.set_trace()
    demogs_oud_yes = pd.read_csv(demogs_oud_yes_path)
    demogs_oud_yes_sampled = demogs_oud_yes.sample(num_sample)
    demogs_oud_yes_sampled.to_csv(demogs_oud_yes_path[:-4]+'_sampled.csv', index=False)
    return demogs_oud_yes_path[:-4]+'_sampled.csv'


def cohort_matching_big_data( demogs_oud_yes_path
                    , demogs_oud_no_path
                    , pos_to_negs_ratio
                    , k
                    ):
    
    match_based_on = ['DOB', 'NUM_MONTHLY_OPIOID_PRESCS', 'NUM_MONTHS_IN_DATA']
    
    demog_oud_yes_data = pd.read_csv(demogs_oud_yes_path)
    demog_oud_no_data = pd.read_csv(demogs_oud_no_path)

    # Normalizing the demographic data (I will only use those that we matched based on)
    demog_oud_yes_data_filtered_normed, demog_oud_no_data_filtered_normed = normalize_min_max(demog_oud_yes_data[match_based_on]
                                                                                            , demog_oud_no_data[match_based_on])
    # Copy normalized value into the original data frame for columns in the match_based_on list
    demog_oud_yes_data[match_based_on] = demog_oud_yes_data_filtered_normed
    demog_oud_no_data[match_based_on] = demog_oud_no_data_filtered_normed
    
    # Deviding oud_yes cohort to two sub-cohort based on SEX
    demog_oud_yes_data_sex_1 = demog_oud_yes_data[demog_oud_yes_data['SEX'] == 1]
    demog_oud_yes_data_sex_2 = demog_oud_yes_data[demog_oud_yes_data['SEX'] == 2]

    # Deviding oud_no cohort to two sub-cohort based on SEX
    demog_oud_no_data_sex_1 = demog_oud_no_data[demog_oud_no_data['SEX'] == 1]
    demog_oud_no_data_sex_2 = demog_oud_no_data[demog_oud_no_data['SEX'] == 2]
    
    
    # ==== For SEX = 1 ===
    # Find the optimum number of clusters using KElbowVisualizer in Yellow Brick
    elbow_range=30

    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(2,elbow_range))
    visualizer.fit(demog_oud_yes_data_sex_1[match_based_on]) 
    visualizer.show() 
    plt.savefig('results/visualization_results/kmeans_KElbowVisualizer_sex1_'+str(elbow_range)+'.png', dpi=600)
    plt.close()

    # wcss = []
    # range_k = 8
    # for i in range(1, range_k):
    #     k_itr = 2 ** i
    #     kmeans = KMeans(n_clusters=k_itr, init='k-means++', max_iter=600, n_init=10, random_state=1234)
    #     kmeans.fit(demog_oud_yes_data_sex_1[match_based_on])
    #     wcss.append(kmeans.inertia_)
    #     print(k_itr)                
    # plt.plot([2**i for i in range(1,range_k)], wcss)
    # plt.xlabel('Number of clusters')
    # plt.savefig('results/visualization_results/kmeans_xcss_sex1_'+str(range_k)+'.png', dpi=600)
    # plt.close()
    # np.savetxt('wcss_list_sex1.csv', wcss)

    kmeans = KMeans(n_clusters=visualizer.elbow_value_, init='k-means++', max_iter=600, n_init=10, random_state=1234)
    pred_y = kmeans.fit_predict(demog_oud_yes_data_sex_1[match_based_on])
    cluster_centers = pd.DataFrame(data=kmeans.cluster_centers_, columns=demog_oud_yes_data_sex_1[match_based_on].columns)

    # Randomly select K (=30) oud_yes patients
    cluster_centers.to_csv('outputs/cluster_centers_sex1_k_'+str(k)+'_ratio'+str(pos_to_negs_ratio)+'.csv', index=False)
    
    # Find the cosine similarity between all oud_no patients to the k anchors 
    similarities = cdist(cluster_centers, demog_oud_no_data_sex_1[match_based_on], metric='euclidean')
    
    # Compute the arithmetic/geometric mean of the K cosine similarities for all rows in D 
    similarities_avg = similarities.mean(axis=0) 

    # Create a dataframe, A (in the code I call it dist_to_anchor_oud_no_sex1), whith two columns: oud_no patients ENROLID, and their averaged distance to the k anchors
    dist_to_anchor_oud_no_sex1 = pd.DataFrame({'ENROLID':demog_oud_no_data_sex_1['ENROLID'].values, 'AVG_DIST_TO_K':similarities_avg})
    # This MATCHED column will be used to implement matching without replacement
    dist_to_anchor_oud_no_sex1['MATCHED'] = 0

    # Sort A (in the code I used the name dist_to_anchor_oud_no_sex1 instead of A) based on distance to 
    dist_to_anchor_oud_no_sex1.sort_values('AVG_DIST_TO_K', inplace=True)

    line_counter = 0
    start_time = time.time()
    # For each oud_yes sample p_i
    for index, row in demog_oud_yes_data_sex_1.iterrows(): 
        if line_counter%10000==0:
            print(line_counter)
            print("--- %s seconds ---" % (time.time() - start_time))
        line_counter +=1   
        #  Find the average cosine similarities, x_i, between this patient P_i to the K oud_yes anchor samples.
        similarities_oud_yes_to_anchor = cdist(np.reshape(row[match_based_on].values, (1,len(match_based_on))), np.reshape(cluster_centers.values, (-1,len(match_based_on))), metric='euclidean')
        avg_sim_oud_yes_to_anch = np.mean(similarities_oud_yes_to_anchor)
        
        # Use binary search to find the closests positions to the average distance computed above
        matched_oud_no_samples = find_closest_sorted(dist_to_anchor_oud_no_sex1[dist_to_anchor_oud_no_sex1['MATCHED'] == 0], avg_sim_oud_yes_to_anch, pos_to_negs_ratio)
        if len(matched_oud_no_samples) == 0:
            pdb.set_trace()
        dist_to_anchor_oud_no_sex1.loc[matched_oud_no_samples.index, 'MATCHED'] = 1

    # ============ For SEX=2  
    # pdb.set_trace()  
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(2,elbow_range))
    visualizer.fit(demog_oud_yes_data_sex_2[match_based_on]) 
    visualizer.show() 
    plt.savefig('results/visualization_results/kmeans_KElbowVisualizer_sex2_'+str(elbow_range)+'.png', dpi=600)
    plt.close()    
    # pdb.set_trace()
    # wcss_sex2 = []
    # range_k = 8
    # for i in range(1, range_k):
    #     k_itr = 2 ** i
    #     kmeans = KMeans(n_clusters=k_itr, init='k-means++', max_iter=600, n_init=10, random_state=1234)
    #     kmeans.fit(demog_oud_yes_data_sex_2[match_based_on])
    #     wcss_sex2.append(kmeans.inertia_)
    #     print(k_itr)                
    # plt.plot([2**i for i in range(1,range_k)], wcss_sex2)
    # plt.xlabel('Number of clusters')
    # plt.savefig('results/visualization_results/kmeans_xcss_sex2_'+str(range_k)+'.png', dpi=600)
    # plt.close()
    # np.savetxt('wcss_list_sex2.csv', wcss_sex2)

    kmeans = KMeans(n_clusters=visualizer.elbow_value_, init='k-means++', max_iter=600, n_init=10, random_state=1234)
    pred_y = kmeans.fit_predict(demog_oud_yes_data_sex_2[match_based_on])
    cluster_centers = pd.DataFrame(data=kmeans.cluster_centers_, columns=demog_oud_yes_data_sex_2[match_based_on].columns)

    # Randomly select K (=30) oud_yes patients
    cluster_centers.to_csv('outputs/cluster_centers_sex2_k_'+str(k)+'_ratio'+str(pos_to_negs_ratio)+'.csv', index=False)
    
    similarities_sex2 = cdist(cluster_centers, demog_oud_no_data_sex_2[match_based_on], metric='euclidean')
    
    # Compute the arithmetic/geometric mean of the K cosine similarities for all rows in D 
    similarities_sex2_avg = similarities_sex2.mean(axis=0) 

    # Create a dataframe whith two columns: oud_no patients ENROLID, averaged distance to the k anchors
    dist_to_anchor_oud_no_sex2 = pd.DataFrame({'ENROLID':demog_oud_no_data_sex_2['ENROLID'].values, 'AVG_DIST_TO_K':similarities_sex2_avg})
    dist_to_anchor_oud_no_sex2['MATCHED'] = 0

    # Sort A
    dist_to_anchor_oud_no_sex2.sort_values('AVG_DIST_TO_K', inplace=True)

    line_counter = 0
    for index, row in demog_oud_yes_data_sex_2.iterrows(): 
        if line_counter%10000==0:
            print(line_counter)
            print("--- %s seconds ---" % (time.time() - start_time))
        line_counter +=1
    
        #  Find the average cosine similarities, x_i, between this patient P_i to the K oud_yes anchor samples.
        similarities_oud_yes_to_anchor = cdist(np.reshape(row[match_based_on].values, (1,len(match_based_on))), np.reshape(cluster_centers.values, (-1,len(match_based_on))), metric='euclidean')
        avg_sim_oud_yes_to_anch = np.mean(similarities_oud_yes_to_anchor)
        
        # Use binary search to find the closests positions
        matched_oud_no_samples = find_closest_sorted(dist_to_anchor_oud_no_sex2[dist_to_anchor_oud_no_sex2['MATCHED'] == 0], avg_sim_oud_yes_to_anch, pos_to_negs_ratio)
        if len(matched_oud_no_samples) == 0:
            pdb.set_trace()        
        dist_to_anchor_oud_no_sex2.loc[matched_oud_no_samples.index, 'MATCHED'] = 1

    print('-------- Finished matching-----------')    
    print("--- %s seconds ---" % (time.time() - start_time))  
    #pdb.set_trace()
    matched_oud_no_all = pd.concat([dist_to_anchor_oud_no_sex2[dist_to_anchor_oud_no_sex2['MATCHED']==1], dist_to_anchor_oud_no_sex1[dist_to_anchor_oud_no_sex1['MATCHED']==1]])
    dist_to_anchor_oud_no_sex1[dist_to_anchor_oud_no_sex1['MATCHED']==1].to_csv(demogs_oud_no_path[:-4]+'_matched_SEX1.csv', index=False)
    dist_to_anchor_oud_no_sex2[dist_to_anchor_oud_no_sex2['MATCHED']==1].to_csv(demogs_oud_no_path[:-4]+'_matched_SEX2.csv', index=False)

    demog_oud_no_data[demog_oud_no_data.ENROLID.isin(matched_oud_no_all.ENROLID)].to_csv(demogs_oud_no_path[:-4]+'_matched.csv', index=False)    
    return demogs_oud_no_path[:-4]+'_matched.csv'

def cohort_matching( demogs_oud_yes_path
                    , demogs_oud_no_path
                    , pos_to_negs_ratio
                    ):
    #pdb.set_trace()
    match_based_on = ['DOB', 'NUM_MONTHLY_OPIOID_PRESCS', 'NUM_MONTHS_IN_DATA']
    demogs_oud_yes = pd.read_csv(demogs_oud_yes_path)
    demogs_oud_yes.columns = demogs_oud_yes.columns.str.replace("'",'')
    demogs_oud_no = pd.read_csv(demogs_oud_no_path)
    demogs_oud_no.columns = demogs_oud_no.columns.str.replace("'",'')
    demogs_oud_no['MATCHED'] = 0
    matched_negatives_enrolids = list()
    line_counter = 0 
    # with open('outputs/similarity_matrix.csv')
    for index, row in demogs_oud_yes.iterrows():
        # print(line_counter)
        if line_counter % 1000 == 0:
            print('Finished matching oud_no samples to {} oud_yes patients out of {} oud_yes patients. The ratio is {}.'.format(line_counter, len(demogs_oud_yes), pos_to_negs_ratio))
        line_counter +=1    
        oud_no_data_filtered_sex = demogs_oud_no[(demogs_oud_no['SEX'] == row['SEX']) & (demogs_oud_no['MATCHED'] == 0)]
        if len(oud_no_data_filtered_sex) == 0:
            # if len(demogs_oud_no[demogs_oud_no['MATCHED'] == 0]) == 0:
                # pdb.set_trace()
            # print('Warning: could not find a match based on patients sex') 
            similarities = cdist(np.reshape(row[match_based_on].values, (1,len(match_based_on))), np.reshape(demogs_oud_no[demogs_oud_no['MATCHED'] == 0][match_based_on].values, (-1,len(match_based_on))), metric='cosine')
            matched_negs = demogs_oud_no[demogs_oud_no['MATCHED'] == 0].iloc[similarities[0].argsort()[:pos_to_negs_ratio]]
            demogs_oud_no.loc[demogs_oud_no['ENROLID'].isin(matched_negs['ENROLID'].values),'MATCHED'] = 1
        else:
            similarities = cdist(np.reshape(row[match_based_on].values, (1,len(match_based_on))), np.reshape(oud_no_data_filtered_sex[match_based_on].values, (-1,len(match_based_on))), metric='cosine')
            matched_negs = oud_no_data_filtered_sex.iloc[similarities[0].argsort()[:pos_to_negs_ratio]]
            demogs_oud_no.loc[demogs_oud_no['ENROLID'].isin(matched_negs['ENROLID'].values),'MATCHED'] = 1
    # pdb.set_trace()
    demogs_oud_no[demogs_oud_no['MATCHED']==1].loc[:, demogs_oud_no.columns != 'MATCHED'].to_csv(demogs_oud_no_path[:-4]+'_matched.csv', index=False)
    return demogs_oud_no_path[:-4]+'_matched.csv'

def blind_data(line_meds_splitted
            , line_diags_splitted
            , line_procs_splitted
            , line_demogs_splitted
            , prediction_win_size
            , enrolid
            , date_idx = 5
            , label_idx = 6):
    # pdb.set_trace()
    line_meds_blinded = [enrolid]
    line_diags_blinded = [enrolid]
    line_procs_blinded = [enrolid]

    line_meds_splitted = [list(y) for x, y in itertools.groupby(line_meds_splitted[1:], lambda z: z == 'EOV') if not x]            
    line_diags_splitted = [list(y) for x, y in itertools.groupby(line_diags_splitted[1:], lambda z: z == 'EOV') if not x]            
    line_procs_splitted = [list(y) for x, y in itertools.groupby(line_procs_splitted[1:], lambda z: z == 'EOV') if not x]                

    for i in range(len(line_meds_splitted)):
        current_date = int(line_meds_splitted[i][0]) 
        if 'DIAGNOSES_DATE' in line_demogs_splitted.columns:
            diag_or_last_data = int(line_demogs_splitted['DIAGNOSES_DATE'])
        elif 'LAST_RECORD_DATE' in line_demogs_splitted.columns:
            diag_or_last_data = int(line_demogs_splitted['LAST_RECORD_DATE'])
        else:
            pdb.set_trace()
            print('Warning: demographics table includes unknown column.')    
        # if line_meds_splitted[i][0] 
        diff_times = diff_month(datetime(diag_or_last_data//100,diag_or_last_data%100, 1 ), datetime(current_date//100,current_date%100, 1))
        if diff_times > prediction_win_size:
            line_meds_blinded.extend(line_meds_splitted[i])
            line_meds_blinded.extend(['EOV'])
    #  diags        
    for i in range(len(line_diags_splitted)):
        current_date = int(line_diags_splitted[i][0]) 
        if 'DIAGNOSES_DATE' in line_demogs_splitted.columns:
            diag_or_last_data = int(line_demogs_splitted['DIAGNOSES_DATE'])
        elif 'LAST_RECORD_DATE' in line_demogs_splitted.columns:
            diag_or_last_data = int(line_demogs_splitted['LAST_RECORD_DATE'])
        else:
            pdb.set_trace()
            print('Warning: demographics table includes unknown column.')    
        diff_times = diff_month(datetime(diag_or_last_data//100,diag_or_last_data%100, 1 ), datetime(current_date//100,current_date%100, 1))
        if diff_times > prediction_win_size:
            line_diags_blinded.extend(line_diags_splitted[i])
            line_diags_blinded.extend(['EOV'])

    # procs        
    for i in range(len(line_procs_splitted)):
        current_date = int(line_procs_splitted[i][0]) 
        if 'DIAGNOSES_DATE' in line_demogs_splitted.columns:
            diag_or_last_data = int(line_demogs_splitted['DIAGNOSES_DATE'])
        elif 'LAST_RECORD_DATE' in line_demogs_splitted.columns:
            diag_or_last_data = int(line_demogs_splitted['LAST_RECORD_DATE'])
        else:
            pdb.set_trace()
            print('Warning: demographics table includes unknown column.')    
        diff_times = diff_month(datetime(diag_or_last_data//100,diag_or_last_data%100, 1 ), datetime(current_date//100,current_date%100, 1))
        if diff_times > prediction_win_size:
            line_procs_blinded.extend(line_procs_splitted[i])
            line_procs_blinded.extend(['EOV'])

    return line_meds_blinded, line_diags_blinded, line_procs_blinded

def split_train_validation_test(meds_oud_yes_path
                                   , diags_oud_yes_path
                                   , procs_oud_yes_path
                                   , demogs_oud_yes_path
                                   , meds_oud_no_path
                                   , diags_oud_no_path
                                   , procs_oud_no_path
                                   , demogs_oud_no_path
                                   , train_ratio
                                   , validation_ratio
                                   , prediction_win_size
                                   , display_step):
    
    oud_yes_demographics = pd.read_csv(demogs_oud_yes_path)
    oud_no_demographics = pd.read_csv(demogs_oud_no_path)
    # Remove existing files
    # if matched ==1:
    #pdb.set_trace()
    #     demogs_oud_no_path = demogs_oud_no_path[:-4]+'_matched.csv'
    #     print('You are using matched negative cohort: {}'.format(demogs_oud_no_path))
    remove_existing_train_val_test()
    enrolid_ind = 0
    
    # Positive samples   
    with open(meds_oud_yes_path) as medications_oud_yes_file, open(diags_oud_yes_path) as diagnoses_oud_yes_file, open(procs_oud_yes_path) as procedures_oud_yes_file, open(demogs_oud_yes_path) as demographics_oud_yes_file, open('outputs/train_medications.csv', 'a') as train_meds_file, open('outputs/train_diagnoses.csv', 'a') as train_diags_file, open('outputs/train_procedures.csv', 'a') as train_procs_file, open('outputs/train_demographics.csv', 'a') as train_demogs_file, open('outputs/train_labels.csv','a') as train_labels_file, open('outputs/validation_medications.csv', 'a') as valid_meds_file, open('outputs/validation_diagnoses.csv', 'a') as valid_diags_file, open('outputs/validation_procedures.csv', 'a') as valid_procs_file, open('outputs/validation_demographics.csv', 'a') as valid_demogs_file, open('outputs/validation_labels.csv','a') as valid_labels_file, open('outputs/test_medications.csv', 'a') as test_meds_file, open('outputs/test_diagnoses.csv', 'a') as test_diags_file, open('outputs/test_procedures.csv', 'a') as test_procs_file, open('outputs/test_demographics.csv', 'a') as test_demogs_file, open('outputs/test_labels.csv','a') as test_labels_file:
        demogs_header = next(demographics_oud_yes_file)
        # pdb.set_trace()
        train_demogs_file.write(demogs_header.replace('\n','').replace("'","") + '_OR_last_record_Date'+', Label')
        train_demogs_file.write('\n')
        valid_demogs_file.write(demogs_header.replace('\n','').replace("'","") + '_OR_last_record_Date'+', Label')
        valid_demogs_file.write('\n')
        test_demogs_file.write(demogs_header.replace('\n','').replace("'","") + '_OR_last_record_Date'+', Label')
        test_demogs_file.write('\n')

        line_counter = 0 
        for line_meds in medications_oud_yes_file:
            if line_counter % display_step == 0:
                print('Finished analyzing {} oud_yes patients data'.format(line_counter))
            line_counter +=1
            line_meds_splitted = line_meds.split(',')
            line_meds_splitted = [i.replace("'","") for i in line_meds_splitted]

            line_diags = diagnoses_oud_yes_file.readline()   
            line_diags_splitted = line_diags.split(',')
            line_diags_splitted = [i.replace("'","") for i in line_diags_splitted]

            line_procs = procedures_oud_yes_file.readline()   
            line_procs_splitted=line_procs.split(',')
            line_procs_splitted = [i.replace("'","") for i in line_procs_splitted]

            # line_demogs = demographics_oud_yes_file.readline().rstrip('\n')   
            # line_demogs_splitted = line_demogs.split(',')
            # line_demogs_splitted = [i.replace("'","") for i in line_demogs_splitted]
            # line_demogs_splitted.append('1')
            
            # Check if all the streams belong to the same patient
            if not(float(line_meds_splitted[enrolid_ind].replace("'",'')) == float(line_diags_splitted[enrolid_ind].replace("'",'')) == float(line_procs_splitted[enrolid_ind].replace("'",''))):
               pdb.set_trace()
               print("Warning: current streams don't match!")
            current_enrolid = float(line_meds_splitted[enrolid_ind].replace("'",''))            
            # pdb.set_trace()
            current_demographics = oud_yes_demographics[oud_yes_demographics['ENROLID'] == current_enrolid]
            if current_demographics.empty == True:
                continue
            if len(current_demographics) > 1:
                print('Warning: there are more than 1 record in the demographics')
                pdb.set_trace()            
            rand_temp=random.randint(1,100)

            # Blind the data before writing in into to train, validation and test
            line_meds_blinded, line_diags_blinded, line_procs_blinded = blind_data(line_meds_splitted
                                                                                    , line_diags_splitted
                                                                                    , line_procs_splitted
                                                                                    , current_demographics
                                                                                    , prediction_win_size
                                                                                    , current_enrolid)    


            if line_diags_blinded[-1] == "'EOV\\n'\n'":
                pdb.set_trace()
            # num_oud_in_val = 0
            if (rand_temp > train_ratio *100) and (rand_temp <= (train_ratio+validation_ratio) * 100):     
                # num_oud_in_val+=1
                valid_meds_file.write((','.join(map(repr, line_meds_blinded))).replace("'","").replace('\n',''))
                valid_meds_file.write('\n')
                valid_diags_file.write((','.join(map(repr, line_diags_blinded))).replace("'","").replace('\n',''))
                valid_diags_file.write('\n')  
                valid_procs_file.write((','.join(map(repr, line_procs_blinded))).replace("'","").replace('\n',''))
                valid_procs_file.write('\n')  
                valid_demogs_file.write((','.join(map(repr, current_demographics.values[0]))).replace("'","").replace('\n',''))
                # 1 means oud-yes patient
                valid_demogs_file.write(', 1')
                valid_demogs_file.write('\n')
                valid_labels_file.write(str(current_enrolid))
                valid_labels_file.write(',')
                valid_labels_file.write('1, 0')
                valid_labels_file.write('\n')
            elif rand_temp > (train_ratio+validation_ratio) * 100:
                #======== Testing set
                # num_oud_in_test+=1
                test_meds_file.write((','.join(map(repr, line_meds_blinded))).replace("'","").replace('\n',''))
                test_meds_file.write('\n')
                test_diags_file.write((','.join(map(repr, line_diags_blinded))).replace("'","").replace('\n',''))
                test_diags_file.write('\n')  
                test_procs_file.write((','.join(map(repr, line_procs_blinded))).replace("'","").replace('\n',''))
                test_procs_file.write('\n')  
                test_demogs_file.write((','.join(map(repr, current_demographics.values[0]))).replace("'","").replace('\n',''))
                # 1 means oud-yes patient
                test_demogs_file.write(', 1')
                test_demogs_file.write('\n')
                test_labels_file.write(str(current_enrolid))
                test_labels_file.write(',')
                test_labels_file.write('1, 0')
                test_labels_file.write('\n')
            else:
                #======== train set
                # num_oud_in_train+=1
                train_meds_file.write((','.join(map(repr, line_meds_blinded))).replace("'","").replace('\n',''))
                train_meds_file.write('\n')
                train_diags_file.write((','.join(map(repr, line_diags_blinded))).replace("'","").replace('\n',''))
                train_diags_file.write('\n')  
                train_procs_file.write((','.join(map(repr, line_procs_blinded))).replace("'","").replace('\n',''))
                train_procs_file.write('\n')  
                train_demogs_file.write((','.join(map(repr, current_demographics.values[0]))).replace("'","").replace('\n',''))
                # 1 means oud-yes patient
                train_demogs_file.write(', 1')
                train_demogs_file.write('\n')
                train_labels_file.write(str(current_enrolid))
                train_labels_file.write(',')
                train_labels_file.write('1, 0')
                train_labels_file.write('\n')


    # Negative samples   
    with open(meds_oud_no_path) as medications_oud_no_file, open(diags_oud_no_path) as diagnoses_oud_no_file, open(procs_oud_no_path) as procedures_oud_no_file, open(demogs_oud_no_path) as demographics_oud_no_file, open('outputs/train_medications.csv', 'a') as train_meds_file, open('outputs/train_diagnoses.csv', 'a') as train_diags_file, open('outputs/train_procedures.csv', 'a') as train_procs_file, open('outputs/train_demographics.csv', 'a') as train_demogs_file, open('outputs/train_labels.csv','a') as train_labels_file, open('outputs/validation_medications.csv', 'a') as valid_meds_file, open('outputs/validation_diagnoses.csv', 'a') as valid_diags_file, open('outputs/validation_procedures.csv', 'a') as valid_procs_file, open('outputs/validation_demographics.csv', 'a') as valid_demogs_file, open('outputs/validation_labels.csv','a') as valid_labels_file, open('outputs/test_medications.csv', 'a') as test_meds_file, open('outputs/test_diagnoses.csv', 'a') as test_diags_file, open('outputs/test_procedures.csv', 'a') as test_procs_file, open('outputs/test_demographics.csv', 'a') as test_demogs_file, open('outputs/test_labels.csv','a') as test_labels_file:
        demogs_oud_no_header = next(demographics_oud_no_file)
        # pdb.set_trace()
        # demog_read_flag = True
        line_counter = 0
        for line_meds in medications_oud_no_file:
            if line_counter % display_step == 0:
                print('Finished analyzing {} oud_no patients data'.format(line_counter))
            line_counter +=1            
            line_meds_splitted = line_meds.split(',')
            line_meds_splitted = [i.replace("'","") for i in line_meds_splitted]

            line_diags = diagnoses_oud_no_file.readline()   
            line_diags_splitted = line_diags.split(',')
            line_diags_splitted = [i.replace("'","") for i in line_diags_splitted]

            line_procs = procedures_oud_no_file.readline()   
            line_procs_splitted=line_procs.split(',')
            line_procs_splitted = [i.replace("'","") for i in line_procs_splitted]

            # if demog_read_flag == True:
            #     line_demogs = demographics_oud_no_file.readline().rstrip('\n')   
            #     line_demogs_splitted = line_demogs.split(',')
            #     line_demogs_splitted = [i.replace("'","") for i in line_demogs_splitted]
            #     line_demogs_splitted.append('0')
            # pdb.set_trace()
            # if int(line_meds_splitted[enrolid_ind].replace("'",'')) == 33149430201:
            #     pdb.set_trace()
            if not(float(line_meds_splitted[enrolid_ind].replace("'",'')) == float(line_diags_splitted[enrolid_ind].replace("'",'')) == float(line_procs_splitted[enrolid_ind].replace("'",''))):
               pdb.set_trace()
               print("Warning: current streams don't match!")
            # if (int(line_meds_splitted[enrolid_ind].replace("'",'')) < int(line_demogs_splitted[enrolid_ind]) ):
            #     pdb.set_trace()
            #     demog_read_flag = False
            #     continue
            # else:
            #     demog_read_flag = True
            if not(float(line_meds_splitted[enrolid_ind].replace("'",'')) == float(line_diags_splitted[enrolid_ind].replace("'",'')) == float(line_procs_splitted[enrolid_ind].replace("'",''))):
               pdb.set_trace()
               print("Warning: current streams don't match!")
            # pdb.set_trace()
            current_enrolid = float(line_meds_splitted[enrolid_ind].replace("'",''))
            current_demographics = oud_no_demographics[oud_no_demographics['ENROLID'] == current_enrolid]
            if current_demographics.empty == True:
                continue            
            if len(current_demographics) > 1:
                print('Warning: there are more than 1 record in the demographics')
                pdb.set_trace()
            rand_temp=random.randint(1,100)

            line_meds_blinded, line_diags_blinded, line_procs_blinded = blind_data(line_meds_splitted
                                                                                    , line_diags_splitted
                                                                                    , line_procs_splitted
                                                                                    , current_demographics
                                                                                    , prediction_win_size
                                                                                    , current_enrolid)    


            # num_oud_in_val = 0
            if (rand_temp > train_ratio *100) and (rand_temp <= (train_ratio+validation_ratio) * 100):     
                # num_oud_in_val+=1
                valid_meds_file.write((','.join(map(repr, line_meds_blinded))).replace("'","").replace('\n',''))
                valid_meds_file.write('\n')
                valid_diags_file.write((','.join(map(repr, line_diags_blinded))).replace("'","").replace('\n',''))
                valid_diags_file.write('\n')  
                valid_procs_file.write((','.join(map(repr, line_procs_blinded))).replace("'","").replace('\n',''))
                valid_procs_file.write('\n')  
                valid_demogs_file.write((','.join(map(repr, current_demographics.values[0]))).replace("'","").replace('\n',''))
                # 0 means oud-no patient
                valid_demogs_file.write(', 0')
                valid_demogs_file.write('\n')
                valid_labels_file.write(str(current_enrolid))
                valid_labels_file.write(',')
                valid_labels_file.write('0, 1')
                valid_labels_file.write('\n')
            elif rand_temp > (train_ratio+validation_ratio) * 100:
                #======== Testing set
                # num_oud_in_test+=1
                test_meds_file.write((','.join(map(repr, line_meds_blinded))).replace("'","").replace('\n',''))
                test_meds_file.write('\n')
                test_diags_file.write((','.join(map(repr, line_diags_blinded))).replace("'","").replace('\n',''))
                test_diags_file.write('\n')  
                test_procs_file.write((','.join(map(repr, line_procs_blinded))).replace("'","").replace('\n',''))
                test_procs_file.write('\n')  
                test_demogs_file.write((','.join(map(repr, current_demographics.values[0]))).replace("'","").replace('\n',''))
                # 0 means oud-no patient
                test_demogs_file.write(', 0')
                test_demogs_file.write('\n')
                test_labels_file.write(str(current_enrolid))
                test_labels_file.write(',')
                test_labels_file.write('0, 1')
                test_labels_file.write('\n')
            else:
                #======== train set
                # num_oud_in_train+=1
                train_meds_file.write((','.join(map(repr, line_meds_blinded))).replace("'","").replace('\n',''))
                train_meds_file.write('\n')
                train_diags_file.write((','.join(map(repr, line_diags_blinded))).replace("'","").replace('\n',''))
                train_diags_file.write('\n')  
                train_procs_file.write((','.join(map(repr, line_procs_blinded))).replace("'","").replace('\n',''))
                train_procs_file.write('\n')  
                train_demogs_file.write((','.join(map(repr, current_demographics.values[0]))).replace("'","").replace('\n',''))
                # 0 means oud-no patient
                train_demogs_file.write(', 0')
                train_demogs_file.write('\n')
                train_labels_file.write(str(current_enrolid))
                train_labels_file.write(',')
                train_labels_file.write('0, 1')
                train_labels_file.write('\n')




