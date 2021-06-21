from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import os
import numpy as np
import pdb
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import csv
import pandas as pd



def read_stationary_data(train_data_path
                        , validation_data_path
                        , test_data_path):
    train_data_stationary = pd.read_csv(train_data_path)
    validation_data_stationary = pd.read_csv(validation_data_path)
    test_data_stationary = pd.read_csv(test_data_path)

    return train_data_stationary, validation_data_stationary, test_data_stationary

def performance_evaluation(rf_predictions
                        ,  labels):
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

    return tn, tp, fn, fp, accuracy, precision, recall, specificity, F1    

def write_results(tn
                , tp
                , fn
                , fp
                , accuracy
                , precision
                , recall
                , F1
                , model
                    ):
    with open('results/classical_ml_models/'+model+'_prediction_performance.csv', 'w') as f_results:
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
def random_forest_model(train_data_path
                        , validation_data_path
                        , test_data_path
                        ):

    pdb.set_trace()
    train_data = pd.read_csv(train_data_path)
    validation_data = pd.read_csv(validation_data_path)
    test_data = pd.read_csv(test_data_path)
    training_all = pd.concat([train_data, validation_data], axis=0, ignore_index = True)
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    hyperparameters = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    with open('saved_classical_ml_models/rf_hyperparameters.csv', 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in hyperparameters.items():
           writer.writerow([key, value])
    pdb.set_trace()       
    randomCV = RandomizedSearchCV(estimator=RandomForestClassifier(verbose=1), param_distributions=hyperparameters, n_iter=2, cv=3,scoring="f1")
    randomCV.fit(training_all.iloc[:,1:-1], training_all['Label'])

    best_rf_model= randomCV.best_estimator_
    feat_importances = pd.Series(randomCV.best_estimator_.feature_importances_, index=train_header.replace('\n','').split(',')[1:-1])

    fig = feat_importances.nlargest(len(training_all.columns)-2).plot(kind='barh', grid=True,figsize=(12,10))
    fig.set_xlabel("Importance Score")
    fig.set_ylabel("Features")
    fig.get_figure().savefig("results/classical_ml_models/feature_importance.png", dpi=300)

    rf_predictions = best_rf_model.predict(test_data.iloc[:,1:-1])

    rf_test_auc=roc_auc_score(test_data['Label'], best_rf_model.predict_proba(test_data.iloc[:,1:-1])[:,1])

    tn, tp, fn, fp, accuracy, precision, recall, specificity, F1 = performance_evaluation(rf_predictions
                                                                            , test_data['Label']
                                                                            )   



