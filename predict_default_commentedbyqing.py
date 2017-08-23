#!/data/apps/Anaconda2-4.3.1/bin/python

import csv
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
from sklearn import tree
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pylab as pl
import matplotlib.pyplot as plt
import pickle

def load_data(keep_vars=None):
    """
    This is the function to load the data as 'dataframe' which is similar to the data format in matlab
    keep_vars is the parameter input: if None, all the columns will be imported, if type columns 'A','B,'...will be imported
    """
    df = pd.read_csv('/Users/qing/github/example/data.csv', sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
    #
    #"this is to choose the file and load the data, file format is csv and directory of file needs be adapted "
    #
#   df = df.loc[(df['investor_type'] == '7'), ]
    #df = df.loc[(df['loan_type_mcdash'] == 'C') | (df['loan_type_mcdash'] == 'D'), ]
    df = df.loc[(df['int_type'] == '1'), ]
    'select from the table by label where int_type == 1'
    df['income_miss'] = (df['applicant_income'] == "-3")*1
    'create a ne new column income_miss'
    if keep_vars is not None:
        df = df[keep_vars]
    return df

def clean_variables(df):
    """
    This is a loop function 
    For each variable in the miss_vars, run the function fill_vars
    """
    miss_vars = ['fico_orig','ltv_ratio', 'dti_ratio', 'arm_init_rate', 'orig_year']
    for variable in miss_vars:
        df = fill_vars(variable, df)
    return df


def fill_vars(variable_name, df):
    """
    This function to fill col of input name 'variable_name' in the original dataframe
    If the variable name exist in the dataframe(table), 
    create two new columns :
    variable_miss: for the selected column, if the value is null, then return 0, if is not null, return 1
    variable_fill: for all the null value in the selected column, fill the cell with 0 and keep values in other cells.
    """
    vars = list(df.columns)
    variable_miss = variable_name + "_miss"
    variable_fill = variable_name + "_fill"
    if variable_name in vars:
        miss = pd.isnull(df[variable_name]).astype(int)
        df[variable_miss] = miss
        df[variable_fill] = df[variable_name]
        df.ix[df[variable_miss]==1, variable_fill] = 0
    return df

#*****************************************************************************************************************************
# The main function 
#*****************************************************************************************************************************
if __name__ == "__main__":
    KEEP_VARS = ['applicant_income',
                 'applicant_race_1',
                 'applicant_race_pre2004',
                 'applicant_sex',
                 'application_qtr',
                 'arm_init_rate',
                 'coapplicant_sex',
                 'cur_int_rate',
                 'document_type',
                 'dti_ratio',
                 'ever90dpd',
                 'fico_orig',
                 'investor_type',
                 'int_type',
                 'io_flg',
                 'jumbo_flg',
                 'lien_type',
                 'loan_amount',
                 'loan_purpose',
                 'loan_type_mcdash',
                 'ltv_ratio',
                 'margin_rate',
                 'mort_type',
                 'occupancy_type',
                 'orig_year',
                 'orig_amt',
                 'pp_pen_flg',
                 'optionarm_flg',
                 'prop_state',
                 'state_code',
                 'race_n',
                 'coapplicant',
                 'msa_id']

    #df = load_data(KEEP_VARS)
    df = load_data() # call the load data function and create the dataframe
    df = clean_variables(df) # call the clean_variables, all in missing value in ['fico_orig','ltv_ratio', 'dti_ratio', 'arm_init_rate', 'orig_year'] is filled with 0
    
    outcome = 'ever90dpd' # name of the colunm to store the true value which you want to predict
    int_rate = 1 # 'This is where to define a int_rate which seems affect the results'
    data = [] # a list
    #----------------------------------------------------------------------------------------------
    'categroise variables into seperate lists'
    for race in [0]:
        explanatory_vars_linear = ['applicant_income',
                                   'income_miss',
                                   'ltv_ratio_miss',
                                   'ltv_ratio_fill',
                                   'fico_orig_miss',
                                   'fico_orig_fill',
                                   'cur_int_rate',
                                   'orig_amt']
        explanatory_vars_nonlinear = ['cur_int_rate',
                                      'orig_amt']

        dummy_vars_linear = ['document_type', 'occupancy_type', 'jumbo_flg',
                             'prod_type', 'msa_id', 'investor_type',
                             'loan_purpose', 'coapplicant', 'term_nmon']

        dummy_vars_nonlinear = ['document_type', 'occupancy_type', 'jumbo_flg', 'prod_type',
                                'msa_id', 'investor_type', 'loan_purpose', 'coapplicant', 'term_nmon',
                                'fico_bin', 'ltv_bin', 'ltv_80', 'income_bin']
        #---------------------------------------------------------------------------------------------
        'Create seperate tables for types of variables:  df2 is outcomes + vars; df3 is vars, y_linear/y_nonlinear is outcomes'
        df2_linear = df[[outcome] + explanatory_vars_linear].dropna() # dropna values
        df2_nonlinear = df[[outcome] + explanatory_vars_nonlinear].dropna() # dropna values
        df2_linear = df2_linear.apply(pd.to_numeric) # make the data numeric format 
        df2_nonlinear = df2_nonlinear.apply(pd.to_numeric) # make the data numeric format
        df3_linear = df2_linear[explanatory_vars_linear]
        df3_nonlinear = df2_nonlinear[explanatory_vars_nonlinear]
        df3_linear.loc[:,'log_orig_amt'] = np.log(df3_linear.loc[:,'orig_amt'])
        df3_nonlinear.loc[:,'log_orig_amt'] = np.log(df3_linear.loc[:,'orig_amt'])
        y_linear = df2_linear[outcome] # results/true y  
        y_nonlinear = df2_nonlinear[outcome] # results/true y  
        #----------------------------------------------------------------------------------------------
        "Here if we want to include the race as a dummy variable"
        if race == 1:
            dummy_vars_linear = dummy_vars_linear + ['race']
            dummy_vars_nonlinear = dummy_vars_nonlinear + ['race']
        for variable in dummy_vars_linear: 
            #'create two dummy columns with variale_dum_0 and variable_dum_1'
            prefix = variable + "_dum"
            dummy_ranks = pd.get_dummies(df[variable], prefix=prefix)
            df3_linear = df3_linear.join(dummy_ranks.ix[:, 1:])
        for variable in dummy_vars_nonlinear:
            prefix = variable + "_dum"
            dummy_ranks = pd.get_dummies(df[variable], prefix=prefix)
            df3_nonlinear = df3_nonlinear.join(dummy_ranks.ix[:, 1:])
        
        #----------------------------------------------------------------------------------------------
        'Split the data into train and test data for linear and nonlinear'
        X_train, X_test, y_train, y_test = train_test_split(df3_linear, y_linear,
                                                            test_size=0.33, random_state=42)
        X_train2, X_test2, y_train2, y_test2 = train_test_split(df3_nonlinear, y_nonlinear,
                                                                test_size=0.33, random_state=42)
        #predicted_vals = pd.DataFrame(data = {"Default" : y_test, "InterestRate" : X_test['cur_int_rate'], "LTV" : X_test['ltv_ratio_fill']})
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #'This is the prediction part'
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        predicted_vals_all = pd.DataFrame() # Create a new dataframe
        if int_rate == 0:
            del X_test['cur_int_rate']
            del X_train['cur_int_rate']
            del X_test2['cur_int_rate']
            del X_train2['cur_int_rate']
        # if int_rate ==0 some ['cur_int_rate'] will not be uses in the regression 
        #--------------------------------------------------------------------------------------------------------------
        # Create the container to store the results 
        # dict() is the {key:value} 
        #--------------------------------------------------------------------------------------------------------------
        seed = 7
        fpr= dict()
        tpr = dict()
        output = dict()
        models = dict()
        cv = True # if cross validate or not 
        cv_logit = list() 
        weights = np.ones(np.size(y_train))* (np.size(y_train) /float(np.sum(y_train)))* (y_train == 1)
        weights = weights + np.ones(np.size(y_train))* (y_train == 0)
        #--------------------------------------------------------------------------------------------------------------
        # Cross-Validate Decision Tree on min_samples_leaf,
        # Two models are used: tree and forest, cross validation se is 3 
        # Search the best param in the combination of min_sample_split and min_sample_leaf
        # model params in the clf_tree and clf_fores
        # This part is using the linear vars
        #--------------------------------------------------------------------------------------------------------------
        if cv:
            param_grid = {
                'min_samples_split': [10,  100 ],
                'min_samples_leaf':  [1, 2, 10]
                }
            #--------------------------------------------------------------------------------------------------------------
            clf = DecisionTreeClassifier(random_state=seed,
                                         class_weight="balanced") #'here is to define a classifier model: tree'
            print "Decision Tree CV"
            #clf = CalibratedClassifierCV(clf, cv=3, method='isotonic'),
            grid_clf = GridSearchCV(clf, param_grid, cv=3, verbose=1, n_jobs = 40) #'here is to pass the possible parameters to the model'
            grid_clf.fit(X_train, y_train) # train the model with the train data 
            clf_tree = grid_clf.best_estimator_ # store the model params in the clf_tree
            #---------------------------------------------------------------------------------------------------------------
            clf = RandomForestClassifier(random_state=seed,
                                         class_weight="balanced",
                                         n_estimators=400) #'here is to define a classifier model: random forest'
            grid_clf = GridSearchCV(clf, param_grid, cv=3, verbose=1, n_jobs = 40) #'here is to pass the possible parameters to the model'

            print "Forest CV"
            grid_clf.fit(X_train, y_train) # train the model with the train data 
            clf_forest = grid_clf.best_estimator_  # store the model params in the clf_forest
            param_grid = {
                'n_estimators' : [40,80],
                'base_estimator__max_depth' : [20, 50, 80],
                'base_estimator__min_samples_leaf': [2,5,10]
                } #'here he includes one more parameter space: n_estimators, he does not perform it yet in this script'
            
            # clf = AdaBoostClassifier(random_state=seed,
            #                          base_estimator = clf_tree,
            #                          n_estimators = 100)
            # grid_clf = GridSearchCV(clf, param_grid, cv=3, verbose=2, n_jobs = -1)
            # print "Adaboost CV"
            # grid_clf.fit(X_train, y_train)
            # clf_ada = grid_clf.best_estimator_
            
        # If do not do cross validation 
        # Two models are used Ramdonforest and trees 
        # Two trees are used: different paramters 
        else:
            clf_forest = RandomForestClassifier(n_estimators=300,
                                                max_depth=None,
                                                min_samples_split=25,
                                                min_samples_leaf = 2,
                                                random_state=seed,
                                                class_weight = "balanced",
                                                verbose = 1,
                                                n_jobs=-1)
            clf_tree = DecisionTreeClassifier(random_state=seed,
                                              max_depth=None,
                                              min_samples_split=25,
                                              min_samples_leaf = 2,
                                              class_weight = "balanced" )
            clf_tree2 = DecisionTreeClassifier(random_state=seed,
                                              max_depth=10,
                                              min_samples_split=500,
                                              min_samples_leaf = 50,
                                              class_weight = "balanced" )
            #clf_ada = AdaBoostClassifier(random_state=seed,
            #                             base_estimator = clf_tree2)
        #---------------------------------------------------------------------------------------------------------------
        # Run the model, doing the prediction
        # Store the results in dic 
        # fpr['name of a model'] false positive rate 
        # tpr['name of a model'] true positive rate
        #
        #---------------------------------------------------------------------------------------------------------------
        names =  ["Logit",
                  "LogitNonLinear",
                  "DecisionTreeIsotonic",
                  "RandomForestIsotonic"
                  #"AdaBoostIsotonic"
                  ]

        classifiers = [LogisticRegression(C=1e12, tol=1e-6),
                       LogisticRegression(C=1e12, tol=1e-6),
                       CalibratedClassifierCV(clf_tree, cv=2,
                                              method='isotonic'),
                       CalibratedClassifierCV(clf_forest, cv=2,
                                              method='isotonic')
                       #CalibratedClassifierCV(clf_ada, cv=3,
                       #                        method='isotonic')
                       ]
        for name, clf in zip(names, classifiers):
            if name == "LogitNonLinear":
                clf.fit(X_train2, y_train2) # only nonlinear vars perform this regression
                result = pd.DataFrame(data=clf.predict_proba(X_test2)) # predict results as a dataframe using test data
                result_all = pd.DataFrame(data=clf.predict_proba(df3_nonlinear)) # predict results as a dataframe using all data
                fpr[name], tpr[name], _ = roc_curve(y_test, result[1]) # Compute Receiver operating characteristic (ROC)
                fpr[name], tpr[name], _ = precision_recall_curve(y_test, result[1]) #Compute precision-recall pairs for different probability thresholds using test data this overwrite the roc_curve 
                #output[name] =  metrics.roc_auc_score(y_test, result[1])
                output[name] = average_precision_score(y_test, result[1]) #Compute average precision (AP) from prediction scores using test data
                print("%s, score: %f " % (name, output[name]))
            else:
                clf.fit(X_train, y_train)

                result = pd.DataFrame(data=clf.predict_proba(X_test))
                result_all = pd.DataFrame (data=clf.predict_proba(df3_linear))

                test_vals = pd.concat([y_test.reset_index(drop=True), X_test.reset_index(drop=True)], axis=1, ignore_index=True)
                test_vals.columns = ["Default"] + list(X_test.columns)
                #test_vals2 = resample(test_vals[test_vals["Default"]==1], replace=True, n_samples= np.shape(test_vals[test_vals["Default"]== 0])[0])
                #test_vals = test_vals.append(test_vals2, ignore_index=True)
                results_roc = pd.DataFrame(data=clf.predict_proba(test_vals.iloc[:,1:]))
                #-------------------------------------------------------------------------------------------------------------
                # Evaluating the results
                #-------------------------------------------------------------------------------------------------------------
                fpr[name], tpr[name], _ = roc_curve(test_vals.iloc[:,0], results_roc[1]) #Compute Receiver operating characteristic 
                fpr[name], tpr[name], _ = precision_recall_curve(test_vals.iloc[:,0], results_roc[1])
                #output[name] = metrics.roc_auc_score(test_vals.iloc[:,0], results_roc[1])
                output[name] = average_precision_score(y_test, result[1])
                print("%s, score: %f " % (name, output[name]))

            #predicted_vals = pd.concat([predicted_vals.reset_index(drop=True), result[1].reset_index(drop=True)], axis=1, ignore_index=True)
            #-------------------------------------------------------------------------------------------------------------
            predicted_vals_all = pd.concat([predicted_vals_all.reset_index(drop=True), result_all[1].reset_index(drop=True)], axis=1, ignore_index=True) # store all the results in predicted_vals_all for data for all the results 
            
            # with open("../output/%s_race%d_interestrate%d.pkl"
            #           % (name, race, int_rate), 'wb') as f:
            #     pickle.dump(clf,f)
            # with open("../output/median_input_%s_race%d_interestrate%d.pkl"
            #            % (name, race, int_rate), 'wb') as f:
            #     if name == "LogitNonLinear":
            #         pickle.dump(X_test2.median(),f)
            #     else:
            #         pickle.dump(X_test.median(),f)


        data.append(output)
        # data is a list, every element is a dict, the key of the dict is the name of a model
        # ["Logit", "LogitNonLinear", "DecisionTreeIsotonic", "RandomForestIsotonic", #"AdaBoostIsotonic"(is commented by the author)]
        # value for each key is the score for the model 
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #'Plotting the results'
        # 'Each model has a roc curve 
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        plt.figure(1)
        for name in names:
            plt.plot(tpr[name], fpr[name],
                     label='%s ROC curve' % name)
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            # plt.plot(fpr[name], tpr[name],
            #          label='%s ROC curve' % name)
            # plt.xlim([0, 1])
            # plt.ylim([0, 1])
            # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')

        #plt.title('Receiver operating characteristic example')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="upper right")
        #plt.show()
        plt.savefig("../output/roc_auc_big_race%d_interestrate%d.pdf" % (race,int_rate), bbox_inches='tight')
        plt.close(1)
        
        #-------------------------------------------------------------------------------------------------------------
        # To illustrate which true y is used for cross validation, the indicated one is the y_test data used to evaluate the model 
        #-------------------------------------------------------------------------------------------------------------
        cv_indicator = pd.DataFrame(y_linear).merge(pd.DataFrame(y_test), left_index=True, right_index=True, how='left', indicator=True)
        cv_indicator = cv_indicator[['ever90dpd_x', '_merge']]
        #predicted_vals_all = pd.concat([cv_indicator.reset_index(drop=True), df3_linear.reset_index(drop=True), predicted_vals_all.reset_index(drop=True)], axis=1, ignore_index=True)
        # predicted_vals.columns = ["Default", "IntRate", "LTV"] + names
        # predicted_vals.to_csv("../output/vals_race%d_interestrate%d.csv" % (race,int_rate))
        #predicted_vals_all.columns = ["Default", "Test"] +   list(df3_linear.columns) + list(names)
        #predicted_vals_all.to_csv("../output/all_vals_race%d_interestrate%d.csv" % (race,int_rate))

#-------------------------------------------------------------------------------------------------------
#"This is to save the results with different int_rate, dataframe: 'data' is the predicted results"
# data is a list, every element is a dict, the key of the dict is the name of a model
# ["Logit", "LogitNonLinear", "DecisionTreeIsotonic", "RandomForestIsotonic", #"AdaBoostIsotonic"(is commented by the author)]
# value for each key is the score for the model 
#-------------------------------------------------------------------------------------------------------
with open("../output/scores_interestrate%d.csv" % (int_rate), 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(data[0].keys())
    for row in data:
        writer.writerow(row.values())
