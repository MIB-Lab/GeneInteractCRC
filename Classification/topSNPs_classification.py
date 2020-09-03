"""
File:       topSNPs_classification.py
Author:     Somayeh Kafaie
Date:       September 2018
Purpose:    To train the model for disease classification using SNPs
            selected by several methods.
To run: python topSNPs_classification.py --file_name=IG_50.csv
"""
import pandas as pd
import numpy as np
import sys
import logging
import argparse
import datetime
from sklearn.preprocessing import OneHotEncoder, scale
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from scipy.stats import randint as sp_randint
#from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

LOG_FILE_NAME = ""

def log(content, add_time=True):
    """
    """
    runtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    if (add_time):
        content = "["+runtime+"] "+content
    with open("output/"+LOG_FILE_NAME, "a+") as myfile:
        myfile.write(content+"\n")
    print(content)


def initialize_logger(file_name = 'logfile.log'):
    logging.basicConfig(level=logging.DEBUG)
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    fileHandler = logging.FileHandler(file_name)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)


def parametr_tuning_grid(model, params, scores, X_train, Y_train, X_test, Y_test):
    """
    """
    for score in scores:
        log("# Tuning hyper-parameters for %s: " % score)
        log("", False)
        #clf = GridSearchCV(model, params, cv=3, scoring='%s_macro' % score)
        clf = GridSearchCV(model, params, cv=5, scoring=score)
        clf.fit(X_train, Y_train)

        log("Best parameters set found on development set:")
        log(str(clf.best_params_), False)
        log("Grid search score _ TEST set:")
        log(str(clf.score(X_test, Y_test)*100), False)
        log("", False)
        log("Grid scores on development set:")
        log(str(clf.grid_scores_), False)
        log("", False)
        log("Detailed classification report:")
        log("", False)
        y_true, y_pred = Y_test, clf.predict(X_test)
        log(classification_report(y_true, y_pred), False)
        log("", False)

def parametr_tuning_random(model, params, scores, X_train, Y_train, X_test, Y_test, n_iter_search=10):
    """
    """
    for score in scores:
        log("# Tuning hyper-parameters for %s: " % score)
        log("", False)
        rnd_tune = RandomizedSearchCV(model, params, n_iter=n_iter_search, cv=5,
                                        scoring=score)
        rnd_tune.fit(X_train, Y_train)

        log("Best parameters set found on development set:")
        log(str(rnd_tune.best_params_), False)
        log("random search score _ TEST set:")
        log(str(rnd_tune.score(X_test, Y_test)*100), False)
        log("", False)
        log("random search scores on development set:")
        log(str(rnd_tune.grid_scores_), False)
        log("", False)
        log("Detailed classification report:")
        log("", False)
        y_true, y_pred = Y_test, rnd_tune.predict(X_test)
        log(classification_report(y_true, y_pred), False)
        log("", False)


def genshuffle(X, Y):
    #by setting random_state parameter to a fixed number, it alwayse uses the
    #same seed for random generator and the final order after siffeling will
    #be the same. We want this so that we can later use k-fold cross validation
    #to train for different combinations of test and train sets and pick the best
    return shuffle(X, Y, random_state=100)

def do_preprocessing(file_name, test_percent=0.1):
    """
    """
    data = pd.read_csv("data/"+file_name)
    #first column represents phenotypes
    Y = data.iloc[:,0]
    #next columns represent features
    X_raw = data.iloc[:,1:]

    #apply oneHot Encoding
    enc=OneHotEncoder(sparse=False)
    enc.fit(X_raw)
    X  = enc.transform(X_raw)

    #divid test and train sets
    log("constructing training/testing split...")
    (X_train, X_test, Y_train, Y_test) = train_test_split(X, Y,
        test_size=test_percent, random_state=42)

    return X_train, Y_train, X_test, Y_test

def train_randomForest(X_train, Y_train, X_test, Y_test):
    """
    """
    log("", False)
    log("----------------------------------------------------", False)
    log("------------------Random Forest---------------------", False)
    scores = ['accuracy']
    # construct the set of hyperparameters to tune
    params = {"max_depth": [2, 3],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(2, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"],
              "n_estimators":sp_randint(9, 101)}
    model = RandomForestClassifier(random_state=0)
    log("Random search for parametr tuning:")
    parametr_tuning_random(model, params, scores, X_train, Y_train, X_test, Y_test,
                            n_iter_search=20)
    log("", False)
    # use a full grid over all parameters
    param_grid = {"max_depth": [2, 3, None],
                "max_features": [1, 3, 10],
                "min_samples_split": [2, 3, 10],
                "bootstrap": [True, False],
                "criterion": ["gini", "entropy"],
                "n_estimators":[10, 30, 50, 70, 90]}
    log("Grid search for parametr tuning:")
    parametr_tuning_grid(model, param_grid, scores, X_train, Y_train, X_test, Y_test)


def train_mlp(X_train, Y_train, X_test, Y_test):
    """
    """
    log("", False)
    log("----------------------------------------------------", False)
    log("-----------------------MLP--------------------------", False)
    scores = ['accuracy']
    # construct the set of hyperparameters to tune
    params = {'hidden_layer_sizes': [(100,),(500,),(250,),(10,)],
                'solver': ('sgd','adam', 'lbfgs'), 'max_iter': [500,1000,1500],
                'learning_rate_init':[0.005,0.05,0.001, 0.01]}
    # parameters = {'solver': ['lbfgs'], 'max_iter': [500,1000,1500],
    # 'alpha': 10.0 ** -np.arange(1, 7), 'hidden_layer_sizes':np.arange(5, 12),
    # 'random_state':[0,1,2,3,4,5,6,7,8,9]}

    model = MLPClassifier(early_stopping=True)
    log("Random search for parametr tuning:")
    parametr_tuning_random(model, params, scores, X_train, Y_train, X_test, Y_test,
                            n_iter_search=40)
    log("", False)
    # use a full grid over all parameters
    log("Grid search for parametr tuning:")
    parametr_tuning_grid(model, params, scores, X_train, Y_train, X_test, Y_test)

def train_svm(X_train, Y_train, X_test, Y_test):
    """
    """
    log("", False)
    log("----------------------------------------------------", False)
    log("-----------------------SVM--------------------------", False)
    scores = ['accuracy']
    # construct the set of hyperparameters to tune
    params = {'gamma': [0.1, 1e-2, 1e-3, 1e-4, 'auto'], 'C': [1, 10, 100, 1000]}
    model = svm.SVC(probability=True)

    log("Random search for parametr tuning:")
    parametr_tuning_random(model, params, scores, X_train, Y_train, X_test, Y_test,
                            n_iter_search=20)
    log("", False)
    # use a full grid over all parameters
    log("Grid search for parametr tuning:")
    parametr_tuning_grid(model, params, scores, X_train, Y_train, X_test, Y_test)


def train_knn(X_train, Y_train, X_test, Y_test):
    """
    """
    log("", False)
    log("----------------------------------------------------", False)
    log("------------------K Nearest Neighbors---------------", False)
    #scores = ['accuracy', 'precision', 'recall']
    scores = ['accuracy']
    # construct the set of hyperparameters to tune
    params = {"n_neighbors": np.arange(1, 31, 2),
	           "metric": ['euclidean', 'cityblock']}
    model=KNeighborsClassifier(weights="distance")
    log("Random search for parametr tuning:")
    parametr_tuning_random(model, params, scores, X_train, Y_train, X_test, Y_test)
    log("", False)
    log("Grid search for parametr tuning:")
    parametr_tuning_grid(model, params, scores, X_train, Y_train, X_test, Y_test)

def train_logisticRegression2(X_train, Y_train, X_test, Y_test):
    """
    In this type, learning rate can be set
    """
    log("", False)
    log("----------------------------------------------------", False)
    log("------------------LR_SGDClassifier------------------", False)
    #scores = ['accuracy', 'precision', 'recall']
    scores = ['accuracy']
    # construct the set of hyperparameters to tune
    params = {'loss': ('log', 'hinge', 'modified_huber'),
    'penalty': ['l1', 'l2', 'elasticnet'], 'max_iter': [10, 50, 100],
    'alpha': [10 ** x for x in range(-6, -1)]}
    # Fitting a logistic regression model
    model=SGDClassifier(loss="modified_huber", penalty="l2", max_iter=10)
    log("Random search for parametr tuning:")
    parametr_tuning_random(model, params, scores, X_train, Y_train, X_test,
                            Y_test, n_iter_search=30)
    log("", False)
    log("Grid search for parametr tuning:")
    parametr_tuning_grid(model, params, scores, X_train, Y_train, X_test, Y_test)

def train_logisticRegression1(X_train, Y_train, X_test, Y_test):
    """
    """
    log("", False)
    log("----------------------------------------------------", False)
    log("-------------------LogisticRegression---------------", False)
    #scores = ['accuracy', 'precision', 'recall']
    scores = ['accuracy']
    # construct the set of hyperparameters to tune
    params = {'penalty': ['l1','l2'], 'C': [0.001,0.01,0.1,1,10,100,1000],
    'max_iter': [100, 500]}
    # Fitting a logistic regression model
    model=LogisticRegression()
    log("Random search for parametr tuning:")
    parametr_tuning_random(model, params, scores, X_train, Y_train, X_test,
                            Y_test, n_iter_search=15)
    log("", False)
    log("Grid search for parametr tuning:")
    parametr_tuning_grid(model, params, scores, X_train, Y_train, X_test, Y_test)

def apply_ml(file_name, test_percent):
    '''
    '''
    runtime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    name = str(file_name[:-4])+"_"+runtime
    global LOG_FILE_NAME
    LOG_FILE_NAME = 'logger_'+name+'.log'
    #initialize_logger(LOG_FILE_NAME)

    #logging.info("Preprosessing:")
    log("Preprosessing:")
    X_train, Y_train, X_test, Y_test = do_preprocessing(file_name, test_percent)

    #import pdb; pdb.set_trace()
    train_logisticRegression1(X_train, Y_train, X_test, Y_test)
    train_logisticRegression2(X_train, Y_train, X_test, Y_test)
    train_knn(X_train, Y_train, X_test, Y_test)
    train_mlp(X_train, Y_train, X_test, Y_test)
    train_svm(X_train, Y_train, X_test, Y_test)
    train_randomForest(X_train, Y_train, X_test, Y_test)
    #train_vote(X_train, Y_train, X_test, Y_test)

if __name__ == '__main__':
    """
    Handle running the program directly from the command line.
    It processes potential arguments entered by the user.
    """
    #files = ["IG_50.csv", "IG_100.csv", "Centrality_50.csv", "Centrality_100.csv",
    #    "Graphlet_50.csv", "Graphlet_100.csv", "union_50.csv", "union_100.csv",
    #    "surf_50.csv", "surf_100.csv", "oddRatio_50.csv", "oddRatio_100.csv"]

    files = [ "fastepi_35.csv"] #before applying filtering


    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, default='',
                       help='file-name to read the database')
    parser.add_argument('--test_percent', type=int, default=0.1,
                       help='percantage of dataset assigned to test-set')
    args = parser.parse_args()
    if (args.file_name==''):
        for file in files:
            apply_ml(file, args.test_percent)
    else:
        apply_ml(args.file_name, args.test_percent)
