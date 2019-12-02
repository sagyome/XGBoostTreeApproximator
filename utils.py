"""
This module contains several functions that are used in various stages of the process
"""
import numpy as np
from sklearn.metrics import roc_curve, auc
import xgboost as xg
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import random

RANDOM_SEED = 1

def softmax(x):
    """
    This function is useful for converting the aggregated results come from the different trees into class probabilities
    :param x: Numpy k-dimensional array
    :return: Softmax of X
    """
    return np.array([np.exp(x)/np.sum(np.exp(x))])

def get_auc(test_y,y_score):
    """

    :param test_y: Labels
    :param y_score: probabilities of labels
    :return: ROC AUC score
    """
    np.random.seed(RANDOM_SEED)
    classes=[i for i in range(y_score.shape[1])]
    y_test_binarize=np.array([[1 if i ==c else 0 for c in classes] for i in test_y])
    fpr, tpr, _ = roc_curve(y_test_binarize.ravel(), y_score.ravel())
    return auc(fpr, tpr)

def train_decision_tree(train,feature_cols,label_col):
    """
    This function gets a dataframe as an input and optimizes a decision tree to the data

    :param train: Pandas dataframe
    :param feature_cols: feature column names
    :param label_col: label column name
    :return: Trained sklearn decision tree
    """
    np.random.seed(RANDOM_SEED)
    parameters = {'criterion': ['entropy', 'gini'],
                  'max_depth': [3, 5, 10, 20, 50],
                  'min_samples_leaf': [1, 2, 5, 10]}
    model = DecisionTreeClassifier()
    clfGS = GridSearchCV(model, parameters, cv=3)
    clfGS.fit(train[feature_cols].values, train[label_col])
    return clfGS.best_estimator_



def train_rf_model(train,feature_cols,label_col):
    """
        This function gets a dataframe as an input and optimizes a random forest classifier to the data

        :param train: Pandas dataframe
        :param feature_cols: feature column names
        :param label_col: label column name
        :return: Trained random forest classifier
        """
    np.random.seed(RANDOM_SEED)
    parameters = {'n_estimators':[50,100],
                  'criterion': ['entropy'],
                  'min_samples_leaf': [1, 10, 100],
                  'max_features':['auto','log2']}
    model = RandomForestClassifier()
    clfGS = GridSearchCV(model, parameters, cv=3)
    clfGS.fit(train[feature_cols].values, train[label_col])
    return clfGS.best_estimator_

def train_xgb_classifier(train,feature_cols,label_col,xgb_params):
    """
    Train an XGBoost to the input dataframe

    :param train: pandas dataframe
    :param feature_cols: feature column names
    :param label_col: label column name
    :param xgb_params: Dict of XGBoost parameters
    :return: label column namened XGboost
    """
    np.random.seed(RANDOM_SEED)
    tuning_params = {'colsample_bytree': [0.3,0.5,0.9],
                  'learning_rate': [0.01,0.1],
                  'max_depth': [2,5,10],
                  'alpha': [1,10],
                     'n_estimators':[50,100]}
    if train[label_col].nunique() > 2:
        xgb_params['objective'] = "multi:softprob"
    else:
        xgb_params['objective'] = "binary:logitraw"
    model = xg.XGBClassifier(xgb_params)
    clfGS = GridSearchCV(model, tuning_params, cv=3)
    clfGS.fit(train[feature_cols], train[label_col])
    return clfGS.best_estimator_

def decision_tree_instance_depth(inst, dt):
    """

    :param inst: Instance to be inferenced - numpy vector
    :param dt: sklearn decision tree
    :return: The depth of the leaf that corresponds the instance
    """
    indx = 0
    depth = 0
    # epsilon: thresholds may be shifted by a very small floating points. For example: x1 <= 2.6 may become x1 <= 2.5999999
    # and then x1 = 2.6 won't be captured
    epsilon = 0.0000001
    t = dt.tree_
    while t.feature[indx] >= 0:
        if inst[t.feature[indx]] <= t.threshold[indx] + epsilon:
            indx = t.children_left[indx]
        else:
            indx = t.children_right[indx]
        depth += 1
    return  depth

def decision_tree_depths(test,feature_cols,dt):
    """
    This function is used for calculatingg the prediction depths of each instance that were inferenced by the input
    decision tree

    :param test: Pandas dataframe
    :param feature_cols: feature column names
    :param dt: decision tree
    :return: the depths of leaves that were assigned to each instance
    """
    X = test[feature_cols].values
    return [decision_tree_instance_depth(inst,dt) for inst in X]

#The following are not used:

def train_xgb_classifier2(train,feature_cols,label_col,xgb_params):
    """
    Train an XGBoost to the input dataframe

    :param train: pandas dataframe
    :param feature_cols: feature column names
    :param label_col: label column name
    :param xgb_params: Dict of XGBoost parameters
    :return: label column namened XGboost
    """
    if train[label_col].nunique() > 2:
        obj = "multi:softprob"
    else:
        obj = "binary:logitraw"
    xgb_model = xg.XGBClassifier(**xgb_params)
    xgb_model.fit(train[feature_cols], train[label_col])
    return  xgb_model

def ensemble_prediction_depth(X, rf):
    depths = []
    for inst in X:
        depths.append(np.sum([tree_prediction_depth(inst,base_model.tree_) for base_model in rf.estimators_]))
    return depths

def tree_prediction_depth(inst, t):
    indx = 0
    depth = 0
    epsilon = 0.0000001
    # epsilon: thresholds may be shifted by a very small floating points. For example: x1 <= 2.6 may become x1 <= 2.5999999
    # and then x1 = 2.6 won't be captured
    while t.feature[indx] >= 0:
        if inst[t.feature[indx]] <= t.threshold[indx] + epsilon:
            indx = t.children_left[indx]
        else:
            indx = t.children_right[indx]
        depth += 1
    return depth

def get_features_statistics(data):
    min_values = {col:min(data[col]) for col in data.columns}
    max_values = {col: max(data[col]) for col in data.columns}
    mean_values = {col: np.mean(data[col]) for col in data.columns}
    return min_values, max_values, mean_values