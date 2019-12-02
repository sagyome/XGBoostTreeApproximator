""" Read datasets

This script contain all the functions that enable reading datasets for experiments

Each get_xxx function returns train set, test set, names of the feature columns and label column.
Some of the datasets available from sklearn while others are read from files that are stored in \datasets
In some cases required data cleansing and processing is conducted

"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes, load_wine
from sklearn.feature_extraction import DictVectorizer
import calendar
month_dict = dict((v,k) for k,v in enumerate(calendar.month_abbr))
day_dict = dict((v,k) for k,v in enumerate(calendar.day_name))

def get_sklearn_ds(data,train_test_ratio):
    """
    This function read, split and process datasets that come from sklearn

    :param data: sklearn.utils.Bunch object that rperesents the dataset
    :param train_test_ratio: The proportion on training instances
    :return: train data and test data (pandas dataframes), names of feature columns and label column
    """
    feature_cols, label_col = data.feature_names, 'class'
    X, y = data.data, data.target
    data = pd.DataFrame(X, columns=data.feature_names)
    data['class'] = y
    train = data.sample(frac=train_test_ratio)
    test = data.drop(train.index)
    return train, test, feature_cols, label_col

def dummify_data(data,feature_cols, label_col):
    """
    This function identify categorical features and convert them into multiple binary features (one for each category)

    :param data: pandas dataframe
    :param feature_cols: feature names
    :param label_col: label column name
    :return: pandas dataframe with dummy features, names of the new features
    """

    #The following line identifies the categorical features in the dataset
    char_cols = data[feature_cols].dtypes.pipe(lambda x: x[x == 'object']).index

    #convert the categorical features to multiple dummy variables and add them to the dataframe
    for col in char_cols:
        dummies = pd.get_dummies(data[col])
        dummies.columns = [col + '=' + category for category in dummies.columns]
        data = pd.concat([data, dummies], axis=1)

    #Remove the original categorical features from the dataframe
    data = data.drop(char_cols,axis=1)
    feature_cols = [col for col in data.columns if col != label_col]
    return data, feature_cols

def data_processing(data,train_test_ratio,label_col,random_state):
    """
    This function changes the label names as enumerates and divide data to train-test

    :param data: pandas dataframe
    :param train_test_ratio: proportion of training data
    :param label_col: label column name
    :return: train and test sets as pandas dataframes
    """
    labels_dict = {v: k for k, v in enumerate(data[label_col].unique())}
    data[label_col] = [labels_dict[i] for i in data[label_col]]
    train = data.sample(frac=train_test_ratio,random_state=random_state)
    test = data.drop(train.index)
    return train, test

def get_iris_data(random_state, train_test_ratio=0.8):
    return get_sklearn_ds(load_iris(),train_test_ratio)

def get_breast_cancer_data(random_state, train_test_ratio=0.8):
    return get_sklearn_ds(load_breast_cancer(), train_test_ratio)

def get_winery_data(random_state, train_test_ratio=0.8):
    train, test, feature_cols, label_col = get_sklearn_ds(load_wine(), train_test_ratio)
    train['magnesium'], test['magnesium'] = train['magnesium'].astype(int), test['magnesium'].astype(int)
    return train, test, feature_cols, label_col

def get_diabetes(random_state, train_test_ratio=0.8):
    return get_sklearn_ds(load_diabetes(),train_test_ratio)

def get_kohkilyeh(random_state, train_test_ratio=0.8):
    data = pd.read_csv('datasets/kohkiloyeh.csv')
    label_col = 'pb'
    data,feature_cols = dummify_data(data,data.columns[:-1],label_col)
    train, test = data_processing(data, train_test_ratio, label_col,random_state)
    return train, test, feature_cols, label_col

def get_haberman(random_state, train_test_ratio=0.8):
    feature_cols = ['Age', 'year_of_operation', 'number_of_positive_axiilary_nodes']
    label_col = 'class'
    data = pd.read_csv('datasets/haberman.data',names= feature_cols+[label_col])
    train, test = data_processing(data, train_test_ratio, label_col,random_state)
    return train, test, feature_cols, label_col


def get_balance_scale(random_state, train_test_ratio=0.8):
    feature_cols = ['left-weight', 'left-dist', 'right-weight', 'right-dist']
    label_col = 'class'
    data = pd.read_csv('datasets/balance-scale.data', names=[label_col]+feature_cols)
    train, test = data_processing(data, train_test_ratio, label_col,random_state)
    return train, test, feature_cols, label_col

def get_spambase(random_state, train_test_ratio=0.8):
    feature_cols = []
    with open('datasets/spambase.names', 'r') as f:
        for line in f:
            feature_cols.append(line.replace('\n', '').replace(']','xxx').replace('[','xxx'))
    label_col = 'is_spam'
    data = pd.read_csv('datasets/spambase.data', names=feature_cols + [label_col])
    train, test = data_processing(data, train_test_ratio, label_col,random_state)
    return train, test, feature_cols, label_col

def get_zoo(random_state, train_test_ratio=0.8):
    feature_cols = ['x' + str(i) for i in range(17)]
    label_col = 'class'
    data = pd.read_csv('datasets/zoo.data', names=feature_cols + [label_col])
    train, test = data_processing(data, train_test_ratio, label_col,random_state)
    return train, test, feature_cols[1:], label_col

def get_german(random_state, train_test_ratio=0.8):
    feature_cols = ['x'+str(i) for i in range(20)]
    label_col = 'class'
    data = pd.read_csv('datasets/german.data',sep=' ', names = feature_cols+[label_col])
    data, feature_cols = dummify_data(data, feature_cols, label_col)
    for col in data.columns:
        data[col] = data[col].astype(int)
    train, test = data_processing(data, train_test_ratio, label_col,random_state)
    return train, test, feature_cols, label_col

def get_pima(random_state, train_test_ratio=0.8):
    feature_cols = ['x'+str(i) for i in range(8)]
    label_col = 'class'
    data = pd.read_csv('datasets/pima.scv', names=feature_cols+[label_col])
    train, test = data_processing(data, train_test_ratio, label_col,random_state)
    return train, test, feature_cols, label_col

def get_bank(random_state, train_test_ratio=0.8):
    data = pd.read_csv('datasets/bank.txt', sep=';')
    label_col = 'y'
    data, feature_cols = dummify_data(data, data.columns[:-1], label_col)
    train, test = data_processing(data, train_test_ratio, label_col,random_state)
    return train, test, feature_cols, label_col

def get_banknote(random_state, train_test_ratio=0.8):
    feature_cols = ['x' + str(i) for i in range(4)]
    label_col = 'class'
    data = pd.read_csv('datasets/banknote.txt', names=feature_cols + [label_col])
    train, test = data_processing(data, train_test_ratio, label_col,random_state)
    return train, test, feature_cols, label_col

def get_car(random_state, train_test_ratio=0.8):
    feature_cols = ['x' + str(i) for i in range(6)]
    label_col = 'class'
    data = pd.read_csv('datasets/car.data', names=feature_cols + [label_col])
    data, feature_cols = dummify_data(data, feature_cols, label_col)
    train, test = data_processing(data, train_test_ratio, label_col,random_state)
    return train, test, feature_cols, label_col

def get_credit(random_state, train_test_ratio=0.8):
    feature_cols = ['x' + str(i) for i in range(15)]
    label_col = 'class'
    data = pd.read_csv('datasets/credit.data', names=feature_cols + [label_col])
    for col in ['x1','x2','x7','x13']:
        data[col] = data[col].replace('?', -1000).astype(float).values
        data = data[data[col] > -1000]
    data = data[data['class'].apply(lambda x: isinstance(x, str))]
    data, feature_cols = dummify_data(data, feature_cols, label_col)
    train, test = data_processing(data, train_test_ratio, label_col,random_state)
    return train, test, feature_cols, label_col

def get_ctherapy(random_state, train_test_ratio=0.8):
    data = pd.read_csv('datasets/cryotherapy.csv')
    data = data.rename({'Result_of_Treatment':'class'},axis=1)
    label_col = 'class'
    feature_cols = data.columns[:-1]
    train, test = data_processing(data, train_test_ratio, label_col,random_state)
    return train, test, feature_cols, label_col

def get_internet_trust(random_state, train_test_ratio=0.8):
    feature_cols = ['x' + str(i) for i in range(4)]
    label_col = 'class'
    data = pd.read_csv('datasets/disshonest_internet.txt.txt', names=feature_cols+ [label_col], sep=' ')
    data,feature_cols = dummify_data(data,feature_cols,label_col)
    train, test = data_processing(data, train_test_ratio, label_col,random_state)
    return train, test, feature_cols, label_col

def get_forest(random_state, train_test_ratio=0.8):
    data = pd.read_csv('datasets/forsttypes.csv')
    feature_cols = data.columns[1:]
    label_col = 'class'
    train, test = data_processing(data, train_test_ratio, label_col,random_state)
    return train, test, feature_cols, label_col

def get_glass(random_state, train_test_ratio=0.8):
    feature_cols = ['x'+str(i) for i in range(10)][1:]
    label_col='class'
    data = pd.read_csv('datasets/glass.data',names=feature_cols+[label_col])
    train, test = data_processing(data, train_test_ratio, label_col,random_state)
    return train, test, feature_cols, label_col

def get_liver(random_state, train_test_ratio=0.8):
    feature_cols = ['x' + str(i) for i in range(10)]
    label_col = 'class'
    data = pd.read_csv('datasets/liver.csv', names=feature_cols+ [label_col]).dropna()
    data, feature_cols = dummify_data(data, feature_cols, label_col)
    train, test = data_processing(data, train_test_ratio, label_col,random_state)
    return train, test, feature_cols, label_col

def get_vegas(random_state, train_test_ratio=0.8):
    data = pd.read_csv('datasets/LasVegasTripAdvisorReviews-Dataset.csv', sep=';')
    data = data.rename({'Score': 'class'}, axis=1)
    data = data.drop(['User country', 'Hotel name'], axis=1)
    data['Hotel stars'] = data['Hotel stars'].apply(lambda x: float(x.replace(',', '.')))
    data['Review month'] = data['Review month'].apply(lambda x: month_dict[x[:3]])
    data['Review weekday'] = data['Review weekday'].apply(lambda x: day_dict[x])
    label_col = 'class'
    feature_cols = [col for col in data.columns if col != 'class']
    data, feature_cols = dummify_data(data, feature_cols, label_col)
    train, test = data_processing(data, train_test_ratio, label_col,random_state)
    return train, test, feature_cols, label_col

def get_magic(random_state, train_test_ratio=0.8):
    feature_cols = ['x' + str(i) for i in range(10)]
    label_col = 'class'
    data = pd.read_csv('datasets/magic04.data', names=feature_cols + [label_col])
    train, test = data_processing(data, train_test_ratio, label_col,random_state)
    return train, test, feature_cols, label_col

def get_mamo(random_state, train_test_ratio=0.8):
    feature_cols = ['x' + str(i) for i in range(5)]
    label_col = 'class'
    data = pd.read_csv('datasets/mammographic_masses.data', names=feature_cols + [label_col])
    data = data[['?' not in str(row) for indx, row in data.iterrows()]]
    for col in feature_cols:
        data[col] = data[col].astype(int)
    train, test = data_processing(data, train_test_ratio, label_col,random_state)
    return train, test, feature_cols, label_col

def get_occupancy(random_state, train_test_ratio=0.8):
    data = pd.read_csv('datasets/occupancy.txt')
    data = data.drop(['date'], axis=1)
    data = data.rename({'Occupancy': 'class'},axis=1)
    feature_cols = data.columns[:-1]
    label_col = 'class'
    train, test = data_processing(data, train_test_ratio, label_col,random_state)
    return train, test, feature_cols, label_col

def get_biodeg(random_state, train_test_ratio=0.8):
    feature_cols = ['x' + str(i) for i in range(41)]
    label_col = 'class'
    data = pd.read_csv('datasets/biodeg.txt', sep=';', names=feature_cols+ [label_col])
    train, test = data_processing(data, train_test_ratio, label_col,random_state)
    return train, test, feature_cols, label_col

def get_seismic(random_state, train_test_ratio=.8):
    feature_cols = ['x' + str(i) for i in range(18)]
    label_col = 'class'
    data = pd.read_csv('datasets/seismic.arff', names=feature_cols + [label_col])
    data, feature_cols = dummify_data(data,feature_cols,label_col)
    train, test = data_processing(data, train_test_ratio, label_col,random_state)
    return train, test, feature_cols, label_col

def get_abalone(random_state, train_test_ratio=.8):
    feature_cols = ['x' + str(i) for i in range(8)]
    label_col = 'class'
    data = pd.read_csv('datasets/abalone.data', names=feature_cols + [label_col])
    data['x0'] = [1 if i == 'M' else 0 for i in data['x0']]
    train, test = data_processing(data, train_test_ratio, label_col,random_state)
    return train, test, feature_cols, label_col

def get_ecoli(random_state, train_test_ratio=0.8):
    feature_cols = ['x'+str(i) for i in range(8)]
    label_col = 'class'
    data = pd.read_csv('datasets/ecoli1.data',names=feature_cols+[label_col])
    feature_cols = feature_cols[1:]
    train, test = data_processing(data, train_test_ratio, label_col,random_state)
    return train, test, feature_cols, label_col

def get_australian(random_state, train_test_ratio=0.8):
    feature_cols = ["A" + str(i) for i in range(14)]
    label_col ='class'
    data = pd.read_csv("datasets/australian.dat", sep=" ", names=feature_cols+[label_col])
    train, test = data_processing(data, train_test_ratio, label_col,random_state)
    return train, test, feature_cols, label_col

def get_nurse(random_state, train_test_ratio=0.8):
    feature_cols = ['x' + str(i) for i in range(8)]
    label_col ='class'
    data = pd.read_csv("datasets/post-operative.data", names = feature_cols + [label_col])
    data, feature_cols = dummify_data(data, feature_cols, label_col)
    data[label_col] = [i[0] for i in data[label_col]]
    train, test = data_processing(data, train_test_ratio, label_col,random_state)
    return train, test, feature_cols, label_col

def get_tic_tac_toe(random_state, train_test_ratio=0.8):
    feature_cols=['top - left - square','top - middle - square','top - right - square','middle - left - square',
               'middle - middle - square','middle - right - square','bottom - left - square','bottom - middle - square','bottom - right - square']
    label_col='class'
    data=pd.read_csv('datasets/tic-tac-toe.data',names=feature_cols+[label_col])
    data, feature_cols = dummify_data(data, feature_cols, label_col)
    train, test = data_processing(data, train_test_ratio, label_col,random_state)
    return train, test, feature_cols, label_col

def get_aust_weather_data(random_state, train_test_ratio=0.8):
    data = pd.read_csv('datasets/weatherAUS.csv')
    data = data[[k for k, v in data.isnull().sum().sort_values(ascending=False).items() if v < 1000]]
    label_col = data.columns[-1]
    feature_cols = data.columns[1:-1]
    data, feature_cols = dummify_data(data, feature_cols, label_col)
    data = data.sample(10000)
    for k, v in data.isnull().sum().sort_values(ascending=False).items():
        if v > 0:
            mean_val = data[k].mean()
            data[k] = data[k].fillna(mean_val)
    train, test = data_processing(data, train_test_ratio, label_col,random_state)
    return train, test, feature_cols, label_col
def get_thyroid_data(random_state, train_test_ratio=0.8):
    f = open('datasets/allbp.txt', 'r')
    records = []
    for line in f:
        line = line.split('|')[0].split(',')
        records.append({i: val for i, val in enumerate(line)})
    f.close()
    data = pd.DataFrame(records)
    numeric_idices = [0, 17, 19, 21, 23, 25]
    for indx in numeric_idices:
        data[indx] = data[indx].replace('?', np.nan).astype(float)
        data[indx] = data[indx].replace(np.nan,data[indx].mean())
    data.columns = ['x' + str(col) for col in data.columns[:-1]] + ['class']
    feature_cols = data.columns[:-1]
    label_col = data.columns[-1]
    data, feature_cols = dummify_data(data, feature_cols, label_col)
    train, test = data_processing(data, train_test_ratio, label_col,random_state)
    train, test = train[train[label_col] < 2], test[test[label_col] < 2]
    return train, test, feature_cols, label_col

def get_shuttle_data(random_state, train_test_ratio=0.8):
    data = pd.read_csv('datasets/shuttle.txt', sep=' ', names=['x' + str(i) for i in range(9)] + ['class'])
    data = data[data['class'].isin([1, 4, 5])]
    feature_cols = data.columns[:-1]
    label_col = data.columns[-1]
    data, feature_cols = dummify_data(data, feature_cols, label_col)
    train, test = data_processing(data, train_test_ratio, label_col,random_state)
    return train, test, feature_cols, label_col

def get_seeds_data(random_state, train_test_ratio=0.8):
    feature_cols = ['x' + str(i) for i in range(7)]
    label_col = 'class'
    f = open('datasets/seeds_dataset.txt', 'r')
    records = []
    for line in f:
        d = []
        record = line.split('\t')
        for item in record:
            try:
                item = float(item)
                d.append(item)
            except:
                continue
        records.append(d)
    data = pd.DataFrame(records)
    data.columns = ['x' + str(i) for i in data.columns[:-1]] + ['class']
    data['class'] = data['class'].astype(int)
    train, test = data_processing(data, train_test_ratio, label_col,random_state)
    return train, test, feature_cols, label_col

def get_contraceptive_data(random_state, train_test_ratio=0.8):
    feature_cols = ['x' + str(i) for i in range(9)]
    label_col = 'class'
    data = pd.read_csv('datasets/cmc.txt', names=feature_cols + [label_col])
    train, test = data_processing(data, train_test_ratio, label_col,random_state)
    return train, test, feature_cols, label_col

def get_cardiotocography_data(random_state, train_test_ratio=0.8):
    data = pd.read_csv('datasets/cardiotocography.csv')
    label_col = 'NSP'
    feature_cols = data.columns[:-1]
    train, test = data_processing(data, train_test_ratio, label_col,random_state)
    return train, test, feature_cols, label_col

def get_ionosphere_data(random_state, train_test_ratio=0.8):
    feature_cols = ['x' + str(i) for i in range(34)]
    label_col = 'class'
    data = pd.read_csv('datasets/ionosphere.data', names=feature_cols + [label_col])
    train, test = data_processing(data, train_test_ratio, label_col,random_state)
    return train, test, feature_cols, label_col

def get_hayes_roth_dataset(random_state, train_test_ratio=0.8):
    feature_cols = ['x' + str(i) for i in range(5)]
    label_col = 'class'
    data = pd.read_csv('datasets/hayes-roth.data', names=feature_cols + [label_col])
    train, test = data_processing(data, train_test_ratio, label_col,random_state)
    return train, test, feature_cols, label_col

def get_waveform_data(random_state, train_test_ratio=0.8):
    feature_cols = ['x' + str(i) for i in range(21)]
    label_col = 'class'
    data = pd.read_csv('datasets/waveform.data', names=feature_cols + [label_col])
    train, test = data_processing(data, train_test_ratio, label_col,random_state)
    return train, test, feature_cols, label_col

def get_divorce_data(random_state, train_test_ratio=0.8):
    data = pd.read_csv('datasets/divorce.csv',sep=';')
    feature_cols = data.columns[:-1]
    label_col = data.columns[-1]
    train, test = data_processing(data, train_test_ratio, label_col,random_state)
    return train, test, feature_cols, label_col
#def get_happiness_data(random_state, train_test_ratio=0.8):


datasets_dict = {'iris':get_iris_data,'breast_cancer':get_breast_cancer_data,
                 'winery':get_winery_data ,'kohkiloyeh':get_kohkilyeh,'thyroid':get_thyroid_data,
                 'haberman':get_haberman,'balance_scale':get_balance_scale,'cardiotocography':get_cardiotocography_data,
                 'spambase':get_spambase,'zoo':get_zoo, 'german':get_german,'bank':get_bank,'ionosphere':get_ionosphere_data,
                 'banknote':get_banknote,'car':get_car,'credit':get_credit,'ctherapy':get_ctherapy, 'internet':get_internet_trust,'shuttle':get_shuttle_data,
                 'forest':get_forest,'glass':get_glass,'liver':get_liver,'vegas':get_vegas, 'magic':get_magic,'mamographic':get_mamo,
                 'hayes_roth':get_hayes_roth_dataset, 'waveform':get_waveform_data, 'divorce':get_divorce_data,
                 'occupancy':get_occupancy,'biodeg':get_biodeg,'seismic':get_seismic,'ecoli':get_ecoli,'contraceptive':get_contraceptive_data,
                 'aust':get_australian,'nurse':get_nurse,'tic_tac_toe':get_tic_tac_toe, 'abalone':get_abalone,'pima':get_pima, 'seeds':get_seeds_data}