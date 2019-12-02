"""
This module contains the ConjunctionSet class
"""

from statsmodels.distributions.empirical_distribution import ECDF
from TreesExtraction import *
from collections import Counter
from pruning import *
from pyod.models.knn import KNN
from pyod.models.lof import LOF

class ConjunctionSet():
    """
    ConjunctionSet is a class that represents a set of conjunctions.

    This is the output of stage 1.
    Each conjunction at the given set represents a possible combination of leaves from the source decision forests.
    """

    def __init__(self, max_number_of_conjunctions=np.inf, filter_method='probability'):
        """
        :param max_number_of_conjunctions: Number of maximum allowed conjunctions at each iteration
        :param filter_method: The approach that will be takes for filtering conjunctions
        """
        self.filter_method = filter_method
        self.max_number_of_conjunctions = max_number_of_conjunctions

    def fit(self,trees_conjunctions,data,feature_cols,label_col, int_features = []):
        """

        :param trees_conjunctions: Decision forest given as a list of lists of conjunction objects.
        :param data: pandas dataframe that was used for training the decision forest
        :param feature_cols: Feature names in the dataframe
        :param label_col: label column name
        :param int_features: list of integer feartures
        :return: set a list of conjunction set that best represents the decision forest
        """
        self.feature_cols = feature_cols
        self.labels = data[label_col].unique()
        self.trees_conjunctions = trees_conjunctions
        self.int_features = int_features

        #Create an ECDF for each feature
        self.set_probability_ecdf(data)


        #Extract all the leaf combinations that were applied for training data:
        print('Create conjunction set from training data instances')
        self.create_conjunction_set_from_data(data)


        #set maximum number of conjunctions per label
        print('Create complete conjunction set')
        self.calculate_max_conjunctions_per_label(data,label_col)

        # Run the algorithm of creating the complete conjunction set:
        self.createConjunctionSetFromTreeConjunctions()

        #Merge the two conjunctions:
        self.conjunctions = self.conjunctions + self.training_conjunctions

        #Get the ordered splitting points for creating the hierarchy at stage 2
        self.set_ordered_splitting_points()

    def createConjunctionSetFromTreeConjunctions(self):
        """
        This method generates the conjunction set (stage 1) from the decision forest
        """
        self.conjunctions = self.trees_conjunctions[0] #Define the first tree as the current conjunction set
        i = 1
        self.size_per_iteration = [len(self.conjunctions)]
        while i < len(self.trees_conjunctions): #At each iteration we merge the next tree with the current conjunction set
            self.conjunctions= merge_two_conjunction_sets(self.conjunctions, self.trees_conjunctions[i])
            i+=1
            self.filter() #Filter redundant conjunction according to the filtering strategy
            self.size_per_iteration.append(len(self.conjunctions))
            print('Size at iteration '+str(i)+': '+str(len(self.conjunctions)))
    def filter(self):
        """
        This method filters the current conjunction set according to the filtering strategy.

        At the first stage it filters conjunctions that contain irrelevant integer rules.
        For example: If x is an integer then a conjunction that contains  5.5 >= x < 6 is filtered out
        """
        self.conjunctions = [conj for conj in self.conjunctions if self.int_filter(conj)]
        if len(self.conjunctions)<=self.max_number_of_conjunctions:
            return
        if self.filter_method == 'probability':
            self.filter_by_probability()
        if self.filter_method == 'probability_label':
            self.filter_by_probability_labels()
        elif self.filter_method == 'knn':
            self.filter_by_knn()
        elif self.filter_method == 'LOF':
            self.filter_by_lof()

    def filter_by_probability(self,EPSILON=0.00001):
        """
        This method filters conjunctions according to the product of each rule ECDF

        :param EPSILON: Prevent a case of probability = 0

        """
        independent_probs = []
        for conj in self.conjunctions:
            independent_probs.append(np.sum(np.log([self.ecdf[col](conj.features_upper[col])-self.ecdf[col](conj.features_lower[col])+EPSILON for
                                                         col in range(len(self.feature_cols))])))
        max_value = sorted(independent_probs, reverse=True)[self.max_number_of_conjunctions]
        self.conjunctions = [c for c,val in zip(self.conjunctions,independent_probs) if val >= max_value] #It actually includes a little more than max_number_of conjunctions due to the >=

    def predict(self,X):
        return [np.argmax(i) for i in self.predict_proba(X)]

    def predict_proba(self,X):
        predictions = []
        if isinstance(X,pd.DataFrame):
            X = X[self.feature_cols].values
        for inst in X:
            for conjunction in self.conjunctions:
                if conjunction.containsInstance(inst):
                    predictions.append(np.exp(conjunction.label_probas) / np.sum(np.exp(conjunction.label_probas), axis=0)) #softmax
                    break
        return np.array(predictions)

    #Filtering functions

    def set_probability_ecdf(self,data):
        self.ecdf = {i:ECDF(data[col].values) for i,col in enumerate(self.feature_cols)}

    def set_minimum_intervals(self,data):
        intervals = {col: data[col].diff().sort_values().dropna().values for col in self.feature_cols}
        self.minimum_intervals = [x[x > 0].min()*self.min_interval_ratio for col, x in intervals.items()]

    def int_filter(self,conj,EPSILLON=0.00001):
        for i,feature in enumerate(self.feature_cols):
            if feature in self.int_features:
                if conj.features_upper[i]-conj.features_lower[i]-EPSILLON <= 0.5 and (conj.features_lower[i] % 1) > 0:
                    return False
        return True

    def create_conjunction_set_from_data(self,X):
        """

        :param X: Pandas dataframe (or matrix)
        :return: training_conjunctions - all the conjunctions that were applied for X
        """
        participated_leaves = []
        self.training_conjunctions = []
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_cols].values
        for inst in X:
            s=''
            conj = Conjunction(self.feature_cols,self.labels,leaf_index=[],label_probas=np.zeros(len(self.labels))) #Define the conjunction
            for tree_index,tree in enumerate(self.trees_conjunctions):
                for leaf_index,leaf in enumerate(tree):
                    if leaf.containsInstance(inst):
                        conj = conj.merge(leaf)
                        s+=str(tree_index)+'|'+str(leaf_index)+'_'
            if s not in participated_leaves:
                self.training_conjunctions.append(conj)
                participated_leaves.append(s)
        print('Number of conjunctions created from data: '+str(len(self.training_conjunctions)))

    def set_ordered_splitting_points(self):
        """
        This method creates the splitting points for stage 2 (order the conjunctions in a hierarchical order)
        """
        self.splitting_points = {i:[] for i in range(len(self.feature_cols))}
        for tree in self.trees_conjunctions:
            for leaf in tree:
                for i,lower,upper in zip(range(len(self.feature_cols)),leaf.features_lower,leaf.features_upper):
                    self.splitting_points[i].extend([upper,lower])
        for i in self.splitting_points:
            self.splitting_points[i] = [v[0] for v in Counter(self.splitting_points[i]).most_common() if np.abs(v[0]) < np.inf]

    def filter_by_knn(self):
        """
        Filter by KNN ANomaly detection method. Doesn't seem to be better than probability filtering for now
        :return:
        """
        data_points = np.array([conj.get_data_point(self.min_values, self.max_values, self.mean_values)for conj in self.conjunctions]).reshape(len(self.conjunctions),len(self.feature_cols))
        anomaly_probas = [i[1] for i in self.knn_clf.predict_proba(data_points)]
        max_value = sorted(anomaly_probas)[self.max_number_of_conjunctions]
        self.conjunctions = [c for c, val in zip(self.conjunctions, anomaly_probas) if val <= max_value]
    def filter_by_lof(self):
        """
        Filter by LOF ANomaly detection method. Doesn't seem to be better than probability filtering for now
        :return:
        """
        data_points = np.array([conj.get_data_point(self.min_values, self.max_values, self.mean_values)for conj in self.conjunctions]).reshape(len(self.conjunctions),len(self.feature_cols))
        anomaly_probas = [i[1] for i in self.lof_clf.predict_proba(data_points)]
        max_value = sorted(anomaly_probas)[self.max_number_of_conjunctions]
        self.conjunctions = [c for c, val in zip(self.conjunctions, anomaly_probas) if val <= max_value]

    def calculate_max_conjunctions_per_label(self,data,label_col):
        self.max_conjunctions_per_label = dict((data[label_col].value_counts(normalize=True)*self.max_number_of_conjunctions).astype(int))

    def filter_by_probability_labels(self,EPSILON=0.00001):
        """

        :param EPSILON: Added to the denominator to prevent devision by zero
        :return:
        """
        conjunctions_dict = {}
        probs_dict = {}

        for indx,conj in enumerate(self.conjunctions):
            conjunctions_dict[indx] = conj
            probs_dict[indx] = np.sum(np.log(
                [self.ecdf[col](conj.features_upper[col]) - self.ecdf[col](conj.features_lower[col]) + EPSILON for
                 col in range(len(self.feature_cols))]))
        probs_dict = dict(sorted(probs_dict.items(), key=lambda kv: kv[1], reverse=True))
        conjs_per_label = {label:0 for label in self.labels}
        conjunctions = []
        for indx in probs_dict:
            conj = conjunctions_dict[indx]
            label = np.argmax(conj.label_probas)
            if conjs_per_label[label] < self.max_conjunctions_per_label[label]:
                conjunctions.append(conj)
                conjs_per_label[label]+=1
            if len(conjunctions) == self.max_number_of_conjunctions:
                break
        self.conjunctions = conjunctions

