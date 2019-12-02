"""
This module contains the conjunction class

"""
import numpy as np
from utils import *

class Conjunction():
    """
    A conjunction is a combination of feature bounds mapped into a class probability vector
    """
    def __init__(self,feature_names,label_names,leaf_index=None,label_probas=None):
        """
        :param feature_names: list of strings. Also determine the dimensionality
        :param label_names: list of labels. Determines the number of labels too
        :param leaf_index: This feature is optional. Can be relevant if we'd like to document the leaves that were used from the input forest
        :param label_probas: also optional. Relevant if we'd like to determine the class probabilities within the constructor
        """
        self.feature_names = feature_names
        self.number_of_features = len(feature_names)
        self.label_names = label_names

        # upper and lower bounds of the feature for each rule
        self.features_upper = [np.inf] * len(feature_names)
        self.features_lower = [-np.inf] * len(feature_names)

        self.label_probas = np.array(label_probas)
        self.leaf_index = leaf_index

        #The following dict is used for excluding irrelevant merges of different dummy variables that come from the same categorical feature
        self.categorical_features_dict={}

    def addCondition(self, feature, threshold, bound):
        """
        This method adds a condition to the conjunction if relevant (rule isn't already contained in the conjunction)

        :param feature: relevant feature
        :param threshold: upper\lower bound
        :param bound: bound direction

        """
        #Check if the rule isn't already contained in the conjunction
        if bound == 'lower':
            if self.features_lower[feature] < threshold:
                self.features_lower[feature] = threshold
        else:
            if self.features_upper[feature] > threshold:
                self.features_upper[feature] = threshold

        #Address categorical features:
        if '=' in self.feature_names[feature] and threshold >= 1 and bound == 'lower':
            splitted = self.feature_names[feature].split('=')
            self.categorical_features_dict[splitted[0]] = splitted[1]

    def isContradict(self, other_conjunction):
        """
        :param other_conjunction: conjunction object
        :return: True if other and self have at least one contradiction, otherwise False
        """

        #Check upper and lower bounds contradiction
        for i in range(self.number_of_features):
            if self.features_upper[i] <= other_conjunction.features_lower[i] or self.features_lower[i] >=  other_conjunction.features_upper[i]:
                return True

        # check for categorical features contradiction
        for feature in self.feature_names:
            if feature in self.categorical_features_dict and feature in other_conjunction.categorical_features_dict:
                if self.categorical_features_dict[feature] != other_conjunction.categorical_features_dict[feature]:
                    return True

    def merge(self, other):
        """
        :param other: conjunction
        :return: new_conjunction - a merge of the self conjunction with other
        """
        new_conjunction = Conjunction(self.feature_names,self.label_names,
                                      self.leaf_index+other.leaf_index,self.label_probas+other.label_probas)
        new_conjunction.features_upper = [min(i,j) for i,j in zip(self.features_upper,other.features_upper)]
        new_conjunction.features_lower = [max(i, j) for i, j in zip(self.features_lower, other.features_lower)]
        new_conjunction.categorical_features_dict = self.categorical_features_dict
        new_conjunction.categorical_features_dict.update(other.categorical_features_dict)
        return new_conjunction

    def containsInstance(self,inst):
        """
        Checks whether the input instance falls under the conjunction

        :param inst:
        :return: True if
        """
        for i, lower, upper in zip(range(len(inst)), self.features_lower, self.features_upper):
            if inst[i] >= upper or inst[i] < lower:
                return False
        return True

    def has_low_interval(self,lowest_intervals):
        for lower,upper,interval in zip(self.features_lower,self.features_upper,lowest_intervals):
            if upper-lower<interval:
                return True
        return False

    def predict_probas(self):
        """
        :return: softmax of the result vector
        """

        return softmax(self.label_probas)

    def toString(self):
        """
        This function creates a string representation of the conjunction (only for demonstration purposes)
        """
        s = ""
        #save lower bounds
        for feature, threshold in enumerate(self.features_lower):
            if threshold != (-np.inf):
                s +=  self.feature_names[feature] + ' >= ' + str(np.round(threshold,3)) + ", "
        #save upper bounds
        for feature, threshold in enumerate(self.features_upper):
            if threshold != np.inf:
                s +=  self.feature_names[feature] + ' < ' + str(np.round(threshold,3)) + ", "
        #save labels
        s += 'labels: ['
        s+=str(self.label_probas)
        s += ']'
        return s

    #From here on everything is still tested
    def get_data_point(self, min_values, max_values, mean_values):
        X = []
        for i,feature in enumerate(self.feature_names):
            if self.features_lower[i]==-np.inf and self.features_upper[i]==np.inf:
                X.append(mean_values[feature])
            else:
                X.append(np.mean([max(min_values[feature],self.features_lower[i]), min(max_values[feature],self.features_upper[i])]))
        return np.array(X)