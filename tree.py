"""
This module contain a tree class and several functions that are used for constructing the decision tree (stage 2 of the FBT algorithm)
"""

from scipy.stats import entropy
from utils import *

class Tree():
    """
    A decision tree that is based on hierarchical ordering of conjunction set

    Essentialy, the tree is a node with 2 descendents in case of an internal node and a prediction vector if its a leaf
    """

    def __init__(self,conjunctions, splitting_values,max_depth):
        """
        :param conjunctions: A list of conjunctions
        :param splitting_values: A dictionary in ehich keys are features and values are splitting values ordered by frequency
        :param max_depth: Tree maximum depth
        """

        self.conjunctions = conjunctions
        self.splitting_values = splitting_values
        self.max_depth = max_depth

    def split(self):
        # 1. Spliting is stopped if:
        #    a. there's a single conjunctions
        #    b. Entropy doesn't improved
        # 2. Splitting values - at each iteration we selrct the most common value for each feature and selects
        #    The one with the highest information gain
        # 3. Information gain is calculated as the mean emtropy across the different feature dimensions
        if len(self.conjunctions) == 1 or self.max_depth == 0:
            self.selected_feature = None
            self.left = None
            self.right = None
            return
        if len(set([np.argmax(conj.label_probas) for conj in self.conjunctions])) > 1:
            self.selected_feature, self.selected_value, self.entropy, \
            l_conjunctions, r_conjunctions = select_splitting_feature_by_entropy(self.conjunctions, self.splitting_values)
        else:
            self.selected_feature, self.selected_value, self.entropy, \
            l_conjunctions, r_conjunctions = select_splitting_feature_by_max_splitting(self.conjunctions,
                                                                                 self.splitting_values)
        if self.selected_feature is None:
            return
        descending_splitting_values = {k:([i for i in v if i!=self.selected_value] if k == self.selected_feature else v) for k,v in self.splitting_values.items()}
        self.left = Tree(l_conjunctions,descending_splitting_values,max_depth = self.max_depth-1)
        self.right = Tree(r_conjunctions, descending_splitting_values,max_depth = self.max_depth-1)
        self.left.split()
        self.right.split()

    def predict_instance_proba(self,inst):
        """
        Predicte class probabilities for a given instance

        :param inst: Numpy array. Each dimension is a feature
        :return: class probabilities

        This is a recursive method that routes the instance to its relevant leaf
        """
        if self.selected_feature == None:
            #return softmax(np.array([c.label_probas for c in self.conjunctions]).sum(axis=0))
            return np.array([softmax(c.label_probas) for c in self.conjunctions]).mean(axis=0)[0]
        if inst[self.selected_feature] >= self.selected_value:
            return self.left.predict_instance_proba(inst)
        else:
            return self.right.predict_instance_proba(inst)

    def get_instance_decision_path(self, inst,result=[]):
        """

        :param inst: numpy array represents an instance to be inferenced
        :param result: a list where each item represents a node
        :return:
        """
        result=list(result)
        if self.selected_feature == None:
            result.append('labels: '+str(np.array([softmax(c.label_probas) for c in self.conjunctions]).mean(axis=0)[0]))
            return result
        else:
            if inst[self.selected_feature] >= self.selected_value:
                result.append(str(self.selected_feature)+'>='+str(self.selected_value))
                return self.left.get_instance_decision_path(inst,result)
            else:
                result.append(str(self.selected_feature) + '<' + str(self.selected_value))
                return self.right.get_instance_decision_path(inst, result)

    def predict_proba(self,data):
        """
        Predicted class probabilities for each data instance

        :param data: pandas dataframe
        :return: numpy array with calss probabilities for each data instance
        """
        probas=[]
        for inst in data.values:
            probas.append(self.predict_instance_proba(inst))
        return np.array(probas)

    def get_decision_paths(self,data):
        """

        :param data: matrix of [numer_of_instances, number_of_features] dimensions
        :return: A list where each item corresponds to the decision path of one insance
        """
        paths = []
        for inst in data.values:
            paths.append(self.get_instance_decision_path(inst))
        return paths

    # The following methods are relevant for the experimental evaluation. Enable calculating the depth of leaves used for predictions
    def predict_proba_and_depth(self,data):
        probas = []
        depths = []
        for inst in data.values:
            proba, depth = self.predict_instance_proba_and_depth(inst)
            probas.append(proba)
            depths.append(depth)
        return np.array(probas),depths

    def predict_instance_proba_and_depth(self,inst):
        if self.selected_feature == None:
            #return softmax(np.array([c.label_probas for c in self.conjunctions]).sum(axis=0))
            return np.array([softmax(c.label_probas) for c in self.conjunctions]).mean(axis=0)[0], 0
        if inst[self.selected_feature] >= self.selected_value:
            probas, depth = self.left.predict_instance_proba_and_depth(inst)
            return probas, depth + 1
        else:
            probas, depth = self.right.predict_instance_proba_and_depth(inst)
            return probas, depth + 1


def select_splitting_feature_by_entropy(conjunctions, splitting_values):
    """
    :param conjunctions: List of conjunctions
    :param splitting_values: A dictionary. Keys are features and values are splitting points, ordered by frequency
    :return: selected feature, splitting value, weighted entropy stemmed from the split, conjunctions of the left node, conjunctions of the right node

    Splitting algorithm:
    1. Define the best entropy as the current entropy of the class probability vectors
    2. For each feature - get the most frequent spliiting value (first item of the dict) and calculate weighted entropy of split
    3. Based on the best entropy - return the derived variables
    """
    conjunctions_len = len(conjunctions)
    best_entropy = get_entropy([c.label_probas for c in conjunctions])
    selected_feature,selected_value,l_conjunctions, r_conjunctions = None, None, None, None
    for feature,values in splitting_values.items():
        if len(values)==0:
            continue
        for i in range(len(values)):#We iterate over all the values within the feature to find the best splitting point
            temp_l_conjunctions, temp_r_conjunctions,temp_entropy = calculate_entropy_for_split(conjunctions,feature, values[i])
            # We want to prevent a case where all the conjunctions are going to one of the descendent
            if temp_entropy < best_entropy and len(temp_l_conjunctions) < conjunctions_len and  len(temp_r_conjunctions) < conjunctions_len:
                best_entropy = temp_entropy
                selected_feature = feature
                selected_value = values[i]
                l_conjunctions = temp_l_conjunctions
                r_conjunctions = temp_r_conjunctions
    return selected_feature,selected_value,best_entropy, l_conjunctions, r_conjunctions

def select_splitting_feature_by_max_splitting(conjunctions,splitting_values):
    """

    :param conjunctions: List of conjunctions
    :param splitting_values: A dictionary. Keys are features and values are splitting points, ordered by frequency
    :return: selected feature, splitting value, weighted entropy stemmed from the split, conjunctions of the left node, conjunctions of the right node

    Splitting algorithm:
    1. Define the best entropy as the current entropy of the class probability vectors
    2. For each feature - get the most frequent spliiting value (first item of the dict) and calculate weighted entropy of split
    3. Based on the best entropy - return the derived variables
    """
    conjunctions_len = len(conjunctions)
    #best_entropy = get_entropy([c.label_probas for c in conjunctions])
    best_value = len(conjunctions)
    selected_feature,selected_value,l_conjunctions, r_conjunctions = None, None, None, None
    for feature,values in splitting_values.items():
        if len(values)==0:
            continue
        for i in range(len(values)):#We iterate over all the values within the feature to find the best splitting point
            temp_l_conjunctions, temp_r_conjunctions, temp_value = calculate_max_for_split(conjunctions, feature, values[i])
            if temp_value < best_value:
                best_value = temp_value
                selected_feature = feature
                selected_value = values[i]
                l_conjunctions = temp_l_conjunctions
                r_conjunctions = temp_r_conjunctions

    return selected_feature,selected_value,0, l_conjunctions, r_conjunctions

def calculate_entropy_for_split(conjunctions,feature,value):
    """
    Calculate the entropy of splitting the conjunctions according to the given feature vale

    :param conjunctions: List of conjunctions
    :param feature: splitting feature
    :param value: splitting value
    :return: conjunctions of left and right nodes, weighted entropy
    """
    l_conjunctions = []
    r_conjunctions = []
    l_probas = []
    r_probas = []
    for conj in conjunctions:
        if conj.features_upper[feature] <= value:
            r_conjunctions.append(conj)
            r_probas.append(conj.label_probas)
        elif conj.features_lower[feature] >= value:
            l_conjunctions.append(conj)
            l_probas.append(conj.label_probas)
        else:
            r_conjunctions.append(conj)
            r_probas.append(conj.label_probas)
            l_conjunctions.append(conj)
            l_probas.append(conj.label_probas)
    return l_conjunctions, r_conjunctions, calculate_weighted_entropy(l_probas, r_probas)

def calculate_weighted_entropy(l_probas,r_probas):
    """

    :param l_probas: numpy array wehre each item is a probability vector
    :param r_probas: numpy array wehre each item is a probability vector
    :return: weighted entropy
    """
    l_entropy, r_entropy = get_entropy(l_probas), get_entropy(r_probas)
    l_size,r_size = len(l_probas),len(r_probas)
    overall_size = l_size+r_size
    return(l_size*l_entropy+r_size*r_entropy)/overall_size


def get_entropy(probas):
    """
    Calculate antropy of an array of class probability vectors
    :param probas: An array of class probability vectors
    :return: the average entropy of each class vector
    """
    values = np.array([np.argmax(x) for x in probas])
    values, counts = np.unique(values, return_counts=True)
    probas = counts / np.sum(counts)
    return entropy(probas)

def calculate_max_for_split(conjunctions,feature,value):
    l_conjunctions = []
    r_conjunctions = []
    l_probas = []
    r_probas = []
    for conj in conjunctions:
        if conj.features_upper[feature] <= value:
            r_conjunctions.append(conj)
            r_probas.append(conj.label_probas)
        elif conj.features_lower[feature] >= value:
            l_conjunctions.append(conj)
            l_probas.append(conj.label_probas)
        else:
            r_conjunctions.append(conj)
            r_probas.append(conj.label_probas)
            l_conjunctions.append(conj)
            l_probas.append(conj.label_probas)
    return l_conjunctions, r_conjunctions, max(len(l_conjunctions),len(r_conjunctions))



