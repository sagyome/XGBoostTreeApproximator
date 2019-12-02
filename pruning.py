"""
This module contain the Pruner function for pruning a decision forest
"""
import pandas as pd
import numpy as np
from utils import *
from sklearn.metrics import cohen_kappa_score

class Pruner():
    """
    A static class that supports the pruning of a decision forest
    """
    def predict_probas_tree(self,conjunctions,X):
        """
        Predict probabilities for X using a tree, represented as a conjunction set

        :param conjunctions: A list of conjunctions
        :param X: numpy array of data instances
        :return: class probabilities for each instance of X
        """

        probas = []
        for inst in X:
            for conj in conjunctions:
                if conj.containsInstance(inst):
                    probas.append(conj.label_probas)
        return np.array(probas)
    def predict_probas(self,forest,X):
        """
        Predict probabilities of X, using a decision forest

        :param forest: A list of decision trees where each tree is a list of conjunctions
        :param X: Numpy array of data instances
        :return: List of class probabilities vector
        """
        predictions = []
        if isinstance(X, pd.DataFrame):
            X = X.values
        for t in forest:
            predictions.append(self.predict_probas_tree(t, X))
        return np.array([softmax(pred)[0] for pred in np.array(predictions).sum(axis=0)])

    def predict(self,forest,X):
        """
            Predict labels of X, using a decision forest

            :param forest: A list of decision trees where each tree is a list of conjunctions
            :param X: Numpy array of data instances
            :return: class vector
        """
        return np.argmax(self.predict_probas(forest,X),axis=1)

    def get_forest_auc(self,forest,X,Y):
        """
        Calculates predictions ROC AUC

        :param forest: A list of lists of conjunctions
        :param X: Numpy array of data instances
        :param Y: Label vector
        :return: ROC AUC
        """
        y_probas = self.predict_probas(forest,X)
        return get_auc(Y,y_probas)

    def forests_kappa_score(self,probas1,probas2):
        """
        Calculates Cohen's kappa of the predictions divided from two vectors of class probabilities

        :param probas1: list of class probabilities
        :param probas2: list of class probabilities
        :return: Cohen's kappa
        """

        predictions1 = np.array([np.argmax(i) for i in probas1])
        predictions2 = np.array([np.argmax(i) for i in probas1])
        return cohen_kappa_score(predictions1,predictions2)

    def kappa_based_pruning(self,forest,X,Y,min_forest_size=10):
        """
        This method conduct a kappa-based ensemble pruning.

        :param forest: A list of lists of conjunctions (a decision forest)
        :param X: Numpy array (data instances)
        :param Y: Label vector
        :param min_forest_size: minimum size of the pruned ensemble
        :return: list of lists of conjunctions - represents the pruned ensemble

        The algorithm contains the following stages:
        1. Add the tree with the highest AUC for X to the new (empty) forest
        2. At each iteration add the tree with the highest cohen's kappa in relation to the new forest
        3. Stop when the new forest AUC doesn't improve and minimum forest size was reached
        """

        selected_indexes = [np.argmax([self.get_forest_auc([t],X,Y) for t in forest])] #Include only the tree with the best AUC
        previous_auc = 0
        current_auc = get_auc(Y,self.predict_probas([forest[selected_indexes[0]]],X))
        new_forest = [forest[selected_indexes[0]]]
        while current_auc > previous_auc or len(new_forest) <= min_forest_size:
            kappas = [1 if i in selected_indexes else self.forests_kappa_score(new_forest,[t],X) for i,t in enumerate(forest)]
            new_index = np.argmin(kappas)
            if new_index in selected_indexes:
                break
            selected_indexes.append(new_index)
            previous_auc = current_auc
            new_forest.append(forest[new_index])
            current_auc = get_auc(Y,self.predict_probas(new_forest,X))
        return new_forest

    def max_auc_pruning(self, forest, X, Y, min_forest_size=10):
        """
        This method conduct an ensemble pruning using a greedy algorithm that maximizes the AUC on the given dataset.

        :param forest: A list of lists of conjunctions (a decision forest)
        :param X: Numpy array (data instances)
        :param Y: Label vector
        :param min_forest_size: minimum size of the pruned ensemble
        :return: list of lists of conjunctions - represents the pruned ensemble
        """
        X = X.values
        trees_predictions = {i: self.predict_probas_tree(forest[i],X) for i in range(len(forest))} #predictions are stored beforehand for efficiency purposes
        selected_indexes = [np.argmax([get_auc(Y,trees_predictions[i]) for i in trees_predictions])] #get the tree with the highest AUC for the given dataset
        previous_auc = 0
        best_auc = get_auc(Y,trees_predictions[selected_indexes[0]])
        while best_auc > previous_auc or len(selected_indexes) <= min_forest_size:
            previous_auc = best_auc
            best_index = None
            for i in range(len(forest)):
                if i in selected_indexes:
                    continue
                probas = np.array([trees_predictions[indx] for indx in selected_indexes + [i]]) #get the probas given by each tree, included the tested one
                probas = np.array([softmax(prob)[0] for prob in probas.sum(axis=0)]) #aggregate the predictions
                temp_auc = get_auc(Y,probas)
                if temp_auc > best_auc or best_index==None:
                    best_auc = temp_auc
                    best_index = i
            selected_indexes.append(best_index)
        print('Pruned forest training set AUC: '+str(best_auc))
        return [t for i,t in enumerate(forest) if i in selected_indexes]


