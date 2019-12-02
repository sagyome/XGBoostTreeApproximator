"""
This module contains a forest based tree class (FBT).

The class takes an XGBoost as an input and generates a decision aims at preserving the predictive performance of
the XGboost model
"""
from conjunctionset import *
from tree import *
from pruning import *

class FBT():
    """
    This class creates a decision tree from an XGboost
    """
    def __init__(self,max_depth,min_forest_size,max_number_of_conjunctions,pruning_method=None):
        """

        :param max_depth: Maximum allowed depths of the generated tree
        :param min_forest_size: Minimum size of the pruned forest (relevant for the pruning stage)
        :param max_number_of_conjunctions:
        :param pruning_method: Pruning method. If None then there's no pruning. 'auc' is for greedy auc-bsed pruning
        :param xgb_model: Trained XGboost model
        """
        self.min_forest_size = min_forest_size
        self.max_number_of_conjunctions = max_number_of_conjunctions
        self.pruning_method = pruning_method
        self.max_depth = max_depth

    def fit(self,train,feature_cols,label_col, xgb_model, pruned_forest=None, trees_conjunctions_total=None):
        """
        Generates the decision tree by applying the following stages:
        1. Generating a conjunction set that represents each tree of the decision forest
        2. Prune the decision forest according to the given pruning approach
        3. Generate the conjunction set (stage 1 in the algorithm presented)
        4. Create a decision tree out of the generated conjunction set

        :param train: pandas dataframe that was used for training the XGBoost
        :param feature_cols: feature column names
        :param label_col: label column name
        :param xgb_model: XGBoost
        :param pruned_forest: A list of trees, represnt a post-pruning forest. Relevant mostly for the experiment presented in the paper
        :param tree_conjunctions: This para
        """
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.int_cols = [k for k,v in train[feature_cols].dtypes.items() if 'int' in str(v)]
        self.xgb_model = xgb_model
        if pruned_forest is None or trees_conjunctions_total is None:
            self.trees_conjunctions_total = extractConjunctionSetsFromForest(self.xgb_model,train[self.label_col].unique(),self.feature_cols)
            print('Start pruning')
            self.prune(train)
        else:
            self.pruner = Pruner()
            self.trees_conjunctions_total = trees_conjunctions_total
            self.trees_conjunctions = pruned_forest
        self.cs = ConjunctionSet(max_number_of_conjunctions=self.max_number_of_conjunctions)
        self.cs.fit(self.trees_conjunctions,train, feature_cols,label_col,int_features=self.int_cols)
        print('Start ordering conjunction set in a tree structure')
        self.tree = Tree(self.cs.conjunctions, self.cs.splitting_points,self.max_depth)
        self.tree.split()
        print('Construction of tree has been completed')

    def prune(self,train):
        """

        :param train: pandas dataframe used as a pruning dataset
        :return: creates a pruned decision forest (include only the relevant trees)
        """
        if self.pruning_method == None:
            self.trees_conjunctions = self.trees_conjunctions_total
        self.pruner = Pruner()
        if self.pruning_method == 'auc':
            self.trees_conjunctions = self.pruner.max_auc_pruning(self.trees_conjunctions_total, train[self.feature_cols],
                                                                      train[self.label_col], min_forest_size=self.min_forest_size)

    def predict_proba(self,X):
        """
        Returns class probabilities

        :param X: Pandas dataframe or a numpy matrix
        :return: class probabilities for the corresponding data
        """
        return self.tree.predict_proba(X)

    def predict(self, X):
        """
        Get predictions vector

        :param X: Pandas dataframe or a numpy matrix
        :return: Predicted classes
        """
        return np.argmax(self.predict_proba(X), axis=1)

    def get_decision_paths(self, X):
        """

        :param X: Pandas data frame of [number_of_instances, number_of_features] dimension
        :return: A list of decision paths where each decision path represented as a string of nodes. one node for the leaf and the other for the decision nodes
        """
        paths = self.tree.get_decision_paths(X)
        processed_paths = []
        for path in paths:
            temp_path = []
            for node in path:
                if node.startswith('label'):
                    temp_path.append(node)
                else:
                    if '<' in node:
                        splitted = node.split('<')
                        temp_path.append(self.feature_cols[int(splitted[0])]+' < '+splitted[1])
                    else:
                        splitted = node.split('>=')
                        temp_path.append(self.feature_cols[int(splitted[0])] + ' >= ' + splitted[1])
            processed_paths.append(temp_path)
        return processed_paths

    #######################################################################
    #The following functions are only relevant for the experiment
    # They should be excluded from the documentation of the package
    ########################################################################

    def predict_proba_and_depth(self,X):
        """
        Get class probabilities and depths for each instance

        :param X: Pandas dataframe or a numpy matrix
        :return: class probabilities and the depth of each prediction
        """
        return self.tree.predict_proba_and_depth(X)

    def predict_proba_pruned_forest(self,X):
        """
        Predict_proba using the pruned forest

        :param X: Pandas dataframe or a numpy matrix
        :return: Class probabilities according to the pruned forest
        """
        return self.pruner.predict_probas(self.trees_conjunctions,X)

    def predict_proba_and_depth_forest(self,X):
        """
                Predict_proba and depth using the original forest

                :param X: Pandas dataframe or a numpy matrix
                :return: Class probabilities according to the forest and corresponding depths
        """
        probas = []
        depths = []
        for inst in X.values:
            proba=[]
            depth = 0
            for t in self.trees_conjunctions_total:
                for conj in t:
                    if conj.containsInstance(inst):
                        depth+= np.sum(np.abs(conj.features_upper)!=np.inf) + np.sum(np.abs(conj.features_lower)!=np.inf)
                        proba.append(conj.label_probas)
            depths.append(depth)
            probas.append(softmax(np.array(proba).sum(axis=0)))
        return np.array([i[0] for i in probas]), depths

    def predict_proba_and_depth_pruned_forest(self,X):
        """
        Predict_proba and depth using the pruned forest

        :param X: Pandas dataframe or a numpy matrix
        :return: Class probabilities according to the pruned forest and corresponding depths
        """
        probas = []
        depths = []
        for inst in X.values:
            proba=[]
            depth = 0
            for t in self.trees_conjunctions:
                for conj in t:
                    if conj.containsInstance(inst):
                        depth+= np.sum(np.abs(conj.features_upper)!=np.inf) + np.sum(np.abs(conj.features_lower)!=np.inf)
                        proba.append(conj.label_probas)
            depths.append(depth)
            probas.append(softmax(np.array(proba).sum(axis=0)))
        return np.array([i[0] for i in probas]), depths


