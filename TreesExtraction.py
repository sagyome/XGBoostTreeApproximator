"""
This module contains functions for extracting information of individual trees from XGBoost
"""

import re
from conjunction import *
#internal node parser:
feature_regex = re.compile('\D+(?P<node_index>\d+):\[(?P<feature>[^<]+)<(?P<value>[^\]]+)\D+(?P<left>\d+)\D+(?P<right>\d+)\D+(?P<missing>\d+)')

#leaf parser:
leaf_regex = re.compile('\D+(?P<node_index>\d+)[^\=]+=(?P<prediction>.+)')

def extractNodesFromModel(model):
    """
    Extract decision trees from XGBoost.

    :param model: XGBoost model
    :param feature_dict: {feature_name: feature_index}
    :return: trees: List of trees where trees represented as lists of dictionaries. Each dictionary represents a node within the corresponding tree
    """
    trees= []
    for tree_string in model._Booster.get_dump():
        nodes = [feature_regex.search('t' + node).groupdict() if '[' in node else leaf_regex.search('t' +node).groupdict() for node in tree_string.split('\n')[:-1]]
        trees.append(nodes)
    return trees

def extractClassValue(tree,leaf_index,label_names,class_index):
    """
    This function takes a leaf index and convert the class logit into a probability

    :param tree: dictionary that represents a decision tree
    :param leaf_index: leaf index - integer
    :param label_names: list of strings - labels
    :param class_index: index of the addressed class
    :return: class probabilitis
    """
    pred = float(tree[leaf_index]['prediction'])
    if len(label_names)>2:
        return [pred if i == class_index else 0 for i in range(len(label_names))]
    else:
        p = 1 / (1 + np.exp(pred))
        return [p,1-p]
def extractConjunctionsFromTree(tree, tree_index,leaf_index, feature_dict, label_names, class_index):
    """
    Covert the leaves of a tree into a set of conjunctions

    :param tree: list of dictionaries where each dictionary represents a node within a tree
    :param leaf_index: index of the currently processed node
    :param feature_dict: {feature name: feature index} - for converting xgboost feature names to conjunction feature indices
    :param label_names: possible class values
    :param class_index: currently addressed class - since each model is basically a binary classification of tree of a single class it's impoertant to know the relevant class
    :return: A set of conjunctions
    """
    if 'prediction' in tree[leaf_index]:
        probas = extractClassValue(tree,leaf_index,label_names,class_index)
        return [Conjunction(list(feature_dict.keys()),label_names,
                            leaf_index=[str(tree_index)+'_'+str(leaf_index)],label_probas=probas)]
    l_conjunctions = extractConjunctionsFromTree(tree,tree_index,int(tree[leaf_index]['left']),feature_dict,label_names,class_index)
    r_conjunctions = extractConjunctionsFromTree(tree,tree_index,int(tree[leaf_index]['right']),feature_dict,label_names,class_index)
    for c in l_conjunctions:
        c.addCondition(feature_dict[tree[leaf_index]['feature']],float(tree[leaf_index]['value']),'upper')
    for c in r_conjunctions:
        c.addCondition(feature_dict[tree[leaf_index]['feature']],float(tree[leaf_index]['value']),'lower')
    return l_conjunctions + r_conjunctions

def merge_two_conjunction_sets(conj_list1,conj_list2):
    """
    Gets two conjunction sets and return a set that is a cartesian product of the two input sets

    :param conj_list1:
    :param conj_list2:
    :return:
    """
    new_conjunction_list=[]
    for c1 in conj_list1:
        for c2 in conj_list2:
            if not c1.isContradict(c2):
                new_conjunction_list.append(c1.merge(c2))
    return new_conjunction_list

def postProcessTrees(conjunction_sets, num_of_labels):
    """
    This function is used for integrating mulitple binary trees into a single tree of multiple labels

    :param conjunction_sets: list of lists of conjunctions
    :param num_of_labels: number of labels in the dataset that was used for training
    :return: new list of conjunctions
    """

    new_conj_list = []
    for i in range(0, len(conjunction_sets), num_of_labels):
        conj = conjunction_sets[i]
        for j in range(i + 1, i + num_of_labels):
            conj = merge_two_conjunction_sets(conj, conjunction_sets[j])
        new_conj_list.append(conj)
    return new_conj_list

def extractConjunctionSetsFromForest(model,unique_labels,features):
    """
    This function takes XGBoost model and returns a list of trees where each tree is represented as a list of conjunctions.
    Each of the tree conjunctions stands for a single decision path

    :param model: XGBoost model
    :param unique_labels: label names
    :param features: feature names
    :return: a list of conjunctions
    """

    trees = extractNodesFromModel(model)
    num_of_labels = len(unique_labels)
    feature_dict = {v:k for k,v in enumerate(features)}
    conjunction_sets = {}
    for i,t in enumerate(trees): #i stands for the corresponding class index
        indexed_tree = {int(v['node_index']): v for v in t}
        conjunction_sets[i] = extractConjunctionsFromTree(indexed_tree,i,0, feature_dict, unique_labels, i % num_of_labels)
    if num_of_labels > 2:
        return postProcessTrees(conjunction_sets,num_of_labels)
    else:
        return list(conjunction_sets.values())
