import pandas as pd 
import numpy as np 
import math

class Node:
    def __init__(self, ids = [],  depth = 0, children = [], entropy = 0):
#        print(ids[:10])
        self.ids = ids
        self.depth = depth
        self.children = children
        self.entropy = entropy
        self.attribute = None
        self.label = None 
        self.value_order = None

    def setAttribute(self, attribute, value_order):
        self.attribute = attribute
        self.value_order = value_order

    def setLabel(self, label):
        self.lable = label

class DecisionTree:
    def __init__(self, depthNode = 20, thredhold = 0.05):
        self.depthNode = depthNode
        self.thredhold = thredhold

    def fit(self, data, target):
        self.data = data
        self.attributes = list(data)
        self.target = target
        self.labels = target.unique()
        
        self.root = Node(ids= range(self.data.shape[0]),entropy= self.info_entropy(self.target), depth= 0)

        queue = [self.root]
        while queue:
            node = queue.pop()
            if node.depth < self.depthNode and node.entropy > self.thredhold:
                node.children = self.split(node)
                queue += node.children
            else:
                self.setLabel(node)
    
    def predict(self, X_train):
        y_predict = [None]*X_train.size
        for i in range(X_train.size):
            x = X_train.iloc[i,:]
            node = self.root
            while node.children:
                node = node.children[ node.value_order.index[x[node.attribute]]]
            y_predict[i] = node.label
        return y_predict

    def info_entropy(self, rows):
        entropy = 0.0
#        print(rows.unique())
        for i in rows.unique():
            p = rows[rows == i].size / rows.size
#             print(p)
            entropy += - p * math.log(p)
        return entropy

    def split(self, node):
        ids = node.ids
        data = self.data.iloc[ids,:]
        best_gain = 0
        best_attribute = None
        best_split = []
        value_order = None
        for i,attr in enumerate(self.attributes):
            if attr != node.attribute:
                data_i = data.iloc[:, i]
                value_unique = data_i.unique()
                split_idx= []
                for value in value_unique:
                    sub_ids = data_i.index[data_i == value].tolist()
                    split_idx.append(sub_ids)
                entropy_sub = 0.0
                print(split_idx)
                for sub_ids in split_idx:
                    entropy_sub += len(sub_ids)* self.info_entropy(self.target[sub_ids]) / len(ids)
                gain = node.entropy - entropy_sub
                if gain > best_gain:
                    best_gain = gain
                    best_attribute = attr
                    best_split = split_idx
                    value_order = value_unique
        node.setAttribute(best_attribute, value_order)
        childrenNode = [Node(ids = sub_ids, depth= node.depth + 1, entropy=self.info_entropy(self.target[sub_ids])) for sub_ids in best_split]
        return childrenNode


    def setLabel(self, node):
        ids = node.ids
        node.setLabel(self.target[ids].value_counts()[0])