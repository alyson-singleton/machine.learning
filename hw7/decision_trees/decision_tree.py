import numpy as np
import random
import copy
import math

def prob(dataset):
    if len(dataset) == 0:
        return 0
    return sum(dataset)/len(dataset)

def train_error(dataset):
    '''
        TODO:
        Calculate the train error of the subdataset and return it.
        For a dataset with two classes:
        C(p) = min{p, 1-p}
    '''
    p = prob(dataset)
    return min(p, 1-p)

def entropy(dataset):
    '''
        TODO:x
        Calculate the entropy of the subdataset and return it.
        This function is used to calculate the entropy for a dataset with 2 classes.
        Mathematically, this function return:
        C(p) = -p*log(p) - (1-p)log(1-p)
    '''

    p = prob(dataset)
    if p == 0 or p == 1:
        return 0
    return -p*math.log(p) - (1-p)*math.log(1-p)


def gini_index(dataset):
    '''
        TODO:
        Calculate the gini index of the subdataset and return it.
        For dataset with 2 classes:
        C(p) = 2*p*(1-p)
    '''

    p = prob(dataset)
    return 2*p*(1-p)

class Node:
    '''
    Helper to construct the tree structure.
    '''
    def __init__(self, left=None, right=None, depth=0, index_split_on=0, isleaf=False, label=1, info=None):
        self.left = left
        self.right = right
        self.depth = depth
        self.index_split_on = index_split_on
        self.isleaf = isleaf
        self.label = label
        self.info = {} if info is None else info


class DecisionTree:

    def __init__(self, data, validation_data=None, gain_function=entropy, max_depth=40):
        self.max_depth = max_depth
        self.root = Node()
        self.gain_function = gain_function

        indices = list(range(1, len(data[0])))

        self._split_recurs(self.root, data, indices)

        # Pruning
        if not (validation_data is None):
            self._prune_recurs(self.root, validation_data)

    def predict(self, features):
        '''
        Helper function to predict the label given a row of features.
        You do not need to modify this.
        '''
        return self._predict_recurs(self.root, features)

    def accuracy(self, data):
        '''
        Helper function to calculate the accuracy on the given data.
        You do not need to modify this.
        '''
        return 1 - self.loss(data)

    def loss(self, data):
        '''
        Helper function to calculate the loss on the given data.
        You do not need to modify this.
        '''
        cnt = 0.0
        test_Y = [row[0] for row in data]
        for i in range(len(data)):
            prediction = self.predict(data[i])
            if (prediction != test_Y[i]):
                cnt += 1.0
        return cnt/len(data)

    def _predict_recurs(self, node, row):
        '''
        Helper function to predict the label given a row of features.
        Traverse the tree until leaves to get the label.
        You do not need to modify this.
        '''
        if node.isleaf or node.index_split_on == 0:
            return node.label
        split_index = node.index_split_on
        if row[split_index]:
            return self._predict_recurs(node.left, row)
        else:
            return self._predict_recurs(node.right, row)


    def _prune_recurs(self, node, validation_data):
        '''
        TODO:
        Prune the tree bottom up recursively. Nothing needs to be returned.
        Do not prune if the node is a leaf.
        Do not prune if the node is non-leaf and has at least one non-leaf child.
        Prune if deleting the node could reduce loss on the validation data.'''

        if node.isleaf:
            return
        if node.right.isleaf==False and node.right is not None:
            self._prune_recurs(node.right, validation_data)
        if node.left.isleaf==False and node.left is not None:
            self._prune_recurs(node.left, validation_data)
        else:
            no_prune_loss = self.loss(validation_data)

            node.isleaf = True

            count_ones, count_zeros = validation_data[0].count(1), validation_data[0].count(0)
            if count_ones > count_zeros:
                node.label = 1
            if count_zeros > count_ones:
                node.label = 0
            else: node.label = random.randint(0,1)

            prune_loss = self.loss(validation_data)

            if prune_loss > no_prune_loss:
                node.isleaf = False
                node.label = -1
            else:
                node.left = None
                node.right = None




    def _is_terminal(self, node, data, indices):
        '''
        TODO:
        Helper function to determine whether the node should stop splitting.
        Stop the recursion:
            1. The dataset is empty.
            2. There are no more indices to split on.
            3. All the instances in this dataset belong to the same class
            4. The depth of the nodex exceede the maximum depth.
        Return:
            - A boolean, True indicating the current node should be a leaf.
            - A label, indicating the label of the leaf (-1 if False)'''

        if len(data) == 0:
            return True, random.randint(0,1)
        if len(indices) == 0 or len(set([row[0] for row in data]))==1 or node.depth > self.max_depth:
            return True, self._majority_label(data)
        else: return False, -1

    def _majority_label(self, data):
        '''
        Finds the best label for the remaining data
        '''
        threshold = len(data)/2
        label_sum = sum([row[0] for row in data])
        if label_sum > threshold:
            return 1
        else:
            return 0


    def _split_recurs(self, node, rows, indices):
        '''
        TODO:
        Recursively split the node based on the rows and indices given.
        Nothing needs to be returned.
        First use _is_terminal() to check if the node needs to be splitted.
        Then select the column that has the maximum infomation gain to split on.
        Also store the label predicted for this node.
        Then split the data based on whether satisfying the selected column.
        The node should not store data, but the data is recursively passed to the children.
        '''
        terminal_bool, terminal_label = self._is_terminal(node, rows, indices)

        if terminal_bool:
            node.label = terminal_label
            node.isleaf = True
        else:
            updatedGain = 0
            updatedAttribute = None
            gain_array = np.zeros(len(indices))
            for index in range(len(indices)):
                gain_array[index] = self._calc_gain(rows, indices[index], self.gain_function)

            splitting_index = np.argmax(gain_array)
            updatedAttribute = indices[np.argmax(gain_array)]
            updatedGain = np.max(gain_array)

            node.index_split_on = updatedAttribute
            node.info['cost'] = updatedGain
            node.info['data_size'] = len(rows)
            node.label = self._majority_label(rows)
            node.isleaf = False

            left = [row for row in rows if row[updatedAttribute]==1]
            right = [row for row in rows if row[updatedAttribute]==0]

            node.left = Node(depth = node.depth + 1)
            node.right = Node(depth = node.depth + 1)
            updatedIndicesleft = indices[:]
            updatedIndicesright = indices[:]
            updatedIndicesleft.remove(updatedAttribute)
            updatedIndicesright.remove(updatedAttribute)

            self._split_recurs(node.left, left, updatedIndicesleft)
            self._split_recurs(node.right, right, updatedIndicesright)

    def _calc_gain(self, data, split_index, gain_function):
        '''
        TODO:
        Calculate the gain of the proposed splitting and return it.
        Gain = C(P[y=1]) - (P[x_i=True] * C(P[y=1|x_i=True]) + P[x_i=False]C(P[y=1|x_i=False)])
        Here the C(p) is the gain_function. For example, if C(p) = min(p, 1-p), this would be
        considering training error gain. Other alternatives are entropy and gini functions.
        '''
        
        x_i_true = sum([row[split_index] for row in data])/len(data)
        left, right = self._split_data(data, split_index)
        return gain_function([row[0] for row in data]) - (x_i_true * gain_function(left) +
            (1- x_i_true) * gain_function(right))

    def _split_data(self, data, split_index):
        left, right = [], []
        for row in data:
            if row[split_index] == 1:
                left.append(row[0])
            else:
                right.append(row[0])
        return left, right


    def print_tree(self):
        '''
        Helper function for tree_visualization.
        Only effective with very shallow trees.
        You do not need to modify this.
        '''
        temp = []
        output = []
        print('---START PRINT TREE---')
        def print_subtree(node, indent=''):
            if node is None:
                return str("None")
            if node.isleaf:
                return str(node.label)
            else:
                decision = 'split attribute = %d; cost = %f; sample size = %d' % (node.index_split_on, node.info['cost'], node.info['data_size'])
            left = indent + 'T -> '+ print_subtree(node.left, indent + '\t\t')
            right = indent + 'F -> '+ print_subtree(node.right, indent + '\t\t')
            return (decision + '\n' + left + '\n' + right)

        print(print_subtree(self.root))
        print('----END PRINT TREE---')




    def loss_plot_vec(self, data):
        '''
        Helper function to visualize the loss when the tree expands.
        You do not need to modify this.
        '''
        self._loss_plot_recurs(self.root, data, 0)
        loss_vec = []
        q = [self.root]
        num_correct = 0
        while len(q) > 0:
            node = q.pop(0)
            num_correct = num_correct + node.info['curr_num_correct']
            loss_vec.append(num_correct)
            if node.left != None:
                q.append(node.left)
            if node.right != None:
                q.append(node.right)

        return 1 - np.array(loss_vec)/len(data)



    def _loss_plot_recurs(self, node, rows, prev_num_correct):
        '''
        Helper function to visualize the loss when the tree expands.
        You do not need to modify this.
        '''
        labels = [row[0] for row in rows]
        curr_num_correct = labels.count(node.label) - prev_num_correct
        node.info['curr_num_correct'] = curr_num_correct

        if not node.isleaf:
            left_data, right_data = [], []
            left_num_correct, right_num_correct = 0, 0
            for row in rows:
                if row[node.index_split_on]:
                    left_data.append(row)
                else:
                    right_data.append(row)

            left_labels = [row[0] for row in left_data]
            left_num_correct = left_labels.count(node.label)
            right_labels = [row[0] for row in right_data]
            right_num_correct = right_labels.count(node.label)

            if node.left != None:
                self._loss_plot_recurs(node.left, left_data, left_num_correct)
            if node.right != None:
                self._loss_plot_recurs(node.right, right_data, right_num_correct)
