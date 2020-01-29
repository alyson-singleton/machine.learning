import random
import numpy as np
import matplotlib.pyplot as plt

from get_data import get_data
from decision_tree import DecisionTree, train_error, entropy, gini_index


def loss_plot(ax, title, tree, pruned_tree, train_data, test_data):
    '''
        Example plotting code. This plots four curves: the training and testing
        average loss using tree and pruned tree.
        You do not need to change this code!
        Arguments:
            - ax: A matplotlib Axes instance.
            - title: A title for the graph (string)
            - tree: An unpruned DecisionTree instance
            - pruned_tree: A pruned DecisionTree instance
            - train_data: Training dataset returned from get_data
            - test_data: Test dataset returned from get_data
    '''
    fontsize=8
    ax.plot(tree.loss_plot_vec(train_data), label='train non-pruned')
    ax.plot(tree.loss_plot_vec(test_data), label='test non-pruned')
    ax.plot(pruned_tree.loss_plot_vec(train_data), label='train pruned')
    ax.plot(pruned_tree.loss_plot_vec(test_data), label='test pruned')


    ax.locator_params(nbins=3)
    ax.set_xlabel('number of nodes', fontsize=fontsize)
    ax.set_ylabel('loss', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    legend = ax.legend(loc='upper center', shadow=True, fontsize=fontsize-2)

def loss_plot_depth(loss, title):
    '''
    plot the loss vs depth
    '''
    depth = range(1, 16)
    plt.scatter(depth, loss)
    plt.title(title)
    plt.xlabel("depth")
    plt.ylabel("loss")
    plt.show()

def generate_loss(train_data, validation_data, gain_function):
    '''
    find the loss for the different depths of a given gain function
    '''
    loss = []
    for i in range(1, 16):
        tree = DecisionTree(train_data, validation_data=validation_data, gain_function=gain_function, max_depth=i)
        loss.append(tree.loss(train_data))
    return loss

def explore_depth(train_data, validation_data):
    '''
    generate the 6 graphs of depth vs. loss
    '''
    #entropy_loss = generate_loss(train_data, None, entropy)
    #loss_plot_depth(entropy_loss, "Entropy without Pruning")
    #entropy_loss = generate_loss(train_data, validation_data, entropy)
    #loss_plot_depth(entropy_loss, "Entropy with Pruning")

def explore_dataset(filename, class_name):
    train_data, validation_data, test_data = get_data(filename, class_name)

    # TODO: Print 12 loss values associated with the dataset.
    # For each measure of gain (training error, entropy, gini):
    #      (a) Print average training loss (not-pruned)
    #      (b) Print average test loss (not-pruned)
    #      (c) Print average training loss (pruned)
    #      (d) Print average test loss (pruned)

    train_error_tree = DecisionTree(train_data, gain_function=train_error, max_depth=40)
    entropy_tree = DecisionTree(train_data, gain_function=entropy, max_depth=40)
    gini_index_tree = DecisionTree(train_data, gain_function=gini_index, max_depth=40)

    train_error_tree_prune = DecisionTree(train_data, validation_data=validation_data, gain_function=train_error, max_depth=40)
    entropy_tree_prune = DecisionTree(train_data, validation_data=validation_data, gain_function=entropy, max_depth=40)
    gini_index_tree_prune = DecisionTree(train_data, validation_data=validation_data, gain_function=gini_index, max_depth=40)

    print("Train error training loss wo pruning: " + str(train_error_tree.loss(train_data)))
    print("Train error test loss wo pruning: " + str(train_error_tree.loss(test_data)))
    print("Train error training loss w pruning: " + str(train_error_tree_prune.loss(train_data)))
    print("Train error test loss w pruning: " + str(train_error_tree_prune.loss(test_data)))

    print("Entropy training loss wo pruning: " + str(entropy_tree.loss(train_data)))
    print("Entropy test loss wo pruning: " + str(entropy_tree.loss(test_data)))
    print("Entropy training loss w pruning: " + str(entropy_tree_prune.loss(train_data)))
    print("Entropy test loss w pruning: " + str(entropy_tree_prune.loss(test_data)))

    print("Gini training loss wo pruning: " + str(gini_index_tree.loss(train_data)))
    print("Gini test loss wo pruning: " + str(gini_index_tree.loss(test_data)))
    print("Gini training loss w pruning: " + str(gini_index_tree_prune.loss(train_data)))
    print("Gini test loss w pruning: " + str(gini_index_tree_prune.loss(test_data)))

    #explore_depth(train_data, validation_data)

    # TODO: Feel free to print or plot anything you like here. Just comment
    # make sure to comment it out, or put it in a function that isn't called
    # by default when you hand in your code!

def main():
    ########### PLEASE DO NOT CHANGE THESE LINES OF CODE! ###################
    random.seed(1)
    np.random.seed(1)
    #########################################################################

    explore_dataset('data/chess.csv', 'won')
    explore_dataset('data/spam.csv', '1')

main()
