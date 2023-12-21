import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from statistics import mode

def generate_bootstrap(xTrain, yTrain):
    """
    Helper function to generate a bootstrap sample from the data. Each
    call should generate a different random bootstrap sample!

    Parameters
    ----------
    xTrain : nd-array with shape n x d
        Training data 
    yTrain : 1d array with shape n
        Array of responses associated with training data.

    Returns
    -------
    xBoot : nd-array with shape n x d
        Bootstrap sample from xTrain
    yBoot : 1d array with shape n
        Array of responses associated with xBoot
    oobIdx : 1d array with shape k (which can be 0-(n-1))
        Array containing the out-of-bag sample indices from xTrain 
        such that using this array on xTrain will yield a matrix 
        with only the out-of-bag samples (i.e., xTrain[oobIdx, :]).
    """
    n, d = xTrain.shape
    indices = np.random.choice(n, n) 

    # Create bootstrap sample
    xBoot = xTrain[indices]
    yBoot = yTrain[indices]

    # Identify out-of-bag indices
    unique_indices = np.unique(indices)
    oobIdx = np.setdiff1d(np.arange(n), unique_indices)

    return xBoot, yBoot, oobIdx
    
    


def generate_subfeat(xTrain, maxFeat):
    """
    Helper function to generate a subset of the features from the data. Each
    call is likely to yield different columns (assuming maxFeat is less than
    the original dimension)

    Parameters
    ----------
    xTrain : nd-array with shape n x d
        Training data 
    maxFeat : int
        Maximum number of features to consider in each tree

    Returns
    -------
    xSubfeat : nd-array with shape n x maxFeat
        Subsampled features from xTrain
    featIdx: 1d array with shape maxFeat
        Array containing the subsample indices of features from xTrain
    """
    n, d = xTrain.shape
    # Randomly choose maxFeat features without replacement
    featIdx = np.random.choice(d, maxFeat, replace=False)  

    # Create feature subset
    xSubfeat = xTrain[:, featIdx]

    return xSubfeat, featIdx


class RandomForest(object):
    nest = 0           # number of trees
    maxFeat = 0        # maximum number of features
    maxDepth = 0       # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None   # splitting criterion
    model = {}         # keeping track of all the models developed, where
                       # the key is the bootstrap sample. The value should be a dictionary
                       # and have 2 keys: "tree" to store the tree built
                       # "feat" to store the corresponding featIdx used in the tree


    def __init__(self, nest, maxFeat, criterion, maxDepth, minLeafSample):
        """
        Decision tree constructor

        Parameters
        ----------
        nest: int
            Number of trees to have in the forest
        maxFeat: int
            Maximum number of features to consider in each tree
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int 
            Maximum depth of the decision tree
        minLeafSample : int 
            Minimum number of samples in the decision tree
        """
        self.nest = nest
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample
        self.maxFeat = maxFeat
        self.model = {}

    def train(self, xFeat, y):
        """
        Train the random forest using the data

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of responses associated with training data.

        Returns
        -------
        stats : object
            Keys represent the number of trees and
            the values are the out of bag errors
        """
        stats = {}
        # Train random forest
        for tree in range(self.nest):
            # Use helper functions to get bootstrapped data and subfeatures
            xBoot, yBoot, oobIdx = generate_bootstrap(xFeat, y)
            xSubfeat, featIdx = generate_subfeat(xBoot, self.maxFeat)

            # Create a decision tree
            decisionTree = DecisionTreeClassifier(criterion = self.criterion, max_depth = self.maxDepth, min_samples_leaf= self.minLeafSample)
            # Fit decision tree to the xsubfeatures
            decisionTree.fit(xSubfeat, yBoot)

            oobX = xFeat[oobIdx,:]
            oobY = y[oobIdx]

             # Calculate class probabilities for the OOB samples
            oobProbs = decisionTree.predict_proba(oobX[:, featIdx])

            # Extract probabilities for the positive class
            positiveClassProbs = oobProbs[:, 1]

            # Convert probabilities to predicted class labels (0 or 1)
            oobPredictions = np.round(positiveClassProbs)

            # Calculate oob error
            oobError = 1.0 - np.mean(oobPredictions == oobY)

            self.model[tree] = {"tree": decisionTree, "feat": featIdx}
            stats[tree + 1] = oobError



        return stats

    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict 
        what class the values will have.

        Parameters
        ----------
        xFeat : nd-array with shape m x d
            The data to predict.  

        Returns
        -------
        yHat : 1d array or list with shape m
            Predicted response per sample
        """
        yHat = []

        for obs in xFeat:
            predictions = []
            for key, value in self.model.items():
                tree = value['tree']
                featIdx = value['feat']
                xSub = obs[featIdx].reshape(1, -1)

                # Use predict_proba to get class probabilities
                class_probs = tree.predict_proba(xSub)

                # Choose the class with the highest probability as the prediction
                prediction = np.argmax(class_probs)
                predictions.append(prediction)

            # Use mode to get the majority vote
            majority = mode(predictions)
            yHat.append(majority)

        return yHat


def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("xTrain",
                        help="filename for features of the training data")
    parser.add_argument("yTrain",
                        help="filename for labels associated with training data")
    parser.add_argument("xTest",
                        help="filename for features of the test data")
    parser.add_argument("yTest",
                        help="filename for labels associated with the test data")
    parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")
    
    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    np.random.seed(args.seed)   
    model = RandomForest(maxFeat=10, criterion="gini", maxDepth=12, minLeafSample=6,  nest=args.epoch)
    trainStats = model.train(xTrain, yTrain)
    print(trainStats)
    yHat = model.predict(xTest)
    print(accuracy_score(yTest, yHat))


    # # CODE TO FIND OPTIMAL PARAMETERS
    # # Function to train and evaluate the random forest model
    # def train_and_evaluate_model(max_feat, max_depth, min_leaf, nest_size):
    #     model = RandomForest(maxFeat=max_feat, criterion="gini", maxDepth=max_depth, minLeafSample=min_leaf, nest=nest_size)
    #     model.train(xTrain, yTrain)
    #     yHat = model.predict(xTest)
    #     classError = 1 - accuracy_score(yTest, yHat)
    #     return classError

    # # Plotting function
    # def plot_classification_error(x_values, y_values, xlabel, title):
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(x_values, y_values, marker='^', linestyle='--', color='r')
    #     plt.title(title)
    #     plt.xlabel(xlabel)
    #     plt.ylabel('Classification Error')
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.show()

    # # Plot Classification Error vs Number of Max Features
    # maxFeatOpts = range(1, xTrain.shape[1]+1)
    # maxFeatAccs = [train_and_evaluate_model(i, 5, 4, 50) for i in maxFeatOpts]
    # plot_classification_error(maxFeatOpts, maxFeatAccs, 'Number of Max Features', 'Classification Error vs Number of Max Features')

    # # Plot Classification Error vs Depth of Trees
    # maxDepthOpts = range(1, 20)
    # maxDepthAccs = [train_and_evaluate_model(11, i, 4, 50) for i in maxDepthOpts]
    # plot_classification_error(maxDepthOpts, maxDepthAccs, 'Tree Depth', 'Classification Error vs Depth of Trees')

    # # Plot Classification Error vs Minimum Leaf Size
    # maxLeafOpts = range(1, 11)
    # maxLeafAccs = [train_and_evaluate_model(11, 12, i, 50) for i in maxLeafOpts]
    # plot_classification_error(maxLeafOpts, maxLeafAccs, 'Leaf Size', 'Classification Error vs Minimum Leaf Size')

    # # Plot Classification Error vs Nest Size
    # nestOpts = range(1, 150)
    # nestAccs = [train_and_evaluate_model(11, 12, 1, i) for i in nestOpts]
    # plot_classification_error(nestOpts, nestAccs, 'Nest Size', 'Classification Error vs Nest Size')

    


if __name__ == "__main__":
    main()