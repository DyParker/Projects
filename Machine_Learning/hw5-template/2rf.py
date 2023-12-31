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
    n = xTrain.shape[0]
    bootstrapInd = np.random.choice(n, n)
    xBoot = xTrain[bootstrapInd]
    yBoot = yTrain[bootstrapInd]

    ones = np.ones(n, dtype=bool)
    ones[bootstrapInd] = False
    oobIdx = np.where(ones)[0]



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

    colInds = np.random.choice(range(d), maxFeat, replace = False)
    xSubfeat = xTrain[:,colInds]


    return xSubfeat, colInds


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

        for i in range(self.nest):
            xBoot, yBoot, oobIdx = generate_bootstrap(xFeat, y)

            xSubfeat, featIdx = generate_subfeat(xBoot, self.maxFeat)

            tree = DecisionTreeClassifier(
                criterion = self.criterion,
                max_depth = self.maxDepth,
                min_samples_leaf= self.minLeafSample
            )

            tree.fit(xSubfeat, yBoot)

            oobX = xFeat[oobIdx,:]
            oobY = y[oobIdx]

            oobError = 1 - tree.score(oobX[:, featIdx], oobY)

            self.model[i] = {"tree":tree, "feat":featIdx}
            stats[i+1] = oobError



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

        for sample in xFeat:
            treePredictions = []
            for i in self.model.keys():
                tree = self.model[i]["tree"]
                xSub = sample[self.model[i]["feat"]].reshape(1,-1)
                prediction = tree.predict(xSub)
                treePredictions.append(prediction[0])

            majorityVote = mode(treePredictions)
            yHat.append(majorityVote)

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
    parser.add_argument("epoch",
                        type=int, help="number of epochs to train the model")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")
    
    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    np.random.seed(args.seed)   
    model = RandomForest(maxFeat=10, criterion="gini", maxDepth=11, minLeafSample=1,  nest=args.epoch)
    trainStats = model.train(xTrain, yTrain)
    print(trainStats)
    yHat = model.predict(xTest)
    print(accuracy_score(yTest, yHat))


    ## CODE TO FIND OPTIMAL PARAMETERS

    # maxFeatOpts = range(1, xTrain.shape[1])
    # maxFeatAccs = []

    # for i in maxFeatOpts:
    #     model = RandomForest(maxFeat=i, criterion="gini", maxDepth=5, minLeafSample=4, nest=50)
    #     model.train(xTrain, yTrain)
    #     yHat = model.predict(xTest)
    #     classError = 1 - accuracy_score(yTest, yHat)
    #     maxFeatAccs.append(classError)

    # plt.figure(figsize=(8, 6))
    # plt.plot(maxFeatOpts, maxFeatAccs, marker='o', linestyle='-', color='b')
    # plt.title('Classification Error vs Number of Max Features')
    # plt.xlabel('Number of Max Features')
    # plt.ylabel('Classification Error')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # maxDepthOpts = range(1, 20)
    # maxDepthAccs = []
    
    # for i in maxDepthOpts:
    #     model = RandomForest(maxFeat=10, criterion="gini", maxDepth=i, minLeafSample=4, nest=50)
    #     model.train(xTrain, yTrain)
    #     yHat = model.predict(xTest)
    #     classError = 1 - accuracy_score(yTest, yHat)
    #     maxDepthAccs.append(classError)

    # plt.figure(figsize=(8, 6))
    # plt.plot(maxDepthOpts, maxDepthAccs, marker='o', linestyle='-', color='b')
    # plt.title('Classification Error vs Depth of Trees')
    # plt.xlabel('Tree Depth')
    # plt.ylabel('Classification Error')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # maxLeafOpts = range(1, 10)
    # maxLeafAccs = []

    # for i in maxLeafOpts:
    #     model = RandomForest(maxFeat=10, criterion="gini", maxDepth=12, minLeafSample=i, nest=50)
    #     model.train(xTrain, yTrain)
    #     yHat = model.predict(xTest)
    #     classError = 1 - accuracy_score(yTest, yHat)
    #     maxLeafAccs.append(classError)

    # plt.figure(figsize=(8, 6))
    # plt.plot(maxLeafOpts, maxLeafAccs, marker='o', linestyle='-', color='b')
    # plt.title('Classification Error vs Minimum Leaf Size')
    # plt.xlabel('Leaf Size')
    # plt.ylabel('Classification Error')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # nestOpts = range(1, 150)
    # nestAccs = []

    # for i in nestOpts:
    #     model = RandomForest(maxFeat=10, criterion="gini", maxDepth=12, minLeafSample=1, nest=i)
    #     model.train(xTrain, yTrain)
    #     yHat = model.predict(xTest)
    #     classError = 1 - accuracy_score(yTest, yHat)
    #     nestAccs.append(classError)

    # plt.figure(figsize=(8, 6))
    # plt.plot(nestOpts, nestAccs, marker='o', linestyle='-', color='b')
    # plt.title('Classification Error vs Nest Size')
    # plt.xlabel('Nest Size')
    # plt.ylabel('Classification Error')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()



if __name__ == "__main__":
    main()