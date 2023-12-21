import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class Node:
    def __init__(self, feature=None, left=None, right=None, split=None, value=None):
        # For decision node
        self.feature = feature
        self.left = left
        self.right = right
        self.split = split

        # For leaf node
        self.value = value

def calculate_split_score(y, criterion):
    """
    Given a numpy array of labels associated with a node, y, 
    calculate the score based on the criterion specified.

    Parameters
    ----------
    y : numpy.1d array with shape n
        Array of labels associated with a node
    criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
    Returns
    -------
    score : float
        The gini or entropy associated with a node
    """
    # Get total count
    total_samples = len(y)
    if total_samples == 0:
        return 0  # Return 0 if the array is empty
    
    # Count the number of samples in each class (0 and 1)
    class_counts = np.bincount(y)
    
    # Calculate the probabilities of each class
    class_probabilities = class_counts / total_samples
    if criterion == 'gini':
        # Calculate the Gini score
        gini_score = 1.0 - np.sum(np.square(class_probabilities))
        return gini_score
    elif criterion == 'entropy':
        # Calculate the Entropy score
        entropy_score = -np.sum(class_probabilities * np.log2(class_probabilities + 1e-10))
        return entropy_score
    else:
        return ValueError("Neither gini or entropy stated") # Return Error if neither gini or entropy is stated





class DecisionTree(object):
    maxDepth = 0       # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None   # splitting criterion

    def __init__(self, criterion, maxDepth, minLeafSample):
        """
        Decision tree constructor

        Parameters
        ----------
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int 
            Maximum depth of the decision tree
        minLeafSample : int 
            Minimum number of samples in the decision tree
        """
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample
        # Add a root
        self.root=None

    # Get the best split for each decision node
    def get_best_split(self, xFeat, y):
        
        # Initialize score
        bestScore = float('inf')

        # Loop through the features
        for featureIndex in range(len(xFeat[0])):

            # Check each split and choose the lowest score
            for i in range(len(xFeat)):
                yLeft = y[xFeat[:, featureIndex] <= xFeat[i, featureIndex]]
                yRight = y[xFeat[:, featureIndex] > xFeat[i, featureIndex]]

                # Weight splits
                weightedLeftScore = np.size(yLeft)/np.size(y) * calculate_split_score(yLeft, self.criterion)
                weightedRightScore = np.size(yRight)/np.size(y) * calculate_split_score(yRight, self.criterion)
                score = weightedRightScore + weightedLeftScore

                # Update is new split is more optimal
                if score < bestScore:
                    self.root.split = xFeat[i, featureIndex]
                    self.root.feature = featureIndex
                    bestScore = score

        # Use the split
        leftXFeat = xFeat[xFeat[:, self.root.feature] <= self.root.split]
        rightXFeat = xFeat[xFeat[:, self.root.feature] > self.root.split]

        yLeft = y[xFeat[:, self.root.feature] <= self.root.split]
        yRight = y[xFeat[:, self.root.feature] > self.root.split]

        # Return the split values
        return leftXFeat, rightXFeat, yLeft, yRight

    def train(self, xFeat, y):
        """
        Train the decision tree model.

        Parameters
        ----------
        xFeat : numpy.nd-array with shape n x d
            Training data 
        y : numpy.1d array with shape n
            Array of labels associated with training data.

        Returns
        -------
        self : object
        """
        # TODO do whatever you need
        self.xFeat = np.array(xFeat)
        self.y = np.array(y)

        if len(y) > 0:
            counts = np.bincount(self.y)
            self.root = Node(value = np.argmax(counts))
        else:
            self.root = Node(value = 0)

        # Stop criteria
        if not calculate_split_score(self.y, self.criterion) or len(self.xFeat) < 2*self.minLeafSample or not self.maxDepth: 
            return self
        
        leftXFeat, rightXFeat, yLeft, yRight = self.get_best_split(xFeat, y)
        
        # Recursively build the decision tree
        self.root.left = DecisionTree(self.criterion, self.maxDepth - 1, self.minLeafSample).train(leftXFeat, yLeft)
        self.root.right = DecisionTree(self.criterion, self.maxDepth - 1, self.minLeafSample).train(rightXFeat, yRight)
        return self
        


    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict what class the values will have.

        Parameters
        ----------
        xFeat : numpy.nd-array with shape m x d
            The data to predict.

        Returns
        -------
        yHat : numpy.1d array with shape m
            Predicted class label per sample
        """
        yHat = np.array([])  # Initialize an array to store the predicted class labels
        # Use a pd to reference features
        xFeat = pd.DataFrame(xFeat)
        for x in range(len(xFeat)):
            yHat = np.append(yHat, self.onePredict(xFeat.iloc[x]))
        return yHat
    
    # Get predictions
    def onePredict(self, sample):
        if self.root.left is None and self.root.right is None:
            return self.root.value
        elif sample[self.root.feature] <= self.root.split:
            return self.root.left.onePredict(sample)
        else:
            return self.root.right.onePredict(sample)
        

def dt_train_test(dt, xTrain, yTrain, xTest, yTest):
    """
    Given a decision tree model, train the model and predict
    the labels of the test data. Returns the accuracy of
    the resulting model.

    Parameters
    ----------
    dt : DecisionTree
        The decision tree with the model parameters
    xTrain : numpy.nd-array with shape n x d
        Training data 
    yTrain : numpy.1d array with shape n
        Array of labels associated with training data.
    xTest : numpy.nd-array with shape m x d
        Test data 
    yTest : numpy.1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    acc : float
        The accuracy of the trained knn model on the test data
    """
    # train the model
    dt.train(xTrain, yTrain)
    # predict the training dataset
    yHatTrain = dt.predict(xTrain)
    trainAcc = accuracy_score(yTrain, yHatTrain)
    # predict the test dataset
    yHatTest = dt.predict(xTest)
    testAcc = accuracy_score(yTest, yHatTest)
    return trainAcc, testAcc


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("md",
                        type=int,
                        help="maximum depth")
    parser.add_argument("mls",
                        type=int,
                        help="minimum leaf samples")
    parser.add_argument("--xTrain",                        
                        default="q4xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q4yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q4xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q4yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain).to_numpy()
    yTrain = pd.read_csv(args.yTrain).to_numpy().flatten()
    xTest = pd.read_csv(args.xTest).to_numpy()
    yTest = pd.read_csv(args.yTest).to_numpy().flatten()
    # create an instance of the decision tree using gini
    dt1 = DecisionTree('gini', args.md, args.mls)
    trainAcc1, testAcc1 = dt_train_test(dt1, xTrain, yTrain, xTest, yTest)
    print("GINI Criterion ---------------")
    print("Training Acc:", trainAcc1)
    print("Test Acc:", testAcc1)
    dt = DecisionTree('entropy', args.md, args.mls)
    trainAcc, testAcc = dt_train_test(dt, xTrain, yTrain, xTest, yTest)
    print("Entropy Criterion ---------------")
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)


    # Get accuracies for a mls = 1 as md increases
    giniTrainAcc = []
    giniTestAcc = []

    entropyTrainAcc = []
    entropyTestAcc = []
    for i in range(1, 11):
        dtGini = DecisionTree('gini', i, 1)
        trainAccGini, testAccGini = dt_train_test(dtGini, xTrain, yTrain, xTest, yTest)
        giniTrainAcc.append(trainAccGini)
        giniTestAcc.append(testAccGini)
        dtEntropy = DecisionTree('entropy', i, 1)
        trainAccEntropy, testAccEntropy = dt_train_test(dtEntropy, xTrain, yTrain, xTest, yTest)
        entropyTrainAcc.append(trainAccEntropy)
        entropyTestAcc.append(testAccEntropy)

    # Plot the Gini accuracies
    plt.plot(range(1, 11), giniTrainAcc, label='Train Accuracy (Gini)',marker='o')
    plt.plot(range(1, 11), giniTestAcc, label='Test Accuracy (Gini)',marker='o')
    plt.title('Gini Train Accuracies vs. Max Depth')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot the Entropy accuracies
    plt.plot(range(1, 11), entropyTrainAcc, label='Train Accuracy (Entropy)',marker='o')
    plt.plot(range(1, 11), entropyTestAcc, label='Test Accuracy (Entropy)',marker='o')
    plt.title('Entropy Train and Test Accuracies vs. Max Depth')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Get accuracies for a md = 7 as mls increases
    giniTrainAcc = []
    giniTestAcc = []

    entropyTrainAcc = []
    entropyTestAcc = []
    for i in range(1, 11):
        dtGini = DecisionTree('gini', 7, i)
        trainAccGini, testAccGini = dt_train_test(dtGini, xTrain, yTrain, xTest, yTest)
        giniTrainAcc.append(trainAccGini)
        giniTestAcc.append(testAccGini)
        dtEntropy = DecisionTree('entropy', 7, i)
        trainAccEntropy, testAccEntropy = dt_train_test(dtEntropy, xTrain, yTrain, xTest, yTest)
        entropyTrainAcc.append(trainAccEntropy)
        entropyTestAcc.append(testAccEntropy)
    
    # Plot the Gini accuracies
    plt.plot(range(1, 11), giniTrainAcc, label='Train Accuracy (Gini)',marker='o')
    plt.plot(range(1, 11), giniTestAcc, label='Test Accuracy (Gini)',marker='o')
    plt.title('Gini Train and Test Accuracies vs. Minimum Leaf Samples')
    plt.xlabel('Minimum Leaf Samples')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot the Entropy accuracies
    plt.plot(range(1, 11), entropyTrainAcc, label='Train Accuracy (Entropy)',marker='o')
    plt.plot(range(1, 11), entropyTestAcc, label='Test Accuracy (Entropy)',marker='o')
    plt.title('Entropy Train and Test Accuracies vs. Minimum Leaf Samples')
    plt.xlabel('Minimum Leaf Samples')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()


