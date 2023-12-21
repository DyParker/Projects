import argparse
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

class Perceptron(object):
    mEpoch = 1000  # maximum epoch size
    w = None       # weights of the perceptron

    def __init__(self, epoch):
        self.mEpoch = epoch

    def train(self, xFeat, y):
        """
        Train the perceptron using the data

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of responses associated with training data.

        Returns
        -------
        stats : object
            Keys represent the epochs and values the number of mistakes
        """
        stats = {}
        # TODO implement this

        # Set weights randomly
        self.w = np.random.random(xFeat.shape[1])
        mistakes = float('inf')
        epoch = 0
        
        while mistakes != 0 and epoch < self.mEpoch:

            # Shuffle the data at each epoch
            shuffle = np.random.permutation(xFeat.shape[0])
            xFeat = xFeat[shuffle]
            y = y[shuffle]
            # Reset mistakes
            mistakes = 0
            for i in range(xFeat.shape[0]):

                # Go through each sample and update weights and count mistakes
                xi = xFeat[i]
                yi = y[i]
                wnew, mistake = self.sample_update(xi, yi)
                self.w = wnew
                mistakes += mistake

            # Update epoch and enter stats at the epoch
            epoch+=1
            stats[epoch] = mistakes




        return stats

    def sample_update(self, xi, yi):
        """
        Given a single sample, give the resulting update to the weights

        Parameters
        ----------
        xi : numpy array of shape 1 x d
            Training sample 
        y : single value (-1, +1)
            Training label

        Returns
        -------
            wnew: numpy 1d array
                Updated weight value
            mistake: 0/1 
                Was there a mistake made 
        """
        mistake = 0
        wnew = self.w

        # Make predictions
        yHat = np.dot(xi, wnew)

        if(yHat * yi <=0):
            wnew = wnew +yi*xi
            mistake = 1

        return wnew, mistake

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

        # Make predictions
        predictions = np.dot(xFeat, self.w)

        # Update if positive or negative
        yHat = np.where(predictions > 0, 1, -1)
        return yHat


def transform_y(y):
    """
    Given a numpy 1D array with 0 and 1, transform the y 
    label to be -1 and 1

    Parameters
    ----------
    y : numpy 1-d array with labels of 0 and 1
        The true label.      

    Returns
    -------
    y : numpy 1-d array with labels of -1 and 1
        The true label but 0->1
    """

    # Replace 0 with -1, and replace with 1 if not 0
    y = np.where(y == 0, -1, 1)

    return y


def calc_mistakes(yHat, yTrue):
    """
    Calculate the number of mistakes
    that the algorithm makes based on the prediction.

    Parameters
    ----------
    yHat : 1-d array or list with shape n
        The predicted label.
    yTrue : 1-d array or list with shape n
        The true label.      

    Returns
    -------
    err : int
        The number of mistakes that are made
    """

    err = 0
    for i in range(len(yHat)):
        if(yHat[i] != yTrue[i]):
            err += 1

    return err


def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()



def tune_perceptron(trainx, trainy, epochList, k = 3):
    """
    Tune the preceptron to find the optimal number of epochs

    Parameters
    ----------
    trainx : a nxd numpy array
        The input from either binary / count matrix
    trainy : numpy 1d array of shape n
        The true label.    
    epochList: a list of positive integers
        The epoch list to search over  

    Returns
    -------
    epoch : int
        The optimal number of epochs
    """
    bestAcc = -float('inf')
    bestEpoch = None

    # Use KFold from sklearn
    kfold = KFold(n_splits=k, shuffle=True, random_state=10)

    for epoch in epochList:
        totalAcc = 0

        for trainSplit, testSplit in kfold.split(trainx):
            # Get kfold splits
            xTrain, xTest = trainx[trainSplit], trainx[testSplit]
            yTrain, yTest = trainy[trainSplit], trainy[testSplit]

            # Train a perceptron model
            model = Perceptron(epoch)
            model.train(xTrain, yTrain)

            # Make predictions
            yHat = []
            for x in xTest:
                yHat.append(model.predict(x))
            accuracy = accuracy_score(yTest, yHat)
            totalAcc += accuracy

        # Update bestEpoch if average accuracy is better than current bestAcc 
        if ((totalAcc / k) > bestAcc):
            bestAcc = totalAcc/k
            bestEpoch = epoch

    return bestEpoch


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
    # transform to -1 and 1
    yTrain = transform_y(yTrain)
    yTest = transform_y(yTest)

    np.random.seed(args.seed)   
    model = Perceptron(args.epoch)
    trainStats = model.train(xTrain, yTrain)
    print(trainStats)
    yHat = model.predict(xTest)
    # add predictions for train dataset
    yHat_train = model.predict(xTrain)
    # print out the number of mistakes
    print("Number of mistakes on the test dataset")
    print(calc_mistakes(yHat, yTest))
    print("Number of mistakes on the training dataset")
    print(calc_mistakes(yHat_train, yTrain))



if __name__ == "__main__":
    main()