import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Knn(object):
    k = 0    # number of neighbors to use

    def __init__(self, k):
        """
        Knn constructor

        Parameters
        ----------
        k : int 
            Number of neighbors to use.
        """
        self.k = k

    def train(self, xFeat, y):
        """
        Train the k-nn model.

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of labels associated with training data.

        Returns
        -------
        self : object
        """
        # TODO do whatever you need
        # cast to np arrays
        self.xFeat_train = np.array(xFeat)
        self.y_train = np.array(y)
        return self


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
            Predicted class label per sample
        """
        yHat = [] # variable to store the estimated class label
        # TODO
        # cast to np arrays
        self.xFeat_train = np.array(self.xFeat_train) 
        xFeat = np.array(xFeat)
            
        
        for i in range(len(xFeat)):

            # get feature
            x_i = xFeat[i]

            # get euclidean distance
            dist = np.linalg.norm(self.xFeat_train - x_i, axis=1)
            
            
            # get indices of k-nearest neighbors
            nearest_indices = np.argsort(dist)[:self.k]

            # get labels of k-nearest neighbors
            nearest_labels = self.y_train[nearest_indices]

            # predict the label by getting unique label counts and taking majority, if a tie default to min value label (i.e. label 0 vs label 1 is a tie, choose label 0)
            unique_labels, counts = np.unique(nearest_labels, return_counts=True)
            predicted_label = unique_labels[np.argmax(counts)]

            # add to yHat
            yHat.append(predicted_label)


    

        return yHat


def accuracy(yHat, yTrue):
    """
    Calculate the accuracy of the prediction

    Parameters
    ----------
    yHat : 1d-array with shape n
        Predicted class label for n samples
    yTrue : 1d-array with shape n
        True labels associated with the n samples

    Returns
    -------
    acc : float between [0,1]
        The accuracy of the model
    """
    # TODO calculate the accuracy
    totalCorrect = 0
    for i in range(len(yHat)):
        if yHat[i] == yTrue[i]:
            totalCorrect+=1
    acc = totalCorrect/len(yHat)

    return acc

def plotAccuracies(k_values, train_accuracies, test_accuracies):
    # Plotting the accuracies
    plt.figure()
    plt.plot(k_values, train_accuracies, label='Train Accuracy')
    plt.plot(k_values, test_accuracies, label='Test Accuracy')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. k for k-NN')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("k",
                        type=int,
                        help="the number of neighbors")
    parser.add_argument("--xTrain",
                        default="q3xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q3yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q3xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q3yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)
    # create an instance of the model
    knn = Knn(args.k)
    knn.train(xTrain, yTrain['label'])
    # predict the training dataset
    yHatTrain = knn.predict(xTrain)
    trainAcc = accuracy(yHatTrain, yTrain['label'])
    # predict the test dataset
    yHatTest = knn.predict(xTest)
    testAcc = accuracy(yHatTest, yTest['label'])
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)


    # make plot for accuracy
    k_values = range(1, knn.k+1) 
    train_accuracies = []    # store train accuracies
    test_accuracies = []     # store test accuracies

    for k in k_values:
        knn = Knn(k)
        knn.train(xTrain, yTrain['label'])
         # predict the training dataset
        yHatTrain = knn.predict(xTrain)
        train_acc = accuracy(yHatTrain, yTrain['label'])
        train_accuracies.append(train_acc)
        # predict the test dataset
        yHatTest = knn.predict(xTest)
        test_acc = accuracy(yHatTest, yTest['label'])
        test_accuracies.append(test_acc)

    plotAccuracies(k_values, train_accuracies, test_accuracies)
    


if __name__ == "__main__":
    main()
