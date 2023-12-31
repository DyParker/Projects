import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import knn


def standard_scale(xTrain, xTest):
    """
    Preprocess the training data to have zero mean and unit variance.
    The same transformation should be used on the test data. For example,
    if the mean and std deviation of feature 1 is 2 and 1.5, then each
    value of feature 1 in the test set is standardized using (x-2)/1.5.

    Parameters
    ----------
    xTrain : nd-array with shape n x d
        Training data 
    xTest : nd-array with shape m x d
        Test data 

    Returns
    -------
    xTrain : nd-array with shape n x d
        Transformed training data with mean 0 and unit variance 
    xTest : nd-array with shape m x d
        Transformed test data using same process as training.
    """
    # TODO FILL IN
    # Calculate mean and standard deviation for each feature in the training data
    mean_train = np.mean(xTrain, axis=0)
    std_train = np.std(xTrain, axis=0)

    # Standardize the training data
    xTrain = (xTrain - mean_train) / std_train

    # Standardize the test data using the mean and std deviation from the training data
    xTest = (xTest - mean_train) / std_train

    return xTrain, xTest


def minmax_range(xTrain, xTest):
    """
    Preprocess the data to have minimum value of 0 and maximum
    value of 1. The same transformation should be used on the test data.
    For example, if the minimum and maximum of feature 1 is 0.5 and 2, then
    then feature 1 of test data is calculated as:
    (1 / (2 - 0.5)) * x - 0.5 * (1 / (2 - 0.5))

    Parameters
    ----------
    xTrain : nd-array with shape n x d
        Training data 
    xTest : nd-array with shape m x d
        Test data 

    Returns
    -------
    xTrain : nd-array with shape n x d
        Transformed training data with min 0 and max 1.
    xTest : nd-array with shape m x d
        Transformed test data using same process as training.
    """
    # TODO FILL IN
    min_train = np.min(xTrain, axis=0)
    max_train = np.max(xTrain, axis=0)
    xTrain = (  (1 / (max_train - min_train))  *  xTrain - min_train  *  (1 / (max_train - min_train))    )
    xTest = (  (1 / (max_train - min_train))  *  xTest - min_train  *  (1 / (max_train - min_train))    )


    return xTrain, xTest


def add_irr_feature(xTrain, xTest):
    """
    Add 2 features using Gaussian distribution with 0 mean,
    standard deviation of 1.

    Parameters
    ----------
    xTrain : nd-array with shape n x d
        Training data 
    xTest : nd-array with shape m x d
        Test data 

    Returns
    -------
    xTrain : nd-array with shape n x (d+2)
        Training data with 2 new noisy Gaussian features
    xTest : nd-array with shape m x (d+2)
        Test data with 2 new noisy Gaussian features
    """
    # TODO FILL IN
    xTrain = pd.DataFrame(data=xTrain)
    xTest = pd.DataFrame(data=xTest)
    new_features_train1 = np.random.normal(0, 1, len(xTrain))
    new_features_train2 = np.random.normal(0, 1, len(xTrain))
    new_features_test1 = np.random.normal(0, 1, len(xTest))
    new_features_test2 = np.random.normal(0, 1, len(xTest))

    xTrain['f1'] = new_features_train1
    xTrain['f2'] = new_features_train2
    xTest['f1'] = new_features_test1
    xTest['f2'] = new_features_test2

    return xTrain, xTest


def knn_train_test(k, xTrain, yTrain, xTest, yTest):
    """
    Given a specified k, train the knn model and predict
    the labels of the test data. Returns the accuracy of
    the resulting model.

    Parameters
    ----------
    k : int
        The number of neighbors
    xTrain : nd-array with shape n x d
        Training data 
    yTrain : 1d array with shape n
        Array of labels associated with training data.
    xTest : nd-array with shape m x d
        Test data 
    yTest : 1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    acc : float
        The accuracy of the trained knn model on the test data
    """
    model = knn.Knn(k)
    model.train(xTrain, yTrain['label'])
    # predict the test dataset
    yHatTest = model.predict(xTest)
    return knn.accuracy(yHatTest, yTest['label'])
    

# plot the accuracies
def plotAccuracies(k_values, no_pre_proc_acc, standard_scale_acc, min_max_scale_acc, irr_feat_acc):
    # plotting the accuracies
    plt.figure()
    plt.plot(k_values, no_pre_proc_acc, label='No Pre-Processing Accuracy')
    plt.plot(k_values, standard_scale_acc, label='Standard Scale Accuracy')
    plt.plot(k_values, min_max_scale_acc, label='Min-Max Scale Accuracy')
    plt.plot(k_values, irr_feat_acc, label='Irrelevant Features Accuracy')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Different Pre-Processing Techniques vs. k for k-NN')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("k",
                        type=int,
                        help="the number of neighbors")
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
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)

    # no preprocessing
    acc1 = knn_train_test(args.k, xTrain, yTrain, xTest, yTest)
    print("Test Acc (no-preprocessing):", acc1)
    # preprocess the data using standardization scaling
    xTrainStd, xTestStd = standard_scale(xTrain, xTest)
    acc2 = knn_train_test(args.k, xTrainStd, yTrain, xTestStd, yTest)
    print("Test Acc (standard scale):", acc2)
    # preprocess the data using min max scaling
    xTrainMM, xTestMM = minmax_range(xTrain, xTest)
    acc3 = knn_train_test(args.k, xTrainMM, yTrain, xTestMM, yTest)
    print("Test Acc (min max scale):", acc3)
    # add irrelevant features
    xTrainIrr, yTrainIrr = add_irr_feature(xTrain, xTest)
    acc4 = knn_train_test(args.k, xTrainIrr, yTrain, yTrainIrr, yTest)
    print("Test Acc (with irrelevant feature):", acc4)


    # plot the 4 accuracies
    k_values = range(1, args.k+1) 
    # store 4 accuracies
    no_pre_proc_acc = []    
    standard_scale_acc = []    
    min_max_scale_acc = []
    irr_feat_acc = [] 

    # gather accuracies
    for k in k_values:
        acc1 = knn_train_test(k, xTrain, yTrain, xTest, yTest)
        no_pre_proc_acc.append(acc1)
        # preprocess the data using standardization scaling
        xTrainStd, xTestStd = standard_scale(xTrain, xTest)
        acc2 = knn_train_test(k, xTrainStd, yTrain, xTestStd, yTest)
        standard_scale_acc.append(acc2)
        # preprocess the data using min max scaling
        xTrainMM, xTestMM = minmax_range(xTrain, xTest)
        acc3 = knn_train_test(k, xTrainMM, yTrain, xTestMM, yTest)
        min_max_scale_acc.append(acc3)
        # add irrelevant features
        xTrainIrr, yTrainIrr = add_irr_feature(xTrain, xTest)
        acc4 = knn_train_test(k, xTrainIrr, yTrain, yTrainIrr, yTest)
        irr_feat_acc.append(acc4)

    # plot accuracies
    plotAccuracies(k_values, no_pre_proc_acc, standard_scale_acc, min_max_scale_acc, irr_feat_acc)


if __name__ == "__main__":
    main()
