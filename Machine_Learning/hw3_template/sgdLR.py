import argparse
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from lr import LinearRegression, file_to_numpy


def grad_pt(beta, xi, yi):
    """
    Calculate the gradient for a mini-batch sample.

    Parameters
    ----------
    beta : 1d array with shape d
    xi : 2d numpy array with shape b x d
        Batch training data
    yi: 2d array with shape bx1
        Array of responses associated with training data.

    Returns
    -------
        grad : 1d array with shape d
    """
    # Get the mean of the gradients after calculating them
    grad = np.transpose(np.mean((yi - np.dot(xi, beta)) * xi, axis=0, keepdims=True))
    return grad


class SgdLR(LinearRegression):
    lr = 1  # learning rate
    bs = 1  # batch size
    mEpoch = 1000 # maximum epoch size

    def __init__(self, lr, bs, epoch):
        self.lr = lr
        self.bs = bs
        self.mEpoch = epoch

    def train_predict(self, xTrain, yTrain, xTest, yTest):
        """
        See definition in LinearRegression class
        """
        trainStats = {}
        # TODO: DO SGD

        num_samples, num_features = xTrain.shape
        self.beta = np.zeros((num_features, 1))
        # Start time
        start = time.time()
        
        for epoch in range(self.mEpoch):
            # Shuffle the training data
            indices = np.random.permutation(num_samples)
            xTrain = xTrain[indices]
            yTrain = yTrain[indices]

            # Split batches
            xBatches = [xTrain[b:b+self.bs] for b in range(0, len(xTrain), self.bs)]
            yBatches = [yTrain[b:b+self.bs] for b in range(0, len(yTrain), self.bs)]
            
            for b in range(len(xBatches)):
                # Get the gradients
                grad = grad_pt(self.beta, xBatches[b], yBatches[b])
                
                # Update betas
                self.beta += self.lr * grad

                # Calculate MSE
                train_mse = self.mse(xTrain, yTrain)
                test_mse = self.mse(xTest, yTest)     
                # Append to dictionary
                trainStats[(epoch)*len(yBatches)+b] = {"time":time.time()-start, "train-mse": train_mse, "test-mse": test_mse}      

        
        return trainStats


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
    parser.add_argument("lr", type=float, help="learning rate")
    parser.add_argument("bs", type=int, help="batch size")
    parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")

    args = parser.parse_args()
    # load the train and test data
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    # setting the seed for deterministic behavior
    np.random.seed(args.seed)   
    model = SgdLR(args.lr, args.bs, args.epoch)
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    print(trainStats)





if __name__ == "__main__":
    main()