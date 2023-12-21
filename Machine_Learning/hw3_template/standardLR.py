import argparse
import numpy as np
import pandas as pd
import time

from lr import LinearRegression, file_to_numpy


class StandardLR(LinearRegression):

    def train_predict(self, xTrain, yTrain, xTest, yTest):
        """
        See definition in LinearRegression class
        """
        trainStats = {}
        # TODO: DO SOMETHING
        startTime = time.time()
        # Add a column of ones
        xTrain = np.hstack(((np.ones((xTrain.shape[0],1))), xTrain))
        xTest = np.hstack(((np.ones((xTest.shape[0],1))), xTest))
        
        # Calculate beta
        self.beta = np.matmul(np.linalg.inv(np.matmul(np.transpose(xTrain), xTrain)), np.matmul(np.transpose(xTrain), yTrain))
        

        # Get predictions
        mse_train = self.mse(xTrain, yTrain)
        mse_test = self.mse(xTest, yTest)

        # Get time
        endTime = time.time() - startTime
        
        trainStats[0] = {"time": endTime,
                    "train-mse": mse_train,
                    "test-mse": mse_test}
        
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

    args = parser.parse_args()
    # load the train and test data
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    model = StandardLR()
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    print(trainStats)


if __name__ == "__main__":
    main()
