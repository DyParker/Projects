import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, roc_auc_score
from rf import file_to_numpy

def normalize_feat(xTrain, xTest):
    """
    Helper function to normalize the passed datasets to have
    mean 0 and unit variance

    Parameters
    ----------
    xTrain : nd-array with shape n x d
        Training data
    xTest : nd-array with shape n x d
        Test data

    Returns
    -------
    newTrain : nd-array with shape n x d
        Normalized training data
    newTest : nd-array with shape n x d
        Normalized test data
    """
    trainScaler = StandardScaler().fit(xTrain)
    testScaler = StandardScaler().fit(xTest)

    newTrain = trainScaler.transform(xTrain)
    newTest = testScaler.transform(xTest)

    return newTrain, newTest

def unreg_log(xTrain, yTrain, xTest, yTest):
    """
    Function to train an unregularized logistic
    regression model and return false positive and true
    positive rates on test data.

    Parameters
    ----------
    xTrain : nd-array with shape n x d
        Training data
    yTrain : 1d array with shape n
        Array of responses associated with training data.
    xTest : nd-array with shape n x d
        Test data
    yTest : 1d array with shape n
        Array of responses associated with testing data

    Returns
    -------
    fpr : float
        False positive rate of the model on testing data
    tpr : float
        True positive rate of the model on testing data
    """
    model = LogisticRegression(penalty=None)
    model.fit(xTrain, yTrain)

    yHat = model.predict_proba(xTest)[:,1]

    fpr, tpr, _ = roc_curve(yTest, yHat)

    aucScore = roc_auc_score(yTest, yHat)


    return fpr, tpr, aucScore

def run_pca(xTrain, xTest):
    """
    Function to run PCA on a normalized dataset.

    Parameters
    ----------
    xTrain : nd-array with shape n x d
        Training data, already normalized
    xTest : nd-array with shape n x d
        Test data, already normalized

    Returns
    -------
    xTrainNew : nd-array with shape n x p
        Training data with p principal components
    xTestNew : nd-array with shape n x p
        Testing data with p principal components
    pcaComponents : 1d array with shape p
        p principal components from PCA
    """
    pca = PCA(n_components=0.95)
    pca.fit(xTrain)

    xTrainNew = pca.transform(xTrain)
    xTestNew = pca.transform(xTest)
    pcaComponents = pca.components_


    return xTrainNew, xTestNew, pcaComponents

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

    xTemp = pd.read_csv(args.xTrain)
    colNames = np.array(xTemp.columns)

    # load the train and test data assumes you'll use numpy
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    xTrain, xTest = normalize_feat(xTrain, xTest)

    fpr, tpr, rocAUCNorm = unreg_log(xTrain, yTrain, xTest, yTest)


    print("False Positive Rate:", fpr, "True Positive Rate:", tpr)

    xTrainNew, xTestNew, pcaComps = run_pca(xTrain, xTest)

    fprPCA, tprPCA, rocAUCPCA = unreg_log(xTrainNew, yTrain, xTestNew, yTest)

    # plt.figure(figsize=(8, 6))
    # plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (Normalized) (AUC = %0.2f)' % rocAUCNorm)
    # plt.plot(fprPCA, tprPCA, color='red', lw=2, label='ROC curve (PCA) (AUC = %0.2f)' % rocAUCPCA)
    # plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')  # Diagonal line
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC)')
    # plt.legend(loc='lower right')
    # plt.grid(True)
    # plt.show()


    # # Extract the top 3 principal components
    # top_3_components = pcaComps[:3, :]

    # # Visualize the top 3 components' loadings
    # plt.figure(figsize=(10, 6))

    # for i, component in enumerate(top_3_components):
    #     plt.subplot(3, 1, i + 1)
    #     sorted_indices = np.argsort(component)
    #     plt.barh(range(len(component)), component[sorted_indices], align='center')
    #     plt.yticks(range(len(component)), sorted_indices)
    #     plt.xlabel('Loading Value')
    #     plt.title(f'Principal Component {i+1} Loadings')

    # plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()