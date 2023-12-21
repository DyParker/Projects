from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import sklearn.metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import argparse
from rf import file_to_numpy


def normalize_feat(xTrain, xTest):
    """
    Helperfunction to normalize the features of the wine quality dataset (where applicable), 
    standardizes features by removing the mean and scaling to unit variance.

    Parameters
    ----------
    xTrain: nd-array with shape n x d
        Training data 
    xTest: nd-array with shape n x d
        Test data

    Returns
    -------
    xTrain: normalized dataset for training
    xTest: normalized dataset for testing
    """
    scaler = StandardScaler()

    xTrain = scaler.fit(xTrain).transform(xTrain)
    xTest = scaler.fit(xTest).transform(xTest)
    
    return xTrain, xTest

def unreg_log(xTrain, yTrain, xTest, yTest):
    """
    Helper function that trains an unregularized logistic 
    regression model on the dataset and predicts the 
    probabilities on the test data and calculate the ROC. 

    Parameters
    ----------
    xTrain: nd-array with shape n x d
        Training data 
    yTrain: 1d array with shape n
        Array of responses associated with training data.
    xTest: nd-array with shape n x d
        Test data
    yTest: 1d array with shape n
        Array of responses associated with testing data.

    Returns
    -------
    fpr: float
        False positive rate
    tpr: float
        True positve rate
    auc: float
        Area under the ROC curve
    """

    # Train and predict using the model
    logReg = LogisticRegression(penalty=None)
    clf = logReg.fit(xTrain, yTrain)

    # Make predictions
    predictions = clf.predict_proba(xTest)[:,1]

    # Calculate fpr, tpr, and auc
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(yTest, predictions)
    auc = sklearn.metrics.roc_auc_score(yTest, predictions)

    return fpr, tpr, auc

def run_pca(xTrain, xTest):
    """
    Run PCA on the normalized training dataset. 
    Finds how many components are needed to capture at 
    least 95% of the variance in the original data

    Parameters
    ----------
    xTrain: nd-array with shape n x d
        Training data 
    xTest: nd-array with shape n x d
        Test data

    Returns:
    -------
    xTrain: nd-array with shape n x p
        Transformed xTrain with p components
    xTest: nd-array with shape n x p
        Transformed xTest with p components
    components: 1d array with shape p
        PCA model with p components
    """
    pca = PCA(n_components=0.95)
    pca.fit(xTrain)

    xTrain = pca.transform(xTrain)
    xTest = pca.transform(xTest)
    components = pca.components_

    return xTrain, xTest, components

def plot_roc_curve(fpr, tpr, auc, label):
    plt.plot(fpr, tpr, label=f'{label} - AUC = {auc:.2f}')

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

    # Load the files
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)


    # Normalize dataset
    xTrain, xTest = normalize_feat(xTrain, xTest)
    # Unregularized logistic regression on normalized dataset
    fpr, tpr, auc = unreg_log(xTrain, yTrain, xTest, yTest)
    print("False positive rate: ",fpr)   
    print("True positive rate",tpr)


    # PCA 
    xTrainPCA, xTestPCA, componentsPCA = run_pca(xTrain, xTest)
    print("Components needed for 95% of explained variance:",len(componentsPCA))
    # Unregularized logistic regression on normalized PCA dataset
    fprPCA, tprPCA, aucPCA = unreg_log(xTrainPCA, yTrain, xTestPCA, yTest)

    

    # # Plot ROC curves
    # plot_roc_curve(fpr, tpr, auc, 'Normalized')
    # plot_roc_curve(fprPCA, tprPCA, aucPCA, 'PCA Normalized')

    # plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Plot the random chance line
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC) Curve')
    # plt.legend(loc="lower right")
    # plt.show()

if __name__ == "__main__":
    main()
