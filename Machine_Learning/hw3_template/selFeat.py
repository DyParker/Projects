import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def min_max(df):
    min = np.min(df, axis=0)
    max = np.max(df, axis=0)

    minMaxDF = (1 / (max - min)) * df - min * (1 / (max - min))

    return minMaxDF


def extract_features(df):
    """
    Given a pandas dataframe, extract the relevant features
    from the date column

    Parameters
    ----------
    df : pandas dataframe
        Training or test data 
    Returns
    -------
    df : pandas dataframe
        The updated dataframe with the new features
    """
    # TODO do more than this
    # Specify the date format explicitly
    df['date'] = pd.to_datetime(df['date'], format="%m/%d/%y %H:%M")
    # Hour of the day, Month of the year (season), Day of the week (work schedule)
    df['hour'] = df['date'].dt.hour
    df['month'] = df['date'].dt.month
    df['day_of_week'] =df['date'].dt.day_of_week
    df = df.drop(columns=['date'])

    return df


def cal_corr(df):
    """
    Given a pandas dataframe (include the target variable at the last column), 
    calculate the correlation matrix (compute pairwise correlation of columns)

    Parameters
    ----------
    df : pandas dataframe
        Training or test data (with target variable)
    Returns
    -------
    corrMat : pandas dataframe
        Correlation matrix
    """
    # TODO
    # calculate the correlation matrix and perform the heatmap
    # .corr() defaults to pearson
    corrMat = df.corr()
    return corrMat




def select_features(df):
    """
    Select the features to keep

    Parameters
    ----------
    df : pandas dataframe
        Training or test data 
    Returns
    -------
    df : pandas dataframe
        The updated dataframe with a subset of the columns
    """
    # TODO
    cols = ['lights','T2','RH_2','T3','RH_4','T5','RH_6','RH_7','T8','RH_8','T9','T_out','Press_mm_hg','RH_out','Windspeed','Visibility','Tdewpoint','hour']
    return df[cols]


def preprocess_data(trainDF, testDF):
    """
    Preprocess the training data and testing data

    Parameters
    ----------
    trainDF : pandas dataframe
        Training data 
    testDF : pandas dataframe
        Test data 
    Returns
    -------
    trainDF : pandas dataframe
        The preprocessed training data
    testDF : pandas dataframe
        The preprocessed testing data
    """
    # TODO do something
    # Min-Max scaling
    trainDF, testDF = min_max(trainDF), min_max(testDF)
    return trainDF, testDF


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("outTrain",
                        help="filename of the updated training data")
    parser.add_argument("outTest",
                        help="filename of the updated test data")
    parser.add_argument("--trainFile",
                        default="eng_xTrain.csv",
                        help="filename of the training data")
    parser.add_argument("--testFile",
                        default="eng_xTest.csv",
                        help="filename of the test data")
    parser.add_argument("--yTrainFile",
                        default="eng_yTrain.csv",
                        help="filename of the training data")
    parser.add_argument("--yTestFile",
                        default="eng_yTest.csv",
                        help="filename of the test data")
    args = parser.parse_args()
    # load the x train and test data
    xTrain = pd.read_csv(args.trainFile)
    xTest = pd.read_csv(args.testFile)

    # Load the y train and test data
    yTrain = pd.read_csv(args.yTrainFile)
    yTest = pd.read_csv(args.yTestFile)

    # extract the new features
    xNewTrain = extract_features(xTrain)
    xNewTest = extract_features(xTest)

    # Add target features
    xNewTrain['target'] = yTrain
    xNewTest['target'] = yTest

    # select the features
    xNewTrain = select_features(xNewTrain)
    xNewTest = select_features(xNewTest)
    # preprocess the data
    xTrainTr, xTestTr = preprocess_data(xNewTrain, xNewTest)
   
    # save it to csv
    xTrainTr.to_csv(args.outTrain, index=False)
    xTestTr.to_csv(args.outTest, index=False)

    # 1d) plot
    # # Using cal_corr
    # correlation_matrix = cal_corr(xNewTrain)
    # # Create a heatmap
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, annot_kws={"size": 6})

    # # Customize the plot
    # plt.title("Correlation Heatmap")
    # plt.show()

if __name__ == "__main__":
    main()
