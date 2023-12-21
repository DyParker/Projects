import argparse
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import random
from collections import Counter


def model_assessment(filename):
    """
    Given the entire data, split it into training and test set 
    so you can assess your different models 
    to compare perceptron, logistic regression,
    and naive bayes. 
    """
    df = pd.read_csv(filename, names=['data'], index_col=False)

    # Separate data file into labels and emails
    df['labels'] = df['data'].str[0].astype(int)
    df['emails'] = df['data'].str[2:].apply(lambda text: text.split())
  
    # Split train and test
    features = df['emails'].to_numpy()
    labels = df['labels'].to_numpy()

    # Use sk to split and create pd dataframes
    x_train,x_test,y_train,y_test=train_test_split(features,labels,test_size=0.2)
    train = pd.DataFrame({'y': y_train, 'text': x_train})
    test = pd.DataFrame({'y': y_test, 'text': x_test})

    return train, test


def build_vocab_map(traindf):
    """
    Construct the vocabulary map such that it returns
    (1) the vocabulary dictionary contains words as keys and
    the number of emails the word appears in as values, and
    (2) a list of words that appear in at least 30 emails.

    ---input:
    dataset: pandas dataframe containing the 'text' column
             and 'y' label column

    ---output:
    dict: key-value is word-count pair
    list: list of words that appear in at least 30 emails
    """
    wordsCount = {}
    wordsOver30 = []

    # Iterate through the emails and count the unique words per email, updating wordsCount
    for email in traindf['text']:
        unique_words = set(email)  # Convert to a set to ensure uniqueness
        for word in unique_words:
            if word in wordsCount:
                wordsCount[word] += 1
            else:
                wordsCount[word] = 1

    # Iterate through wordsCount and add a word if its count is >= 30 to wordsOver30
    for word, count in wordsCount.items():
        if count>=30:
            wordsOver30.append(word)


    return wordsCount, wordsOver30


def construct_binary(dataset, freq_words):
    """
    Construct email datasets based on
    the binary representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is 1 if the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise

    ---input:
    dataset: pandas dataframe containing the 'text' column

    freq_word: the vocabulary map built in build_vocab_map()

    ---output:
    numpy array
    """
    # Initialize vector as all 0's
    rows = len(dataset)
    cols = len(freq_words)
    binaryVector = np.zeros((rows, cols), dtype = int)


    for n in range(rows):
        # Split words in email
        email_words = dataset['text'][n]
        for d in range(cols):
            # Check if a frequent word (>=30) is present in the email and set binary value "1" if so
            if freq_words[d] in email_words:
                binaryVector[n, d] = 1
    
    return binaryVector


def construct_count(dataset, freq_words):
    """
    Construct email datasets based on
    the count representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is the number of times the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise

    ---input:
    dataset: pandas dataframe containing the 'text' column

    freq_word: the vocabulary map built in build_vocab_map()

    ---output:
    numpy array
    """
    # Initialize vector as all 0's
    rows = len(dataset)
    cols = len(freq_words)
    countVector = np.zeros((rows, cols), dtype = int)


    for n in range(rows):
        # Split words in email
        email_words = dataset['text'][n]
        for d in range(cols):
            # Get the counts and add them to the vector
            countVector[n,d] = email_words.count(freq_words[d])
    
    return countVector


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        default="spamAssassin.data",
                        help="filename of the input data")
    args = parser.parse_args()

    # Get train and test dataframes
    train, test = model_assessment(args.data)

    # Get dictionary of each word and the amount of emails that have that word, and a list of the words that appear in >=30 emails 
    wordsCount, wordsOver30 = build_vocab_map(train)

    # Get the binary vector sets of train and test datasets
    trainBinaryVector = construct_binary(train, wordsOver30)
    testBinaryVector = construct_binary(test, wordsOver30)

    # Get the count vector sets of train and test datasets
    trainCountVector = construct_count(train, wordsOver30)
    testCountVector = construct_count(test, wordsOver30)

    # Make everything a pd dataframe
    xTrain = pd.DataFrame(data = train['text'])
    yTrain = pd.DataFrame(data = train['y'])

    xTest = pd.DataFrame(data = test['text'])
    yTest = pd.DataFrame(data = test['y'])

    binaryTrain = pd.DataFrame(data = trainBinaryVector, columns = wordsOver30)
    binaryTest = pd.DataFrame(data = testBinaryVector, columns = wordsOver30)

    countTrain = pd.DataFrame(data = trainCountVector, columns = wordsOver30)
    countTest = pd.DataFrame(data = testCountVector, columns = wordsOver30)

    # Write csv files
    xTrain.to_csv('xTrain.csv', index=False)
    yTrain.to_csv('yTrain.csv', index=False)

    xTest.to_csv('xTest.csv', index=False)
    yTest.to_csv('yTest.csv', index=False)

    binaryTrain.to_csv('binaryTrain.csv', index=False)
    binaryTest.to_csv('binaryTest.csv', index=False)

    countTrain.to_csv('countTrain.csv', index=False)
    countTest.to_csv('countTest.csv', index=False)

    # wordsOver30DF = pd.DataFrame(data=wordsOver30)
    # wordsOver30DF.to_csv('listWords.csv', index=False)

if __name__ == "__main__":
    main()
