import argparse
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from perceptron import file_to_numpy, transform_y

def main():
    parser = argparse.ArgumentParser()
    
    binTrain = file_to_numpy("binaryTrain.csv")
    binTest = file_to_numpy("binaryTest.csv")

    countTrain = file_to_numpy("countTrain.csv")
    countTest = file_to_numpy("countTest.csv")

    yTest = file_to_numpy("yTest.csv")
    yTrain = file_to_numpy("yTrain.csv")

    # transform to -1 and 1
    yTrain = transform_y(yTrain)
    yTest = transform_y(yTest)

    bern = BernoulliNB()
    mult = MultinomialNB()
    binaryLRModel = LogisticRegression()
    countLRModel = LogisticRegression()

    bern.fit(binTrain, yTrain)
    mult.fit(countTrain, yTrain)
    binaryLRModel.fit(binTrain, yTrain)
    countLRModel.fit(countTrain, yTrain)


    bernPred = bern.predict(binTest)
    multPred = mult.predict(countTest)
    binaryLRPred = binaryLRModel.predict(binTest)
    countLRPred = countLRModel.predict(countTest)

    bernCM = confusion_matrix(yTest, bernPred)
    multCM = confusion_matrix(yTest, multPred)
    binLRCM = confusion_matrix(yTest, binaryLRPred)
    countLRCM = confusion_matrix(yTest, countLRPred)

    bernMistakes = bernCM[0][1] + bernCM[1][0]
    multMistakes = multCM[0][1] + multCM[1][0]
    binLRMistakes = binLRCM[0][1] + binLRCM[1][0]
    countLRMistakes = countLRCM[0][1] + countLRCM[1][0]

    print("bernoulli mistakes:", bernMistakes)
    print("multinomial mistakes:", multMistakes)
    print("linear regression on binary mistakes:", binLRMistakes)
    print("linear regression on count mistakes:", countLRMistakes)

if __name__ == "__main__":
    main()