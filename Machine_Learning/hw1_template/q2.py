import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd


def load_iris():
    # load the dataset
    iris = datasets.load_iris()
    # create a dataframe
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    # add the labels
    df['target'] = iris.target
    return df

def boxplot(df):
    # get the features
    features = df.columns[:-1]
    
    # make 4 plots
    for feature in features:
        df.boxplot(column=feature, by='target')
    

def scatterplot(df):
    # get the features
    features = df.columns[:-1]

    # create a scatter plot for each pair of features
    for i in range(0, features.shape[0], 2):
        x_feature = features[i]
        y_feature = features[i + 1]
        
        plt.figure(figsize=(8, 6))  # create a new figure for each pair of features
        # details of the plot
        plot = plt.scatter(
            df[x_feature],
            df[y_feature],
            c=df['target'],
            label=df['target'],
            alpha=0.7  
        )
        plt.title(f'Scatter Plot of {x_feature} vs {y_feature}')
        plt.xlabel(x_feature)
        plt.ylabel(y_feature)
        plt.legend(*plot.legend_elements(), title='Classification')




def main():
    # part a
    iris = load_iris()
    # part b
    boxplot(iris)
    # part c
    scatterplot(iris)
    plt.show()



main()