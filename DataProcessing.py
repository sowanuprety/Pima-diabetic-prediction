import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer


def read_data(path):
    """
    This function reads the csv file from the path given and returns a DataFrame object
    :param path:
    :return DataFrame:
    """
    return pd.read_csv(path)


class DataDescription:
    """
    This class is used for Understanding Data
    """
    def __init__(self, path):
        print("Data descripton init called")
        np.random.seed(42)
        print(path)
        self.data = read_data(path)

    # Peeking into data
    def peek(self):
        print(self.data.head(20))

    # Dimension of data
    def shape(self):
        print(self.data.shape)

    # Datatype for each attribute
    def type(self):
        print(self.data.dtype)

    def descriptive_stat(self):
        print(self.data.describe())

    # Class Disribution
    def class_distribution(self):
        print(self.data.groupby('Outcome').size())

    # Correlation among attributes
    def corr(self):
        pd.set_option('display.width', 100)
        pd.set_option('precision', 3)
        print(self.data.corr(method = 'pearson'))

    #skewness
    def skew(self):
        print(self.data.skew())


class DataVisualization(DataDescription):

    def plot(self):
        # Histogram, Density plots, box Plots
        self.data.hist()
        self.data.plot(kind='density', subplots=True, layout=(3, 3), sharex=False)
        self.data.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False)
        plt.show()

    def multivariate_plots(self):

        #Correlation Matrix
        correlations = self.data.corr()
        names = self.data[0:1]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(correlations, vmin=-1, vmax=1)
        fig.colorbar(cax)
        ticks = np.arange(0, 9, 1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(names)
        ax.set_yticklabels(names)
        plt.show()

        # ScatterPlot
        sns.pairplot(self.data)

class Preprocessing(DataDescription):


    def rescaling(self):
        array = self.data.values

        # separate array into input and output components
        X = array[:, 0:8]
        Y = array[:, 8]

        scaler = MinMaxScaler(feature_range=(0, 1))
        rescaledX = scaler.fit_transform(X)

        # summarize transformed data
        np.set_printoptions(precision=3)
        print(rescaledX[0:5, :])

    def normalization(self):
        array = self.data.values

        # separate array into input and output components
        X = array[:, 0:8]
        Y = array[:, 8]

        scaler = Normalizer().fit(X)
        normalizedX = scaler.transform(X)

        # summarize transformed data
        np.set_printoptions(precision=3)
        print(normalizedX[0:5, :])







pima = DataVisualization('diabetes.csv')
pima.plot()
# pima2 = Rescale('diabetes.csv')
# pima2.multivariate_plots()
