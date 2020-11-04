import numpy as np
import math
import pandas as pd
from scipy.fftpack import dct
from sklearn.pipeline import FeatureUnion


class RollingTransformer:
    def __init__(self, lookback, topk=3):
        self.topk = topk  # top k largest absolute coefficients
        self.lookback = lookback
        self.df = pd.DataFrame()

    # compute the top k coefficients
    def compute(self, data):
        dct_out = dct(list(data), type=2)  # compute DCT-2 transform
        topkcoeffs = sorted(dct_out, key=abs, reverse=True)  # sort by absolute values
        if self.topk - len(topkcoeffs) > 0:  # pad with np.nan at the end
            return pd.Series(np.append(topkcoeffs, np.zeros(self.topk - np.size(topkcoeffs)) + np.nan))

        self.df = self.df.append(pd.Series(topkcoeffs[:self.topk]), ignore_index=True)
        return 0

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        df.rolling(window=self.lookback).apply(self.compute)
        # prepend with nan rows
        arr = np.empty((self.lookback - 1,self.topk,))
        arr[:] = np.nan
        dft = pd.DataFrame(arr)
        self.df = pd.concat([dft, self.df], ignore_index=True)
        return self.df.values


class ColumnTransformer:
    def __init__(self, base=3, topk=3, max_lookback=None):
        self.topk = topk  # top k largest coefficients
        self.base = base
        self.max_lookback = max_lookback
        self.transformer = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        translist = list()
        if self.max_lookback is None:
            max_exponent = int(math.log(int(X.shape[0]/3), self.base)) + 1
        else:
            max_exponent = int(math.log(int(self.max_lookback), self.base)) + 1

        for i in range(1, max_exponent):
            translist.append((f'rolling_transformer_{i}', RollingTransformer(lookback=self.base**i, topk=self.topk)))

        self.transformer = FeatureUnion(transformer_list=translist)
        Xt = self.transformer.transform(X)
        return Xt


class TSTransformer:
    def __init__(self, base=3, topk=3):
        self.topk = topk  # top k largest coefficients
        self.base = base

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xt = None
        for i in range(X.shape[1]):
            ct = ColumnTransformer(base=self.base, topk=self.topk)
            if Xt is None:
                Xt = ct.transform(X=X[:, i])
            else:
                Xt = np.concatenate([Xt, ct.transform(X=X[:, i])], axis=1)

        return Xt


def test_uts():
    infile = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/AirPassengers.csv"
    cols = ["ID", "time", "AirPassengers"]
    df = pd.read_csv(infile, names=cols, sep=r',', index_col='ID', engine='python', skiprows=1)
    trainnum = 100
    Xtrain = df.iloc[:trainnum, 1].values
    Xtrain = Xtrain.reshape(-1, 1)

    tf = TSTransformer()
    tf = tf.fit(Xtrain)
    Xt = tf.transform(Xtrain)
    print(f'UTS Transformer, Xt: {Xt.shape}')


def test_mts():
    infile = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/AirPassengers.csv"
    cols = ["ID", "time", "AirPassengers"]
    df = pd.read_csv(infile, names=cols, sep=r',', index_col='ID', engine='python', skiprows=1)
    trainnum = 100
    Xtrain = df.iloc[:trainnum, 1].values
    Xtrain = Xtrain.reshape(-1, 1)
    X = np.concatenate([Xtrain, Xtrain], axis=1)

    tf = TSTransformer()
    tf = tf.fit(X)
    Xt = tf.transform(X)
    print(f'MTS Transformer, Xt: {Xt.shape}')


if __name__ == '__main__':
    test_uts()
    test_mts()


