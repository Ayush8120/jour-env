from sklearn.datasets import make_circles
import pandas as pd

import constants

class createDataset():
    def __init__(self):
        self.n_samples = constants.N_SAMPLES
        self.X,self.Y = make_circles(self.n_samples,noise=0.3,random_state=42)

    def giveDataset(self):
        X_train = self.X[:0.6*self.n_samples,:] 
        Y_train = self.Y[:0.6*self.n_samples,:]

        X_valid = self.X[0.6*self.n_samples:0.8*self.n_samples,:]
        Y_valid = self.Y[0.6*self.n_samples:0.8*self.n_samples,:]

        X_test = self.X[0.8*self.n_samples:,:]
        Y_test = self.Y[0.8*self.n_samples:,:]
        return X_train,Y_train,X_valid,Y_valid,X_test,Y_test
    