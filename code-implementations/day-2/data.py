from sklearn.datasets import make_circles
import pandas as pd
import math,pdb,numpy,torch
import constants
from sklearn.model_selection import train_test_split
torch.manual_seed(42)
class createDataset():
    def __init__(self):
        self.n_samples = constants.N_SAMPLES
        self.X,self.Y = make_circles(self.n_samples,noise=0.3,random_state=42)
        self.Y = numpy.expand_dims(self.Y,axis=1)
        self.X = torch.from_numpy(self.X).type(torch.float)
        self.Y = torch.from_numpy(self.Y).type(torch.float)
    
    def giveDataset(self):
        '''
        do manually
        '''
        # X_train = self.X[:math.floor(0.6*self.n_samples),:] 
        # Y_train = self.Y[:math.floor(0.6*self.n_samples),:]

        # X_valid = self.X[math.floor(0.6*self.n_samples):math.floor(0.8*self.n_samples),:]
        # Y_valid = self.Y[math.floor(0.6*self.n_samples):math.floor(0.8*self.n_samples),:]

        # X_test = self.X[math.floor(0.8*self.n_samples):,:]
        # Y_test = self.Y[math.floor(0.8*self.n_samples):,:]
        
        '''
        or using train_test split twice
        '''

        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=1)
        X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2
        
        return X_train,Y_train,X_valid,Y_valid,X_test,Y_test
    