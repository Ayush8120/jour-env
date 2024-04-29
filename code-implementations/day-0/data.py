import torch
class createDataset():
    def __init__(self) -> None:
        
        # Create *known* parameters
        self.weight = 0.7
        self.bias = 0.3

        # Create data
        self.start = 0
        self.end = 1
        self.step = 0.02
        self.X = torch.arange(self.start, self.end, self.step).unsqueeze(dim=1) #adds a dimension at 1th index
        self.y = self.weight * self.X + self.bias

        print(self.X[:10], self.y[:10])

    def giveSplits(self, ratio: float):
        train_split = int(ratio * len(self.X)) # 80% of data used for training set, 20% for testing 
        X_train, y_train = self.X[:train_split], self.y[:train_split]
        X_test, y_test = self.X[train_split:], self.y[train_split:]
        return (X_train,y_train,X_test,y_test)

# print(len(X_train), len(y_train), len(X_test), len(y_test))



