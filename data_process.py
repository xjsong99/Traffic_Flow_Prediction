import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
import torch

class Data_processer:
    def __init__(self, train_file, test_file, lag):
        self.train_file = train_file
        self.test_file = test_file
        self.lag = lag

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        attr = "Lane 1 Flow (Veh/5 Minutes)"
        df = pd.read_csv(self.train_file, encoding="utf-8").fillna(0)
        self.scaler.fit(df[attr].values.reshape(-1, 1))

    def get_trainDataset(self):
        print("Generating train dataset...")
        
        attr = "Lane 1 Flow (Veh/5 Minutes)"
        df = pd.read_csv(self.train_file, encoding="utf-8").fillna(0)
        
        flow = self.scaler.transform(df[attr].values.reshape(-1, 1)).reshape(1, -1)[0,:]
        
        trainset = []
        for i in range(self.lag, len(flow)):
            trainset.append(flow[i - self.lag:i + 1])

        random.shuffle(trainset) #training data需要shuffle；test data不用shuffle

        trainset = np.array(trainset, dtype=np.float32)

        trainX = torch.from_numpy(trainset[:,:-1])
        trainY = torch.from_numpy(trainset[:, -1]).unsqueeze(1)

        print("Done.")
        
        return trainX, trainY

    def get_testDataset(self):
        print("Generating test dataset...")

        attr = "Lane 1 Flow (Veh/5 Minutes)"
        df = pd.read_csv(self.test_file, encoding="utf-8").fillna(0)
        
        flow = self.scaler.transform(df[attr].values.reshape(-1, 1)).reshape(1, -1)[0,:]
        
        testset = []
        for i in range(self.lag, len(flow)):
            testset.append(flow[i - self.lag:i + 1])

        testset = np.array(testset, dtype=np.float32)

        testX = torch.from_numpy(testset[:,:-1])
        testY = torch.from_numpy(testset[:, -1]).unsqueeze(1)

        print("Done.")
        
        return testX, testY

def main():
    data = Data_processer("data/train.csv", "data/test.csv", 12)
    trainX, trainY = data.get_trainDataset()
    testX, testY = data.get_testDataset()
    print(trainX, trainY)
    print(testX, testY)

if __name__ == "__main__":
    main()
