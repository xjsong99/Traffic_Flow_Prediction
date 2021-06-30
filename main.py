from data_process import Data_processer
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import argparse
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(MyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True).cuda()
        self.linear = nn.Linear(hidden_size, 1).cuda()
        self.sigmoid = nn.Sigmoid().cuda()
    
    def forward(self, x):
        _, (hn, _) = self.lstm(x.cuda())
        hn = hn[-1,:,:]
        return self.sigmoid(self.linear(hn))

def get_parsers():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default="test", help="train/test")
    parser.add_argument('--epoch', default=600, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--time_lag', default=12, type=int)
    parser.add_argument('--hidden_size', default=64, type=int)
    parser.add_argument('--num_layers', default=2, type=int)

    return parser.parse_args()

def main():
    args = get_parsers()

    data = Data_processer("data/train.csv", "data/test.csv", args.time_lag)
    lstm = MyLSTM(1, args.hidden_size, args.num_layers)

    if args.mode == "test":
        print('==== test ====')
        testX, testY = data.get_testDataset()
        
        epoch_index = 0
        while os.path.isfile("./model/epoch_" + str(epoch_index + 10)):
            epoch_index += 10
        
        print("load model from epoch ", epoch_index, "...")
        path = "./model/epoch_" + str(epoch_index)
        lstm.load_state_dict(torch.load(path))

        x = testX[:,:].unsqueeze(2)
        y_predict = lstm.forward(x).detach().cpu().numpy()
        y_predict = data.scaler.inverse_transform(y_predict).reshape(1, -1)[0,:]

        y_label = testY.numpy()
        y_label = data.scaler.inverse_transform(y_label).reshape(1, -1)[0,:]

        F = plt.figure()
        fig = F.add_subplot(111)

        x_index = pd.date_range('2016-3-4 00:00', periods=288, freq='5min')[args.time_lag:]
        fig.plot(x_index, y_label[:len(x_index)], linestyle='-', linewidth=1, color="red", label="real flow")
        fig.plot(x_index, y_predict[:len(x_index)], linestyle='-', linewidth=1, color="blue", label="predicted flow")
        
        plt.xlabel("time [min]")
        plt.ylabel("flow [veh/5min]")
        plt.legend()
        
        fig.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H:%M"))
        
        plt.savefig("./test.png")
        plt.show()

    elif args.mode == "train":
        print('==== train ====')

        trainX, trainY = data.get_trainDataset()
        optimizer = torch.optim.Adam(lstm.parameters(), lr=args.lr)

        loss_record = []

        for epoch_num in range(args.epoch):
            for index in range(args.batch_size, len(trainX), args.batch_size):
                # [batch_size, time_lag].unsqueeze -> [batch_size, time_lag, 1]
                x = trainX[index - args.batch_size:index,:].unsqueeze(2)

                # [batch_size, 1]
                y_label = trainY[index - args.batch_size:index,:]
                
                # [batch_size, 1]
                y_predict = lstm.forward(x)
                
                loss_func = nn.MSELoss()
                loss = loss_func(y_predict, y_label.cuda())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_record.append(float(loss.detach().cpu()))

            if epoch_num % 10 == 0:
                print("epoch:%3d | loss:%6.4f"%(epoch_num, loss))
                path = "./model/epoch_" + str(epoch_num)
                torch.save(lstm.state_dict(), path)
        
        F = plt.figure()
        fig = F.add_subplot(111)
        fig.plot(range(args.epoch), loss_record, linestyle='-')
        
        fig.plot(args.epoch-1, loss_record[-1], marker='o')
        plt.annotate("(%.4f)" % (loss_record[-1]), xy=(args.epoch-1, loss_record[-1]), xytext=(-20, 15), textcoords='offset points')

        plt.xlabel("epoch")
        plt.ylabel("MSE")

        plt.savefig("./train_loss.png")
        plt.show()
        
    else:
        print("please choose train/test.")

if __name__ == '__main__':
    main()