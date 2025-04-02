import sklearn.model_selection
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from work1 import x_train, y_train, x_val, x_test, y_val, y_test, FeedForwardNN, output_dim, hidden_dim, num_layers, \
    activation, predictions


def target_function(x):    #定义目标函数
    return torch.log2(x) + torch.cos(torch.pi*x/2)

#生成数据集
def generate_data(num_samples = 1000,seed = 42):
    torch.manual_seed(seed)   #pytorch随机数固定
    np.random.seed(seed)      #numpy随机数固定

    x = torch.linspace(1,16,num_samples).unsqueeze(1)
    #torch.linspace(start, end ,num_samples)表示生成num_samples个【start，end】里均匀分布的数
    #unsqueeze（1）表示增加一个维度
    y = target_function(x)

    x_train, x_temp, y_train, y_temp = train_test_split(x,y,test_size=0.2, random_state=seed)
    x_val, x_test, y_val, y_test = train_test_split(x_temp,y_temp,test_size=0.5,random_state=seed)

    return x_train,y_train,x_val,y_val,x_test,y_test

x_train, y_train, x_val, y_val, x_test, y_test = generate_data(num_samples=2000)
plt.scatter(x_train.numpy(),y_train.numpy(),label = "Train Data",color = "blue", s=5)
plt.scatter(x_val.numpy(),y_val.numpy(),label = "Validation Data", color = "green", s=5)
plt.scatter(x_test.numpy(),y_test.numpy(),label="Test Data",color = "red", s = 5)
plt.legend()
plt.ylabel("y")
plt.xlabel("x")
plt.title("Dataset Visualization")
plt.show()

import torch.nn as nn
import torch.optim as optim

class FeedForwardNN(nn.Module):#继承父类模块
    def __init__(self,input_dim,hidden_dim,output_dim,num_layers,activation):   #__init__方法：在实例化对象时，将 input_dim 等参数传入并存储，方便后续在类的方法中使用
        super(FeedForwardNN,self).__init__()

        layers = []
        layers.append(nn.Linear(input_dim,hidden_dim))
        layers.append(activation())

        for _ in range(num_layers-1):
            layers.append(nn.Linear(hidden_dim,hidden_dim))
            layers.append(activation())

        layers.append(nn.Linear(hidden_dim,output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self,x):
        return self.model(x)

input_dim = 1
output_dim = 1
hidden_dim = 64
num_layers = 3
activation = nn.ReLU

model = FeedForwardNN(input_dim,hidden_dim,output_dim,num_layers,activation)
print(model)

#训练
import torch.optim as optim
from sklearn.metrics import mean_squared_error

def train(model,x_train,y_train,epochs=100,lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=lr)


    x_train = torch.Tensor(x_train).float()
    y_train = torch.Tensor(y_train).float()

    for epoch in range(epochs):
        outputs = model(x_train)
        loss = criterion(outputs,y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1)%100==0:
            print(f'Epoch[{epoch+1}/{epochs}],Loss:{loss.item():.4f}')

train(model,x_train,y_train,epochs=100,lr=0.001)

#验证
def validate(model,x_val,y_val):
    model.eval() #切换到评估模式
    with torch.no_grad():
        x_val = torch.Tensor(x_val).float()
        y_val = torch.Tensor(y_val).float()

        predictions = model(x_val)
        mse = mean_squared_error(y_val.numpy(),predictions.numpy())
        return mse,predictions

mse,predictions = validate(model,x_val,y_val)
print(f'Validation MSE: {mse:.4f}')


#测试
def evaluate(model,x_test,y_test):
    model.eval()
    with torch.no_grad():
        x_test = torch.Tensor(x_val).float()
        y_test = torch.Tensor(y_val).float()
        predictions = model(x_test)

        # 计算均方误差 (MSE)
        mse = mean_squared_error(y_test.numpy(), predictions.numpy())
        return mse, predictions

    # 在测试集上测试模型
mse_test, predictions_test = evaluate(model, x_test, y_test)
print(f'Test MSE: {mse_test:.4f}')








