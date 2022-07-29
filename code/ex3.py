import numpy as np;
import scipy.io as sc
from predict import predict;


# 加载数据
data = sc.loadmat('Machine Learning/Neural Network/Feedforward Propagation/ex3data1.mat');
theta = sc.loadmat('Machine Learning/Neural Network/Feedforward Propagation/ex3weights.mat');
X = data['X'];
y = data['y'];
theta1 = theta['Theta1'];
theta2 = theta['Theta2'];


p = predict(theta1, theta2, X);

print('训练模型的精度为: ', np.mean(p==y)*100, '%');


