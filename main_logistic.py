# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
import numpy as np
import scipy.io
learning_rate=0.009
iterations=1000
weights=None
bias=None


#reading data

numpyfile= scipy.io.loadmat('mnist_data.mat') 
train_y=numpyfile['trY']
test_y=numpyfile['tsY']

train_x=numpyfile['trX']
test_x=numpyfile['tsX']

#taking mean and standard deviation fature for all the images
train_x_mean=np.mean(train_x,axis=1)
train_x_std=np.std(train_x,axis=1)

test_x_mean=np.mean(test_x,axis=1)
test_x_std=np.std(test_x,axis=1)

train_x=np.column_stack([train_x_mean, train_x_std])
test_x=np.column_stack([test_x_mean,test_x_std])


#loss function
def loss(h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

#sigmoid function
def sigmoid(a):
    return 1/(np.exp(-a)+1)

#fit function
def fit(train_x,train_y,learning_rate,iterations):
    global weights
    global bias
    features=len(train_x[0])
    samples=len(train_x)
    
    weights=np.zeros(features)
    bias=0
    for i in range(iterations):
        #approximating value of y with wieghts,x and bias
        lm=np.dot(train_x,weights)+bias
        #applying sigmoid function
        y_pred=sigmoid(lm)
        
        #applying gradient
        gradient_w=np.dot(train_x.T,(y_pred-train_y))*(1/samples)
        gradient_b=np.sum(y_pred-train_y)*(1/samples)
        
        weights-=learning_rate*gradient_w
        bias-=learning_rate*gradient_b

def predict(test_x):
    global weights
    global bias
    lm=np.dot(test_x,weights)+bias
    y_pred=sigmoid(lm)
    y_pred_values=[0 if i <= 0.48 else 1 for i in y_pred]
    return np.array(y_pred_values)

#function for confusion matrix
def cm(y_test,y_pred):
    tp=len([i for i in range(0,y_test.shape[0]) if y_test[i]==0 and y_pred[i]==0])
    tn=len([i for i in range(0,y_test.shape[0]) if y_test[i]==0 and y_pred[i]==1])
    fp=len([i for i in range(0,y_test.shape[0]) if y_test[i]==1 and y_pred[i]==0])
    fn=len([i for i in range(0,y_test.shape[0]) if y_test[i]==1 and y_pred[i]==1])
    return np.array([[tp,tn],[fp,fn]])


#fitting training datset
fit(train_x,train_y[0].T,0.04,15000)
#predicting test dataset
y_pred=predict(test_x)

#confusion matrix and accurancy
confusion_matrix=cm(test_y[0],y_pred)
accuracy=((confusion_matrix[0][0]+confusion_matrix[1][1])/len(test_y[0]))*100