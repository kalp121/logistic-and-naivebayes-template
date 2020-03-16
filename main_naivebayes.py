# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import scipy.io
import numpy as np

#function for finding relative probability 
def relative_prob(x, mean, std):
	expo = np.exp(-((x-mean)**2 / (2 * std**2 )))
	return (1 / (np.sqrt(2 * np.pi) * std)) * expo

#function for confusion matrix
def cm(y_test,y_pred):
    tp=len([i for i in range(0,y_test.shape[0]) if y_test[i]==0 and y_pred[i]==0])
    tn=len([i for i in range(0,y_test.shape[0]) if y_test[i]==0 and y_pred[i]==1])
    fp=len([i for i in range(0,y_test.shape[0]) if y_test[i]==1 and y_pred[i]==0])
    fn=len([i for i in range(0,y_test.shape[0]) if y_test[i]==1 and y_pred[i]==1])
    return np.array([[tp,tn],[fp,fn]])


#reading data
num_class=2
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


#splitting seven digit images array and eight images array
seven_index=[]
eight_index=[]

#loop to find indexes of seven and eight
for i in range(len(train_y[0])):
    if train_y[0][i]==0:
        seven_index.append(i)
    else:
        eight_index.append(i)

#applied direct split since all the eight digit images are after index 6265
train_seven_mean=train_x_mean[:6265]
train_eight_mean=train_x_mean[6265:]

train_seven_std=train_x_std[:6265]
train_eight_std=train_x_std[6265:]

seven_train=np.column_stack([train_seven_mean, train_seven_std])
eight_train=np.column_stack([train_eight_mean, train_eight_std])


#mean and varience of the training data(used to calculate probability)
seven_mean=np.mean(seven_train,axis=0)
seven_var=np.var(seven_train,axis=0)
seven_mv=np.column_stack([seven_mean,seven_var])

eight_mean=np.mean(eight_train,axis=0)
eight_var=np.var(eight_train,axis=0)
eight_mv=np.column_stack([eight_mean,eight_var])

dic={}
dic[0]=seven_mv
dic[1]=eight_mv

#calculating p(digit7) and p(digit8)
p=[]
p.append(len(train_seven_mean)/len(train_x_mean))
p.append(len(train_eight_mean)/len(train_x_mean))

#function to predict label for the input data
def nb(input_mean,input_std,dic,p):
    all_prob=[]
    for i in range(num_class):
        a=relative_prob(input_mean,dic[i][0][0],np.sqrt(dic[i][0][1]))
        b=relative_prob(input_std,dic[i][1][0],np.sqrt(dic[i][1][1]))
        
        all_prob.append(a*b*p[i])
    return np.argmax(all_prob)

#predicting labels for all the test data
predict=[]
for i in range(len(test_x_mean)):
    predict.append(nb(test_x_mean[i],test_x_std[i],dic,p))

#confusion matrix and accuracy measures 
confusion_matrix=cm(test_y[0],predict)
accuracy=((confusion_matrix[0][0]+confusion_matrix[1][1])/len(test_y[0]))*100
