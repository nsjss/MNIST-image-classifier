import numpy as np
import pandas as pd
import matplotlib as plt
import math
import random


df=pd.read_csv('train.csv')

train=np.array(df).T
print(train.shape)
#print(train)

y_train=train[0]
x_train=train[1:train.shape[1]]
x_train=x_train/255
print(x_train.shape,'ss')

print(y_train.shape,'sssss')

b1=0
b2=0

w1=random.randint(0,10)
print(w1)

print(x_train.shape)
m,n=x_train.shape

def parameters():
    w1=np.random.rand(10,784)-0.5
    b1=np.random.rand(10,1)-0.5
    w2=np.random.rand(10,10)-0.5
    b2=np.random.rand(10,1)-0.5
    return w1,b1,w2,b2

def sigmoid(z1):
    return 1/(1+(math.e**-z1))
#print(sigmoid(1))

#ef ReLU(Z):
    #return np.maximum(Z, 0)

#def softmax(Z):
    #A = np.exp(Z) / sum(np.exp(Z))
    #return A

def forward_prop(w1,b1,w2,b2,X):
    z1=w1.dot(X)+b1
    a1=sigmoid(z1)
    z2=w2.dot(a1)+b2
    a2=sigmoid(z2)
    return z1,a1,z2,a2

def Cost(y):
    cost=np.zeros((y.size,y.max()+1))
    cost[np.arange(y.size),y]=1
    cost=cost.T
    return cost

'''def ReLU_deriv(Z):
    return Z > 0'''

def back_prop(z1,a1,z2,a2,w2,x,y):
    m=y.size
    cost=Cost(y)
    dz2=a2-cost
    dw2=2/m*dz2.dot(a1.T)
    db2=2/m*np.sum(dz2)
    dz1=w2.T.dot(dz2)*math.e**-z1/(1+math.e**-z1)**2
    dw1=2/m*dz1.dot(x.T)
    db1=2/m*np.sum(dz1)
    return dw1,db1,dw2,db2

def update_params(w1,b1,w2,b2,dw1,db1,dw2,db2,lr):
    #print('sdsds',w1,w2,b1,b2)
    w1=w1-dw1*lr
    b1=b1-db1*lr
    w2=w2-dw2*lr
    b2=b2-db2*lr
    return w1,b1,w2,b2

def get_predictions(a2):
    return np.argmax(a2,0)

def get_accuracy(predictions,y):
    print(predictions,y)
    return np.sum(predictions==y)/(y.size)

def gradient_desc(x,y,iter,lr):
    w1,b1,w2,b2=parameters()
    for i in range(iter):
        z1,a1,z2,a2=forward_prop(w1,b1,w2,b2,x)
        dw1,db1,dw2,db2=back_prop(z1,a1,z2,a2,w2,x,y)
        w1,b1,w2,b2=update_params(w1,b1,w2,b2,dw1,db1,dw2,db2,lr)
        if i%10==0:
            print('iter',i)
            predictions = get_predictions(a2)
            print(get_accuracy(predictions, y))
    return w1,b1,w2,b2

iter=500
lr=0.5
w1,b1,w2,b2=gradient_desc(x_train,y_train,iter,lr)

df_test=pd.read_csv('test.csv')

test=np.array(df_test).T
y_test=test[0]
x_test=test[1:train.shape[1]]
x_test=x_test/255

Z1,A1,Z2,A2=forward_prop(w1,b1,w2,b2,x_test)
pred=get_predictions(A2)
print('acc',get_accuracy(pred,y_test))





    









