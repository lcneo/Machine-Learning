import scineo as sn
import numpy as np

train,train_y,test,test_y = np.load("DataSet/No4.npy")
train_x,test_x = train.reshape(-1,128*128),test.reshape(-1,128*128)
X = np.concatenate([train_x,test_x])
X = sn.pca(X)
#print(X.shape)
train_x,test_x = X[:len(train_y)],X[len(train_y):]
sn.predict_knn(train_x,train_y,test_x,test_y,show=True)