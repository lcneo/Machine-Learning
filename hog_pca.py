from sklearn.decomposition import PCA
from hog import get_svm_predict
import numpy as np

def fun_pca(X, n = 50):
    pca = PCA(n_components = n)
    pca.fit(X)
    return pca.explained_variance_ratio_


def read_data():
<<<<<<< HEAD
<<<<<<< HEAD
	train,train_y,test,test_y = np.load("hog.npy")
=======
	train,train_y,test,test_y = np.load("DataSet/No4_hog_flatten.npy")
>>>>>>> parent of 30be9a2... 1_17_5:13
=======
	train,train_y,test,test_y = np.load("DataSet/No4_hog_flatten.npy")
>>>>>>> parent of 30be9a2... 1_17_5:13
	X = np.concatenate([train,test])
	return X,train_y, test_y

def get_predict():
<<<<<<< HEAD
<<<<<<< HEAD
	TT,train_y, test_y = read_data()
	X= TT
	i = 16
	print("n = %f"%(i))
	X = fun_pca(TT,n= i)
	X = preprocessing.scale(X)
	train_x,test_x =X[:len(train_y)],X[len(train_y):]
	get_svm_predict(train_x,train_y,test_x,test_y)
=======
	X,train_y, test_y = read_data()
	X = fun_pca(X)
	get_svm_predict(X[:len(train_y)],train_y,X[len(train_y):],test_y)
>>>>>>> parent of 30be9a2... 1_17_5:13
=======
	X,train_y, test_y = read_data()
	X = fun_pca(X)
	get_svm_predict(X[:len(train_y)],train_y,X[len(train_y):],test_y)
>>>>>>> parent of 30be9a2... 1_17_5:13

if __name__ == "__main__":
	get_predict()

