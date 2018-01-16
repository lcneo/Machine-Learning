from sklearn.decomposition import PCA
from hog import get_svm_predict
import numpy as np

def fun_pca(X, n = 50):
    pca = PCA(n_components = n)
    pca.fit(X)
    return pca.explained_variance_ratio_


def read_data():
	train,train_y,test,test_y = np.load("DataSet/No4_hog_flatten.npy")
	X = np.concatenate([train,test])
	return X,train_y, test_y

def get_predict():
	X,train_y, test_y = read_data()
	X = fun_pca(X)
	get_svm_predict(X[:len(train_y)],train_y,X[len(train_y):],test_y)

if __name__ == "__main__":
	get_predict()

