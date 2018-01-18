#私人的机器学习实用工具库,
import numpy as np
from sklearn import metrics
import time

#数据归一化,输入矩阵,返回矩阵;
def z_score(array):
	from sklearn import preprocessing
	#输入一个mxn的数组进行标准化然后返回
	return preprocessing.scale(array)


#SVM分类器模型,输入参数为train_x,train_y,kernel,输出为模型;
def model_svm(train_x, train_y,kernel = 'rbf'):
    model = SVC(kernel=kernel)
    model.fit(train_x, train_y)
    return model

#KNN分类器模型,输入参数为train_x,train_y,neighbors=1,输出为模型;
def model_knn(train_x, train_y,neighbors=1):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(metric='euclidean',n_neighbors=neighbors)
    model.fit(train_x, train_y)
    return model

#直接返回识别率
def predict_knn(train_x,train_y,test_x,test_y,neighbors = 1,show = False):
    start_time = time.time()
    model = model_knn(train_x,train_y,neighbors = neighbors)
    predict = model.predict(test_x)
    accuracy = metrics.accuracy_score(test_y, predict)
    if show == True:
    	print ('training took %fs!' % (time.time() - start_time))
    	print("识别率为:%.2f%%"%(accuracy*100))
    return accuracy

def predict(train_x,train_y,test_x,test_y,model = model_knn,show = False):
    start_time = time.time()
    model = model(train_x,train_y)
    predict = model.predict(test_x)
    accuracy = metrics.accuracy_score(test_y, predict)
    if show == True:
    	print ('training took %fs!' % (time.time() - start_time))
    	print("识别率为:%.2f%%"%(accuracy*100))
    return accuracy

#PCA 输入为矩阵,输出为矩阵;
def pca(array,n_components = '0.38',svd_solver ='randomized',whiten=True):
	from sklearn.decomposition import PCA
	model_pca = PCA(n_components=n_components, svd_solver=svd_solver,whiten=whiten)
	model_pca.fit(array)
	tra = model_pca.transform(array)
	return tra


if __name__ == '__main__':
	pass