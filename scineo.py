#私人的机器学习实用工具库,
import numpy as np
from sklearn import metrics
import time


#数据归一化,输入矩阵,返回矩阵;
def z_score(Mat):
	from sklearn import preprocessing
	#输入一个mxn的数组进行标准化然后返回
	return preprocessing.scale(Mat)




#SVM分类器模型,输入参数为train_x,train_y,kernel,输出为模型;
def model_svm(train_x, train_y,kernel = 'rbf'):
	from sklearn.svm import SVC
	model = SVC(kernel=kernel)
	model.fit(train_x, train_y)
	return model

#KNN分类器模型,输入参数为train_x,train_y,neighbors=1,输出为模型;
def model_knn(train_x, train_y,neighbors=1):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(metric='euclidean',n_neighbors=neighbors)
    model.fit(train_x, train_y)
    return model




#分别输入样本,直接返回knn的识别率
def predict_knn(train_x,train_y,test_x,test_y,neighbors = 1,show = False):
    start_time = time.time()
    model = model_knn(train_x,train_y,neighbors = neighbors)
    predict = model.predict(test_x)
    accuracy = metrics.accuracy_score(test_y, predict)
    if show == True:
    	print ('training took %fs!' % (time.time() - start_time))
    	print("识别率为:%.2f%%"%(accuracy*100))
    return accuracy
#输入的为矩阵
def predict_knn_mat(mat,neighbors = 1,show = False):
    train_x,train_y,test_x,test_y = mat
    start_time = time.time()
    model = model_knn(train_x,train_y,neighbors = neighbors)
    predict = model.predict(test_x)
    accuracy = metrics.accuracy_score(test_y, predict)
    if show == True:
        print ('training took %fs!' % (time.time() - start_time))
        print("识别率为:%.2f%%"%(accuracy*100))
    return accuracy

#分别输入样本,直接返回svm的识别率
def predict_svm(train_x,train_y,test_x,test_y,kernel = 'rbf',show = False):
    start_time = time.time()
    model = model_svm(train_x, train_y,kernel = kernel)
    predict = model.predict(test_x)
    accuracy = metrics.accuracy_score(test_y, predict)
    if show == True:
        print ('training took %fs!' % (time.time() - start_time))
        print("识别率为:%.2f%%"%(accuracy*100))
    return accuracy
#输入样本矩阵
def predict_svm_mat(mat,kernel = 'rbf',show = False):
    train_x,train_y,test_x,test_y = mat
    start_time = time.time()
    model = model_svm(train_x, train_y,kernel = kernel)
    predict = model.predict(test_x)
    accuracy = metrics.accuracy_score(test_y, predict)
    if show == True:
        print ('training took %fs!' % (time.time() - start_time))
        print("识别率为:%.2f%%"%(accuracy*100))
    return accuracy

#分别输入样本和模型,直接返回识别率
def predict(train_x,train_y,test_x,test_y,model = model_knn,show = False):
    start_time = time.time()
    model = model(train_x,train_y)
    predict = model.predict(test_x)
    accuracy = metrics.accuracy_score(test_y, predict)
    if show == True:
    	print ('training took %fs!' % (time.time() - start_time))
    	print("识别率为:%.2f%%"%(accuracy*100))
    return accuracy
#输入样本矩阵
def predict(mat,model = model_knn,show = False):
    train_x,train_y,test_x,test_y = mat
    start_time = time.time()
    model = model(train_x,train_y)
    predict = model.predict(test_x)
    accuracy = metrics.accuracy_score(test_y, predict)
    if show == True:
        print ('training took %fs!' % (time.time() - start_time))
        print("识别率为:%.2f%%"%(accuracy*100))
    return accuracy



#PCA 输入为矩阵,输出为矩阵;
def pca(mat,n_components = 0.38,svd_solver ='full',whiten=True):
	from sklearn.decomposition import PCA
	model_pca = PCA(n_components=n_components, svd_solver=svd_solver,whiten=whiten)
	model_pca.fit(mat)
	mat = model_pca.transform(mat)
	return mat

#提取hog特征
def hog(mat,orientations=9, pixels_per_cell=(8, 8),cells_per_block=(3, 3)):
    from skimage.feature import hog as fun_hog
    hog_mat = []
    for im in mat:
        fd = fun_hog(im,orientations=orientations, pixels_per_cell=pixels_per_cell,cells_per_block=cells_per_block,block_norm = 'L2-Hys',visualise=False)
        hog_mat.append(fd)
    return np.array(hog_mat)
#
def hog_mat(mat,orientations=9, pixels_per_cell=(8, 8),cells_per_block=(3, 3)):
    start = time.time()
    train,train_y,test,test_y = mat
    train_x = hog(train,orientations = orientations,pixels_per_cell = pixels_per_cell,cells_per_block = cells_per_block)
    test_x = hog(test,orientations = orientations,pixels_per_cell = pixels_per_cell,cells_per_block = cells_per_block)
    print("transform time : %.2f S\t detail : %d"%(time.time()- start,train_x.shape[1]))
    return train_x,train_y,test_x,test_y


if __name__ == '__main__':
	pass