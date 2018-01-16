from sklearn import metrics
from skimage.feature import hog
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

#设置HOG的参数
orientations=8
pixels_per_cell=(4, 4)
cells_per_block=(1, 1)
block_norm = 'L2-Hys'
visualise=True


#分类器模型svm
def svm(train_x, train_y):
    model = SVC(kernel='rbf', probability=True)
    model.fit(train_x, train_y)
    return model


def knn(train_x, train_y):
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model

#训练模型及获取识别率
def get_svm_predict(train_x,train_y,test_x,test_y):
    model = knn(train_x,train_y)
    predict = model.predict(test_x)
    accuracy = metrics.accuracy_score(test_y, predict)
    print("识别率为:%.2f%%"%(accuracy*100))
    

#提取HOG特征
def fun_hog(im,orientations=orientations, pixels_per_cell=pixels_per_cell,cells_per_block=cells_per_block,block_norm = block_norm,visualise=visualise):
    fd, hog_image = hog(im,orientations=orientations, pixels_per_cell=ixels_per_cell,cells_per_block=cells_per_block,block_norm = block_norm,visualise=visualise)
    return hog_image

#将图片全部转换为一维的数组
def get_data():
    train_x,test_x = [],[]
    train,train_y,test,test_y = np.load("DataSet/No4.npy")
    for i in train:
        train_x.append(i.flatten())
    for i in test:
        test_x.append(i.flatten())
    return np.array(train_x), train_y, np.array(test_x),test_y
        
def fun_svm():
    print ('reading training and testing data...')
    train_x,train_y,test_x,test_y = get_data()
    
    get_svm_predict(train_x,train_y,test_x,test_y)

fun_svm()