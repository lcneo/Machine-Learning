from sklearn import metrics
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import time
from tqdm import tqdm

#设置HOG的参数
orientations=16
pixels_per_cell=(16, 16)
cells_per_block=(3, 3)
block_norm = 'L2-Hys'
visualise=True


#分类器模型svm
def svm(train_x, train_y):
    model = SVC(kernel='rbf')
    model.fit(train_x, train_y)
    return model


def knn(train_x, train_y):
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model

#训练模型及获取识别率
def get_svm_predict(train_x,train_y,test_x,test_y):
    start_time = time.time()
    model = knn(train_x,train_y)   
    predict = model.predict(test_x)
    accuracy = metrics.accuracy_score(test_y, predict)
    print ('training took %fs!' % (time.time() - start_time))
    print("识别率为:%.2f%%"%(accuracy*100))
    

#提取HOG特征
def hog(im,orientations=orientations, pixels_per_cell=pixels_per_cell,cells_per_block=cells_per_block,block_norm = block_norm):
    from skimage.feature import hog as fun_hog
    return fun_hog(im,orientations=orientations, pixels_per_cell=pixels_per_cell,cells_per_block=cells_per_block,block_norm = block_norm,visualise=False)

#将图片全部转换为一维的数组
def get_data():
    train_x,test_x = [],[]
    train,train_y,test,test_y = np.load("hog.npy")
    for i in train:
        train_x.append(i.flatten())
    for i in test:
        test_x.append(i.flatten())
    return np.array(train_x), train_y, np.array(test_x),test_y
        
def get_hog_data():
    train_x,test_x = [],[]
    start_time = time.time()
    train,train_y,test,test_y = np.load("DataSet/No4.npy")
    for i in tqdm(train):

        train_x.append(fun_hog(i))
    for i in tqdm(test):
        test_x.append(fun_hog(i))
    print("transformation time:%.2s"%(time.time() - start_time))
    return np.array(train_x), train_y, np.array(test_x),test_y

def fun_svm():
    print ('reading training and testing data...')
    train_x,train_y,test_x,test_y = get_hog_data()
    num_train, num_test, num_feat = len(train_x),len(test_x),len(train_x[0])
    print ('******************** Data Info *********************')
    print ('#training data: %d, #testing_data: %d, dimension: %d' % (num_train, num_test, num_feat))
    get_svm_predict(train_x,train_y,test_x,test_y)

#fun_svm()
    
def fun_test():
    print ('reading training and testing data...')
    train_x,train_y,test_x,test_y = np.load("DataSet/No4_hog_flatten.npy")
    num_train, num_test, num_feat = len(train_x),len(test_x),len(train_x[0])
    print ('******************** Data Info *********************')
    print ('#training data: %d, #testing_data: %d, dimension: %d' % (num_train, num_test, num_feat))
    get_svm_predict(train_x,train_y,test_x,test_y)
 
#fun_test()
if __name__ == "__main__":
    npy = get_hog_data()
    np.save("hog",npy)