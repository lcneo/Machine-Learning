import matplotlib.pyplot as plt
import pywt
import numpy as np
from PIL import Image
from glob import glob
import os
from time import time
import time
from sklearn import metrics


path = '/Users/neo/Downloads/IMG_0033.JPG'
im = Image.open(path)
im = im.resize((138,189))
im = im.convert('L')
im = np.array(im)
CC = pywt.dwt2(im,'haar')
cA,cH,cV,cD = CC[0],CC[1][0],CC[1][1],CC[1][2]


#读取文件,将训练样本和测试样本进行分离
def LoadImage(path):
    train_x, train_y, test_x, test_y = [],[],[],[]
    #tarinlist = ['1.pgm','2.pgm','3.pgm','4.pgm','5.pgm']
    tarinlist = ['1.pgm','2.pgm','3.pgm','4.pgm','5.pgm','6.pgm','7.pgm']
    print(">>>Loading ROL Data")
    impath = glob(path)
    for i in impath:
        im,name = readim(i)
        if os.path.basename(i) in tarinlist:
            train_x.append(sampling(im))
            train_y.append(name)
        else:
            test_x.append(sampling(im))
            test_y.append(name)
    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)

#读取图片,将图片转换为矩阵,读取类别
def readim(path):
    im = Image.open(path)
    name = os.path.basename(os.path.dirname(path))
    return np.array(im),name

def Knn_classifier(train_x, train_y,neighbors=1):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(metric='euclidean',n_neighbors=neighbors)
    model.fit(train_x, train_y)
    return model


def get_predict(train_x,train_y,test_x,test_y,k):
    test_classifiers = ['KNN']
    classifiers = {
                   'KNN':Knn_classifier
                   }
    print ('reading training and testing data...')  
    num_train, num_feat = train_x.shape  
    num_test, num_feat = test_x.shape  
    print ('******************** Data Info *********************')
    #显示训练样本的个数和测试样本的个数
    print ('#training data: %d, #testing_data: %d, dimension: %d,k :%d' % (num_train, num_test, num_feat,k))
    for classifier in test_classifiers:  
        #分类器类型
        print ('******************* %s ********************' % classifier)
        start_time = time.time()  
        #训练模型
        model = classifiers[classifier](train_x, train_y,neighbors=k)
        #训练模型所用时间
        print ('training took %fs!' % (time.time() - start_time))
        
        #predict为预测的结果,返回的为一个ndarray类型的列表,为对应test_x类型的预测结果.
        predict = model.predict(test_x)
        #成功识别的概率      
        #将test_y,与预测结果进行比对,得到预测的准确率.
        accuracy = metrics.accuracy_score(test_y, predict)
        print ('accuracy: %.2f%%' % (100 * accuracy))
    return accuracy
#将矩阵直接打印为图片
def printim(im):
    #cmap = 'gray' 打印的图片的类型灰度化
    plt.imshow(im,cmap = 'gray')
    plt.show()

#返回输入矩阵的均值,平方差,绝对值和,平方和
def Features(A):
    return A.mean(), A.var(), np.abs(A).sum(), (A*A).sum()


#  PCA 主成成分分析 选取特征值大的特征向量作为主成分
def pca(data,k):                   # data为训练样本
    data = np.float32(np.mat(data)) 
    rows,cols = data.shape        # 训练样本的维度
    data_mean = np.mean(data,0)    #对列求均值
    data_mean_all = np.tile(data_mean,(rows,1))
    Z = data - data_mean_all
    T1 = Z*Z.T                    #使用矩阵计算，所以前面mat
    D,V = np.linalg.eig(T1)       #特征值与特征向量
    V1 = V[:,0:k]                 #取前k个特征向量
    V1 = Z.T*V1
    for i in np.arange(k):        #特征向量归一化
        L = np.linalg.norm(V1[:,i])
        V1[:,i] = V1[:,i]/L
        
    data_new = Z*V1               # 降维后的数据
    return data_new,data_mean,V1  # 降维后的数据 ， 平均脸  ，


def sampling(im):
    trait = []
    CC = pywt.dwt2(im,'haar')
    cA,cH,cV = CC[0],CC[1][0],CC[1][1]
    cc = [cA,cH,cV]
    for i in cc:
        for j in Features(i):
            trait.append(j)
    return np.array(trait)
path = r"/Users/neo/Desktop/ORL/*/*.pgm"
train_x, train_y, test_x, test_y  = LoadImage(path)
#get_predict(train_x, train_y, test_x, test_y,1)
a = []
for i in range(1,150):
    a.append(get_predict(train_x, train_y, test_x, test_y,i))
    plt.plot(a)
#print(len(c))
