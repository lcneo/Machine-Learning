import matplotlib.pyplot as plt
import pywt
import numpy as np
from PIL import Image
from glob import glob
import os
from time import time
import time
from sklearn import metrics
from sklearn import preprocessing


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

def Knn_classifier(train_x, train_y,neighbors=4):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(metric='euclidean',n_neighbors=neighbors)
    model.fit(train_x, train_y)
    return model


def get_predict(train_x,train_y,test_x,test_y):
    test_classifiers = ['KNN']
    classifiers = {
                   'KNN':Knn_classifier
                   }
    print ('reading training and testing data...')  
    num_train, num_feat = train_x.shape  
    num_test, num_feat = test_x.shape  
    print ('******************** Data Info *********************')
    #显示训练样本的个数和测试样本的个数
    print ('#training data: %d, #testing_data: %d, dimension: %d' % (num_train, num_test, num_feat))
    for classifier in test_classifiers:  
        #分类器类型
        print ('******************* %s ********************' % classifier)
        start_time = time.time()  
        #训练模型
        model = classifiers[classifier](train_x, train_y,neighbors=1)
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

def sampling(im):
    trait = []
    CC = pywt.dwt2(im,'haar')
    cA,cH,cV = CC[0],CC[1][0],CC[1][1]
    cc = [cA,cH,cV]
    for i in cc:
        for j in Features(i):
            trait.append(j)
    return np.array(trait)

def train_pca(path):
    impath = glob(path)
    pca_x,pca_name = [],[]
    for i in impath:
        im,name = readim(i)
        pca_x.append(sampling(im))
        pca_name.append(name)
    return np.array(pca_x), np.array(pca_name)


def get_train(x,y):
    train = [x*10+y for x in range(40) for y in range(7)]
    test = list(set(range(400)) - set(train))
    return x[train], y[train], x[test], y[test]


def fun_wl(path):
    X,Y = train_pca(path)
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    train_x, train_y, test_x, test_y = get_train(X,Y)  
    return get_predict(train_x,train_y,test_x,test_y)



path = r"/Users/neo/Desktop/ORL/*/*.pgm"
fun_wl(path)

#train_x, train_y, test_x, test_y  = LoadImage(path)
#get_predict(train_x, train_y, test_x, test_y,1)
def fun_test(m,n):
    a = []
    for i in range(m,n):
        a.append(get_predict(train_x, train_y, test_x, test_y,i))
        plt.plot(a)

#print(len(c))
