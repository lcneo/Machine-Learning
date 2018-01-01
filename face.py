'''
整理样本:
读取ROL数据的集中的所有样本图片,将图片分别转换为一位数组,所有数组组合成一个ndarray类型的二维矩阵X;
将与X依次对应的图片的类型的集合转换为一个ndarray类型的一位矩阵Y;

PCA分析:
.
.
.


选取测试样本和训练样本:
每个类型的前五张图片作为训练样本组成train_x,与之对应的类型train_y;
剩下的后六张图片作为测试样本test_x,与之对应的类型train_y;

训练模型生成结果:
选取KNN分类器模型,距离参数选这欧式距离
model = KNeighborsClassifier(metric='euclidean')
训练模型
model.fit(train_x, train_y)
预测结果
redict = model.predict(test_x)
计算识别率
accuracy = metrics.accuracy_score(test_y, predict)


'''




from glob import glob
from PIL import Image
import os
from sklearn.decomposition import PCA 
import time
import scipy.misc as sm
import numpy as np
from sklearn import metrics


#读取图片，返回一个一维数组,和所在文件夹的名字
def readim(path):
    im = sm.imread(path)
    name = os.path.basename(os.path.dirname(path))
    return im.flatten(),name
#读取ROL文件夹中所有文件,分别分类,数据进行5:5分类,一般作为训练样本,一半作为测试样本.
#返回值为train_x, train_y, test_x, test_y;分别是五个ndarray类型.
#其中train_x和test_x都是读取图片的一位数组,train_y和teat_y为对应x的类型
def LoadImage(path):
    train_x, train_y, test_x, test_y = [],[],[],[]
    #tarinlist = ['1.pgm','2.pgm','3.pgm','4.pgm','5.pgm']
    tarinlist = ['1.pgm','2.pgm','3.pgm','4.pgm','5.pgm','6.pgm','8.pgm']
    print(">>>Loading ROL Data")
    impath = glob(path)
    for i in impath:
        im,name = readim(i)
        if os.path.basename(i) in tarinlist:
            train_x.append(im)
            train_y.append(name)
        else:
            test_x.append(im)
            test_y.append(name)
    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)

def fun_pca(X):
    pca = PCA(n_components = 50)
    pca.fit(X)
    return pca.explained_variance_ratio_

def imgshow(filename):
    img =Image.open(filename);
    #img.show();
    #cdata =img.convert('L');这是不是灰度矩阵的意思？
    width,height = img.size
    data = img.getdata();
    data = np.array(data);
    new_data = np.reshape(data,(1,width*height));
    return new_data, width, height;

#naive_bayes作为分类器,训练分类模型,train为每个人相片的1~5,test为每个人相片的6~10.
def naive_bayes_classifier(train_x,train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha = 0.01)
    model.fit(train_x,train_y)
    return model

def Knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(metric='euclidean')

    model.fit(train_x, train_y)
    return model

#运用之前使用过的代码,剩下部分就是对训练好的模型进行测试.
def get_predict(train_x,train_y,test_x,test_y):
    test_classifiers = ['NB','KNN']
    classifiers = {'NB':naive_bayes_classifier,'KNN':Knn_classifier}
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
        model = classifiers[classifier](train_x, train_y)
        #训练模型所用时间
        print ('training took %fs!' % (time.time() - start_time))
        
        #predict为预测的结果,返回的为一个ndarray类型的列表,为对应test_x类型的预测结果.
        predict = model.predict(test_x)


        #成功识别的概率      
        #将test_y,与预测结果进行比对,得到预测的准确率.
        accuracy = metrics.accuracy_score(test_y, predict)
        print ('accuracy: %.2f%%' % (100 * accuracy))
        return predict


if __name__ == '__main__':
    path = r"/Users/neo/Desktop/ORL/*/*.pgm"
    train_x, train_y, test_x, test_y = LoadImage(path)
    get_predict(train_x,train_y,test_x,test_y)
#by neo
