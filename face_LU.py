from PIL import Image;
from sklearn.decomposition import PCA  
import numpy as np;
import pandas as pd;
from time import time;
import matplotlib
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn import neighbors 
###  每一个图像最好算一下时间
#读入图像 并且转换为矩阵
def imgshow(filename):
    img =Image.open(filename);
    #img.show();
    #cdata =img.convert('L');这是不是灰度矩阵的意思？
    width,height = img.size
    data = img.getdata();
    data = np.array(data);
    new_data = np.reshape(data,(1,width*height));
    return new_data, width, height;
    
#我希望把他加载成3维的数字
def load_data(Person, startnumber,endnumber):
    t0 = time();
    y =[];
    print(">>>loading data");
    #matrix= np.array
    for i in range(1,Person+1):
        for j in range(startnumber, endnumber+1):
            data,width, height=imgshow(r"C:\Users\neo\Desktop\ORL\s%d\%d.pgm"%(i,j));
            if( i==1 and j ==startnumber):
                matrix = data;
            else:
                matrix = np.append(matrix, data);
            #print(data);
            #print();
            y.append("s"+str(i));
    matrix = matrix.reshape(((Person*(endnumber-startnumber+1),width*height)))
    y = np.array(y);
    print(">>>在读入图像的时间为%0.3f"%(time()-t0));
    return matrix,y;
"""
无监督的计算PCA（特征脸）
接下来测试一下PCA的保留特征 
我的理解就是PCA就是降低维数来的
因为这样在训练比较少量的数据的时候方便输入，减少时间的损耗
参照了sklearn中的svm的例子，n_components的特征数为150
pca降低维数是对所有样本整体进行降低维数的
"""
def fun_pca(x):
    t1 = time();
    n_components = 50;
    pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(x);
    #print(pca.explained_variance_ratio_);
    X_train_pca = pca.transform(x);
    #print(X_train_pca);
    #print(X_train_pca.shape)
    width, height = X_train_pca.shape;
    print(">>>使用PCA特征提取的时间为%0.3f"%(time()-t1));
    return X_train_pca,width, height;

def drawimg(X_train_pca):   
    yy = [];
    tt = [];
    yy = np.array(yy);
    for i in range(1,11):
        tt = [i]*5;
        yy =np.append(yy, tt);
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X_train_pca[:, 0], X_train_pca[:,2],yy,yy);
    plt.show()
    
    
#数据预处理       
Person = 20; #读取前person人
train_number = 7;  #person的前number张图片
test_number = 10;
x_train,y_train= load_data(Person,1, train_number);
x_test,y_test= load_data(Person, train_number+1, test_number);

#pca操作
X= np.vstack((x_train,x_test));
X_all,width, height = fun_pca(X);  
indexloc =train_number*Person;
X_train = X_all[:indexloc,:];
X_test  = X_all[indexloc:,:];
#现在是没有做归一化的情况，继续knn算法
#现在的最大最小值在区间【-5,5】之间

knn = neighbors.KNeighborsClassifier()  
#训练数据集  
knn.fit(X_train, y_train)  
#预测  
count =0;
samplecount = (test_number-train_number)*Person;
for i in range(samplecount):
    predict = knn.predict([X_test[i]]);  
    print (predict, y_test[i],end=" ")
    if(str(predict[0]) == str(y_test[i])):
        count = count+1;
        print("YES");
    else:
        print("No");
print (count/samplecount)  
