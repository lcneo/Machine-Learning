# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 11:44:38 2017

@author: pluto
"""

from PIL import Image;
from sklearn.decomposition import PCA  
import numpy as np;
import pandas as pd;
from time import time;
import matplotlib;
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn import neighbors 
import random;
import glob;



"""
 @method load_img
 @discussion 读取一张图片拉直成一行数据，并且返回新数据，原有长宽
 @param 文件路径
 @result 拉直后数据， 长， 宽
"""
def load_img(filename):
    img =Image.open(filename);
    #img.show(); 测试一下bmp格式的图片
    #cdata =img.convert('L'); 变为灰度矩阵
    width,height = img.size
    data = img.getdata();
    data = np.array(data);
    new_data = np.reshape(data,(1,width*height));
    return new_data, width, height;

"""
 @method load_Cottondata
 @discussion 遍历文件夹里面所有的图片信息（这个适用于已经分好类的文件夹，
 大小不确定）
 @param 文件夹路径，文件夹内再次含有的文件夹
 @result X,Y,具体的样本数量
"""
def load_Cottondata(filefolder, filepath):
    t0 = time();
    y =[];
    sum  =0;
    print(">>>loading data");
    matrix = np.array([]);
    for i in filefolder:
        filename = glob.glob(filepath+r"\%s\*.bmp"%(i));
        for j in filename:
            data,width, height=load_img(j);
            matrix = np.append(matrix, data);
            y.append(str(filefolder.index(i)));
        sum = sum + len(filename);
    matrix = matrix.reshape(((sum,width*height)))
    y = np.array(y);
    print(">>>在读入图像的时间为%0.3f"%(time()-t0));
    return matrix,y,sum;

  
"""
 @method load_Facedata
 @discussion 遍历文件夹里面所有的图片信息（这个适用于已经分好类的文件夹，
 并且大小确定的文件夹）,实际中每次读取数据写的读取数据的方法都会有些不相同。
 @param 文件夹数量(人的数量), 开始的图片标号， 结束的图片标号 
 @result X,Y
"""
def load_Facedata(Person, startnumber,endnumber):
    t0 = time();
    y =[];
    print(">>>loading data");
    matrix = np.array([]);
    for i in range(1,Person+1):
        for j in range(startnumber, endnumber+1):
            data,width, height=imgshow(r"C:\Users\pluto\Desktop\plan\机器学习\十二月计划\人脸识别\ORL人脸库\ORL人脸库\ORL\s%d\%d.pgm"%(i,j));
            matrix = np.append(matrix, data);

            y.append("s"+str(i));
    matrix = matrix.reshape(((Person*(endnumber-startnumber+1),width*height)))
    y = np.array(y);
    print(">>>在读入图像的时间为%0.3f"%(time()-t0));
    return matrix,y;


"""
其实我写这个函数完全是为了测试一下PCA特征提取的用时，实际用到PCA的时候，
直接调用model就可以了,那我直接介绍PCA方法了。

我理解的PCA就是特征提取，降低维数。
eg:
我们见到一个人的时候 (相当于)--> 计算机读取了一张照片
可是我们在判断一个人的时候，我们并不是以他整个人的全部来进行判断的
我们在生活中找到技巧了，PCA就是相当于是计算机通过数据进行分析找到技巧
我们知道大部分的时候 一个人的眉毛是黑的--> 一个人的头发是黑的
我们知道大部分的时候 判断一个人不能通过判断性别来确定人
所以我们真正见到人的时候，我们可能只需要从他的头发入手，我们可以判断他的眉毛
这就是计算机中所说的进行相关性分析提取特征 --> 计算机也可以用大的矩阵中通过
分析行与行之间的线性相关或者是协方差。找到我们(计算机)认为不重要的信息，并且
抛弃。
PCA的好处就是(我认为的):降低维数-->减少时间， 去除噪声-->减少冗余

 @method PCA
 @discussion 建立一个PCA模型
 @param n_components(特征数);
 whiten(白化 是否归一化的意思);
 svd_solver(奇异值分解SVD 可选值{‘auto’(后面三种情况下权衡), 
 ‘full’(全部遍历), ‘arpack’,‘randomized’(随机遍历，适用于数据量大的情况
 )};)
 @result 降低维数之后的矩阵 
"""
def fun_pca(n_components,x):
    t1 = time();
    pca = PCA(n_components=n_components, svd_solver='full',
          whiten=True).fit(x);
              #randomized
    #print(pca.explained_variance_ratio_);
    #降低维数之后的主成分的方差值， 越大说明越重要
    X_train_pca = pca.transform(x);
    width, height = X_train_pca.shape;
    print(">>>使用PCA特征提取的时间为%0.3f"%(time()-t1));
    return X_train_pca,width, height;

"""
这个函数没什么用，我只是用来看看PCA之后不同类别的差异大不大然后画了一张图
"""
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

"""
其实手写knn中的欧式距离也挺简单的不要用循环。
直接两个矩阵之间相减然后平方生成一个矩阵，然后每一行相加，
新生成一个一列数据找出最小的值。然后找到对应的Y值就可以了。
python和matlab一样是矩阵运算，写for循环。
然而我就for了很多此次…时间复杂度也挺高的。

还有就是预处理时候，我喜欢把训练的X,Y分开。
其实在数据量小的时候要用到交叉验证(可以理解为循环找样本)的时候就挺不方便的，
其实就一行代码的差距，就是如果用我的方法就必须再增加一张大表然后shaffle
"""

#以下备注的代码是用于人脸识别

#数据预处理       
#Person = 40; #读取前person人
#train_number = 7;  #person的前number张图片
#test_number = 10;
#x_train,y_train= load_Facedata(Person,1, train_number);
#x_test,y_test= load_Facedata(Person, train_number+1, test_number);
filefolder = ['经向疵点', '块状疵点' ,'纬向疵点','正常样本'];
filepath2 = r"C:\Users\pluto\Desktop\plan\一月计划\机器学习\机器学习\棉纺织布\128-128梯度实验样本\No12\训练样本";
filepath1 = r"C:\Users\pluto\Desktop\plan\一月计划\机器学习\机器学习\棉纺织布\128-128梯度实验样本\No12\测试样本";

x_train, y_train, trainnum = load_Cottondata(filefolder,filepath1);
x_test, y_test, testnum = load_Cottondata(filefolder,filepath2);

#读出来之后变成大的矩阵  
#以下两行代码如果要测试数据量比较小的情况用 用shaffle进行变换
#此示例不写shaffle函数
#xall_train = np.column_stack((x_train,y_train));
#xall_test  = np.column_stack((x_test ,y_test));


#pca操作
x_all= np.vstack((x_train,x_test));

n_components = 0.56
imgvalue = np.array([]);

X_all,width, height = fun_pca(n_components,x_all);  
X_train = X_all[:trainnum,:];
X_test  = X_all[trainnum:,:];        
#现在是没有做归一化的情况，继续knn算法
#现在的最大最小值在区间【-5,5】之间
n_neighbors = 1;
knn = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors,
                                     algorithm="ball_tree")  
#训练数据集  
knn.fit(X_train, y_train)  
#预测
#分辨矩阵  
count =0;
checkmatrix = np.zeros((4,4));
for i in range(testnum):
    predict = knn.predict([X_test[i]]);  
    #print (predict[0], y_test[i][0],end=" ")
    if(str(predict[0]) == str(y_test[i][0])):
        count = count+1;
        checkmatrix[int(predict[0])][int(predict[0])] = checkmatrix[int(predict[0])][int(predict[0])]+1;
    else:
        checkmatrix[int(predict[0])][int(y_test[i][0])] = checkmatrix[int(predict[0])][int(y_test[i][0])]+1;
print("knn%f  pca%f"%( n_neighbors, n_components),end="  ");
print("准确率",end=" ");
accuracy = count/testnum
print(accuracy);
print("分辨矩阵")
print(checkmatrix)
imgvalue = np.append(imgvalue,accuracy);

"""
以下的代码时用来循环PCA和KNN 然后画图的代码
ximg =  np.arange(0.02,0.95,0.02);
imgs = imgvalue.reshape((47,4));
plt.ylim(0.0,1.0)
matplotlib
plt.xlabel("PCA");
plt.ylabel("accuracy");
plt.title('NO4_testsample')
plt.plot(ximg, imgs[:,0], color= "blue", label="knn=1")
plt.plot(ximg, imgs[:,1], color = "orange", label="knn=3")
plt.plot(ximg, imgs[:,2], color = "green", label="knn=5")
plt.plot(ximg, imgs[:,3], color = "red", label="knn=7")
plt.legend()
plt.show();
"""
