from glob import glob
from PIL import Image
import os
from numpy import *
import pandas as pd
from sklearn.decomposition import PCA 
from time import time;
import matplotlib
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn import neighbors 
import scipy.misc as sm
from tqdm import tqdm
import numpy as np


#读取图片，返回一个一维数组
def readim(path):
    im = sm.imread(path)
    name = os.path.basename(os.path.dirname(path))
    return im.flatten(),name
#读取ROL文件夹中所有文件
def LoadImage(path):
    x_train, y_train, x_test, y_test = [],[],[],[]
    tarinlist = ['1.pgm','2.pgm','3.pgm','4.pgm','5.pgm']
    print(">>>Loading ROL Data")
    impath = glob(path)
    for i in impath:
        im,name = readim(i)
        if os.path.basename(i) in tarinlist:
            x_train.append(im)
            y_train.append(name)
        else:
            x_test.append(im)
            y_test.append(name)
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

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


if __name__ == '__main__':
    path = r"C:\Users\neo\Desktop\ORL\*\*.pgm"
    x_train,y_train,x_test,y_test = LoadImage(path)
    #im = LoadImage(path)
    #a = fun_pca(im)
