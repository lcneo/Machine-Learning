from glob import glob
from PIL import Image
import os
import scipy.misc as sm
import numpy as np
from sklearn import metrics
import  matplotlib.pyplot as plt
import pywt

path = r"C:\Users\neo\Desktop\ORL\*\*.pgm"
im=Image.open(path).convert('L')
im=im.resize((133,189))
im=np.array(im)

def readim(path):
    im = Image.open(path)
    name=os.path.basename(os.path.dirname(path))
    return np.array(im),name

def wave(im):
    [CA,CC]=pywt.dwt2(im,'haar')
    CH,CV,CD=CC
    
    
def sampling(im):
    mean=im.mean()
    var=im.var()
    m1=np.abs(im)
    m1=m1.sum()
    m2=(im*im).sum()
    return mean,var,m1,m2
