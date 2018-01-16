from sklearn import metrics
from skimage.feature import hog
import numpy as np
from sklearn.svm import SVC


def svm(train_x, train_y):
    model = SVC(kernel='rbf', probability=True)
    model.fit(train_x, train_y)
    return model

def get_svm_predict(train_x,train_y,test_x,test_y):
    model = sum(train_x,train_y)
    predict = model.predict(test_x)
    accuracy = metrics.accuracy_score(test_y, predict)
    print("识别率为%"%(accuracy))

mn = np.load(r"C:\Users\Administrator\Desktop\No4.npy")
image = mn[0][2]
fd, hog_image = hog(image, orientations=8, pixels_per_cell=(4, 4), 
cells_per_block=(1, 1),transform_sqrt=None, block_norm = 'L2-Hys',visualise=True) 
