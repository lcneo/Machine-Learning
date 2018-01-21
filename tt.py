import numpy as np
import pandas as pd
import scineo as sn
mat = np.load("DataSet/No4.npy")
sn.hog_mat(mat)
# mat[0] = mat[0].reshape(-1,128*128)
# mat[2] = mat[2].reshape(-1,128*128)
# train_x,train_y,test_x,test_y = mat
# sn.predict_knn(train_x,train_y,test_x,test_y,show = True)
#aa = sn.hog(mat[0])
#print(aa.shape)