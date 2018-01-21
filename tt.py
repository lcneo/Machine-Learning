import numpy as np
import pandas as pd
import scineo as sn
mat = np.load("DataSet/No4.npy")
aa = sn.hog(mat[0])
print(aa.shape)