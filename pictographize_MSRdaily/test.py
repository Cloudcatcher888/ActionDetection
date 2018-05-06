from sklearn import svm
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import matplotlib.pyplot as plt
from fv_py3 import *

a = np.array([0,10,20,30,40,45,49,52,60])
b=np.array([82,87,93,94,94,95,96,97,96])
plt.grid(True, linestyle = "--", color = "black", linewidth = "1")
plt.plot(a,b,marker='o', mec='r', mfc='w')
plt.xlim(0,60)
plt.title('scaled trajectory length\'s influence' )
plt.xlabel('scaled trajectory length')
plt.ylabel('accuracy')
plt.show()