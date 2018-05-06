from fv_py3 import *
import numpy as np

a = np.array([[2,1,3],[0,1,1]])
means, covs, weights = generate_gmm(gmm_folder='D:\MSRAction3D\PycharmProjects\Fisher-Vector-master\\',descriptors=a, N=2)
fv = fisher_vector(samples=a, means=means, covs=covs, w=weights)
a = a