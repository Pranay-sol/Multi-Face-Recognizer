#How CNN Works,
#Input matrix (star) Kernel Martix = output Matrix (Kernal is usually of a smaller shape than the input)
# (star) => Cross-correlation, not multiply
# conv(I,K) = I*K = I (star) rotate180(K), * => convulution

#Types of cross-correlation,
#1. Valid - (star)
# Normal cross - correlation where K does not exceed bounds of I.
#Shape of output is [Y = I - K + 1]
#2. Full - (star_full)
# cross - correlation where K exceed bounds of I.
#Shape of o/p is [Y = I + K -1]

#Convulution Layer - Forward

#Input Layer
#Series of matrix used as input

#Kernal Layer
# (can be) Multiple Series of Kernal Matrix & Biases
# Input Layer is convultued with all Kernals.

#Output Layer
# The number of Output matrix is same as # of kernals.

#Forward Propagation Equation
# Y_i = B_i + Sum(X_j * K_ij), i = 1,2,...d or Y = B + K.|*X (dot product if 1-dimensional)

#Backward Propagation Equation
#In order to update kernals & biases, need to calculate gradients
# dE/dY_i, which are 3 derivations - dE/dB_i, dE/dK_ij & dE/dX_j
# E - Error of neural network
#dE/dK_ij = X_j * dE/dY_i
#dE/dB_i = dE/dY_i
#dE/dX_j = sum(dE/dY_i) *_full K_ij , i = 1,2,3...n

#Reshape Layer
# to convert martix to 1D vector for final prediction or anything in between

#Binary Cross Entrophy Loss

#Sigmoid Activation

#MNIST

import pandas as pd
import numpy as np
a = pd.read_pickle(r"C:\Users\solpr\Desktop\Projects\CNN_PAML_SEM_4\face_data.pkl")
with np.printoptions(threshold=np.inf):
    b = np.array(a["Id"].to_list())
    print(b,np.size(b))