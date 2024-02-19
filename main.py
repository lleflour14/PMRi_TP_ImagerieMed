#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 12:28:50 2021

@author: chaari
"""
#from mat4py import loadmat
import h5py
import scipy.io
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from numpy import linalg as LA
from math import log


###########################################
#        sload data for simulation        #
###########################################
loaded = scipy.io.loadmat('reference.mat')
ref = loaded['im']
loaded = scipy.io.loadmat('sens.mat')
S = loaded['s']
###########################################
###########################################



###########################################
#    simulate paralle MRI acquisition     #
###########################################
sigma = 14
R = 4
reduced_FoV = pMRI_simulator(S,ref,sigma,R)
###########################################
###########################################



###########################################
#  reconstruct full field of view image   #
###########################################
[L,C,Nc] = S.shape
psi = sigma * np.eye(Nc)

reconstructed = reconstruct(reduced_FoV,S,psi,0)

###########################################
###########################################

img = plt.figure()

''''
# 8 antennes donc 8 images à afficher
for i in range(8):
    img.add_subplot(5, 2, i+1)
    plt.imshow(reduced_FoV[:,:,i])

# image de référence
img.add_subplot(5, 2, 9)
plt.imshow(ref)

#image reconstruite
img.add_subplot(5, 2, 10)
plt.imshow(reconstructed)

print(SignalToNoiseRatio(ref,reconstructed))
'''

lamb=1
for i in range(10):
   img.add_subplot(5, 2, i+1)
   lamb = lamb*(i+1)
   regularisation = reconstruct(reduced_FoV,S,psi,lamb)
   print(SignalToNoiseRatio(ref,regularisation))
   plt.imshow(regularisation)



plt.show()
