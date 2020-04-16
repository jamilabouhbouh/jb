#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:10:35 2020

@author: ojgurney-champion
"""
import numpy as np
import simulations as sim
from hyperparams import hyperparams as arg

SNR=[15,30,45]
b=np.array([0, 2, 5, 10, 15, 20, 35, 50, 75, 125, 200, 400, 600, 800])

matlsq = np.zeros([len(SNR), 3, 3])
matNN = np.zeros([len(SNR), 3, 3])
a = 0

for aa in SNR:
    print('\n simulation at SNR of {snr}\n'.format(snr=aa))
    matlsq[a, :, :],matNN[a, :, :] = sim.sim(aa, b, arg)
    a = a + 1
    print('\nresults from lsq:')
    print(matlsq)
    print('\nresults from NN:')
    print(matNN)
    