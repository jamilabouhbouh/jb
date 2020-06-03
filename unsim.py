#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:10:35 2020

@author: ojgurney-champion
"""
import numpy as np
import simulations as sim
from hyperparams import hyperparams as arg

SNR=[20]
lrs=[3, 1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]
b=np.array([0, 5, 10, 20, 30, 40, 60, 150, 300, 500, 700])

matlsq = np.zeros([len(SNR)*len(lrs), 3, 3])
matNN = np.zeros([len(SNR)*len(lrs), 3, 3])
a = 0
for arg.lr in lrs:
    for aa in SNR:
        print('\n simulation at SNR of {snr}\n'.format(snr=aa))
        matlsq[a, :, :],matNN[a, :, :], stability = sim.sim(aa, b, arg)
        a = a + 1
        print('\nresults from lsq:')
        print(matlsq)
        print('\nresults from NN:')
        print(matNN)
        if arg.repeats > 1:
            print('\nstability of NN:')
            print(stability)