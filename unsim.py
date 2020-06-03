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
b=np.array([0, 5, 10, 20, 30, 40, 60, 150, 300, 500, 700])

matlsq = np.zeros([len(SNR), 3, 3])
matNN = np.zeros([len(SNR), 3, 3])
a = 0

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