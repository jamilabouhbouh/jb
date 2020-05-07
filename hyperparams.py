#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 16:03:24 2020

@author: ojgurney-champion
"""
class hyperparams:
    optim='adam' #these are the optimisers implementd. Choices are: 'sgd'; 'sgdr'; 'adagrad' adam
    lr = 0.001 # this is the learning rate. adam needs order of 0.005; others order of 0.05? sgdr can do 0.5
    run_net='split' #these are networks I implemented. One can chose abs_con, hwich used the absolute to constrain parameters. Sig_con which used the sigmoid to constrain parameters, free, which has no parameter constraints, split; which has seperate networks per parameter and tiny, which is a smaller network
    patience=10 # this is the number of epochs without improvement that the network waits untill determining it found its optimum
    fixS0=False # indicates whether to fix S0 in the least squares fit.
    batch_size=128