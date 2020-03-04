#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 16:03:24 2020

@author: ojgurney-champion
"""
class hyperparams:
    optim='sgdr' #'sgd'; 'sgdr'; 'adagrad' adam
    lr = 0.5 # adam needs order of 0.005; others order of 0.05? sgdr can do 0.5
    run_net='sig_con'
    patience=10