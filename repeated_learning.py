#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:43:30 2020

@author: ojgurney-champion
"""
import os
import nibabel as nib
import numpy as np
import deep
import torch
from fitting_algorithms import fit_least_squares_array, ivimN, fit_least_squares_S0, fit_segmented_array, goodness_of_fit
import matplotlib.pyplot as plt
from sys import platform
lr = 0.0005

for run_net in ['abs_con','sig_con','free']:
    for dummys in [0]:
        if dummys is 0:
            fixS0 = False
            constrained = True
        elif dummys is 1:
            fixS0 = False
            constrained = False
    
        dolsq=False
        load_lsq=True
    
        segmented=False
    
        load_nn=False
    
    
        testdata=False
    
        print('running with load LSQ as {load_lsq}, fix S0 as {fixS0}, segmented as {segmented}\n'.format(load_lsq=load_lsq,fixS0=fixS0, segmented=segmented))
        print('running with load_nn as {load_nn}, constrained as {constrained}\n'.format(load_nn=load_nn, constrained=constrained))
    
        if platform == 'win32':
            divi = 'L:/basic/divi/'
        else:
            divi = '/home/ojgurney-champion/lood_storage/divi/'
    
        fold1 = divi + 'Projects/mipa/Data/MIPA/nii'
        fold2 = divi + 'Projects/mipa/Data/REMP/REPRO/nii'
        aa1=[1,2,3,4,5]
        aa2=[6,9,11,12,14,15]
    
        bvalues=np.sort(np.unique([0, 50, 50, 50, 50, 50, 50, 50, 50, 50, 40, 40, 40, 40, 40, 40, 40, 40, 40, 30, 30, 30, 30, 30, 30, 30, 30, 30, 250, 250, 250, 250, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 10, 10, 10, 10, 10, 10, 10, 10, 10, 75, 75, 75, 75, 400, 400, 400, 400, 20, 20, 20, 20, 20, 20, 20, 20, 20, 150, 150, 150, 150, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        bvalues2=np.sort(np.unique([0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 20, 20, 20, 20, 20, 20, 20, 20, 20, 30, 30, 30, 30, 30, 30, 30, 30, 30, 40, 40, 40, 40, 40, 40, 40, 40, 40, 50, 50, 50, 50, 50, 50, 50, 50, 50, 75, 75, 75, 75, 100, 100, 100, 100, 150, 150, 150, 150, 250, 250, 250, 250, 400, 400, 400, 400, 0, 0, 0, 0, 0, 0, 100, 100, 100, 100, 100, 100, 100, 100, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600]))
        bb=[1, 2,3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16]
        dattype='remz'
    
        if dattype is 'remz':
            bvalues=np.array([0, 50, 50, 50, 50, 50, 50, 50, 50, 50, 40, 40, 40, 40, 40, 40, 40, 40, 40, 30, 30, 30, 30, 30, 30, 30, 30, 30, 250, 250, 250, 250, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 10, 10, 10, 10, 10, 10, 10, 10, 10, 75, 75, 75, 75, 400, 400, 400, 400, 20, 20, 20, 20, 20, 20, 20, 20, 20, 150, 150, 150, 150, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            bvalues2=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 20, 20, 20, 20, 20, 20, 20, 20, 20, 30, 30, 30, 30, 30, 30, 30, 30, 30, 40, 40, 40, 40, 40, 40, 40, 40, 40, 50, 50, 50, 50, 50, 50, 50, 50, 50, 75, 75, 75, 75, 100, 100, 100, 100, 150, 150, 150, 150, 250, 250, 250, 250, 400, 400, 400, 400, 0, 0, 0, 0, 0, 0, 100, 100, 100, 100, 100, 100, 100, 100, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600])
    
        MR1='MRI1_reg.nii'
    
        MR2='MRI2_reg.nii'
        b_values=[]
        datatot=np.zeros([0,len(bvalues)])
        datatot2=np.zeros([0,len(bvalues2)])
        for fold in aa1:
            for ss in [1, 2]:
                if os.path.isfile('{folder}/CR{fold:02d}/MRI{ss}_{dat}.nii.gz'.format(folder=fold1,fold=fold,ss=ss, dat=dattype)):
                    data=nib.load('{folder}/CR{fold:02d}/MRI{ss}_{dat}.nii.gz'.format(folder=fold1,fold=fold,ss=ss, dat=dattype))
                    datas=data.get_data()
                    sx, sy, sz, n_b_values = datas.shape
                    X_dw_all = np.reshape(datas, (sx * sy * sz, n_b_values))
                    valid_id = np.sum(X_dw_all == 0, axis=1) == 0
                    datatot2 = np.append(datatot2, X_dw_all[valid_id,:],axis=0)
    
        for fold in aa2:
            for ss in [1, 2]:
                if os.path.isfile('{folder}/CR{fold:02d}/MRI{ss}_{dat}.nii.gz'.format(folder=fold1,fold=fold,ss=ss, dat=dattype)):
                    data=nib.load('{folder}/CR{fold:02d}/MRI{ss}_{dat}.nii.gz'.format(folder=fold1,fold=fold,ss=ss, dat=dattype))
                    datas=data.get_data()
                    sx, sy, sz, n_b_values = datas.shape
                    X_dw_all = np.reshape(datas, (sx * sy * sz, n_b_values))
                    valid_id = np.sum(X_dw_all == 0, axis=1) == 0
                    datatot=np.append(datatot,X_dw_all[valid_id,:],axis=0)
                    if fold==aa2[0]:
                        datatot3=X_dw_all[valid_id,:]
    
        for fold in bb:
            for ss in [1, 2]:
                if os.path.isfile('{folder}/{fold}/{ss}/{dat}.nii.gz'.format(folder=fold2,fold=fold,ss=ss, dat=dattype)):
                    data=nib.load('{folder}/{fold}/{ss}/{dat}.nii.gz'.format(folder=fold2,fold=fold,ss=ss, dat=dattype))
                    datas=data.get_data()
                    sx, sy, sz, n_b_values = datas.shape
                    X_dw_all = np.reshape(datas, (sx * sy * sz, n_b_values))
                    valid_id = np.sum(X_dw_all == 0, axis=1) == 0
                    datatot=np.append(datatot,X_dw_all[valid_id,:],axis=0)
    
        for fold in bb:
            for ss in [1, 2]:
                if os.path.isfile('{folder}/{fold}/{ss}/{dat}_intra.nii.gz'.format(folder=fold2, fold=fold, ss=ss, dat=dattype)):
                    data = nib.load('{folder}/{fold}/{ss}/{dat}_intra.nii.gz'.format(folder=fold2, fold=fold, ss=ss, dat=dattype))
                    datas = data.get_data()
                    sx, sy, sz, n_b_values = datas.shape
                    X_dw_all = np.reshape(datas, (sx * sy * sz, n_b_values))
                    valid_id = np.sum(X_dw_all == 0, axis=1) == 0
                    datatot = np.append(datatot, X_dw_all[valid_id, :], axis=0)
        
        del valid_id, X_dw_all, datas, data
    
        index = np.argsort(bvalues)
        index2 = np.argsort(bvalues2)
    
        S0 = np.nanmean(datatot[:,bvalues==0],axis=1)
        datatot = datatot/S0[:,None]
        S0 = np.nanmean(datatot3[:,bvalues==0],axis=1)
        datatot3 = datatot3/S0[:,None]
        datmean=np.nanmean(datatot,axis=0)
        datmean2=np.nanmean(datatot2,axis=0)
    
        res = [i for i, val in enumerate(datatot!=datatot) if not val.any()]
        res2 = [i for i, val in enumerate(datatot2!=datatot2) if not val.any()]
        paramsNN=np.zeros([20,4,np.shape(datatot3)[0]])
        for qq in range(20):
            net = deep.learn_IVIM(datatot[res],bvalues,run_net=run_net,lr=lr)
            paramsNN[qq]=deep.infer_IVIM(datatot3, bvalues, net)
            del net
        # save NN results
        names = ['Dp_NN_{net}_rep'.format(net=run_net), 'D_NN_{net}_2'.format(net=run_net), 'f_NN_{net}_2'.format(net=run_net), 'S0_NN_{net}_2'.format(net=run_net)]
        multiple = [1000., 1000000., 10000., 1000.]
        fold = aa2[0]
        ss = 2
        data=nib.load('{folder}/CR{fold:02d}/MRI{ss}_{dat}.nii.gz'.format(folder=fold1,fold=fold,ss=ss, dat=dattype))
        datas=data.get_data()
        sx, sy, sz, n_b_values = datas.shape
        data.header.set_data_shape((sx,sy,sz,np.shape(paramsNN)[0]))
        X_dw_all = np.reshape(datas, (sx * sy * sz, n_b_values))
        valid_id = np.sum(X_dw_all == 0, axis=1) == 0
        imgtot=np.zeros([sx,sy,sz,np.shape(paramsNN)[0]])
        for k in range(len(names)):
            for rep in range(np.shape(paramsNN)[0]):
                img = np.zeros([sx * sy * sz])
                img[valid_id] = paramsNN[rep,k]  # [Ke, dt, ve, vp]                img[np.isnan(img)] = 0
                img[img < 0] = 0
                img[deep.isnan(img)] = 0
                img = np.reshape(img, [sx, sy, sz])
                imgtot[:,:,:,rep]=img
            nib.save(nib.Nifti1Image(imgtot * multiple[k], data.affine, data.header), '{folder}/CR{fold:02d}/MRI{ss}_{dat}_{name}.nii.gz'.format(folder=fold1,fold=fold,ss=ss, dat=dattype,name=names[k]))
