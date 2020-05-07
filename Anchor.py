#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 09:01:49 2020

@author: ojgurney-champion
"""

import os
import nibabel as nib
import numpy as np
import deep
import torch
#from fitting_algorithms import fit_least_squares_array, ivimN, fit_least_squares_S0, fit_segmented_array, goodness_of_fit
#import matplotlib.pyplot as plt
from sys import platform
from hyperparams import hyperparams as arg
import IO as io
from fitting_algorithms import goodness_of_fit
import matplotlib.pyplot as plt

if platform == 'win32':
    divi = 'L:/basic/divi/'
else:
    divi = '/home/ojgurney-champion/lood_storage/divi/'
maskload=True
bval=np.array([0, 0, 0, 0, 700, 700, 700, 700, 1, 1, 1, 1, 5,  5,  5,  5,  100, 100, 100, 100, 300, 300, 300, 300, 10, 10, 10, 10, 0, 0, 0, 0, 20, 20, 20, 20, 500, 500, 500, 500, 50, 50, 50, 50, 40, 40, 40, 40, 30, 30, 30, 30, 150, 150, 150, 150, 75,  75,  75,  75,  0, 0, 0, 0, 600, 600, 600, 600, 200, 200, 200, 200, 400, 400, 400, 400, 2, 2, 2, 2])
home=divi+'Projects/anchor/analysis/'
#/home/ojgurney-champion/lood_storage/divi/Projects/anchor/analysis/ANCHOR_0001_V1/MR/00701_DWI-IVIM/00701_DWI-IVIM/oud/x20180928_113224WIPDWIIVIMs701a1007.bval

dirlist=io.main(home)

datatot=np.zeros([0,len(bval)])
           
for fold in dirlist:
    print('load file: {file}\n'.format(file=fold))
    data=nib.load(fold)
    datas=data.get_fdata()
    sx, sy, sz, n_b_values = datas.shape
    if n_b_values==len(bval):
        if maskload:
            try:
                dirs,file=os.path.split(fold)
                data=nib.load('{folder}/{name}'.format(folder=dirs,name='Liver.nii.gz'))
                mask=data.get_fdata()>0.5
            except:
                print('liver.nii.gz niet found')
                mdata=np.average(datas[:,:,:,bval==0],axis=3)
                s0=np.average(mdata[mdata>0])
                mask=mdata>s0/2
        else:
            mdata=np.average(datas[:,:,:,bval==0],axis=3)
            s0=np.average(mdata[mdata>0])
            mask=mdata>s0/2
            try:
                dirs,file=os.path.split(fold)
                data=nib.load('{folder}/{name}'.format(folder=dirs,name='Liver.nii.gz'))
                mask=(data.get_fdata()+mask)>0.5
            except:
                print('no mask for this patient')    
        datas[~np.repeat(np.expand_dims(mask,3),np.shape(datas)[3],axis=3)]=0
        X_dw_all = np.reshape(datas, (sx * sy * sz, n_b_values))
        valid_id = np.sum(X_dw_all == 0, axis=1) == 0
        datatot = np.append(datatot, X_dw_all[valid_id,:],axis=0)
del valid_id, X_dw_all, datas, data
bval=np.delete(bval,np.s_[3::4],0)
datatot=np.array(np.delete(datatot,np.s_[3::4],1))

S0 = np.nanmean(datatot[:,bval==0],axis=1)
datatot = datatot/S0[:,None]

datmean=np.nanmean(datatot,axis=0)

res = [i for i, val in enumerate(datatot!=datatot) if not val.any()]
net = deep.learn_IVIM(datatot[res],bval,arg)
torch.save(net, 'network_{nn}.pt'.format(nn=arg.run_net))
paramsNN=deep.infer_IVIM(datatot, bval, net)
del net
gofNN=goodness_of_fit(bval,paramsNN[0],paramsNN[1],paramsNN[2],paramsNN[3],datatot)
names = ['geof_NN_{nn}.nii'.format(nn=arg.run_net), 'Dp_NN_{nn}.nii'.format(nn=arg.run_net), 'D_NN_{nn}.nii'.format(nn=arg.run_net), 'f_NN_{nn}.nii'.format(nn=arg.run_net), 'S0_NN_{nn}.nii'.format(nn=arg.run_net)]

tot=0
bval=np.array([0, 0, 0, 0, 700, 700, 700, 700, 1, 1, 1, 1, 5,  5,  5,  5,  100, 100, 100, 100, 300, 300, 300, 300, 10, 10, 10, 10, 0, 0, 0, 0, 20, 20, 20, 20, 500, 500, 500, 500, 50, 50, 50, 50, 40, 40, 40, 40, 30, 30, 30, 30, 150, 150, 150, 150, 75,  75,  75,  75,  0, 0, 0, 0, 600, 600, 600, 600, 200, 200, 200, 200, 400, 400, 400, 400, 2, 2, 2, 2])

for fold in dirlist:
    print('saving file: {file}\n'.format(file=fold))
    data=nib.load(fold)
    datas=data.get_fdata()
    sx, sy, sz, n_b_values = datas.shape
    if n_b_values==len(bval):
        if maskload:
            try:
                dirs,file=os.path.split(fold)
                data=nib.load('{folder}/{name}'.format(folder=dirs,name='Liver.nii.gz'))
                mask=data.get_fdata()>0.5
            except:
                print('liver.nii.gz niet found')
                mdata=np.average(datas[:,:,:,bval==0],axis=3)
                s0=np.average(mdata[mdata>0])
                mask=mdata>s0/2
        else:
            mdata=np.average(datas[:,:,:,bval==0],axis=3)
            s0=np.average(mdata[mdata>0])
            mask=mdata>s0/2
            try:
                dirs,file=os.path.split(fold)
                data=nib.load('{folder}/{name}'.format(folder=dirs,name='Liver.nii.gz'))
                mask=(data.get_fdata()+mask)>0.5
            except:
                print('no mask for this patient')    
        datas[~np.repeat(np.expand_dims(mask,3),np.shape(datas)[3],axis=3)]=0
        data.header.set_data_shape((sx,sy,sz))
        X_dw_all = np.reshape(datas, (sx * sy * sz, n_b_values))
        valid_id = np.sum(X_dw_all == 0, axis=1) == 0
        for k in range(len(names)):
            img = np.zeros([sx * sy * sz])
            if k == 0:
                img[valid_id] = gofNN[tot:(tot + sum(valid_id))]  # [Ke, dt, ve, vp]
            else:
                img[valid_id] = paramsNN[k-1][tot:(tot+sum(valid_id))]  # [Ke, dt, ve, vp]
            img[np.isnan(img)] = 0
            img[img < 0] = 0
            img = np.reshape(img, [sx, sy, sz])
            dirs,file=os.path.split(fold)
            nib.save(nib.Nifti1Image(img, data.affine, data.header), '{folder}/{name}.gz'.format(folder=dirs,name=names[k]))
        tot = tot+sum(valid_id)
dummy=np.array(paramsNN)
plt.plot(dummy[1,:10000],dummy[2,:10000],'rx')
plt.plot(dummy[1,:10000],dummy[3,:10000],'rx')
plt.plot(dummy[2,:10000],dummy[3,:10000],'rx')