import os
import nibabel as nib
import numpy as np
import deep
import torch
from fitting_algorithms import fit_least_squares_array, ivimN, fit_least_squares_S0, fit_segmented_array, goodness_of_fit
import matplotlib.pyplot as plt
from sys import platform

for dummys in [1]:
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
                #X_dw_temp=X_dw_all[valid_id,:]
                #S0 = np.nanmean(X_dw_temp[:, bvalues == 0], axis=1)
                #index = np.argsort(bvalues)
                #X_dw_temp = X_dw_temp[:,index] / S0[:, None]
                #print(np.nanmean(X_dw_temp, axis=0))
                #print(fold)
                #if max(np.nanmean(X_dw_temp,axis=0)) > 20:
                #    print('oeps')
    for fold in bb:
        for ss in [1, 2]:
            if os.path.isfile('{folder}/{fold}/{ss}/{dat}.nii.gz'.format(folder=fold2,fold=fold,ss=ss, dat=dattype)):
                data=nib.load('{folder}/{fold}/{ss}/{dat}.nii.gz'.format(folder=fold2,fold=fold,ss=ss, dat=dattype))
                datas=data.get_data()
                sx, sy, sz, n_b_values = datas.shape
                X_dw_all = np.reshape(datas, (sx * sy * sz, n_b_values))
                valid_id = np.sum(X_dw_all == 0, axis=1) == 0
                datatot=np.append(datatot,X_dw_all[valid_id,:],axis=0)
                #X_dw_temp=X_dw_all[valid_id,:]
                #S0 = np.nanmean(X_dw_temp[:, bvalues == 0], axis=1)
                #index = np.argsort(bvalues)
                #X_dw_temp = X_dw_temp[:,index] / S0[:, None]
                #print(np.nanmean(X_dw_temp, axis=0))
                #print(fold)
                #if max(np.nanmean(X_dw_temp,axis=0)) > 20:
                #    print('oeps')


    for fold in bb:
        for ss in [1, 2]:
            if os.path.isfile('{folder}/{fold}/{ss}/{dat}_intra.nii.gz'.format(folder=fold2, fold=fold, ss=ss, dat=dattype)):
                data = nib.load('{folder}/{fold}/{ss}/{dat}_intra.nii.gz'.format(folder=fold2, fold=fold, ss=ss, dat=dattype))
                datas = data.get_data()
                sx, sy, sz, n_b_values = datas.shape
                X_dw_all = np.reshape(datas, (sx * sy * sz, n_b_values))
                valid_id = np.sum(X_dw_all == 0, axis=1) == 0
                datatot = np.append(datatot, X_dw_all[valid_id, :], axis=0)
                #X_dw_temp=X_dw_all[valid_id,:]
                #S0 = np.nanmean(X_dw_temp[:, bvalues == 0], axis=1)
                #index = np.argsort(bvalues)
                #X_dw_temp = X_dw_temp[:,index] / S0[:, None]
                #print(np.nanmean(X_dw_temp, axis=0))
                #print(fold)
                #if max(np.nanmean(X_dw_temp,axis=0)) > 20:
                #    print('oeps')
    del valid_id, X_dw_all, datas, data

    index = np.argsort(bvalues)
    index2 = np.argsort(bvalues2)

    S0 = np.nanmean(datatot[:,bvalues==0],axis=1)
    datatot = datatot/S0[:,None]
    S0 = np.nanmean(datatot2[:,bvalues2==0],axis=1)
    datatot2 = datatot2/S0[:,None]
    datmean=np.nanmean(datatot,axis=0)
    datmean2=np.nanmean(datatot2,axis=0)

    if testdata:
        paramslsq = fit_least_squares_S0(bvalues, datmean, S0_output=True)
        paramslsq2 = fit_least_squares_S0(bvalues2, datmean2, S0_output=True)

        plt.plot(datmean[index])
        plt.plot(ivimN(bvalues[index], paramslsq[0]*10, paramslsq[1]*1000, paramslsq[2]*10, paramslsq[3]))

        plt.plot(datmean2[index2])
        plt.plot(ivimN(bvalues[index], paramslsq2[0]*10, paramslsq2[1]*1000, paramslsq2[2]*10, paramslsq2[3]))

        net = deep.learn_IVIM(np.transpose(np.repeat(np.expand_dims(datmean,1),1000,axis=1)), bvalues, run_net='sig_con')
        net2 = deep.learn_IVIM(np.transpose(np.repeat(np.expand_dims(datmean2,1),1000,axis=1)), bvalues2, run_net='sig_con')

        paramsNN=deep.infer_IVIM(np.expand_dims(datmean,0), bvalues, net)
        paramsNN2=deep.infer_IVIM(np.expand_dims(datmean2,0), bvalues2, net2)

        plt.plot(datmean[index])
        plt.plot(ivimN(bvalues[index], paramsNN[0][0]*10, paramsNN[1][0]*1000, paramsNN[2][0]*10, paramsNN[3][0]))

        plt.plot(datmean2[index2])
        plt.plot(ivimN(bvalues[index], paramsNN2[0][0]*10, paramsNN2[1][0]*1000, paramsNN2[2][0]*10, paramsNN2[3][0]))


        net = deep.learn_IVIM(np.transpose(np.repeat(np.expand_dims(datmean,1),1000,axis=1)), bvalues, run_net ='free')
        net2 = deep.learn_IVIM(np.transpose(np.repeat(np.expand_dims(datmean2,1),1000,axis=1)), bvalues2, run_net ='free')

        paramsNN=deep.infer_IVIM(np.expand_dims(datmean,0), bvalues, net)
        paramsNN2=deep.infer_IVIM(np.expand_dims(datmean2,0), bvalues2, net2)

        plt.plot(datmean[index])
        plt.plot(ivimN(bvalues[index], paramsNN[0][0]*10, paramsNN[1][0]*1000, paramsNN[2][0]*10, paramsNN[3][0]))

        plt.plot(datmean2[index2])
        plt.plot(ivimN(bvalues[index], paramsNN2[0][0]*10, paramsNN2[1][0]*1000, paramsNN2[2][0]*10, paramsNN2[3][0]))

    print('least squares fitting\n')
    if dolsq:
        if not load_lsq:
            if segmented:
                print('segmented\n')
                paramslsq=fit_segmented_array(bvalues,datatot)
                paramslsq2=fit_segmented_array(bvalues2,datatot2)
                np.savez(os.path.join(fold1, 'seg.npz'), paramslsq=paramslsq, paramslsq2=paramslsq2)
            else:
                print('normal fitting\n')
                paramslsq = fit_least_squares_array(bvalues, datatot,fixS0=fixS0)
                paramslsq2 = fit_least_squares_array(bvalues2, datatot2,fixS0=fixS0)
                if fixS0:
                    np.savez(os.path.join(fold1, 'fix_s0.npz'), paramslsq=paramslsq, paramslsq2=paramslsq2)
                else:
                    np.savez(os.path.join(fold1, 'res.npz'), paramslsq=paramslsq, paramslsq2=paramslsq2)
        else:
            print('loading fit\n')
            if segmented:
                loads=np.load(os.path.join(fold1, 'seg.npz'))
            else:
                if fixS0:
                    loads = np.load(os.path.join(fold1, 'fix_s0.npz'))
                else:
                    loads = np.load(os.path.join(fold1, 'res.npz'))

            paramslsq=loads['paramslsq']
            paramslsq2=loads['paramslsq2']
            del loads


    if not load_nn:
        res = [i for i, val in enumerate(datatot!=datatot) if not val.any()]
        res2 = [i for i, val in enumerate(datatot2!=datatot2) if not val.any()]

        if constrained:
            net = deep.learn_IVIM(datatot[res],bvalues,run_net='sig_con')
            net2 = deep.learn_IVIM(datatot2[res2],bvalues2,run_net='sig_con')
            torch.save(net, 'network_con.pt')
            torch.save(net2, 'network2_con.pt')
        else:
            net = deep.learn_IVIM(datatot[res],bvalues, run_net='free')
            net2 = deep.learn_IVIM(datatot2[res2],bvalues2, run_net='free')
            torch.save(net, 'network.pt')
            torch.save(net2, 'network2.pt')

    else:
        if constrained:
            net=torch.load('network_con.pt')
            net2=torch.load('network2_con.pt')
        else:
            net=torch.load('network.pt')
            net2=torch.load('network2.pt')

    paramsNN=deep.infer_IVIM(datatot, bvalues, net)
    paramsNN2=deep.infer_IVIM(datatot2, bvalues2, net2)
    del net, net2

    if not fixS0:
        gofNN=goodness_of_fit(bvalues,paramsNN[0],paramsNN[1],paramsNN[2],paramsNN[3],datatot)
        gof2NN=goodness_of_fit(bvalues2,paramsNN2[0],paramsNN2[1],paramsNN2[2],paramsNN2[3],datatot2)
    else:
        gofNN = goodness_of_fit(bvalues, paramsNN[0], paramsNN[1], paramsNN[2], np.ones(len(paramsNN[2])), datatot)
        gof2NN = goodness_of_fit(bvalues2, paramsNN2[0], paramsNN2[1], paramsNN2[2], np.ones(len(paramsNN2[2])), datatot2)

    if dolsq:
        if not fixS0:
            goflsq=goodness_of_fit(bvalues,paramslsq[0],paramslsq[1],paramslsq[2],paramslsq[3],datatot)
            gof2lsq=goodness_of_fit(bvalues2,paramslsq2[0],paramslsq2[1],paramslsq2[2],paramslsq2[3],datatot2)
        else:
            goflsq = goodness_of_fit(bvalues, paramslsq[0], paramslsq[1], paramslsq[2], np.ones(len(paramslsq[2])), datatot)
            gof2lsq = goodness_of_fit(bvalues2, paramslsq2[0], paramslsq2[1], paramslsq2[2], np.ones(len(paramslsq2[2])), datatot2)

    del datatot, datatot2

    # save NN results
    if constrained:
        names = ['geof_NN_con.nii', 'Dp_NN_con.nii', 'D_NN_con.nii', 'f_NN_con.nii', 'S0_NN_con.nii']
    else:
        names = ['geof_NN.nii','Dp_NN.nii', 'D_NN.nii', 'f_NN.nii', 'S0_NN.nii']

    if segmented:
        names_lsq = ['geof_lsq_seg.nii','Dp_lsq_seg.nii', 'D_lsq_seg.nii', 'f_lsq_seg.nii']
    elif fixS0:
        names_lsq = ['geof_lsq_fixS0.nii','Dp_lsq_fixS0.nii', 'D_lsq_fixS0.nii', 'f_lsq_fixS0.nii']
    else:
        names_lsq = ['geof_lsq.nii','Dp_lsq.nii', 'D_lsq.nii', 'f_lsq.nii', 'S0_lsq.nii']

    multiple = [100., 1000., 1000000., 10000, 1000]
    tot=0
    for fold in aa1:
        for ss in [1, 2]:
            if os.path.isfile('{folder}/CR{fold:02d}/MRI{ss}_{dat}.nii.gz'.format(folder=fold1,fold=fold,ss=ss, dat=dattype)):
                data=nib.load('{folder}/CR{fold:02d}/MRI{ss}_{dat}.nii.gz'.format(folder=fold1,fold=fold,ss=ss, dat=dattype))
                datas=data.get_data()
                sx, sy, sz, n_b_values = datas.shape
                data.header.set_data_shape((sx,sy,sz))
                X_dw_all = np.reshape(datas, (sx * sy * sz, n_b_values))
                valid_id = np.sum(X_dw_all == 0, axis=1) == 0
                for k in range(len(names)):
                    img = np.zeros([sx * sy * sz])
                    if k is 0:
                        img[valid_id] = gof2NN[tot:(tot + sum(valid_id))]  # [Ke, dt, ve, vp]
                    else:
                        img[valid_id] = paramsNN2[k-1][tot:(tot+sum(valid_id))]  # [Ke, dt, ve, vp]
                    img[np.isnan(img)] = 0
                    img[img < 0] = 0
                    img = np.reshape(img, [sx, sy, sz])
                    nib.save(nib.Nifti1Image(img * multiple[k], data.affine, data.header), '{folder}/CR{fold:02d}/MRI{ss}_{dat}_{name}.nii.gz'.format(folder=fold1,fold=fold,ss=ss, dat=dattype,name=names[k]))
                for k in range(len(names_lsq)):
                    img = np.zeros([sx * sy * sz])
                    if dolsq:
                        if k is 0:
                            img[valid_id] = gof2lsq[tot:(tot + sum(valid_id))]  # [Ke, dt, ve, vp]
                        else:
                            img[valid_id] = paramslsq2[k-1][tot:(tot+sum(valid_id))]
                        img[np.isnan(img)] = 0
                        img[img < 0] = 0
                        img = np.reshape(img, [sx, sy, sz])
                        nib.save(nib.Nifti1Image(img * multiple[k], data.affine, data.header), '{folder}/CR{fold:02d}/MRI{ss}_{dat}_{name}.nii.gz'.format(folder=fold1,fold=fold,ss=ss, dat=dattype,name=names_lsq[k]))
                tot = tot+sum(valid_id)

    tot=0
    for fold in aa2:
        for ss in [1, 2]:
            if os.path.isfile('{folder}/CR{fold:02d}/MRI{ss}_{dat}.nii.gz'.format(folder=fold1,fold=fold,ss=ss, dat=dattype)):
                data=nib.load('{folder}/CR{fold:02d}/MRI{ss}_{dat}.nii.gz'.format(folder=fold1,fold=fold,ss=ss, dat=dattype))
                datas=data.get_data()
                sx, sy, sz, n_b_values = datas.shape
                data.header.set_data_shape((sx,sy,sz))
                X_dw_all = np.reshape(datas, (sx * sy * sz, n_b_values))
                valid_id = np.sum(X_dw_all == 0, axis=1) == 0
                for k in range(len(names)):
                    img = np.zeros([sx * sy * sz])
                    if k is 0:
                        img[valid_id] = gofNN[tot:(tot + sum(valid_id))]  # [Ke, dt, ve, vp]
                    else:
                        img[valid_id] = paramsNN[k-1][tot:(tot+sum(valid_id))]  # [Ke, dt, ve, vp]                img[np.isnan(img)] = 0
                    img[img < 0] = 0
                    img = np.reshape(img, [sx, sy, sz])
                    nib.save(nib.Nifti1Image(img * multiple[k], data.affine, data.header), '{folder}/CR{fold:02d}/MRI{ss}_{dat}_{name}.nii.gz'.format(folder=fold1,fold=fold,ss=ss, dat=dattype,name=names[k]))
                for k in range(len(names_lsq)):
                    img = np.zeros([sx * sy * sz])
                    if dolsq:
                        if k is 0:
                            img[valid_id] = goflsq[tot:(tot + sum(valid_id))]  # [Ke, dt, ve, vp]
                        else:
                            img[valid_id] = paramslsq[k-1][tot:(tot+sum(valid_id))]
                        img[np.isnan(img)] = 0
                        img[img < 0] = 0
                        img = np.reshape(img, [sx, sy, sz])
                        nib.save(nib.Nifti1Image(img * multiple[k], data.affine, data.header), '{folder}/CR{fold:02d}/MRI{ss}_{dat}_{name}.nii.gz'.format(folder=fold1,fold=fold,ss=ss, dat=dattype,name=names_lsq[k]))
                tot = tot+sum(valid_id)

    for fold in bb:
        for ss in [1, 2]:
            if os.path.isfile('{folder}/{fold}/{ss}/{dat}.nii.gz'.format(folder=fold2,fold=fold,ss=ss, dat=dattype)):
                data=nib.load('{folder}/{fold}/{ss}/{dat}.nii.gz'.format(folder=fold2,fold=fold,ss=ss, dat=dattype))
                datas=data.get_data()
                sx, sy, sz, n_b_values = datas.shape
                X_dw_all = np.reshape(datas, (sx * sy * sz, n_b_values))
                data.header.set_data_shape((sx,sy,sz))
                valid_id = np.sum(X_dw_all == 0, axis=1) == 0
                for k in range(len(names)):
                    img = np.zeros([sx * sy * sz])
                    if k is 0:
                        img[valid_id] = gofNN[tot:(tot + sum(valid_id))]  # [Ke, dt, ve, vp]
                    else:
                        img[valid_id] = paramsNN[k-1][tot:(tot+sum(valid_id))]
                    img[np.isnan(img)] = 0
                    img[img < 0] = 0
                    img = np.reshape(img, [sx, sy, sz])
                    nib.save(nib.Nifti1Image(img * multiple[k], data.affine, data.header), '{folder}/{fold}/{ss}/{dat}_{name}.nii.gz'.format(folder=fold2,fold=fold,ss=ss, dat=dattype,name=names[k]))
                for k in range(len(names_lsq)):
                    img = np.zeros([sx * sy * sz])
                    if dolsq:
                        if k is 0:
                            img[valid_id] = goflsq[tot:(tot + sum(valid_id))]  # [Ke, dt, ve, vp]
                        else:
                            img[valid_id] = paramslsq[k-1][tot:(tot+sum(valid_id))]
                        img[np.isnan(img)] = 0
                        img[img < 0] = 0
                        img = np.reshape(img, [sx, sy, sz])
                        nib.save(nib.Nifti1Image(img * multiple[k], data.affine, data.header), '{folder}/{fold}/{ss}/{dat}_{name}.nii.gz'.format(folder=fold2,fold=fold,ss=ss, dat=dattype,name=names_lsq[k]))
                tot = tot+sum(valid_id)

    for fold in bb:
        for ss in [1, 2]:
            if os.path.isfile('{folder}/{fold}/{ss}/{dat}_intra.nii.gz'.format(folder=fold2,fold=fold,ss=ss, dat=dattype)):
                data=nib.load('{folder}/{fold}/{ss}/{dat}_intra.nii.gz'.format(folder=fold2,fold=fold,ss=ss, dat=dattype))
                datas=data.get_data()
                sx, sy, sz, n_b_values = datas.shape
                X_dw_all = np.reshape(datas, (sx * sy * sz, n_b_values))
                data.header.set_data_shape((sx,sy,sz))
                valid_id = np.sum(X_dw_all == 0, axis=1) == 0
                for k in range(len(names)):
                    img = np.zeros([sx * sy * sz])
                    if k is 0:
                        img[valid_id] = gofNN[tot:(tot + sum(valid_id))]  # [Ke, dt, ve, vp]
                    else:
                        img[valid_id] = paramsNN[k-1][tot:(tot+sum(valid_id))]
                    img[np.isnan(img)] = 0
                    img[img < 0] = 0
                    img = np.reshape(img, [sx, sy, sz])
                    nib.save(nib.Nifti1Image(img * multiple[k], data.affine, data.header), '{folder}/{fold}/{ss}/{dat}_intra_{name}.nii.gz'.format(folder=fold2,fold=fold,ss=ss, dat=dattype,name=names[k]))
                for k in range(len(names_lsq)):
                    img = np.zeros([sx * sy * sz])
                    if dolsq:
                        if k is 0:
                            img[valid_id] = goflsq[tot:(tot + sum(valid_id))]  # [Ke, dt, ve, vp]
                        else:
                            img[valid_id] = paramslsq[k-1][tot:(tot+sum(valid_id))]
                        img[np.isnan(img)] = 0
                        img[img < 0] = 0
                        img = np.reshape(img, [sx, sy, sz])
                        nib.save(nib.Nifti1Image(img * multiple[k], data.affine, data.header), '{folder}/{fold}/{ss}/{dat}_intra_{name}.nii.gz'.format(folder=fold2,fold=fold,ss=ss, dat=dattype,name=names_lsq[k]))
                tot = tot+sum(valid_id)