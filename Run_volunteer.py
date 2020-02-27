import nibabel as nib
import numpy as np
import deep
from fitting_algorithms import fit_least_squares_array, ivimN, fit_least_squares_S0, fit_segmented_array, goodness_of_fit
import sys as sys

lsq_only = False
bvalues = np.array([0, 10, 20, 30, 50, 75, 150, 300, 450, 600])
multiple = [100., 1000., 1000000., 10000, 1000]

if not lsq_only:
    for run_net in ['abs_con']:

        print('network is {}'.format(run_net))
        for ii in range(1):
            data = nib.load('/home/ojgurney-champion/lood_storage/divi/Projects/mipa/Data/deep/PCAMIP.nii')
            datas = data.get_data()
            sx, sy, sz, n_b_values = datas.shape
            X_dw_all = np.reshape(datas, (sx * sy * sz, n_b_values))
            S0 = np.nanmean(X_dw_all[:, bvalues == 0], axis=0)
            valid_id = np.squeeze(X_dw_all[:, bvalues == 0]<(S0/1))
            X_dw_all[np.squeeze(valid_id), :] = 0
            valid_id = np.sum(X_dw_all == 0, axis=1) == 0
            X_dw_sel=X_dw_all[valid_id,:]
            S0 = np.nanmean(X_dw_sel[:, bvalues == 0], axis=1)
            X_dw_sel = X_dw_sel / S0[:, None]
            res = [i for i, val in enumerate(X_dw_sel != X_dw_sel) if not val.any()]
            if run_net == 'loss_con':
                lr = 0.0001
            else:
                lr = 0.0005
            net = deep.learn_IVIM(X_dw_sel[res],bvalues,run_net=run_net,lr=lr)
            paramsNN=deep.infer_IVIM(X_dw_sel, bvalues, net)
            del net
            gofNN=goodness_of_fit(bvalues,paramsNN[0],paramsNN[1],paramsNN[2],paramsNN[3],X_dw_sel)
            names = ['geof_NN_{ii}_{net}_2'.format(ii=ii,net=run_net),'Dp_NN_{ii}_{net}_2'.format(ii=ii,net=run_net), 'D_NN_{ii}_{net}_2'.format(ii=ii,net=run_net), 'f_NN_{ii}_{net}_2'.format(ii=ii,net=run_net), 'S0_NN_{ii}_{net}_2'.format(ii=ii,net=run_net)]
            for k in range(len(names)):
                img = np.zeros([sx * sy * sz])
                if k is 0:
                    img[valid_id] = gofNN
                else:
                    img[valid_id] = paramsNN[k - 1]
                img[np.isnan(img)] = 0
                img = np.reshape(img, [sx, sy, sz])
                nib.save(nib.Nifti1Image(img * multiple[k], data.affine, data.header),
                         '/home/ojgurney-champion/lood_storage/divi/Projects/mipa/Data/deep/output/{name}.nii.gz'.format(name=names[k]))
            del paramsNN, data, datas, X_dw_all, valid_id, img, gofNN, res, X_dw_sel, S0
'''
data = nib.load('C:/Users/ochampion/Documents/Data/ML_IVIM/PCAMIP.nii')
datas = data.get_data()
sx, sy, sz, n_b_values = datas.shape
X_dw_all = np.reshape(datas, (sx * sy * sz, n_b_values))
S0 = np.nanmean(X_dw_all[:, bvalues == 0], axis=0)
valid_id = np.squeeze(X_dw_all[:, bvalues == 0] < (S0 / 1))
X_dw_all[np.squeeze(valid_id), :] = 0
valid_id = np.sum(X_dw_all == 0, axis=1) == 0
X_dw_sel = X_dw_all[valid_id, :]
S0 = np.nanmean(X_dw_sel[:, bvalues == 0], axis=1)
X_dw_sel = X_dw_sel / S0[:, None]
paramslsq = fit_least_squares_array(bvalues, X_dw_sel)
goflsq=goodness_of_fit(bvalues,paramslsq[0],paramslsq[1],paramslsq[2],paramslsq[3],X_dw_sel)
names_lsq = ['geof_lsq.nii', 'Dp_lsq.nii', 'D_lsq.nii', 'f_lsq.nii', 'S0_lsq.nii']

for k in range(len(names_lsq)):
    img = np.zeros([sx * sy * sz])
    if k is 0:
        img[valid_id] = goflsq
    else:
        img[valid_id] = paramslsq[k - 1]
    img[np.isnan(img)] = 0
    img = np.reshape(img, [sx, sy, sz])
    nib.save(nib.Nifti1Image(img * multiple[k], data.affine, data.header),
             'C:/Users/ochampion/Documents/Data/ML_IVIM/{name}.nii.gz'.format(name=names_lsq[k]))'''