import os
import nibabel as nib
import numpy as np
import xlsxwriter
import fitting_algorithms as lsq
from sys import platform

segmented=False
constrained=False
fixS0=False

datatype = 'remz'
if platform == 'win32':
    divi='L:/basic/divi/'
else:
    divi='/home/ojgurney-champion/lood_storage/divi/'

fold1=divi + 'Projects/mipa/Data/MIPA/nii'
fold2=divi + 'Projects/mipa/Data/REMP/REPRO/nii'
aa1=[1,2,3,4,6,12,14,15]
bb=[1, 2,3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16]

MR1='MRI1_reg.nii'
MR2='MRI2_reg.nii'

if constrained:
    names = ['geof_NN_con.nii', 'Dp_NN_con.nii', 'D_NN_con.nii', 'f_NN_con.nii', 'S0_NN_con.nii']
elif fixS0:
    names = ['geof_NN_S0_fix.nii','Dp_NN_S0_fix.nii', 'D_NN_S0_fix.nii', 'f_NN_S0_fix.nii', 'S0_NN_S0_fix.nii']
else:
    names = ['geof_NN.nii','Dp_NN.nii', 'D_NN.nii', 'f_NN.nii', 'S0_NN.nii']

if segmented:
    names_lsq = ['geof_lsq_seg.nii','Dp_lsq_seg.nii', 'D_lsq_seg.nii', 'f_lsq_seg.nii']
elif fixS0:
    names_lsq = ['geof_lsq_fixS0.nii','Dp_lsq_fixS0.nii', 'D_lsq_fixS0.nii', 'f_lsq_fixS0.nii']
else:
    names_lsq = ['geof_lsq.nii','Dp_lsq.nii', 'D_lsq.nii', 'f_lsq.nii', 'S0_lsq.nii']

    bvalues=np.array([0, 50, 50, 50, 50, 50, 50, 50, 50, 50, 40, 40, 40, 40, 40, 40, 40, 40, 40, 30, 30, 30, 30, 30, 30, 30, 30, 30, 250, 250, 250, 250, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 10, 10, 10, 10, 10, 10, 10, 10, 10, 75, 75, 75, 75, 400, 400, 400, 400, 20, 20, 20, 20, 20, 20, 20, 20, 20, 150, 150, 150, 150, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    bvalues2=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 20, 20, 20, 20, 20, 20, 20, 20, 20, 30, 30, 30, 30, 30, 30, 30, 30, 30, 40, 40, 40, 40, 40, 40, 40, 40, 40, 50, 50, 50, 50, 50, 50, 50, 50, 50, 75, 75, 75, 75, 100, 100, 100, 100, 150, 150, 150, 150, 250, 250, 250, 250, 400, 400, 400, 400, 0, 0, 0, 0, 0, 0, 100, 100, 100, 100, 100, 100, 100, 100, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600])

#multiple = [1000., 1000000., 10000]
multiple = [100, 1., 1000., 100]
tot=0
stats=np.zeros((len(aa1),len(names),2,3))
aaa=0
for fold in aa1:
    for ss in [1, 2]:
        if os.path.isfile('{folder}/CR{fold:02d}/MRI{ss}_{dat}.nii.gz'.format(folder=fold1,fold=fold,ss=ss ,dat=datatype)):
            if ss is 1:
                mask = nib.load('{folder}/CR{fold:02d}/ADC600_0-label.nii.gz'.format(folder=fold1,fold=fold))
            else:
                mask = nib.load('{folder}/CR{fold:02d}/ADC600_0_1-label.nii.gz'.format(folder=fold1,fold=fold))
            mask = mask.get_data()
            bbb=0
            dataall=nib.load('{folder}/CR{fold:02d}/MRI{ss}_{dat}.nii.gz'.format(folder=fold1,fold=fold,ss=ss ,dat=datatype))
            dataall = dataall.get_data()
            sx,sy,sz,bvalnum=dataall.shape
            maskarray=np.reshape(mask, (sx*sy*sz))
            dataall=np.reshape(dataall,(sx*sy*sz,bvalnum))
            datafit=np.mean(dataall[maskarray==1],axis=0)
            if bvalnum is 103:
                pars_fit=lsq.fit_least_squares(datafit,bvalues,S0_output=False, fixS0=False)
            else:
                pars_fit = lsq.fit_least_squares(datafit, bvalues, S0_output=False, fixS0=False)
            stats[aaa,:,ss-1,2]=pars_fit
            for k in range(len(names)):
                data = nib.load('{folder}/CR{fold:02d}/MRI{ss}_{dat}_{name}.nii.gz'.format(folder=fold1,fold=fold,ss=ss ,dat=datatype,name=names[k]))
                mask[data==0] = 0
                data = data.get_data()
                sick = data[mask == 1]
                stats[aaa,bbb,ss-1,1]=np.mean(sick)/multiple[k]
                data = nib.load('{folder}/CR{fold:02d}/MRI{ss}_{dat}_{name}.nii.gz'.format(folder=fold1,fold=fold,ss=ss ,dat=datatype,name=names_lsq[k]))
                data = data.get_data()
                sick = data[mask == 1]
                stats[aaa,bbb,ss-1,0]=np.mean(sick)/multiple[k]
                bbb=bbb+1
    aaa=aaa+1

aaa=0
stats2=np.zeros((len(bb),len(names),2,3))
for fold in bb:
    for ss in [1, 2]:
        if os.path.isfile('{folder}/{fold}/{ss}/{dat}.nii.gz'.format(folder=fold2,fold=fold,ss=ss ,dat=datatype)):
            if ss is 1:
                mask = nib.load('{folder}/{fold}/{ss}/ADC600_0-label.nii.gz'.format(folder=fold2, fold=fold,ss=ss))
            else:
                if fold in [5,8]:
                    mask = nib.load('{folder}/{fold}/{ss}/ADC600_0_1-label.nii.gz'.format(folder=fold2, fold=fold,ss=ss))
                else:
                    mask = nib.load(
                        '{folder}/{fold}/{ss}/ADC600_0_2-label.nii.gz'.format(folder=fold2, fold=fold,ss=ss))

            mask = mask.get_data()
            dataall=nib.load('{folder}/{fold}/{ss}/{dat}.nii.gz'.format(folder=fold2,fold=fold,ss=ss ,dat=datatype))
            dataall = dataall.get_data()
            sx,sy,sz,bvalnum=dataall.shape
            maskarray=np.reshape(mask, (sx*sy*sz))
            dataall=np.reshape(dataall,(sx*sy*sz,bvalnum))
            datafit=np.mean(dataall[maskarray==1],axis=0)
            if bvalnum is 103:
                pars_fit=lsq.fit_least_squares(datafit,bvalues,S0_output=False, fixS0=False)
            else:
                pars_fit = lsq.fit_least_squares(datafit, bvalues, S0_output=False, fixS0=False)
            stats2[aaa,:,ss-1,2]=pars_fit
            bbb = 0
            for k in range(len(names)):
                if ss is 2 and fold in [5, 8]:
                    data = nib.load(
                        '{folder}/{fold}/{ss}/{dat}_intra_{name}.nii.gz'.format(folder=fold2, fold=fold, ss=ss,
                                                                                dat=datatype,
                                                                                name=names[k]))
                else:
                    data = nib.load(
                        '{folder}/{fold}/{ss}/{dat}_{name}.nii.gz'.format(folder=fold2, fold=fold, ss=ss, dat=datatype,
                                                                          name=names[k]))
                data = data.get_data()
                mask[data == 0] = 0
                sick = data[mask == 1]
                stats2[aaa, bbb, ss - 1, 1] = np.mean(sick) / multiple[k]
                if ss is 2 and fold in [5,8]:
                    data = nib.load(
                        '{folder}/{fold}/{ss}/{dat}_intra_{name}.nii.gz'.format(folder=fold2, fold=fold,ss=ss ,dat=datatype,
                                                                                name=names_lsq[k]))
                else:
                    data = nib.load(
                        '{folder}/{fold}/{ss}/{dat}_{name}.nii.gz'.format(folder=fold2, fold=fold,ss=ss ,dat=datatype,
                                                                                name=names_lsq[k]))
                data = data.get_data()
                sick = data[mask == 1]
                stats2[aaa, bbb, ss-1, 0] = np.mean(sick)/multiple[k]

                bbb = bbb + 1
    aaa = aaa + 1

# stats is treatment response data, dim1 is patients, dim2 is parameter, dim3 is pre[0]-post[1] treatment and dim4=lsq[0]/NN[1]
print(stats)
# stats2 is test-retest data, dim1 is patients, dim2 is parameter, dim3 is test[0]-retest[1] and dim4=lsq[0]/NN[1]
print(stats2)

for ii in range(3):
    print(stats[:,ii,:,1])

for ii in range(3):
    print(stats2[:, ii, :, 1])

np.savez('results/output.npz',stats=stats,stats2=stats2)

workbook = xlsxwriter.Workbook('results/data.xlsx')
worksheet = workbook.add_worksheet()
row=0
col=0
for aa in range(len(stats)+1):
    if aa is 0:
        worksheet.write(row, col, 'lsq_Gof')
        worksheet.write(row, col + 1, 'lsq_Gof rep')
        worksheet.write(row, col +3, 'lsq_Dp')
        worksheet.write(row, col + 4, 'lsq_Dp rep')
        worksheet.write(row, col + 6, 'lsq_Dt')
        worksheet.write(row, col + 7, 'lsq_Dt_rep')
        worksheet.write(row, col + 9, 'lsq_f')
        worksheet.write(row, col + 10, 'lsq_f_rep')
        worksheet.write(row, col + 12, 'NN_Gof')
        worksheet.write(row, col + 13, 'NN_Gof_rep')
        worksheet.write(row, col + 15, 'NN_Dp')
        worksheet.write(row, col + 16, 'NN_Dp_rep')
        worksheet.write(row, col + 18, 'NN_Dt')
        worksheet.write(row, col + 19, 'NN_Dt_rep')
        worksheet.write(row, col + 21, 'NN_f')
        worksheet.write(row, col + 22, 'NN_f_rep')
        worksheet.write(row, col + 24, 'all_Dp')
        worksheet.write(row, col + 25, 'all_Dp_rep')
        worksheet.write(row, col + 27, 'all_Dt')
        worksheet.write(row, col + 28, 'all_Dt_rep')
        worksheet.write(row, col + 30, 'all_f')
        worksheet.write(row, col + 31, 'all_f_rep')
    else:
        worksheet.write(row,col,     stats[aa-1,0,0,0])
        worksheet.write(row,col+1,   stats[aa-1,0,1,0])
        worksheet.write(row,col+3,   stats[aa-1,1,0,0])
        worksheet.write(row,col+4,   stats[aa-1,1,1,0])
        worksheet.write(row,col+6,   stats[aa-1,2,0,0])
        worksheet.write(row,col+7,   stats[aa-1,2,1,0])
        worksheet.write(row, col + 9, stats[aa - 1, 3, 0, 0])
        worksheet.write(row, col + 1, stats[aa - 1, 3, 1, 0])
        worksheet.write(row,col+12,   stats[aa-1,0,0,1])
        worksheet.write(row,col+13,  stats[aa-1,0,1,1])
        worksheet.write(row,col+15,  stats[aa-1,1,0,1])
        worksheet.write(row,col+16,  stats[aa-1,1,1,1])
        worksheet.write(row,col+18,  stats[aa-1,2,0,1])
        worksheet.write(row,col+19,  stats[aa-1,2,1,1])
        worksheet.write(row,col+21,  stats[aa-1,3,0,1])
        worksheet.write(row,col+22,  stats[aa-1,3,1,1])
        worksheet.write(row, col + 24, stats[aa - 1, 0, 0, 2])
        worksheet.write(row, col + 25, stats[aa - 1, 0, 1, 2])
        worksheet.write(row, col + 27, stats[aa - 1, 1, 0, 2])
        worksheet.write(row, col + 28, stats[aa - 1, 1, 1, 2])
        worksheet.write(row, col + 30, stats[aa - 1, 2, 0, 2])
        worksheet.write(row, col + 31, stats[aa - 1, 2, 1, 2])
    row += 1

worksheet = workbook.add_worksheet()
row = 0
col = 0
for aa in range(len(stats2)):
    if aa is 0:
        worksheet.write(row, col, 'lsq_Gof')
        worksheet.write(row, col + 1, 'lsq_Gof rep')
        worksheet.write(row, col +3, 'lsq_Dp')
        worksheet.write(row, col + 4, 'lsq_Dp rep')
        worksheet.write(row, col + 6, 'lsq_Dt')
        worksheet.write(row, col + 7, 'lsq_Dt_rep')
        worksheet.write(row, col + 9, 'lsq_f')
        worksheet.write(row, col + 10, 'lsq_f_rep')
        worksheet.write(row, col + 12, 'NN_Gof')
        worksheet.write(row, col + 13, 'NN_Gof_rep')
        worksheet.write(row, col + 15, 'NN_Dp')
        worksheet.write(row, col + 16, 'NN_Dp_rep')
        worksheet.write(row, col + 18, 'NN_Dt')
        worksheet.write(row, col + 19, 'NN_Dt_rep')
        worksheet.write(row, col + 21, 'NN_f')
        worksheet.write(row, col + 22, 'NN_f_rep')
        worksheet.write(row, col + 24, 'all_Dp')
        worksheet.write(row, col + 25, 'all_Dp_rep')
        worksheet.write(row, col + 27, 'all_Dt')
        worksheet.write(row, col + 28, 'all_Dt_rep')
        worksheet.write(row, col + 30, 'all_f')
        worksheet.write(row, col + 31, 'all_f_rep')
    else:
        worksheet.write(row,col,     stats2[aa-1,0,0,0])
        worksheet.write(row,col+1,   stats2[aa-1,0,1,0])
        worksheet.write(row,col+3,   stats2[aa-1,1,0,0])
        worksheet.write(row,col+4,   stats2[aa-1,1,1,0])
        worksheet.write(row,col+6,   stats2[aa-1,2,0,0])
        worksheet.write(row,col+7,   stats2[aa-1,2,1,0])
        worksheet.write(row, col + 9, stats2[aa - 1, 3, 0, 0])
        worksheet.write(row, col + 1, stats2[aa - 1, 3, 1, 0])
        worksheet.write(row,col+12,   stats2[aa-1,0,0,1])
        worksheet.write(row,col+13,  stats2[aa-1,0,1,1])
        worksheet.write(row,col+15,  stats2[aa-1,1,0,1])
        worksheet.write(row,col+16,  stats2[aa-1,1,1,1])
        worksheet.write(row,col+18,  stats2[aa-1,2,0,1])
        worksheet.write(row,col+19,  stats2[aa-1,2,1,1])
        worksheet.write(row,col+21,  stats2[aa-1,3,0,1])
        worksheet.write(row,col+22,  stats2[aa-1,3,1,1])
        worksheet.write(row, col + 24, stats2[aa - 1, 0, 0, 2])
        worksheet.write(row, col + 25, stats2[aa - 1, 0, 1, 2])
        worksheet.write(row, col + 27, stats2[aa - 1, 1, 0, 2])
        worksheet.write(row, col + 28, stats2[aa - 1, 1, 1, 2])
        worksheet.write(row, col + 30, stats2[aa - 1, 2, 0, 2])
        worksheet.write(row, col + 31, stats2[aa - 1, 2, 1, 2])
    row += 1

workbook.close()