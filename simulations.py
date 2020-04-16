import numpy as np
import deep as deep
import fitting_algorithms as fit
import time
import sys
import xlsxwriter
from hyperparams import hyperparams as arg

#run_net=sys.argv[1]

#print('network is {}'.format(run_net))

#b=np.array([0,2, 5, 10, 30, 50, 75, 100, 150, 250, 400, 500, 600])
#SNR=25

#load = False

def sim(SNR, b, arg, run_net=None, sims = 100000, num_samples_leval = 5000, Dmin = 0.5 /1000, Dmax = 2.0 /1000, fmin = 0.1, fmax = 0.5, Dsmin= 0.05, Dsmax=0.2, rician = False,segmented=False):

    IVIM_signal_noisy, f, D, Dp = sim_signal(SNR, b, sims=sims,Dmin = Dmin, Dmax = Dmax, fmin = fmin, fmax = fmax, Dsmin= Dsmin, Dsmax=Dsmax, rician=rician)

    D = D[:num_samples_leval]
    Dp = Dp[:num_samples_leval]
    f = f[:num_samples_leval]

    start_time = time.time()
    net = deep.learn_IVIM(IVIM_signal_noisy, b, arg)
    elapsed_time = time.time() - start_time
    print('\ntime elapsed for training: {}\n'.format(elapsed_time))
    IVIM_signal_noisy=IVIM_signal_noisy[:num_samples_leval, :]
    start_time = time.time()
    paramsNN = deep.infer_IVIM(IVIM_signal_noisy, b, net)
    elapsed_time = time.time() - start_time
    print('\ntime elapsed for  inference: {}\n'.format(elapsed_time))
    # ['Ke_NN.nii', 'f_NN.nii', 'tau_NN.nii', 'v_NN.nii']
    del net
    print('results for NN')
    matNN=print_errors(np.squeeze(D), np.squeeze(f), np.squeeze(Dp), paramsNN)
    del paramsNN
    start_time = time.time()
    if segmented:
        paramsf = fit.fit_segmented_array(b,IVIM_signal_noisy)
    else:
        paramsf = fit.fit_least_squares_array(b,IVIM_signal_noisy)

    elapsed_time = time.time() - start_time
    print('\ntime elapsed for lsqfit: {}\n'.format(elapsed_time))
    print('results for lsqfit')
    matlsq=print_errors(np.squeeze(D),np.squeeze(f),np.squeeze(Dp),paramsf)

    del paramsf, IVIM_signal_noisy

    return matlsq, matNN

def sim_signal(SNR, b, sims = 100000, Dmin = 0.5 /1000, Dmax = 2.0 /1000, fmin = 0.1, fmax = 0.5, Dsmin= 0.05, Dsmax=0.2, rician = False, state = 123):
    rg = np.random.RandomState(state)
    test = rg.uniform(0, 1, (sims, 1))
    D = Dmin + (test * (Dmax - Dmin))
    test = rg.uniform(0, 1, (sims, 1))
    f = fmin + (test * (fmax - fmin))
    test = rg.uniform(0, 1, (sims, 1))
    Dp = Dsmin + (test * (Dsmax - Dsmin))

    data_sim = np.zeros([len(D), len(b)])

    b = np.array(b)
    for aa in range(len(D)):
        data_sim[aa, :] = fit.ivim(b, Dp[aa][0], D[aa][0], f[aa][0], 1)


    noise_imag = np.zeros([sims, len(b)])
    noise_real = np.zeros([sims, len(b)])
    for i in range(0, sims - 1):
        noise_real[i,] = rg.normal(0, 1 / SNR,
                                   (1, len(b)))  # wrong! need a SD per input. Might need to loop to maD noise
        noise_imag[i,] = rg.normal(0, 1 / SNR, (1, len(b)))
    if rician:
        IVIM_signal_scaled_noisy = np.sqrt(np.power(data_sim + noise_real, 2) + np.power(noise_imag, 2))
    else:
        IVIM_signal_scaled_noisy = data_sim + noise_imag

    S0_noisy = np.mean(IVIM_signal_scaled_noisy[:, b == 0], axis=1)
    IVIM_signal_noisy = IVIM_signal_scaled_noisy / S0_noisy[:, None]  # check division!
    return IVIM_signal_noisy, f, D, Dp

def print_errors(D,f,Dp,params):
    error_D_lsq = params[1]-D
    randerror_D_lsq=np.std(error_D_lsq)
    syserror_D_lsq=np.mean(error_D_lsq)
    del error_D_lsq

    error_Dp_lsq = params[0]-Dp
    randerror_Dp_lsq=np.std(error_Dp_lsq)
    syserror_Dp_lsq=np.mean(error_Dp_lsq)
    del error_Dp_lsq

    error_f_lsq = params[2]-f
    randerror_f_lsq=np.std(error_f_lsq)
    syserror_f_lsq=np.mean(error_f_lsq)
    del error_f_lsq, params

    normD_lsq = np.mean(D)
    normDp_lsq = np.mean(Dp)
    normf_lsq = np.mean(f)

    print('Sim, random, systematic')
    print([normD_lsq, '  ', randerror_D_lsq, '  ', syserror_D_lsq])
    print([normf_lsq, '  ', randerror_f_lsq, '  ', syserror_f_lsq])
    print([normDp_lsq, '  ', randerror_Dp_lsq, '  ', syserror_Dp_lsq])

    mats=[[normD_lsq,randerror_D_lsq,syserror_D_lsq],
         [normf_lsq,randerror_f_lsq,syserror_f_lsq],
         [normDp_lsq,randerror_Dp_lsq,syserror_Dp_lsq]]

    return mats


#if run_net == 'loss_con':
#    lr = 0.0001
#else:
#    lr = 0.0005
#
#matlsq, matNN = sim(SNR,b,run_net=run_net)
#
#print('network was {}'.format(run_net))