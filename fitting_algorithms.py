'''
Mar 2018 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/deep-learning/ivim
'''

from scipy.optimize import curve_fit, minimize
import numpy as np
from scipy import stats

def goodness_of_fit(b,Dp,Dt,Fp,S0,data):
    datasim=ivim(np.tile(np.expand_dims(b,axis=0),(len(Dp),1)), np.tile(np.expand_dims(Dp,axis=1),(1,len(b))), np.tile(np.expand_dims(Dt,axis=1),(1,len(b))), np.tile(np.expand_dims(Fp,axis=1),(1,len(b))), np.tile(np.expand_dims(S0,axis=1),(1,len(b))))
    norm=np.sum(data,axis=1)
    GOF=np.sum(np.square((datasim-data)/norm[:, None]),axis=1)

    return GOF

def ivimN(b, Dp, Dt, Fp, S0):
    return S0 * (Fp/10 * np.exp(-b * Dp/10) + (1 - Fp/10) * np.exp(-b * Dt/1000))


def ivimN_noS0(b, Dp, Dt, Fp):
    return (Fp/10 * np.exp(-b * Dp/10) + (1 - Fp/10) * np.exp(-b * Dt/1000))

def ivim(b, Dp, Dt, Fp, S0):
    return S0 * (Fp * np.exp(-b * Dp) + (1 - Fp) * np.exp(-b * Dt))

def order(Dp, Dt, Fp):
    if Dp < Dt:
        Dp, Dt = Dt, Dp
        Fp = 1 - Fp
    return Dp, Dt, Fp

def fit_segmented_array(b, x_dw):
    S0=np.mean(x_dw[:,b == 0],axis=1)
    x_dw=x_dw/S0[:,None]
    Dp=np.zeros(len(x_dw))
    Dt = np.zeros(len(x_dw))
    Fp = np.zeros(len(x_dw))
    for aa in range(len(x_dw)):
        Dp[aa], Dt[aa], Fp[aa] = fit_segmented(b,x_dw[aa,:])
    return [Dp, Dt, Fp]

def fit_segmented(b, x_dw):
    try:
        high_b = b[b >= 100]
        high_x_dw = x_dw[b >= 100]
        bounds = ([0, 0], [5, 5])
        # bounds = ([0, 0.4], [0.005, 1])
        params, _ = curve_fit(lambda high_b, Dt, int: int * np.exp(-high_b * Dt/1000), high_b, high_x_dw, p0=(1, 1),
                              bounds=bounds)
        Dt, Fp = params[0]/1000, 1 - params[1]
        x_dw_remaining = x_dw - (1 - Fp) * np.exp(-b * Dt)
        bounds = (0.006, 0.2)
        # bounds = (0.01, 0.3)
        params, _ = curve_fit(lambda b, Dp: Fp * np.exp(-b * Dp), b, x_dw_remaining, p0=(0.1), bounds=bounds)
        Dp = params[0]
        return order(Dp, Dt, Fp)
    except:
        return 0., 0., 0.

def fit_least_squares_array(b, x_dw,S0_output=False,fixS0=False):
    S0=np.mean(x_dw[:,b == 0],axis=1)
    x_dw=x_dw/S0[:,None]
    Dp=np.zeros(len(x_dw))
    Dt = np.zeros(len(x_dw))
    Fp = np.zeros(len(x_dw))
    S0 = np.zeros(len(x_dw))
    if S0_output:
        for aa in range(len(x_dw)):
            Dp[aa], Dt[aa], Fp[aa], S0[aa] = fit_least_squares(b,x_dw[aa,:],S0_output=S0_output)
        return [Dp, Dt, Fp, S0]
    else:
        for aa in range(len(x_dw)):
            Dp[aa], Dt[aa], Fp[aa] = fit_least_squares(b,x_dw[aa,:],fixS0=fixS0)
        return [Dp, Dt, Fp, S0]

def fit_least_squares(b, x_dw, S0_output=False,fixS0=False):
    try:
        #bounds = (0, 1)
        if fixS0:
            bounds = ([0.005, 0, 0], [2, 5, 7])
            params, _ = curve_fit(ivimN_noS0, b, x_dw, p0=[0.1, 0.001, 0.1], bounds=bounds)
        else:
            bounds = ([0.05, 0, 0, 0.8], [5, 5, 8, 1.2])
            params, _ = curve_fit(ivimN, b, x_dw, p0=[0.1, 0.001, 0.1, 1], bounds=bounds)
        Dp, Dt, Fp, S0 = params[0]/10, params[1]/1000, params[2]/10, params[3]
        if S0_output:
            return order(Dp, Dt, Fp), S0
        else:
            return order(Dp, Dt, Fp)
    except:
        if S0_output:
            return fit_segmented(b, x_dw), 1
        else:
            return fit_segmented(b, x_dw)


def fit_least_squares_S0(b, x_dw):
    try:
        # bounds = (0, 1)
        bounds = ([0.06, 0, 0, 0.5], [2, 5, 7, 1.5])
        params, _ = curve_fit(ivimN, b, x_dw, p0=[0.1, 0.001, 0.1, 1], bounds=bounds)
        Dp, Dt, Fp, S0 = params[0] / 10, params[1] / 1000, params[2] / 10, params[3]
        Dp, Dt, Fp = order(Dp, Dt, Fp)
        return [Dp, Dt, Fp, S0]
    except:
        return fit_segmented(b, x_dw), 1


def empirical_neg_log_prior(Dp0, Dt0, Fp0):
    # Dp0, Dt0, Fp0 are flattened arrays
    Dp_valid = (1e-8 < np.nan_to_num(Dp0)) & (np.nan_to_num(Dp0) < 1 - 1e-8)
    Dt_valid = (1e-8 < np.nan_to_num(Dt0)) & (np.nan_to_num(Dt0) < 1 - 1e-8)
    Fp_valid = (1e-8 < np.nan_to_num(Fp0)) & (np.nan_to_num(Fp0) < 1 - 1e-8)
    valid = Dp_valid & Dt_valid & Fp_valid
    Dp0, Dt0, Fp0 = Dp0[valid], Dt0[valid], Fp0[valid]
    Dp_shape, _, Dp_scale = stats.lognorm.fit(Dp0, floc=0)
    Dt_shape, _, Dt_scale = stats.lognorm.fit(Dt0, floc=0)
    Fp_a, Fp_b, _, _ = stats.beta.fit(Fp0, floc=0, fscale=1)

    def neg_log_prior(p):
        Dp, Dt, Fp, = p[0], p[1], p[2]
        if (Dp < Dt):
            return 1e8
        else:
            eps = 1e-8
            Dp_prior = stats.lognorm.pdf(Dp, Dp_shape, scale=Dp_scale)
            Dt_prior = stats.lognorm.pdf(Dt, Dt_shape, scale=Dt_scale)
            Fp_prior = stats.beta.pdf(Fp, Fp_a, Fp_b)
            return -np.log(Dp_prior + eps) - np.log(Dt_prior + eps) - np.log(Fp_prior + eps)

    return neg_log_prior


def neg_log_likelihood(p, b, x_dw):
    return 0.5 * (len(b) + 1) * np.log(np.sum((ivimN(b, p[0], p[1], p[2], p[3]) - x_dw) ** 2))  # 0.5*sum simplified


def neg_log_posterior(p, b, x_dw, neg_log_prior):
    return neg_log_likelihood(p, b, x_dw) + neg_log_prior(p)


def fit_bayesian(b, x_dw, neg_log_prior):
    try:
        bounds = [(0, 1), (0, 1), (0, 1), (0.5,1.5)]
        # bounds = [(0.01, 0.3), (0, 0.005), (0, 0.6)]
        params = minimize(neg_log_posterior, x0=[0.01, 0.001, 0.1, 1], args=(b, x_dw, neg_log_prior), bounds=bounds)
        if not params.success:
            # print(params.message)
            raise (params.message)
        Dp, Dt, Fp = params.x[0], params.x[1], params.x[2]
        return order(Dp, Dt, Fp)
    except:
        return fit_least_squares(b, x_dw)


if __name__ == '__main__':
    # noise 15
    # Dp_truth = array([0.04775578])
    # Dt_truth = array([0.00099359])
    # Fp_truth = array([0.17606327])
    x_dw = np.array([0.92007383, 0.89169505, 0.79940185, 0.73149624, 0.62961453, 0.50390379, 0.30776271])

    # noise 150
    # Dp_truth = array([0.04775578])
    # Dt_truth = array([0.00099359])
    # Fp_truth = array([0.17606327])
    # x_dw = np.array([0.85389307, 1.03929503, 0.91976122, 0.96119655, 0.80588596, 0.53290429, 0.34947694])

    b_values_no0 = [10,20,60,150,300,500,1000]
    t = fit_segmented(b_values_no0, x_dw)
    print(t)
    t = fit_least_squares(b_values_no0, x_dw)
    print(t)
    t = fit_bayesian(b_values_no0, x_dw, np.array([0.01, 0.001, 0.1]),
                     np.array([[10000, 0, 0], [0, 10000, 0], [0, 0, 10000]]))
    print(t)