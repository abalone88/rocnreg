from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm, iqr
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
import pyreadr


def empirical_roc(yh, yd, p=np.linspace(0, 1, 101)):
    n0 = len(yh)
    n1 = len(yd)

    F1_emp = ECDF(yd)
    roc_emp = 1 - F1_emp(np.quantile(yh, 1 - p, method="inverted_cdf"))

    auc_emp = sum(np.less.outer(yh, yd)) / (n0 * n1) + sum(np.equal.outer(yh, yd)) / (2 * n0 * n1)

    res = {'p': p, 'ROC': roc_emp, 'AUC': auc_emp}
    return res


def G(x, y, h):
    return np.mean(norm.cdf((x - y) / h))


def F(p, y, h):
    def toInvert(x, p, y, h):
        return np.mean(norm.cdf((x - y) / h)) - p

    res = opt.root_scalar(toInvert, bracket=(min(y) - 10 ** 10, max(y) + 10 ** 10), args=(p, y, h))
    return res.root


def kernel_roc(yh, yd, p=np.linspace(0, 1, num=101)):
    npoints = len(p)
    roc_k = np.zeros(npoints)

    def silverman_bandwidth(x):
        n = len(x)
        bw = 0.9 * min(np.std(x), iqr(x) / 1.34) * n ** (-0.2)
        return bw

    h0 = silverman_bandwidth(yh)
    h1 = silverman_bandwidth(yd)

    for j in range(npoints):
        roc_k[j] = 1 - G(x=F(p=1 - p[j], y=yh, h=h0), y=yd, h=h1)

    auc = np.sum(roc_k) / npoints

    res = {'p': p, 'ROC': roc_k, 'AUC': auc}
    return res


def bayesian_bootstrap_roc(yh, yd, p=np.linspace(0, 1, 101), B=5000):
    n0 = len(yh)
    n1 = len(yd)

    roc = np.zeros((len(p), B))
    auc = np.zeros(B)

    for j in range(B):
        q = np.random.exponential(scale=1, size=n0)
        weights_h = q / np.sum(q)

        q1 = np.random.exponential(scale=1, size=n1)
        weights_d = q1 / np.sum(q1)

        u = np.zeros(n1)
        for i in range(n1):
            u[i] = np.sum(weights_h * (yh > yd[i]))

        for i in range(len(p)):
            roc[i, j] = np.sum(weights_d * (u <= p[i]))

        auc[j] = np.sum(roc[:, j]) / len(p)

    roc_bb_m = np.mean(roc, axis=1)
    roc_bb_l = np.quantile(roc, q=0.025, axis=1)
    roc_bb_h = np.quantile(roc, q=0.975, axis=1)

    res = {'p': p, 'ROC': roc_bb_m, 'ROCL': roc_bb_l, 'ROCH': roc_bb_h, 'AUC': np.mean(auc),
           'AUCL': np.quantile(auc, q=0.025), 'AUCH': np.quantile(auc, q=0.975)}

    return res

def roc_bb(yh, yd, p=np.linspace(0, 1, num=101), B=5000):
    n0 = len(yh)
    n1 = len(yd)

    roc = np.zeros((len(p), B))
    auc = np.zeros(B)

    for j in range(B):
        q = np.random.exponential(size=n0)
        weights_h = q / np.sum(q)

        q1 = np.random.exponential(size=n1)
        weights_d = q1 / np.sum(q1)

        u = np.zeros(len(yd))
        for i in range(len(yd)):
            u[i] = np.sum(weights_h * (yh > yd[i]))

        for i in range(len(p)):
            roc[i, j] = np.sum(weights_d * (u <= p[i]))

        auc[j] = np.sum(roc[:, j]) / len(p)

    roc_bb_m = np.mean(roc, axis=1)
    roc_bb_l = np.quantile(roc, q=0.025, axis=1)
    roc_bb_h = np.quantile(roc, q=0.975, axis=1)

    res = {}
    res['p'] = p
    res['ROC'] = roc_bb_m
    res['ROCL'] = roc_bb_l
    res['ROCH'] = roc_bb_h
    res['AUC'] = np.mean(auc)
    res['AUCL'] = np.quantile(auc, q=0.025)
    res['AUCH'] = np.quantile(auc, q=0.975)

    return res

