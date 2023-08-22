from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm, iqr
import scipy.optimize as opt
import numpy as np


def empirical_roc(yh, yd, p=np.linspace(0, 1, 101)):
    """
    Empirical estimation of the pooled ROC curve

    @:param yh: Diagnostic test variable for healthy individuals
    @:param yd: Diagnostic test variable for diseased individuals
    @:param p:  Set of false positive fractions (FPF) at which to estimate the pooled ROC curve

    @:return: a list containing p, roc and the AUC (area under the curve).
    """

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


def silverman_bandwidth(x):
    n = len(x)
    bw = 0.9 * min(np.std(x), iqr(x) / 1.34) * n ** (-0.2)
    return bw


def kernel_roc(yh, yd, p=np.linspace(0, 1, num=101)):
    """
    Kernel-based estimation of the pooled ROC curve

    @:param yh: Diagnostic test variable for healthy individuals
    @:param yd: Diagnostic test variable for diseased individuals
    @:param p:  Set of false positive fractions (FPF) at which to estimate the pooled ROC curve

    @:return: a list containing p, roc and the AUC (area under the curve).
    """

    npoints = len(p)
    roc_k = np.zeros(npoints)

    h0 = silverman_bandwidth(yh)
    h1 = silverman_bandwidth(yd)

    for j in range(npoints):
        roc_k[j] = 1 - G(x=F(p=1 - p[j], y=yh, h=h0), y=yd, h=h1)

    auc = np.sum(roc_k) / npoints

    res = {'p': p, 'ROC': roc_k, 'AUC': auc}
    return res


def bayesian_bootstrap_roc(yh, yd, p=np.linspace(0, 1, 101), B=5000):
    """
    Bayesian bootstrap estimation of the pooled ROC curve.

    @:param yh: Diagnostic test variable for healthy individuals
    @:param yd: Diagnostic test variable for diseased individuals
    @:param p:  Set of false positive fractions (FPF) at which to estimate the pooled ROC curve
    @:param B:  An integer value specifying the number of Bayesian bootstrap resamples. Default 5000.

    @:return: a list containing p, roc, AUC (area under the curve) and the 5% CI for the them.
    """

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

# Two mathematically equivalent ways of calculating the youden index

def youden_index1(yh, yd, method):
    """
    Youden index calculated by using the roc and p(FPF) obtained from the pooled roc curve.

    @:param yh:      Diagnostic test variable for healthy individuals
    @:param yd:      Diagnostic test variable for diseased individuals
    @:param method:  "empirical", "kernel estimator" or "bayesian bootstrap"

    @:return: Youden index
    """

    if method == "empirical":
        res = empirical_roc(yh, yd)
    elif method == "kernel estimator":
        res = kernel_roc(yh, yd)
    elif method == "bayesian bootstrap":
        res = bayesian_bootstrap_roc(yh, yd)

    return np.max(np.abs(res['ROC'] - res['p']))


def youden_index2(yh, yd, method, values, B=1000):
    """
    Youden index calculated by using the cdf of the test diagnostic variables

    @:param yh:      Diagnostic test variable for healthy individuals
    @:param yd:      Diagnostic test variable for diseased individuals
    @:param method:  "empirical", "kernel estimator" or "bayesian bootstrap"
    @:param values:  a vector containing the set of values for thich the Youden Index lies in, and grid search will be
    used to find the Youden Index.

    @:return: Youden index
    """

    if method == "empirical":
        def index(c):
            Fh = np.sum([1 if i <= c else 0 for i in yh]) / len(yh)
            Fd = np.sum([1 if j <= c else 0 for j in yd]) / len(yd)
            return Fh - Fd

        score_func = index

    elif method == "kernel estimator":
        h0 = silverman_bandwidth(yh)
        h1 = silverman_bandwidth(yd)

        def index(c):
            return np.mean(norm.cdf((c - yh) / h0)) - np.mean(norm.cdf((c - yd) / h1))

        score_func = index

    elif method == "bayesian bootstrap":
        w, w1 = np.zeros((B, len(yh))), np.zeros((B, len(yd)))
        for j in range(B):
            q = np.random.exponential(scale=1, size=len(yh))
            w[j, ] = q / np.sum(q)
            q1 = np.random.exponential(scale=1, size=len(yd))
            w1[j, ] = q1 / np.sum(q1)

        def index(c):
            Fh, Fd = np.zeros(B), np.zeros(B)
            for k in range(B):
                Fh[k] = np.sum([w[k, i] if j <= c else 0 for i, j in enumerate(yh)])
                Fd[k] = np.sum([w1[k, i] if j <= c else 0 for i, j in enumerate(yd)])
            return np.sum(Fh - Fd) / B

        score_func = index
    else:
        return None

    best_value = values[0]
    best_score = float('-inf')

    # grid search
    for value in values:
        score = score_func(value)
        if score > best_score:
            best_score = score
            best_value = value

    return {"ThreshHold": best_value, "YI": best_score}
