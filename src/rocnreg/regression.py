from scipy.stats import norm
import statsmodels.api as sm
import numpy as np


def cs_roc(marker, group, tag_h, cov_sp, covs, data, p=np.linspace(0, 1, 101)):
    """
    Induced semi-parametric linear model for estimating the covariate specific ROC

    @:param marker  A character string with the name of the diagnostic test variable in the dataframe.
    @:param group   A character string with the name of the variable that distinguishes healthy from
    diseased individuals.
    @:param tag_h   A value assigned to the non-diseased individuals in the group column
    @:param cov_sp  A vector of specific covariates
    @:param covs    A vector of strings of (all) the covariates (column name)
    @:param data    A data frame containing all the variables needed
    @:param p       A vector of probability default 0.0, 0.1, 0.2, ..., 1.0

    @:return        A list containing the p-values and the corresponding ROC
    """

    def load_data():
        # outcomes for healthy and diseased
        yh = data.loc[data[group] == tag_h, marker].values.ravel()
        yd = data.loc[data[group] != tag_h, marker].values.ravel()

        # covariate(s)
        xh = data.loc[data[group] == tag_h, covs].values
        xd = data.loc[data[group] != tag_h, covs].values

        return yh, yd, xh, xd

    def estimate_ols(xh, xd):
        # OLS for coefficient estimation
        xh, xd = sm.add_constant(xh), sm.add_constant(xd) #intercept
        model_h = sm.OLS(yh, xh).fit()
        model_d = sm.OLS(yd, xd).fit()

        coeff_h = model_h.params
        coeff_d = model_d.params

        return xh, xd, coeff_h, coeff_d

    yh, yd, xh, xd = load_data()
    xh, xd, coeff_h, coeff_d = estimate_ols(xh, xd)
    q = len(covs)

    # estimated variance of residuals
    var_h = np.sum((yh - np.dot(xh, coeff_h)) ** 2) / (len(yh) - q - 1)
    var_d = np.sum((yd - np.dot(xd, coeff_d)) ** 2) / (len(yd) - q - 1)

    # Roc
    c = np.insert(cov_sp, 0, 1)
    a = np.dot(c, (coeff_h - coeff_d) / np.sqrt(var_d))
    b = np.sqrt(var_h / var_d)
    roc = 1 - norm.cdf(a + b * norm.ppf(1 - p))

    return {'p':p, 'ROC':roc}
