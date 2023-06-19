from rocnreg import kernel_roc, empirical_roc, bayesian_bootstrap_roc
import pyreadr
import matplotlib.pyplot as plt
import numpy as np

# test
df = pyreadr.read_r('example_data/diabetes.RData')["diabetes"]
yh = df.loc[df['status'] == 0, 'marker'].values.ravel()
yd = df.loc[df['status'] == 1, 'marker'].values.ravel()


def test_kernel_roc(yh, yd, p=np.linspace(0, 1, 101)):
    test = kernel_roc(yh=yh, yd=yd, p=p)
    plt.plot(test['p'], test['ROC'])
    plt.ylim(0, 1.1)
    plt.xlim(0, 1.1)
    plt.show()


def test_empirical_roc(yh, yd, p=np.linspace(0, 1, 101)):
    test = empirical_roc(yh, yd, p=p)
    plt.plot(test['p'], test['ROC'])
    plt.ylim(0, 1.1)
    plt.xlim(0, 1.1)
    plt.title("Empirical ROC")
    plt.xlabel("p")
    plt.ylabel("ROC(p)")
    plt.show()


def test_bayesian_bootstrap_roc(yh, yd, p=np.linspace(0, 1, num=101), B=5000):
    test = bayesian_bootstrap_roc(yh=yh, yd=yd, p=p, B=B)
    plt.plot(test['p'], test['ROC'], 'b-', label='ROC')
    plt.plot(test['p'], test['ROCL'], 'r--', label='ROC Lower')
    plt.plot(test['p'], test['ROCH'], 'g--', label='ROC Upper')
    plt.ylim(0, 1.1)
    plt.xlim(0, 1.1)
    plt.legend()
    plt.show()
