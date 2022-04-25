# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from scipy.stats import gaussian_kde
from best.modules import ZScoreModule
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, average_precision_score
from sklearn.neighbors import KernelDensity

def kl_divergence(mu1, std1, mu2, std2):
    """
    Parametric KL-Divergence between 2 normal 1-D distributions.

    `Normal Distribution <https://en.wikipedia.org/wiki/Normal_distribution>`_


    """
    return 0.5 * ((std1/std2)**2 + ((mu2-mu1)**2 / std2**2) -1 + 2*np.log(std2/std1))


def kl_divergence_mv(mu1, var1, mu2, var2):
    """
    Multidimensional parametric KL-Divergence between 2 normal distributions.

    `KL-Divergence <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions>`_

    `Trace <https://en.wikipedia.org/wiki/Trace_(linear_algebra)>`_

    """
    return 0.5 * ((np.trace(np.dot(np.linalg.inv(var2), var1))) + np.dot(np.dot((mu2 - mu1), np.linalg.inv(var2)), (mu2-mu1).T) - mu1.shape[1] + np.log(np.linalg.det(var2)/np.linalg.det(var1)))[0, 0]


def combine_gauss_distributions(mu1, std1, N1, mu2, std2, N2):
    """
    Recalculates a normal 1-D distribution given two subsets of data.
    """

    c1 = N1 / (N1 + N2)
    c2 = N2 / (N1 + N2)
    mu_combined = (mu1 * c1) + (mu2 * c2)
    std_combined = np.sqrt(
        (N1*std1**2 + N2*std2**2 + N1*((mu1 - mu_combined)**2) + N2*((mu2 - mu_combined)**2)) / (N1+N2)
    ) #
    # np.sqrt((N1*(std1**2) + N2*(std2**2) + N1*N2*(mu2-mu1)**2/(N1+N2)) / (N1+N2)) # https://prod-ng.sandia.gov/techlib-noauth/access-control.cgi/2008/086212.pdf - 1.4
    return mu_combined, std_combined


def combine_mvgauss_distributions(mu1, var1, N1, mu2, var2, N2):
    """
    Recalculates a normal n-D distribution given two subsets of data.
    """
    c1 = N1 / (N1 + N2)
    c2 = N2 / (N1 + N2)
    mu_combined = (mu1 * c1) + (mu2 * c2)
    var_combined = (N1*(var1) + N2*(var2) + N1*N2*(mu2-mu1)**2/(N1+N2)) / (N1+N2) # np.sqrt((N1*(std1**2) + N2*(std2**2) + N1*N2*(mu2-mu1)**2/(N1+N2)) / (N1+N2)) # https://prod-ng.sandia.gov/techlib-noauth/access-control.cgi/2008/086212.pdf
    for k1 in range(var_combined.shape[0]):
        for k2 in range(k1+1, var_combined.shape[0]):
            var_combined[k2, k1] = var_combined[k1, k2]
    return mu_combined, var_combined


def kl_divergence_nonparametric(pk, qk):
    """
    Calculates non-parametric KL-Divergence between two 1-D distributions given by 2 histograms with same bins.
    """
    l_ = pk / qk
    barr = (~np.isinf(l_)) & (~np.isnan(l_))
    return np.nansum(pk[barr] * np.log(l_[barr]))


def get_class_count(Y, classes=None):
    """
    Returns a number of class appearance in the labels
    """
    if isinstance(classes, type(None)):
        classes = np.unique(Y)
        classes.sort()
    return dict([(cl, Y==cl) for cl in classes])


def compare_datasets(dataset, states=['AWAKE', 'N2', 'N3', 'REM']):
    """
    Compares dataset consistency using KDE and prec-recall curves

    Parameters
    ----------
    dataset: dict
        dict where the key is name of dataset for comparison; each dataset is represented by dict with 2 variables X, Y

    Returns
    -------
    dataset: dict
    """
    # AUPRC, AUROC, likelihood
    # sturcture
    # dataset_name - measure_name: AUPRC, AUROC, av likelihood_per_class, av_likelihood, class_frequency - class

    scores = {}
    for idx1, k1 in enumerate(dataset.keys()):
        for idx2, k2 in enumerate(dataset.keys()):
            x1 = dataset[k1]['X']
            y1 = dataset[k1]['Y']
            x2 = dataset[k2]['X']
            y2 = dataset[k2]['Y']

            #nm = k1 + ' - ' + k2

            ZS = ZScoreModule(trainable=True)
            x1 = ZS.fit_transform(x1)
            x2 = ZS.transform(x2)

            kde = KernelDensity().fit(x1)
            #y1_ = kde.score_samples(x1)
            y2_ = kde.score_samples(x2)

            if not k1 in scores.keys(): scores[k1] = {}

            if not k2 in scores[k1].keys():
                scores[k1][k2] = {}
                scores[k1][k2]['loglikelihood'] = {}
                scores[k1][k2]['auroc'] = {}
                #scores[k1][k2]['auprc'] = {}
                scores[k1][k2]['ap'] = {}

            scores[k1][k2]['loglikelihood']['all'] = y2_.mean()

            for cl in states:
                kde = KernelDensity().fit(x1[y1 == cl])
                sc = kde.score_samples(x2)
                #pr, rc, thresholds = precision_recall_curve(y2 == cl, sc)
                #pr = pr[1:]
                #rc = rc[1:]
                #auprc = auc(rc, pr)
                auroc = roc_auc_score(y2 == cl, sc)
                ap = average_precision_score(y2 == cl, sc)
                #scores[k1][k2]['auprc'][cl] = auprc
                scores[k1][k2]['auroc'][cl] = auroc
                scores[k1][k2]['ap'][cl] = ap
                scores[k1][k2]['loglikelihood'][cl] = sc.mean()
    return scores