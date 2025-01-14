"""
This code is generated by Ridvan Salih KUZU @UNIROMA3
LAST EDITED:  03.04.2020
ABOUT SCRIPT:
It is a script for performance visualization functions.
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import auc

def plot_roc(fpr, tpr, figure_name="roc.png"):
    plt.switch_backend('Agg')

    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='red',
             lw=lw, label='ROC curve (area = %0.8f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='blue', lw=lw, linestyle='--')
    # plt.xlim([-0.01, 1.0])
    x_labels = [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]
    plt.xlim([10 ** -6, 0.1])
    plt.ylim([0., 1.01])
    plt.grid(linestyle='--', linewidth=1)
    plt.xticks(x_labels)
    plt.yticks(np.linspace(0.3, 1.0, 8, endpoint=True))
    plt.xscale('log')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    fig.savefig(figure_name, dpi=fig.dpi)


def plot_DET_with_EER(far, frr, far_optimum, frr_optimum, figure_name):
    """ Plots a DET curve with the most suitable operating point based on threshold values"""
    fig = plt.figure()
    lw = 2
    # Plot the DET curve based on the FAR and FRR values
    EER = float((far_optimum + frr_optimum) / 2)
    plt.plot([0, 1], [0, 1], color='blue', lw=lw, linestyle='--')
    plt.plot(far, frr, color='red', linewidth=lw, label='DET Curve (EER = %0.8f)' % EER)
    # Plot the optimum point on the DET Curve
    plt.plot(far_optimum, frr_optimum, "ko", label="Suitable Operating Point")

    plt.xlim([-0.01, 0.1])
    plt.ylim([-0.01, 0.1])
    plt.xlabel('False Acceptance Rate')
    plt.ylabel('False Rejection Rate')
    plt.title('Detection Error Tradeoff')
    plt.legend(loc="upper right")
    plt.grid(True)
    fig.savefig(figure_name, dpi=fig.dpi)


def plot_density(distances, labels, figure_name):
    fig = plt.figure()
    pos_index = np.where(labels == 1)
    neg_index = np.where(labels == 0)
    p1 = sns.distplot(distances[pos_index], kde=True, norm_hist=False, bins=50, color="r", label="Genuine")
    p1 = sns.distplot(distances[neg_index], kde=True, norm_hist=False, bins=50, color="b", label="Impostor", )
    locs, labels = plt.yticks()
    plt.yticks(locs, list(map(str, locs * 0.01)))
    plt.ylabel('Density Distribution [%]', fontsize=18)
    plt.xlabel('Similarity Distance', fontsize=18)
    fig.savefig(figure_name)
