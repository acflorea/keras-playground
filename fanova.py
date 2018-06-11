# -*- coding: utf-8 -*-
"""
Test for the old fANOVA version
"""
import sys
import numpy as np

from pyfanova.fanova import Fanova
from pyfanova.fanova_from_csv import FanovaFromCSV

import os

path = os.path.dirname(os.path.realpath(__file__))

'''
Online LDA example
'''


def online_lda():
    marginals = []
    f = Fanova(path + '/old_online_lda')
    marginals.append(f.get_pairwise_marginal(0, 1))
    marginals.append(f.get_pairwise_marginal(1, 2))
    marginals.append(f.get_marginal(0))
    marginals.append(f.get_marginal(1))
    marginals.append(f.get_marginal(2))
    marginals.append(f.get_pairwise_marginal(0, 2))

    return marginals


def csv_example():
    marginals = []
    f = FanovaFromCSV("data/heart.csv")
    # f = FanovaFromCSV("data/optim_6conv_pooling_50epoch_na.out.parsed.csv")
    # f = FanovaFromCSV("data/optim_6conv_pooling_50epoch_na.out.parsed_100.csv")

    # marginals.append(f.get_all_pairwise_marginals())
    # marginals.append(f.get_marginal(0))
    # marginals.append(f.get_marginal(1))
    # marginals.append(f.get_marginal(2))
    # marginals.append(f.get_marginal(3))
    # marginals.append(f.get_marginal(4))
    # marginals.append(f.get_marginal(5))
    # marginals.append(f.get_marginal(6))
    # marginals.append(f.get_marginal(7))
    # marginals.append(f.get_marginal(8))
    # marginals.append(f.get_marginal(9))
    # marginals.append(f.get_marginal(10))
    # marginals.append(f.get_marginal(11))

    f.print_all_marginals(1000)

    return marginals

if __name__ == "__main__":
    example = sys.argv[1]
    if example == 'online_lda':
        res = online_lda()
    elif example == 'csv_example':
        res = csv_example()
    print(res)
