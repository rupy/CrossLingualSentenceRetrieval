#!/usr/bin/python
#-*- coding: utf-8 -*-

__author__ = 'rupy'

import logging
import bridged_experiment
import numpy as np

if __name__=="__main__":
    logging.root.setLevel(level=logging.INFO)

    ex = bridged_experiment.BridgedExperiment(False)

    # ex.fit_changing_sample_num(sample_num_list=[2000, 1500, 1000, 500])
    #
    # ex.calc_accuracy_changing_sample_num(sample_num_list=[2000, 1500, 1000, 500])

    sample_list = range(0, 100 + 1, 10)
    distribution_list_list = [
        [100, 100, 700, 100],
        [200, 200, 500, 100],
        [300, 300, 300, 100],
        [400, 400, 100, 100],
        ]

    # calc average of fit & result
    try_num = 50
    cca_avg_list = []
    bcca_avg_list = []
    cca_cor_avg_list = []
    bcca_cor_avg_list = []
    for d in distribution_list_list:
        res_cca_list = []
        res_bcca_list = []
        res_cca_cor_list = []
        res_bcca_cor_list = []
        for i in xrange(try_num):
            ex.fit_changing_sample_num(sample_num_list=sample_list, distribution_list=d)
            res_cca, res_bcca = ex.calc_accuracy_changing_sample_num(sample_num_list=sample_list)
            res_cca_list.append(res_cca)
            res_bcca_list.append(res_bcca)

            res_cor_cca_arr, res_cor_bcca_arr = ex.calc_corrcoef_changing_sample_num(sample_num_list=sample_list)
            res_cca_cor_list.append(res_cor_cca_arr)
            res_bcca_cor_list.append(res_cor_bcca_arr)

        res_cca_arr = np.array(res_cca_list)
        res_bcca_arr = np.array(res_bcca_list)
        cca_avg = res_cca_arr
        bcca_avg = res_bcca_arr
        cca_avg_list.append(cca_avg)
        bcca_avg_list.append(bcca_avg)

        # ex.plot_results(cca_avg, bcca_avg, sample_list, 4)
        # ex.plot_max_results(cca_avg, bcca_avg, sample_list)

        res_cca_cor_arr = np.array(res_cca_cor_list)
        res_bcca_cor_arr = np.array(res_bcca_cor_list)
        cca_cor_avg = res_cca_cor_arr.mean(axis=0)
        bcca_cor_avg = res_bcca_cor_arr.mean(axis=0)
        cca_cor_avg_list.append(cca_cor_avg)
        bcca_cor_avg_list.append(bcca_cor_avg)

    ex.plot_max_list_results(cca_avg_list, bcca_avg_list, sample_list)
    ex.plot_cor_list_results(cca_cor_avg_list, bcca_cor_avg_list, sample_list)
    ex.plot_result(sample_num=100, reg_param=0.01)

