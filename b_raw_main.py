#!/usr/bin/python
#-*- coding: utf-8 -*-

__author__ = 'rupy'

import logging
import bridged_experiment
import numpy as np

if __name__=="__main__":
    logging.root.setLevel(level=logging.INFO)

    ex = bridged_experiment.BridgedExperiment(False)
    ex.process_features()

    # ex.fit_changing_sample_num(sample_num_list=[2000, 1500, 1000, 500])
    #
    # ex.calc_accuracy_changing_sample_num(sample_num_list=[2000, 1500, 1000, 500])

    sample_list = range(0, 100 + 1, 10)

    # calc average of fit & result
    try_num = 1
    res_cca_list = []
    res_bcca_list = []
    for i in xrange(try_num):
        ex.fit_changing_sample_num(sample_num_list=sample_list)
        res_cca, res_bcca = ex.calc_accuracy_changing_sample_num(sample_num_list=sample_list)
        res_cca_list.append(res_cca)
        res_bcca_list.append(res_bcca)
    res_cca_arr = np.array(res_cca_list)
    res_bcca_arr = np.array(res_bcca_list)
    cca_avg = res_cca_arr.mean(axis=0)
    bcca_avg = res_bcca_arr.mean(axis=0)

    ex.plot_results(cca_avg, bcca_avg, sample_list, 5)
    ex.plot_max_results(cca_avg, bcca_avg, sample_list)
    ex.plot_result(sample_num=100, reg_param=0.01)

