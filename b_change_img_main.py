#!/usr/bin/python
#-*- coding: utf-8 -*-

__author__ = 'rupy'

import logging
import bridged_experiment
import numpy as np

if __name__=="__main__":
    logging.root.setLevel(level=logging.INFO)

    line_flag = False
    try_num = 50
    ex = bridged_experiment.BridgedExperiment(line_flag)

    if line_flag:
        sample_list = range(0, 500 + 1, 50)
        distribution_list = [2000, 2000, 500, 498]
    else:
        sample_list = range(0, 100 + 1, 10)
        distribution_list = [400, 400, 100, 100]

    feature_list = ["pascal_features_g.npy", "pascal_features.npy"]
    feature_name_list = ["CNN(GoogLeNet)", "CNN(CaffeNet)"]

    # calc average of fit & result
    bcca_avg_list = []
    bcca_cor_avg_list = []
    for feature_file in feature_list:
        res_bcca_list = []
        res_bcca_cor_list = []
        for i in xrange(try_num):
            ex.fit_changing_sample_num(sample_num_list=sample_list, distribution_list=distribution_list, feature_file=feature_file)
            res_cca, res_bcca = ex.calc_accuracy_changing_sample_num(sample_num_list=sample_list)
            res_bcca_list.append(res_bcca)

            res_cor_cca_arr, res_cor_bcca_arr = ex.calc_corrcoef_changing_sample_num(sample_num_list=sample_list)
            res_bcca_cor_list.append(res_cor_bcca_arr)

        res_bcca_arr = np.array(res_bcca_list)
        bcca_avg = res_bcca_arr
        bcca_avg_list.append(bcca_avg)

        res_bcca_cor_arr = np.array(res_bcca_cor_list)
        bcca_cor_avg = res_bcca_cor_arr.mean(axis=0)
        bcca_cor_avg_list.append(bcca_cor_avg)

    ex.plot_max_list_img_changing(bcca_avg_list, sample_list, feature_name_list)

