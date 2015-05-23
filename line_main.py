#!/usr/bin/python
#-*- coding: utf-8 -*-

__author__ = 'rupy'

import logging
import experiment

if __name__=="__main__":
    logging.root.setLevel(level=logging.INFO)

    ex = experiment.Experiment(True)
    ex.process_features()

    sample_list = range(100, 2500, 100)

    ex.fit_changing_sample_num(sample_num_list=sample_list)

    res_cca, res_gcca = ex.calc_accuracy_changing_sample_num(sample_num_list=sample_list)
    ex.plot_results(res_cca, res_gcca, sample_list, 5)
    ex.plot_result(sample_num=2400, reg_param=0.1)

