#!/usr/bin/python
#-*- coding: utf-8 -*-

__author__ = 'rupy'

import logging
import experiment

if __name__=="__main__":
    logging.root.setLevel(level=logging.INFO)

    ex = experiment.Experiment(False)
    ex.process_features()

    # ex.fit_changing_sample_num(sample_num_list=[2000, 1500, 1000, 500])
    #
    # ex.calc_accuracy_changing_sample_num(sample_num_list=[2000, 1500, 1000, 500])

    sample_list = range(20, 501, 20)

    ex.fit_changing_sample_num(sample_num_list=sample_list)

    ex.calc_accuracy_changing_sample_num(sample_num_list=sample_list)
    ex.plot_result(sample_num=500, reg_param=0.1)

