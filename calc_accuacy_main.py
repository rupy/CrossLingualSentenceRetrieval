#!/usr/bin/python
#-*- coding: utf-8 -*-

__author__ = 'rupy'

import logging
import experiment

if __name__=="__main__":
    logging.root.setLevel(level=logging.INFO)

    ex = experiment.Experiment()

    ex.load_pca_features(False)
    # ex.plot_result(line_flag=False, sample_num=100, reg_param=0.1)

    # line
    # ex.calc_accuracy_changing_step(step_list=range(2, 25, 1), line_flag=True)
    # ex.calc_accuracy_changing_sample_num(step_list=range(50, 1001, 50), line_flag=True)

    # raw
    # ex.calc_accuracy_changing_step(step_list=range(2, 25, 1), line_flag=False)
    ex.calc_accuracy_changing_sample_num(sample_num_list=[100, 50, 10], line_flag=False)
