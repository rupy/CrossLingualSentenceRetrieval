#!/usr/bin/python
#-*- coding: utf-8 -*-

__author__ = 'rupy'

import logging
import experiment

if __name__=="__main__":
    logging.root.setLevel(level=logging.INFO)

    ex = experiment.Experiment(False)

    ex.load_pca_features()
    # ex.plot_result(line_flag=False, sample_num=100, reg_param=0.1)

    ex.calc_accuracy_changing_sample_num(sample_num_list=[100, 50, 10])
