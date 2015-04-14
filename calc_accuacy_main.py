#!/usr/bin/python
#-*- coding: utf-8 -*-

__author__ = 'rupy'

import logging
import experiment

if __name__=="__main__":
    logging.root.setLevel(level=logging.INFO)

    ex = experiment.Experiment()

    # line
    # ex.calc_accuracy_changing_step(step_list=range(2, 25, 1), line_flag=True)
    ex.calc_accuracy_changing_step(step_list=range(50, 1001, 50), line_flag=True)

    # raw
    # ex.calc_accuracy_changing_step(step_list=range(2, 25, 1), line_flag=False)
    ex.calc_accuracy_changing_step(step_list=range(25, 201, 25), line_flag=False)
