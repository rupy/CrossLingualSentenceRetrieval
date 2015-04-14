#!/usr/bin/python
#-*- coding: utf-8 -*-

__author__ = 'rupy'

import logging
import experiment

if __name__=="__main__":
    logging.root.setLevel(level=logging.INFO)

    ex = experiment.Experiment()

    # !!!!!!!IMPORTANT!!!!!!!!!
    # this function must run first
    # if you ran this function run brefore, you don't have to run it again.
    if True:
        ex.process_features()

    # line
    ex.fit_changing_step(step_list=range(1, 25, 1), line_flag=True)
    # ex.fit_changing_step(step_list=range(50, 1001, 50), line_flag=True)

    # raw
    ex.fit_changing_step(step_list=range(1, 25, 1), line_flag=False)
    # ex.fit_changing_step(step_list=range(25, 201, 25), line_flag=False)
