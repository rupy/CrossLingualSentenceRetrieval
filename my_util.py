#!/usr/bin/python
#-*- coding: utf-8 -*-

__author__ = 'rupy'
import numpy as np

def range_skip_every_nth_idx(end, step):

    # indices except for every step-th data
    indices = sorted(reduce(lambda x, y: x + y, [range( i + 1 ,end , step) for i in xrange(step - 1)]))
    return np.array(indices)

def skip_rows_by_step(arr, step):

    # select rows except for every step-th row
    indices = range_skip_every_nth_idx(len(arr), step)
    return arr[indices], indices

def skip_rows_by_step2(arr, step):

    # select rows except for every step-th row
    indices_step_set = set(range(0, len(arr), step))
    indices_all_set = set(range(0, len(arr)))
    indices = np.array(sorted(list(indices_all_set - indices_step_set)))
    return arr[indices], indices
