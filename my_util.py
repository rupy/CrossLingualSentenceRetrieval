#!/usr/bin/python
#-*- coding: utf-8 -*-

__author__ = 'rupy'

def range_skip_every_nth_idx(end, step):

    # indices except for every step-th data
    indices = sorted(reduce(lambda x, y: x + y, [range( i + 1 ,end , step) for i in xrange(step - 1)]))
    return indices

def skip_rows_by_step(arr, step):

    # select rows except for every step-th row
    indices = range_skip_every_nth_idx(len(arr), step)
    return arr[indices]
