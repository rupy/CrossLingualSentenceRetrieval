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

def random_split_list(lst, cut_list):

    # shuffle
    if isinstance(lst, list):
        lst_cpy = list(lst)
    elif isinstance(lst, np.ndarray):
        lst_cpy = lst.copy()

    np.random.shuffle(lst_cpy)

    # create cut indices
    cut_indices_list = [0] + [reduce(lambda x, y: x + y, cut_list[0:(i+1)]) for i, c in enumerate(cut_list)]

    if sum(cut_list) != len(lst_cpy):
        cut_indices_list[-1] = len(lst_cpy)
    # print cut_indices_list
    split_list = []

    # split
    list_len = len(cut_indices_list)
    for cut_from, cut_to in zip(cut_indices_list[:list_len-1], cut_indices_list[1:list_len]):
        partial_lst = lst_cpy[cut_from:cut_to]
        split_list.append(partial_lst)

    return split_list

if __name__ == '__main__':


    a = np.arange(0, 100).reshape(10, 10)
    print random_split_list(a, [2, 2, 6])




