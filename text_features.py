#!/usr/bin/python
#-*- coding: utf-8 -*-

__author__ = 'rupy'

import gensim
from sklearn.feature_extraction.text import CountVectorizer
import os
import numpy as np
import logging
import sys
import my_util as util

def int_sort(a,b):
    head_a, tail_a = os.path.splitext(a)
    head_b, tail_b = os.path.splitext(b)

    return cmp(int(head_a),int(head_b))

class TextFeatures():

    def __init__(self, data_dir, original_dir= None, min_df=1):

        # log setting
        program = os.path.basename(sys.argv[0])
        self.logger = logging.getLogger(program)
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')

        self.data_dir = data_dir
        if original_dir is None:
            self.original_dir = data_dir
        else:
            self.original_dir = original_dir
        self.vectorizer = CountVectorizer(min_df=min_df)
        self.word_mat = None
        self.terms = None
        self.data_num_label = None
        self.line_count = None

    def create_bow_feature(self):

        dirs = os.listdir(self.data_dir)
        dirs_sorted = sorted(dirs, int_sort)
        corpus = [ open(self.data_dir + file_name, 'r').read() for file_name in dirs_sorted]
        x = self.vectorizer.fit_transform(corpus)
        self.word_mat = x.toarray()
        self.terms = np.array(self.vectorizer.get_feature_names())
        print self.word_mat.shape

    def create_bow_feature_with_lines(self):

        dirs = os.listdir(self.data_dir)
        dirs_sorted = sorted(dirs, int_sort)
        corpus = [ open(self.data_dir + file_name, 'r').readlines() for file_name in dirs_sorted]
        corpus_flatten = reduce(lambda x,y:x+y,corpus) # flatten list
        line_count = [[i + 1] * len(line) for i, line in enumerate(corpus)]
        self.data_num_label = reduce(lambda x,y:x+y,line_count)
        self.line_count = [ len(line) for line in corpus]
        x = self.vectorizer.fit_transform(corpus_flatten)
        self.word_mat = x.toarray()
        self.terms = np.array(self.vectorizer.get_feature_names())
        print self.word_mat.shape

    def get_train_data(self, step=2):
        self.logger.info(self.word_mat[::step].shape)
        # select every step-th row
        return self.word_mat[::step]

    def get_test_data(self, step=2):
        self.logger.info(util.skip_rows_by_step(self.word_mat, step).shape)

        # select data except for every step-th data
        return util.skip_rows_by_step(self.word_mat, step)

    def read_text_by_id(self, text_id):
        text = None
        with open(self.original_dir + "%d.txt" % text_id) as f:
            text = f.read()

        return text

if __name__=="__main__":

    japanese_corpus_dir = '../PascalSentenceDataset/line_wakati/'
    txt = TextFeatures(japanese_corpus_dir)
    txt.create_bow_feature_with_lines()

