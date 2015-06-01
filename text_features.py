#!/usr/bin/python
#-*- coding: utf-8 -*-

__author__ = 'rupy'

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import numpy as np
from base_feature import BaseFeature

def int_sort(a,b):
    head_a, tail_a = os.path.splitext(a)
    head_b, tail_b = os.path.splitext(b)

    return cmp(int(head_a),int(head_b))

class TextFeatures(BaseFeature):

    LINE_FEATURE_FILE = 'line_txt_feature.npy'
    RAW_FEATURE_FILE = 'raw_txt_feature.npy'

    def __init__(self, data_dir, original_dir= None, min_df=1, compress_dim=None, feature_name='txt'):

        BaseFeature.__init__(self, data_dir, original_dir, compress_dim, feature_name)

        self.vectorizer = CountVectorizer(min_df=min_df)

        self.terms = None
        self.data_num_label = None
        self.line_count = None

    def create_bow_feature(self):
        self.logger.info("creating bow feature")

        # read text files
        dirs = os.listdir(self.data_dir)
        dirs_sorted = sorted(dirs, int_sort)
        corpus = [ open(self.data_dir + file_name, 'r').read() for file_name in dirs_sorted]

        # create features
        x = self.vectorizer.fit_transform(corpus)
        self.feature = x.toarray()

        # store additional information
        self.terms = np.array(self.vectorizer.get_feature_names())
        self.labels = np.arange(0, self.feature.shape[0])

        self.logger.info("created feature: %s", self.feature.shape)

    def create_bow_feature_with_lines(self):
        self.logger.info("creating bow feature as line data")

        # read text files
        dirs = os.listdir(self.data_dir)
        dirs_sorted = sorted(dirs, int_sort)
        corpus = [ open(self.data_dir + file_name, 'r').readlines() for file_name in dirs_sorted]

        # create line count data
        corpus_flatten = reduce(lambda x,y:x+y,corpus) # flatten list
        line_count = [[i + 1] * len(line) for i, line in enumerate(corpus)]
        self.data_num_label = reduce(lambda x,y:x+y,line_count)
        self.line_count = [ len(line) for line in corpus]
        self.logger.info("line num: %d", len(corpus_flatten))

        # create features
        x = self.vectorizer.fit_transform(corpus_flatten)
        self.feature = x.toarray()

        # store additional information
        self.terms = np.array(self.vectorizer.get_feature_names())
        raw_labels = np.arange(0, len(corpus))
        self.labels = raw_labels.repeat(self.line_count, axis=0)

        self.logger.info("created feature: %s", self.feature.shape)

    def read_text_by_id(self, text_id):
        text = None
        with open(self.original_dir + "%d.txt" % text_id) as f:
            text = f.read()

        return text

if __name__=="__main__":

    japanese_corpus_dir = '../PascalSentenceDataset/line_wakati/'
    txt = TextFeatures(japanese_corpus_dir)
    txt.create_bow_feature_with_lines()

