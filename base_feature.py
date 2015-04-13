#!/usr/bin/python
#-*- coding: utf-8 -*-

__author__ = 'rupy'

import logging
import sys
import os
import numpy as np
import my_util as util
from sklearn.decomposition import PCA

class BaseFeature():

    def __init__(self, data_dir, original_dir=None):

        # log setting
        program = os.path.basename(sys.argv[0])
        self.logger = logging.getLogger(program)
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')

        self.data_dir = data_dir
        if original_dir is None:
            self.original_dir = data_dir
        else:
            self.original_dir = original_dir

        self.feature = None
        self.pca = PCA(n_components=100)
        self.feature_pca = None

    def load_feature(self):
        self.feature = np.load(self.data_dir)
        print self.feature.shape

    def get_train_data(self, step=2):
        self.logger.info(self.feature[::step].shape)

        # select every step-th row
        return self.feature[::step]

    def get_test_data(self, step=2):
        self.logger.info(util.skip_rows_by_step(self.feature, step).shape)

        # select data except for every step-th data
        return util.skip_rows_by_step(self.feature, step)

    def pca_feature(self, n_components=100):
        self.pca.n_components = n_components
        self.feature_pca = self.pca.fit(self.feature).transform(self.feature)
