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

    LINE_FEATURE_FILE = 'line_feature.npy'
    RAW_FEATURE_FILE = 'raw_feature.npy'

    def __init__(self, data_dir, original_dir=None, compress_dim=None):

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
        self.compress_dim = compress_dim
        if compress_dim is not None:
            self.pca = PCA(n_components=compress_dim)
        else:
            self.pca = None
        self.feature_pca = None

    def get_train_data(self, step=2):
        self.logger.info(self.feature[::step].shape)

        # select every step-th row
        return self.feature[::step]

    def get_test_data(self, step=2):
        if step != 1:
            self.logger.info(util.skip_rows_by_step(self.feature, step).shape)

            # select data except for every step-th data
            return util.skip_rows_by_step(self.feature, step)
        else:
            self.logger.info(self.feature.shape)

            # return all data
            return self.feature

    def pca_feature(self):
        self.logger.info("compressing feature")
        return self.pca.fit(self.feature).transform(self.feature)

    def save_processed_feature(self, save_dir, line_flag):
        self.logger.info("saving feature")
        save_file = self.LINE_FEATURE_FILE if line_flag else self.RAW_FEATURE_FILE
        np.save(save_dir + str(self.compress_dim) + '_' + save_file, self.feature)

    def load_processed_feature(self, save_dir, line_flag):
        self.logger.info("load feature")
        save_file = self.LINE_FEATURE_FILE if line_flag else self.RAW_FEATURE_FILE
        self.feature = np.load(save_dir + str(self.compress_dim) + '_' + save_file)
