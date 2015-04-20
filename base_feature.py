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
        logging.basicConfig(format='%(asctime)s : %(filename)s : %(levelname)s : %(message)s')

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
        self.train_feature_pca = None
        self.test_feature_pca = None

    def get_train_data(self, step=2):
        # select every step-th row
        train_feature = self.feature[::step]
        if self.pca is not None:
            train_feature_pca = self.pca_feature(train_feature)
            self.logger.info("train data shape:%s => %s", train_feature.shape, train_feature_pca.shape)
            return train_feature_pca
        else:
            self.logger.info("train data shape:%s", train_feature.shape)
            return train_feature

    def get_test_data(self, step=2):

        if step != 1:
            # select data except for every step-th data
            test_feature = util.skip_rows_by_step(self.feature, step)
            if self.pca is not None:
                test_feature_pca = self.pca_feature(test_feature)
                self.logger.info("test data shape:%s => %s", test_feature.shape, test_feature_pca.shape)
                return test_feature_pca
            else:
                self.logger.info("test data shape:%s", test_feature.shape)
                return test_feature
        else:
            if self.pca is not None:
                test_feature_pca = self.pca_feature(self.feature)
                self.logger.info("test data shape:%s => %s", self.feature.shape, test_feature_pca.shape)
                # return all data
                return test_feature_pca
            else:
                self.logger.info("test data shape:%s", self.feature.shape)
                # return all data
                return self.feature

    def pca_feature(self, feature):
        self.logger.info("compressing feature by compress_dim: %d", self.compress_dim)
        if self.compress_dim > feature.shape[0]:
            self.logger.warn("sample num %d is less than compress_dim %d. so pca cannot compress feature because of lacking of rank. use original feature.", feature.shape[0], self.compress_dim)
            return feature
        else:
            return self.pca.fit_transform(feature)

    def save_pca_data(self, feature_dir, prefix):
        train_data_pca = self.get_train_data(2)
        test_data_pca = self.get_test_data(2)
        self.logger.info("saving pca feature")
        np.save(feature_dir + prefix + '_pca_train.npy', train_data_pca)
        np.save(feature_dir + prefix + '_pca_test.npy', test_data_pca)

    def load_pca_data(self, feature_dir, prefix):
        self.logger.info("loading pca feature")
        train_data_pca = np.load(feature_dir + prefix + '_pca_train.npy')
        test_data_pca = np.load(feature_dir + prefix + '_pca_test.npy')
        self.train_feature_pca = train_data_pca
        self.test_feature_pca = test_data_pca
        self.logger.info("train data shape:%s", self.train_feature_pca.shape)
        self.logger.info("test data shape:%s", self.test_feature_pca.shape)


    def sample_random_train_data(self, sample_num):
        all_indices = range(self.train_feature_pca.shape[0])
        sampled_indices = np.random.choice(all_indices, sample_num, False)
        sampled_data = self.train_feature_pca[sampled_indices]
        self.logger.info(" sampled train data shape:%s => %s", self.train_feature_pca.shape, sampled_data.shape)
        return sampled_data

    def sample_random_test_data(self, sample_num):
        all_indices = range(self.test_feature_pca.shape[0])
        sampled_indices = np.random.choice(all_indices, sample_num, False)
        sampled_data = self.test_feature_pca[sampled_indices]
        self.logger.info(" sampled test data shape:%s => %s", self. test_feature_pca.shape, sampled_data.shape)
        return sampled_data
