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

    def __init__(self, data_dir, original_dir=None, compress_dim=None, feature_name='base'):

        # log setting
        program = os.path.basename(__name__ + "(" + feature_name + ")")
        self.logger = logging.getLogger(program)
        logging.basicConfig(format='%(asctime)s : %(name)s : %(levelname)s : %(message)s')

        self.data_dir = data_dir
        if original_dir is None:
            self.original_dir = data_dir
        else:
            self.original_dir = original_dir

        self.feature_name = feature_name

        self.feature = None
        self.compress_dim = compress_dim
        if compress_dim is not None:
            self.pca = PCA(n_components=compress_dim)
        else:
            self.pca = None
        self.train_feature_pca = None
        self.test_feature_pca = None

        self.train_indices = None
        self.test_indices = None

        self.sampled_train_indices = None
        self.sampled_train_feature = None

        self.labels = None

    def __get_train_data(self, train_indices):
        # select every step-th row
        train_feature = self.feature[train_indices]
        if self.pca is not None:
            train_feature_pca = self.__pca_feature(train_feature, True)
            self.logger.info("train data shape:%s => %s", train_feature.shape, train_feature_pca.shape)
            return train_feature_pca
        else:
            self.logger.info("train data shape:%s", train_feature.shape)
            return train_feature

    def __get_test_data(self, test_indices):

            # select data except for every step-th data
        test_feature = self.feature[test_indices]

        if self.pca is not None:
            test_feature_pca = self.__pca_feature(test_feature, False)
            self.logger.info("test data shape:%s => %s", test_feature.shape, test_feature_pca.shape)
            return test_feature_pca
        else:
            self.logger.info("test data shape:%s", test_feature.shape)
            return test_feature

    def pca_train_and_test_data(self, train_indices, test_indices):
        self.train_feature_pca = self.__get_train_data(train_indices)
        self.train_indices = np.array(train_indices)
        self.test_feature_pca = self.__get_test_data(test_indices)
        self.test_indices = np.array(test_indices)

    def __pca_feature(self, feature, fit_flag):
        self.logger.info("compressing feature by compress_dim: %d", self.compress_dim)
        if self.compress_dim > feature.shape[0]:
            self.logger.warn("sample num %d is less than compress_dim %d. so pca cannot compress feature because of lacking of rank. use original feature.", feature.shape[0], self.compress_dim)
            return feature
        else:
            if fit_flag:
                self.pca.fit(feature)
            return self.pca.transform(feature)

    # def save_pca_data(self, feature_dir, prefix):
    #     train_data_pca = self.get_train_data(2)
    #     test_data_pca = self.get_test_data(2)
    #     self.logger.info("saving pca feature")
    #     np.save(feature_dir + prefix + '_pca_train.npy', train_data_pca)
    #     np.save(feature_dir + prefix + '_pca_test.npy', test_data_pca)
    #
    # def load_pca_data(self, feature_dir, prefix):
    #     self.logger.info("loading pca feature")
    #     train_data_pca = np.load(feature_dir + prefix + '_pca_train.npy')
    #     test_data_pca = np.load(feature_dir + prefix + '_pca_test.npy')
    #     self.train_feature_pca = train_data_pca
    #     self.test_feature_pca = test_data_pca
    #     self.logger.info("train data shape:%s", self.train_feature_pca.shape)
    #     self.logger.info("test data shape:%s", self.test_feature_pca.shape)
    #

    def get_label_indices(self, sampled_indices):
        return np.array(reduce(lambda x,y: x+y, [np.where(self.train_indices == i)[0].tolist() for i in sampled_indices]))

    def sample_train_feature(self, sampled_indices):
        if len(sampled_indices) != 0:
            sampled_label_indices = self.get_label_indices(sampled_indices)
        else:
            sampled_label_indices = []
        print self.train_feature_pca.shape
        self.sampled_train_feature = self.train_feature_pca[sampled_label_indices]
        self.sampled_train_indices = self.train_indices[sampled_label_indices]
        self.logger.info("sampled train data shape:%s => %s", self.train_feature_pca.shape, self.sampled_train_feature.shape)

        return self.sampled_train_feature

    def get_train_data_num(self):
        return self.train_feature_pca.shape[0]

    @staticmethod
    def sample_indices(data_num, sample_num):
        all_indices = range(data_num)
        sampled_indices = np.random.choice(all_indices, sample_num, False)
        return sorted(sampled_indices)

    @staticmethod
    def sample_all_indices(data_num):
        all_indices = range(data_num)
        return all_indices

    @staticmethod
    def all_indices1(data_num):
        all_indices = range(0, data_num, 5) + range(2, data_num, 5)
        return sorted(all_indices)

    @staticmethod
    def all_indices2(data_num):
        all_indices = range(1, data_num, 5) + range(3, data_num, 5)
        return sorted(all_indices)

    @staticmethod
    def sample_indices3(data_num, sample_num):
        all_indices = range(4, data_num, 5)
        sampled_indices = np.random.choice(all_indices, sample_num, False)
        return sorted(sampled_indices)

    @staticmethod
    def sample_indices(data_num, cut_list, print_flag=True):
        indices = np.arange(0, data_num)
        split_indices = util.random_split_list(indices, cut_list)
        if print_flag:
            rate_list = [ 1.0 * c / sum(cut_list)for c in cut_list]
            print "+------------------------------------+"
            print "|    \   |TRAIN1|TRAIN2|TRAIN3| TEST |"
            print "+--------+------+------+------+------+"
            print "|data_num|%6d|%6d|%6d|%6d|" % tuple(cut_list)
            print "|rate    |%5d%%|%5d%%|%5d%%|%5d%%|" % tuple([int(100 * r) for r in rate_list])
            print "+------------------------------------+"
        return split_indices

    @staticmethod
    def sample_indices_with_ratio_list(data_num, ratio_list, print_flag=True):
        denominator = sum(ratio_list)
        rate_list = [ ratio / float(denominator) for ratio in ratio_list]
        cut_list = [ int(data_num * rate) for rate in rate_list]
        if print_flag:
            print "+------------------------------------+"
            print "|    \   |TRAIN1|TRAIN2|TRAIN3| TEST |"
            print "+--------+------+------+------+------+"
            print "|data_num|%6d|%6d|%6d|%6d|" % tuple(cut_list)
            print "|rate    |%5d%%|%5d%%|%5d%%|%5d%%|" % tuple([int(100 * r) for r in rate_list])
            print "+------------------------------------+"
        return BaseFeature.sample_indices(data_num, cut_list, False)

    @staticmethod
    def sample_test(indices, sample_num):
        sampled_indices = np.random.choice(indices, sample_num, False)
        return sorted(sampled_indices)
if __name__ == '__main__':

    print BaseFeature.sample_indices_with_ratio_list(10, [1, 2])

