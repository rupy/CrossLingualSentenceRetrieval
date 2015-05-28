#!/usr/bin/python
#-*- coding: utf-8 -*-

__author__ = 'rupy'

from joint import Joint

import text_features as txt
import image_features as img
import gcca
import logging
from sklearn.neighbors import NearestNeighbors
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import base_feature as feat

class BridgedJoint(Joint):

    OUTPUT_DIR = 'output/'
    CCA_PARAMS_SAVE_DIR = OUTPUT_DIR + 'pca/'

    LINE_BCCA_DIR = CCA_PARAMS_SAVE_DIR + "line_bcca/"
    RAW_BCCA_DIR = CCA_PARAMS_SAVE_DIR + "raw_bcca/"

    def __init__(self, en_dir, img_path, jp_dir, img_original_dir=None, img_correspondence_path=None, jp_original_dir=None, compress_word_dim=100, compress_img_dim=100, line_flag=False):

        Joint.__init__(self, en_dir, img_path, jp_dir, img_original_dir, img_correspondence_path, jp_original_dir, compress_word_dim, compress_img_dim, line_flag)

        self.bcca = gcca.BridgedCCA()

        self.__prep_dir()

    def __prep_dir(self):
        if not os.path.isdir(BridgedJoint.OUTPUT_DIR):
            os.mkdir(BridgedJoint.OUTPUT_DIR)
        if not os.path.isdir(BridgedJoint.CCA_PARAMS_SAVE_DIR):
            os.mkdir(BridgedJoint.CCA_PARAMS_SAVE_DIR)
        if not os.path.isdir(BridgedJoint.LINE_CCA_DIR):
            os.mkdir(BridgedJoint.LINE_CCA_DIR)
        if not os.path.isdir(BridgedJoint.LINE_BCCA_DIR):
            os.mkdir(BridgedJoint.LINE_BCCA_DIR)
        if not os.path.isdir(BridgedJoint.RAW_CCA_DIR):
            os.mkdir(BridgedJoint.RAW_CCA_DIR)
        if not os.path.isdir(BridgedJoint.RAW_BCCA_DIR):
            os.mkdir(BridgedJoint.RAW_BCCA_DIR)
        if not os.path.isdir(BridgedJoint.FEATURE_DIR):
            os.mkdir(BridgedJoint.FEATURE_DIR)

    def __get_bcca_save_dir(self, sample_num, reg_param):
        return (BridgedJoint.LINE_BCCA_DIR if self.line_flag else BridgedJoint.RAW_BCCA_DIR) + str(reg_param).replace(".", "_") + '/' + str(sample_num) + '/'

    def bcca_fit(self, sample_num, reg_param, sampled_indices1, sampled_indices2, sampled_indices3):
        self.logger.info("====== sampling training data ======")
        self.bcca.reg_param = reg_param

        sampled_en1 = self.english_feature.sample_train_feature(sampled_indices1)
        sampled_img1 = self.image_feature.sample_train_feature(sampled_indices1)
        # sampled_jp1 = self.japanese_feature.sample_train_feature(sampled_indices1)

        # sampled_en2 = self.english_feature.sample_train_feature(sampled_indices2)
        sampled_img2 = self.image_feature.sample_train_feature(sampled_indices2)
        sampled_jp2 = self.japanese_feature.sample_train_feature(sampled_indices2)

        sampled_en3 = self.english_feature.sample_train_feature(sampled_indices3)
        sampled_img3 = self.image_feature.sample_train_feature(sampled_indices3)
        sampled_jp3 = self.japanese_feature.sample_train_feature(sampled_indices3)

        self.logger.info("====== fitting by BCCA ======")
        self.logger.info("sample_num:%d reg_param:%f", sample_num, reg_param)
        self.bcca.fit(
            sampled_en1,
            sampled_img1,

            sampled_img2,
            sampled_jp2,

            sampled_en3,
            sampled_jp3,
        )

        save_dir = self.__get_bcca_save_dir(sample_num, reg_param)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        self.bcca.save_params(save_dir + 'params.h5')

    def bcca_transform(self, sample_num, reg_param):
        self.logger.info("====== transforming by BCCA ======")
        self.logger.info("sample_num :%d reg_param:%f", sample_num, reg_param)

        save_dir = self.__get_bcca_save_dir(sample_num, reg_param)
        self.bcca.load_params(save_dir + 'params.h5')

        self.bcca.transform(
            self.english_feature.test_feature_pca,
            self.image_feature.test_feature_pca,
            self.japanese_feature.test_feature_pca
        )

    def bcca_plot(self):
        self.bcca.plot_result('AGCCA')

    def compare_correlation_coefficient(self):
        self.cca.calc_correlations()
        self.bcca.calc_correlations()

    def get_cca_correlatins(self):
        pair_list_cca, cor_list_cca = self.cca.get_correlations()
        return pair_list_cca, cor_list_cca

    def get_bcca_correlatins(self):
        pair_list_bcca, cor_list_bcca = self.bcca.get_correlations()
        return pair_list_bcca, cor_list_bcca


if __name__=="__main__":
    pass