#!/usr/bin/python
#-*- coding: utf-8 -*-

__author__ = 'rupy'

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

class Joint():


    OUTPUT_DIR = 'output/'
    CCA_PARAMS_SAVE_DIR = OUTPUT_DIR + 'pca/'

    LINE_GCCA_DIR = CCA_PARAMS_SAVE_DIR + "line_gcca/"
    LINE_CCA_DIR =  CCA_PARAMS_SAVE_DIR + "line_cca/"
    RAW_GCCA_DIR = CCA_PARAMS_SAVE_DIR + "raw_gcca/"
    RAW_CCA_DIR =  CCA_PARAMS_SAVE_DIR + "raw_cca/"

    FEATURE_DIR = OUTPUT_DIR + 'features/'

    def __init__(self, en_dir, img_path, jp_dir, img_original_dir=None, img_correspondence_path=None, jp_original_dir=None, compress_word_dim=100, compress_img_dim=100, line_flag=False):

        # log setting
        program = os.path.basename(__name__)
        self.logger = logging.getLogger(program)
        logging.basicConfig(format='%(asctime)s : %(name)s : %(levelname)s : %(message)s')

        self.english_feature = txt.TextFeatures(en_dir, compress_dim=compress_word_dim, feature_name='eng')
        self.japanese_feature = txt.TextFeatures(jp_dir, jp_original_dir, compress_dim=compress_word_dim)
        self.image_feature = img.ImageFeatures(img_path, img_original_dir, img_correspondence_path, compress_img_dim)
        self.gcca = gcca.GCCA()
        self.cca = gcca.CCA()
        self.line_flag = line_flag

        self.logger.info("===== initializing =====")
        self.logger.info("line_flag:%s", self.line_flag)

        self.__prep_dir()

    def __prep_dir(self):
        if not os.path.isdir(Joint.OUTPUT_DIR):
            os.mkdir(Joint.OUTPUT_DIR)
        if not os.path.isdir(Joint.CCA_PARAMS_SAVE_DIR):
            os.mkdir(Joint.CCA_PARAMS_SAVE_DIR)
        if not os.path.isdir(Joint.LINE_GCCA_DIR):
            os.mkdir(Joint.LINE_GCCA_DIR)
        if not os.path.isdir(Joint.LINE_CCA_DIR):
            os.mkdir(Joint.LINE_CCA_DIR)
        if not os.path.isdir(Joint.RAW_GCCA_DIR):
            os.mkdir(Joint.RAW_GCCA_DIR)
        if not os.path.isdir(Joint.RAW_CCA_DIR):
            os.mkdir(Joint.RAW_CCA_DIR)
        if not os.path.isdir(Joint.FEATURE_DIR):
            os.mkdir(Joint.FEATURE_DIR)

    def create_features(self, feature_file=None):
        self.logger.info("===== creating features =====")
        if self.line_flag:
            self.english_feature.create_bow_feature_with_lines()
            self.japanese_feature.create_bow_feature_with_lines()
            self.image_feature.load_feature_and_copy_line(self.english_feature.line_count, feature_file)
        else:
            self.english_feature.create_bow_feature()
            self.japanese_feature.create_bow_feature()
            self.image_feature.load_original_feature(feature_file)

    def pca_train_and_test_data(self, train_indices, test_indices):
        self.logger.info("===== compressing features by PCA =====")
        self.english_feature.pca_train_and_test_data(train_indices, test_indices)
        self.japanese_feature.pca_train_and_test_data(train_indices, test_indices)
        self.image_feature.pca_train_and_test_data(train_indices, test_indices)

    def __get_cca_save_dir(self, sample_num, reg_param):
        return (Joint.LINE_CCA_DIR if self.line_flag else Joint.RAW_CCA_DIR) + str(reg_param).replace(".", "_") + '/' + str(sample_num) + '/'

    def __get_gcca_save_dir(self, sample_num, reg_param):
        return (Joint.LINE_GCCA_DIR if self.line_flag else Joint.RAW_GCCA_DIR) + str(reg_param).replace(".", "_") + '/' + str(sample_num) + '/'

    def cca_fit(self, sample_num, reg_param, sampled_indices):
        self.logger.info("====== sampling training data ======")

        self.cca.reg_param = reg_param

        sampled_train_data_en = self.english_feature.sample_train_feature(sampled_indices)
        sampled_train_data_jp = self.japanese_feature.sample_train_feature(sampled_indices)

        self.logger.info("====== fitting by CCA ======")
        self.logger.info("sample_num:%d reg_param:%f", sample_num, reg_param)
        self.cca.fit(sampled_train_data_en, sampled_train_data_jp)

        save_dir = self.__get_cca_save_dir(sample_num, reg_param)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        self.cca.save_params(save_dir + 'params.h5')

    def cca_transform(self, sample_num, reg_param):
        self.logger.info("====== transforming by CCA ======")
        self.logger.info("sample_num:%d reg_param:%f", sample_num, reg_param)

        save_dir = self.__get_cca_save_dir(sample_num, reg_param)
        self.cca.load_params(save_dir + 'params.h5')

        self.cca.transform(
            self.english_feature.test_feature_pca,
            self.japanese_feature.test_feature_pca
        )

    def cca_plot(self):
        self.cca.plot_result()

    def gcca_fit(self, sample_num, reg_param, sampled_indices):
        self.logger.info("====== sampling training data ======")
        self.gcca.reg_param = reg_param

        sampled_train_data_en = self.english_feature.sample_train_feature(sampled_indices)
        sampled_train_data_img = self.image_feature.sample_train_feature(sampled_indices)
        sampled_train_data_jp = self.japanese_feature.sample_train_feature(sampled_indices)

        self.logger.info("====== fitting by GCCA ======")
        self.logger.info("sample_num:%d reg_param:%f", sample_num, reg_param)
        self.gcca.fit(sampled_train_data_en, sampled_train_data_img, sampled_train_data_jp)

        save_dir = self.__get_gcca_save_dir(sample_num, reg_param)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        self.gcca.save_params(save_dir + 'params.h5')

    def gcca_transform(self, sample_num, reg_param):
        self.logger.info("====== transforming by GCCA ======")
        self.logger.info("sample_num :%d reg_param:%f", sample_num, reg_param)

        save_dir = self.__get_gcca_save_dir(sample_num, reg_param)
        self.gcca.load_params(save_dir + 'params.h5')

        self.gcca.transform(
            self.english_feature.test_feature_pca,
            self.image_feature.test_feature_pca,
            self.japanese_feature.test_feature_pca
        )

    def gcca_plot(self):
        self.gcca.plot_result()

    def retrieval_j2e_by_gcca(self, j_id, neighbor_num = 10):

        min_dim = 30

        en_mat, im_mat, jp_mat = self.gcca.z_list[0][:, :min_dim], self.gcca.z_list[1][:, :min_dim], self.gcca.z_list[2][:, :min_dim]

        nn = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(en_mat)
        dists, nn_indices = nn.kneighbors([jp_mat[j_id - 1]], neighbor_num, return_distance=True)
        print nn_indices + 1
        print "*%d****************" % (j_id)
        print self.japanese_feature.read_text_by_id(j_id)
        for idx in nn_indices[0]:
            print "=%d=================" % (idx + 1)
            print self.english_feature.read_text_by_id(idx + 1)

    def retrieval_j2e_by_cca(self, j_id, neighbor_num = 10):

        min_dim = 30

        en_mat, jp_mat = self.cca.z_list[0][:, :min_dim], self.cca.z_list[1][:, :min_dim]

        nn = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(en_mat)
        dists, nn_indices = nn.kneighbors([jp_mat[j_id - 1]], neighbor_num, return_distance=True)
        print nn_indices + 1
        print "*%d****************" % (j_id)
        print self.japanese_feature.read_text_by_id(j_id)
        for idx in nn_indices[0]:
            print "=%d=================" % (idx + 1)
            print self.english_feature.read_text_by_id(idx + 1)

    def retrieval_j2i_by_gcca(self, j_id, neighbor_num=10):

        min_dim = 30

        en_mat, im_mat, jp_mat = self.gcca.z_list[0][:, :min_dim], self.gcca.z_list[1][:, :min_dim], self.gcca.z_list[2][:, :min_dim]

        nn = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(im_mat)
        dists, nn_indices = nn.kneighbors([jp_mat[j_id - 1]], neighbor_num, return_distance=True)
        print nn_indices + 1
        print "*%d*****************" % (j_id)
        print self.japanese_feature.read_text_by_id(j_id)
        # for idx in nn_indices[0]:
        #     print "=%d=================" % (idx + 1)
        #     self.image_feature.plot_img_by_id(idx + 1)
        self.image_feature.plot_img_by_ids(nn_indices[0] + 1)

    def compare_correlation_coefficient(self):
        self.cca.calc_correlations()
        self.gcca.calc_correlations()

    def plot_original_data(self):
        """
        plot original two data.
        :return: None
        """

        pca = PCA(n_components=2)
        x = pca.fit_transform(self.english_feature.feature)
        y = pca.fit_transform(self.image_feature.feature)
        z = pca.fit_transform(self.japanese_feature.feature)

        print x[x!=0]
        print y
        print z[z!=0]

        # plot
        plt.subplot(311)
        plt.plot(x[:, 0], x[:, 1], '.r')
        plt.title('X')

        plt.subplot(312)
        plt.plot(y[:, 0], y[:, 1], '.g')
        plt.title('Y')

        plt.subplot(313)
        plt.plot(z[:, 0], z[:, 1], '.b')
        plt.title('Z')
        plt.show()


if __name__=="__main__":
    pass