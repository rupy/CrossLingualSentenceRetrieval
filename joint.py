#!/usr/bin/python
#-*- coding: utf-8 -*-

__author__ = 'rupy'

import text_features as txt
import image_features as img
import gcca
import cca
import logging
from sklearn.neighbors import NearestNeighbors
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

class Joint():


    OUTPUT_DIR = 'output/'
    CCA_PARAMS_SAVE_DIR = OUTPUT_DIR + 'pca/'

    LINE_GCCA_DIR = CCA_PARAMS_SAVE_DIR + "line_gcca/"
    LINE_CCA_DIR =  CCA_PARAMS_SAVE_DIR + "line_cca/"
    RAW_GCCA_DIR = CCA_PARAMS_SAVE_DIR + "raw_gcca/"
    RAW_CCA_DIR =  CCA_PARAMS_SAVE_DIR + "raw_cca/"

    FEATURE_DIR = OUTPUT_DIR + 'features/'

    def __init__(self, en_dir, img_path, jp_dir, img_original_dir=None, img_correspondence_path=None, jp_original_dir=None, compress_dim=100):

        # log setting
        program = os.path.basename(sys.argv[0])
        self.logger = logging.getLogger(program)
        logging.basicConfig(format='%(asctime)s : %(filename)s : %(levelname)s : %(message)s')

        self.english_feature = txt.TextFeatures(en_dir, compress_dim=compress_dim)
        self.japanese_feature = txt.TextFeatures(jp_dir, jp_original_dir, compress_dim=compress_dim)
        self.image_feature = img.ImageFeatures(img_path, img_original_dir, img_correspondence_path, compress_dim)
        self.gcca = gcca.GCCA()
        self.cca = cca.CCA()

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

    def save_pca_features(self, line_flag):
        self.english_feature.save_pca_data(Joint.FEATURE_DIR, 'eng_' + 'line' if line_flag else 'raw')
        self.image_feature.save_pca_data(Joint.FEATURE_DIR, 'img_' + 'line' if line_flag else 'raw')
        self.japanese_feature.save_pca_data(Joint.FEATURE_DIR, 'jpn_' + 'line' if line_flag else 'raw')

    def load_pca_features(self, line_flag):
        self.english_feature.load_pca_data(Joint.FEATURE_DIR, 'eng_' + 'line' if line_flag else 'raw')
        self.image_feature.load_pca_data(Joint.FEATURE_DIR, 'img_' + 'line' if line_flag else 'raw')
        self.japanese_feature.load_pca_data(Joint.FEATURE_DIR, 'jpn_' + 'line' if line_flag else 'raw')

    def process_features(self, line_flag=False):
        self.logger.info("processing features")
        if line_flag:
            self.english_feature.create_bow_feature_with_lines()
            self.japanese_feature.create_bow_feature_with_lines()
            self.image_feature.load_feature_and_copy_line(self.english_feature.line_count)
        else:
            self.english_feature.create_bow_feature()
            self.japanese_feature.create_bow_feature()
            self.image_feature.load_original_feature()

    def cca_fit(self, sample_num, line_flag=False, reg_param=0.1):
        self.logger.info("fitting CCA line_flag:%s sample_num:%d reg_param:%f", line_flag, sample_num, reg_param)

        self.cca.reg_param = reg_param

        self.cca.fit(
            self.english_feature.sample_random_train_data(sample_num),
            self.japanese_feature.sample_random_train_data(sample_num)
        )
        if line_flag:
            if not os.path.isdir(Joint.LINE_CCA_DIR + str(reg_param).replace(".", "_") + '/' + str(sample_num) + '/'):
                os.makedirs(Joint.LINE_CCA_DIR + str(reg_param).replace(".", "_") + '/' + str(sample_num) + '/')
            self.cca.save_params(Joint.LINE_CCA_DIR + str(reg_param).replace(".", "_") + '/' + str(sample_num) + '/')
        else:
            if not os.path.isdir(Joint.RAW_CCA_DIR + str(reg_param).replace(".", "_") + '/' + str(sample_num) + '/'):
                os.makedirs(Joint.RAW_CCA_DIR + str(reg_param).replace(".", "_") + '/' + str(sample_num) + '/')
            self.cca.save_params(Joint.RAW_CCA_DIR + str(reg_param).replace(".", "_") + '/' + str(sample_num) + '/')

    def cca_transform(self, sample_num, line_flag=False, reg_param=0.1):
        self.logger.info("transforming by CCA line_flag:%s sample_num:%d reg_param:%f", line_flag, sample_num, reg_param)
        if line_flag:
            self.cca.load_params(Joint.LINE_CCA_DIR + str(reg_param).replace(".", "_") + '/' + str(sample_num) + '/')
        else:
            self.cca.load_params(Joint.RAW_CCA_DIR + str(reg_param).replace(".", "_") + '/' + str(sample_num) + '/')
        self.cca.transform(
            self.english_feature.test_feature_pca,
            self.japanese_feature.test_feature_pca
        )
        self.cca.fix_reverse()

    def cca_plot(self):
        self.cca.plot_cca_result(False)

    def gcca_fit(self, sample_num, line_flag=False, reg_param=0.1):
        self.logger.info("fitting GCCA line_flag:%s sample_num:%d reg_param:%f", line_flag, sample_num, reg_param)
        self.gcca.reg_param = reg_param

        self.gcca.fit(
            self.english_feature.sample_random_train_data(sample_num),
            self.image_feature.sample_random_train_data(sample_num),
            self.japanese_feature.sample_random_train_data(sample_num)
        )
        if line_flag:
            if not os.path.isdir(Joint.LINE_GCCA_DIR + str(reg_param).replace(".", "_") + '/' + str(sample_num) + '/'):
                os.makedirs(Joint.LINE_GCCA_DIR + str(reg_param).replace(".", "_") + '/' + str(sample_num) + '/')
            self.gcca.save_params(Joint.LINE_GCCA_DIR + str(reg_param).replace(".", "_") + '/' + str(sample_num) + '/')
        else:
            if not os.path.isdir(Joint.RAW_GCCA_DIR + str(reg_param).replace(".", "_") + '/' + str(sample_num) + '/'):
                os.makedirs(Joint.RAW_GCCA_DIR + str(reg_param).replace(".", "_") + '/' + str(sample_num) + '/')
            self.gcca.save_params(Joint.RAW_GCCA_DIR + str(reg_param).replace(".", "_") + '/' + str(sample_num) + '/')

    def gcca_transform(self, sample_num, line_flag=False, reg_param=0.1):
        self.logger.info("transforming by GCCA line_flag:%s step:%d reg_param:%f", line_flag, sample_num, reg_param)

        if line_flag:
            self.gcca.load_params(Joint.LINE_GCCA_DIR + str(reg_param).replace(".", "_") + '/' + str(sample_num) + '/')
        else:
            self.gcca.load_params(Joint.RAW_GCCA_DIR + str(reg_param).replace(".", "_") + '/' + str(sample_num) + '/')

        self.gcca.transform(
            self.english_feature.test_feature_pca,
            self.image_feature.test_feature_pca,
            self.japanese_feature.test_feature_pca
        )

    def gcca_plot(self):
        self.gcca.plot_gcca_result()

    def retrieval_j2e_by_gcca(self, j_id, neighbor_num = 10):

        min_dim = 30

        en_mat, im_mat, jp_mat = self.gcca.z_1[:, :min_dim], self.gcca.z_2[:, :min_dim], self.gcca.z_3[:, :min_dim]

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

        en_mat, jp_mat = self.cca.x_c[:, :min_dim], self.cca.y_s[:, :min_dim]

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

        en_mat, im_mat, jp_mat = self.gcca.z_1[:, :min_dim], self.gcca.z_2[:, :min_dim], self.gcca.z_3[:, :min_dim]

        nn = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(im_mat)
        dists, nn_indices = nn.kneighbors([jp_mat[j_id - 1]], neighbor_num, return_distance=True)
        print nn_indices + 1
        print "*%d*****************" % (j_id)
        print self.japanese_feature.read_text_by_id(j_id)
        # for idx in nn_indices[0]:
        #     print "=%d=================" % (idx + 1)
        #     self.image_feature.plot_img_by_id(idx + 1)
        self.image_feature.plot_img_by_ids(nn_indices[0] + 1)

    def cca_calc_search_precision(self, min_dim=30, neighbor_num=1):

        en_mat, jp_mat = self.cca.x_c[:, :min_dim], self.cca.y_s[:, :min_dim]
        nn = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(en_mat)
        dists, nn_indices = nn.kneighbors(jp_mat, neighbor_num, return_distance=True)
        hit_count = 0
        for j_idx, nn_indices_row in enumerate(nn_indices):
            # print nn_indices_row
            if j_idx in nn_indices_row:
                # print True
                hit_count += 1
            else:
                pass
                # print False
        return float(hit_count) / len(nn_indices) * 100



    def gcca_calc_search_precision(self, min_dim=30, neighbor_num=1):

        en_mat, im_mat, jp_mat = self.gcca.z_1[:, :min_dim], self.gcca.z_2[:, :min_dim], self.gcca.z_3[:, :min_dim]
        nn = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(en_mat)
        dists, nn_indices = nn.kneighbors(jp_mat, neighbor_num, return_distance=True)
        hit_count = 0
        for j_idx, nn_indices_row in enumerate(nn_indices):
            # print nn_indices_row
            if j_idx in nn_indices_row:
                # print True
                hit_count += 1
            else:
                # print False
                pass
        return float(hit_count) / len(nn_indices) * 100

    def compare_correlation_coefficient(self):
        self.cca.corrcoef()
        self.gcca.corrcoef()

    def plot_results(self, res_cca, res_gcca, title_list, col_num=2, mode=' SAMPLE'):

        data_num = len(res_cca)
        row_num = data_num / col_num
        if row_num - float(data_num)/col_num != 0:
            print row_num
            row_num = row_num + 1

        fig = plt.figure()
        for i, (title, row_cca, row_gcca) in enumerate(zip(title_list, res_cca, res_gcca)):

            plt.subplot(row_num , col_num, i + 1)
            plt.plot(np.arange(len(row_cca)) * 10 + 10, row_cca, '-r')
            plt.plot(np.arange(len(row_gcca)) * 10 + 10, row_gcca, '-b')
            if mode == 'SAMPLE':
                plt.title('Accuracy(sample:%d)' % title)
            elif mode == 'REG':
                plt.title('Accuracy(reg:%f)' % title)
        plt.tight_layout()
        plt.show()

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