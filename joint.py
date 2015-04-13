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

class Joint():


    OUTPUT_DIR = 'output/'
    CCA_PARAMS_SAVE_DIR = OUTPUT_DIR + 'params/'

    LINE_GCCA_ALL_DIR = CCA_PARAMS_SAVE_DIR + "line_gcca_all/"
    LINE_GCCA_TRAIN_DIR = CCA_PARAMS_SAVE_DIR + "line_gcca_train/"
    LINE_CCA_ALL_DIR =  CCA_PARAMS_SAVE_DIR + "line_cca_all/"
    LINE_CCA_TRAIN_DIR =  CCA_PARAMS_SAVE_DIR + "line_cca_train/"
    RAW_GCCA_ALL_DIR = CCA_PARAMS_SAVE_DIR + "raw_gcca_all/"
    RAW_GCCA_TRAIN_DIR = CCA_PARAMS_SAVE_DIR + "raw_gcca_train/"
    RAW_CCA_ALL_DIR =  CCA_PARAMS_SAVE_DIR + "raw_cca_all/"
    RAW_CCA_TRAIN_DIR =  CCA_PARAMS_SAVE_DIR + "raw_cca_train/"

    def __init__(self, en_dir, img_path, jp_dir, img_original_dir=None, img_correspondence_path=None, jp_original_dir=None):

        # log setting
        program = os.path.basename(sys.argv[0])
        self.logger = logging.getLogger(program)
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')

        self.english_feature = txt.TextFeatures(en_dir)
        self.japanese_feature = txt.TextFeatures(jp_dir, jp_original_dir)
        self.image_feature = img.ImageFeatures(img_path, img_original_dir, img_correspondence_path)
        self.gcca = gcca.GCCA()
        self.cca = cca.CCA()

        self.__prep_dir()

    def __prep_dir(self):
        if not os.path.isdir(Joint.OUTPUT_DIR):
            os.mkdir(Joint.OUTPUT_DIR)
        if not os.path.isdir(Joint.CCA_PARAMS_SAVE_DIR):
            os.mkdir(Joint.CCA_PARAMS_SAVE_DIR)
        if not os.path.isdir(Joint.LINE_GCCA_ALL_DIR):
            os.mkdir(Joint.LINE_GCCA_ALL_DIR)
        if not os.path.isdir(Joint.LINE_GCCA_TRAIN_DIR):
            os.mkdir(Joint.LINE_GCCA_TRAIN_DIR)
        if not os.path.isdir(Joint.LINE_CCA_ALL_DIR):
            os.mkdir(Joint.LINE_CCA_ALL_DIR)
        if not os.path.isdir(Joint.LINE_CCA_TRAIN_DIR):
            os.mkdir(Joint.LINE_CCA_TRAIN_DIR)
        if not os.path.isdir(Joint.RAW_GCCA_ALL_DIR):
            os.mkdir(Joint.RAW_GCCA_ALL_DIR)
        if not os.path.isdir(Joint.RAW_GCCA_TRAIN_DIR):
            os.mkdir(Joint.RAW_GCCA_TRAIN_DIR)
        if not os.path.isdir(Joint.RAW_CCA_ALL_DIR):
            os.mkdir(Joint.RAW_CCA_ALL_DIR)
        if not os.path.isdir(Joint.RAW_CCA_TRAIN_DIR):
            os.mkdir(Joint.RAW_CCA_TRAIN_DIR)

    def cca_fit(self, mode='ALL', line_flag=False, step=2, reg_param=0.1):
        self.logger.info("fitting CCA mode:%s line_flag:%s step:%d reg_param:%f", mode, line_flag, step, reg_param)

        self.cca.reg_param = reg_param

        if line_flag:
            self.english_feature.create_bow_feature_with_lines()
            self.japanese_feature.create_bow_feature_with_lines()
        else:
            self.english_feature.create_bow_feature()
            self.japanese_feature.create_bow_feature()
        if mode == 'ALL':
            self.cca.fit(
                self.english_feature.feature,
                self.japanese_feature.feature
            )
            if line_flag:
                self.cca.save_params(Joint.LINE_CCA_ALL_DIR)
            else:
                self.cca.save_params(Joint.RAW_CCA_ALL_DIR)
        else:
            self.cca.fit(
                self.english_feature.get_train_data(step),
                self.japanese_feature.get_train_data(step)
            )
            if line_flag:
                if not os.path.isdir(Joint.LINE_CCA_TRAIN_DIR + str(reg_param).replace(".", "_") + '/' + str(step) + '/'):
                    os.makedirs(Joint.LINE_CCA_TRAIN_DIR + str(reg_param).replace(".", "_") + '/' + str(step) + '/')
                self.cca.save_params(Joint.LINE_CCA_TRAIN_DIR + str(reg_param).replace(".", "_") + '/' + str(step) + '/')
            else:
                if not os.path.isdir(Joint.RAW_CCA_TRAIN_DIR + str(reg_param).replace(".", "_") + '/' + str(step) + '/'):
                    os.makedirs(Joint.RAW_CCA_TRAIN_DIR + str(reg_param).replace(".", "_") + '/' + str(step) + '/')
                self.cca.save_params(Joint.RAW_CCA_TRAIN_DIR + str(reg_param).replace(".", "_") + '/' + str(step) + '/')

    def cca_transform(self, mode='ALL', line_flag=False, step=2, reg_param=0.1):
        if line_flag:
            self.english_feature.create_bow_feature_with_lines()
            self.japanese_feature.create_bow_feature_with_lines()
        else:
            self.english_feature.create_bow_feature()
            self.japanese_feature.create_bow_feature()
        if mode == 'ALL':
            if line_flag:
                self.cca.load_params(Joint.LINE_CCA_ALL_DIR)
            else:
                self.cca.load_params(Joint.RAW_CCA_ALL_DIR)
            self.cca.transform(
                self.english_feature.feature,
                self.japanese_feature.feature
            )
        else:
            if line_flag:
                self.cca.load_params(Joint.LINE_CCA_TRAIN_DIR + str(reg_param).replace(".", "_") + '/' + str(step) + '/')
            else:
                self.cca.load_params(Joint.RAW_CCA_TRAIN_DIR + str(reg_param).replace(".", "_") + '/' + str(step) + '/')
            self.cca.transform(
                self.english_feature.get_test_data(),
                self.japanese_feature.get_test_data()
            )
        self.cca.fix_reverse()

    def cca_plot(self):
        self.cca.plot_cca_result(False)

    def gcca_fit(self, mode='ALL', line_flag=False, step=2, reg_param=0.1):
        self.logger.info("fitting GCCA mode:%s line_flag:%s step:%d reg_param:%f", mode, line_flag, step, reg_param)
        self.gcca.reg_param = reg_param
        if line_flag:
            self.english_feature.create_bow_feature_with_lines()
            self.japanese_feature.create_bow_feature_with_lines()
            self.image_feature.load_feature_and_copy_line(self.english_feature.line_count)
        else:
            self.english_feature.create_bow_feature()
            self.japanese_feature.create_bow_feature()
            self.image_feature.load_feature()

        if mode == 'ALL':
            self.gcca.fit(
                self.english_feature.feature,
                self.image_feature.feature,
                self.japanese_feature.feature
            )
            if line_flag:
                self.gcca.save_params(Joint.LINE_GCCA_ALL_DIR)
            else:
                self.gcca.save_params(Joint.RAW_GCCA_ALL_DIR)
        else:
            if line_flag:
                if not os.path.isdir(Joint.LINE_GCCA_TRAIN_DIR + str(reg_param).replace(".", "_") + '/' + str(step) + '/'):
                    os.makedirs(Joint.LINE_GCCA_TRAIN_DIR + str(reg_param).replace(".", "_") + '/' + str(step) + '/')
            else:
                if not os.path.isdir(Joint.RAW_GCCA_TRAIN_DIR + str(reg_param).replace(".", "_") + '/' + str(step) + '/'):
                    os.makedirs(Joint.RAW_GCCA_TRAIN_DIR + str(reg_param).replace(".", "_") + '/' + str(step) + '/')
            self.gcca.fit(
                self.english_feature.get_train_data(step),
                self.image_feature.get_train_data(step),
                self.japanese_feature.get_train_data(step)
            )
            if line_flag:
                self.gcca.save_params(Joint.LINE_GCCA_TRAIN_DIR + str(reg_param).replace(".", "_") + '/' + str(step) + '/')
            else:
                self.gcca.save_params(Joint.RAW_GCCA_TRAIN_DIR + str(reg_param).replace(".", "_") + '/' + str(step) + '/')

    def gcca_transform(self, mode='ALL', line_flag=False, step=2, reg_param=0.1):

        if line_flag:
            self.english_feature.create_bow_feature_with_lines()
            self.japanese_feature.create_bow_feature_with_lines()
            self.image_feature.load_feature_and_copy_line(self.english_feature.line_count)
        else:
            self.english_feature.create_bow_feature()
            self.japanese_feature.create_bow_feature()
            self.image_feature.load_feature()

        if mode == 'ALL':
            if line_flag:
                self.gcca.load_params(Joint.LINE_GCCA_ALL_DIR)
            else:
                self.gcca.load_params(Joint.RAW_GCCA_ALL_DIR)
            self.gcca.transform(
                self.english_feature.feature,
                self.image_feature.feature,
                self.japanese_feature.feature
            )
        else:
            if line_flag:
                self.gcca.load_params(Joint.LINE_GCCA_TRAIN_DIR + str(reg_param).replace(".", "_") + '/' + str(step) + '/')
            else:
                self.gcca.load_params(Joint.RAW_GCCA_TRAIN_DIR + str(reg_param).replace(".", "_") + '/' + str(step) + '/')
            self.gcca.transform(
                self.english_feature.get_test_data(),
                self.image_feature.get_test_data(),
                self.japanese_feature.get_test_data()
            )

    def gcca_plot(self):
        self.gcca.plot_gcca_result()

    def gcca_nn_j2e(self, j_id, neighbor_num = 10):

        min_dim = min(
            self.gcca.z_1.shape[1] - 1,
            self.gcca.z_2.shape[1] - 1,
            self.gcca.z_3.shape[1] - 1
        )

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

    def cca_nn_j2e(self, j_id, neighbor_num = 10):

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

    def gcca_nn_j2i(self, j_id, neighbor_num=10):

        min_dim = min(
            self.gcca.z_1.shape[1] - 1,
            self.gcca.z_2.shape[1] - 1,
            self.gcca.z_3.shape[1] - 1
        )

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

        # min_dim = min(
        #     self.gcca.z_1.shape[1] - 1,
        #     self.gcca.z_2.shape[1] - 1,
        #     self.gcca.z_3.shape[1] - 1
        # )

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

    def plot_results(self, res_cca, res_gcca, step_list, col_num=2):

        data_num = len(res_cca)
        row_num = data_num / col_num
        if row_num - float(data_num)/col_num != 0:
            print row_num
            row_num = row_num + 1

        fig = plt.figure()
        for i, (step, row_cca, row_gcca) in enumerate(zip(step_list, res_cca, res_gcca)):

            plt.subplot(row_num , col_num, i + 1)
            plt.plot(np.arange(len(row_cca)) * 10 + 10, row_cca, '-r')
            plt.plot(np.arange(len(row_gcca)) * 10 + 10, row_gcca, '-b')
            plt.title('Accuracy(step:%d)' % step)
        plt.tight_layout()
        plt.show()


if __name__=="__main__":
    pass