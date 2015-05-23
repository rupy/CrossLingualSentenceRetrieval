__author__ = 'rupy'

from joint import Joint
import logging
import numpy as np
import base_feature as feat
import os
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
import yaml
from accuracy import Accuracy

class Experiment:

    PCA_COMPRESS_DIM = 100
    SEED_NUM = None

    MAX_DIM = 300
    MIN_DIM = 10
    DIM_STEP = 10

    CONFIG_YAML = 'config.yml'

    def __init__(self, line_flag=False):

        # log setting
        program = os.path.basename(__name__)
        self.logger = logging.getLogger(program)
        logging.basicConfig(format='%(asctime)s : %(name)s : %(levelname)s : %(message)s')

        # load config file
        f = open(Experiment.CONFIG_YAML, 'r')
        self.config = yaml.load(f)
        f.close()

        self.english_corpus_dir = self.config['english_corpus_dir']
        self.japanese_corpus_dir = self.config['japanese_corpus_dir']
        self.japanese_original_corpus_dir = self.config['japanese_original_corpus_dir']
        self.img_features_npy = self.config['img_features_npy']
        self.img_original_dir = self.config['img_original_dir']
        self.img_correspondence_path = self.config['img_correspondence_path']

        self.joint = Joint(
            self.english_corpus_dir,
            self.img_features_npy,
            self.japanese_corpus_dir,
            self.img_original_dir,
            self.img_correspondence_path,
            self.japanese_original_corpus_dir,
            Experiment.PCA_COMPRESS_DIM,
            line_flag
        )
        self.logger.info("<Initilalizing Experiment>")
        if Experiment.SEED_NUM is not None:
            self.logger.info("seed: %s" , Experiment.SEED_NUM)
            np.random.seed(Experiment.SEED_NUM)

    def process_features(self):
        self.joint.create_features()
        self.joint.pca_train_and_test_data()

    def fit_changing_sample_num(self, sample_num_list):
        data_num = self.joint.english_feature.get_train_data_num()
        for s in sample_num_list:
            sampled_indices = feat.BaseFeature.sample_indices(data_num, s)
            self.joint.gcca_fit(s, 0.1, sampled_indices)
            self.joint.cca_fit(s, 0.1, sampled_indices)

    def calc_accuracy(self, start_dim=1, end_dim=100, dim_step=1):
        res_cca_list = []
        res_gcca_list = []

        print "|dim|CCA|GCCA|"
        for i in xrange(start_dim, end_dim, dim_step):
            res_cca = self.cca_calc_search_precision(i)
            res_gcca = self.gcca_calc_search_precision(i)
            print "|%d|%f|%f|" % (i, res_cca, res_gcca)
            res_cca_list.append(res_cca)
            res_gcca_list.append(res_gcca)

        return res_cca_list, res_gcca_list

    def plot_result(self, sample_num=500, reg_param=0.1):
        self.joint.gcca_transform(sample_num, reg_param)
        self.joint.cca_transform(sample_num, reg_param)
        self.joint.cca_plot()
        self.joint.gcca_plot()

    def calc_accuracy_changing_sample_num(self, sample_num_list, reg_param=0.1):

        res_cca_data = []
        res_gcca_data = []

        for sample_num in sample_num_list:
            self.joint.gcca_transform(sample_num, reg_param)
            self.joint.cca_transform(sample_num, reg_param)

            res_cca_list, res_gcca_list = self.calc_accuracy(Experiment.MIN_DIM, Experiment.MAX_DIM + 1, Experiment.DIM_STEP)
            res_cca_data.append(res_cca_list)
            res_gcca_data.append(res_gcca_list)

        res_cca_arr = np.array(res_cca_data)
        res_gcca_arr = np.array(res_gcca_data)
        # np.save('output/results/res_cca_arr.npy', res_cca_arr)
        # np.save('output/results/res_gcca_arr.npy', res_gcca_arr)

        # joint.gcca_transform(mode='PART', line_flag=True, step=5)
        # res_cca_arr = np.load('output/results/res_cca_arr.npy')
        # res_gcca_arr = np.load('output/results/res_gcca_arr.npy')

    def fit_chenging_regparam(self, reg_params, sample_num=500):
        data_num = self.joint.english_feature.get_train_data_num()
        for r in reg_params:
            sampled_indices = feat.BaseFeature.sample_indices(data_num, sample_num)
            self.joint.gcca_fit(sample_num, r, sampled_indices)
            self.joint.cca_fit(sample_num, r, sampled_indices)

    def calc_accuracy_changing_reg_params(self, sample_num, reg_list, col_num=5):

        res_cca_data = []
        res_gcca_data = []

        for reg in reg_list:
            self.joint.gcca_transform(sample_num,reg)
            self.joint.cca_transform(sample_num, reg)

            res_cca_list, res_gcca_list = self.calc_accuracy(Experiment.MIN_DIM, Experiment.MAX_DIM + 1 , Experiment.DIM_STEP)
            res_cca_data.append(res_cca_list)
            res_gcca_data.append(res_gcca_list)

        res_cca_arr = np.array(res_cca_data)
        res_gcca_arr = np.array(res_gcca_data)
        np.save('output/results/res_cca_reg_arr.npy', res_cca_arr)
        np.save('output/results/res_gcca_reg_arr.npy', res_gcca_arr)

        # joint.gcca_transform(line_flag=True, step=5)
        # res_cca_arr = np.load('output/results/res_cca_reg_arr.npy')
        # res_gcca_arr = np.load('output/results/res_gcca_reg_arr.npy')
        self.plot_results(res_cca_arr, res_gcca_arr, reg_list, col_num, 'REG')

    def plot_original_data(self):
        self.joint.plot_original_data()

    def cca_calc_search_precision(self, min_dim, neighbor_num=1):

        en_mat, jp_mat = self.joint.cca.z_list[0][:, :min_dim], self.joint.cca.z_list[1][:, :min_dim]
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

    def gcca_calc_search_precision(self, min_dim, neighbor_num=1):

        en_mat, im_mat, jp_mat = self.joint.gcca.z_list[0][:, :min_dim], self.joint.gcca.z_list[1][:, :min_dim], self.joint.gcca.z_list[2][:, :min_dim]
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

    def plot_results(self, res_cca, res_gcca, title_list, col_num=2, mode='SAMPLE'):

        data_num = len(res_cca)
        row_num = data_num / col_num
        if row_num - float(data_num)/col_num != 0:
            print row_num
            row_num = row_num + 1

        fig = plt.figure()
        # plt.title('Accuracy')
        for i, (title, row_cca, row_gcca) in enumerate(zip(title_list, res_cca, res_gcca)):

            plt.subplot(row_num , col_num, i + 1)
            plt.plot(np.arange(len(row_cca)) * 10 + 10, row_cca, '-r')
            plt.plot(np.arange(len(row_gcca)) * 10 + 10, row_gcca, '-b')
            x_min, x_max = plt.gca().get_xlim()
            y_min, y_max = plt.gca().get_ylim()
            if mode == 'SAMPLE':
                plt.text(0.5 * (x_min + x_max), 0.5 * (y_min + y_max), 'sample:%d' % title, ha='center', va='center', color='gray')
            elif mode == 'REG':
                plt.text(0.5 * (x_min + x_max), 0.5 * (y_min + y_max), 'reg:%s' % title, ha='center', va='center', color='gray')
        plt.tight_layout()
        plt.show()
