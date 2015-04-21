__author__ = 'rupy'

from joint import Joint
import logging
import numpy as np
import base_feature as feat

class Experiment:

    PCA_COMPRESS_DIM = 100
    SEED_NUM =  3

    def __init__(self, line_flag=False):

        english_corpus_dir = '../PascalSentenceDataset/english/'
        japanese_corpus_dir = '../PascalSentenceDataset/line_wakati/'
        japanese_original_corpus_dir = '../PascalSentenceDataset/japanese/'
        img_features_npy = 'pascal_features.npy'
        img_original_dir = '../PascalSentenceDataset/dataset/'
        img_correspondence_path = "../PascalSentenceDataset/correspondence.csv"


        self.joint = Joint(
            english_corpus_dir,
            img_features_npy,
            japanese_corpus_dir,
            img_original_dir,
            img_correspondence_path,
            japanese_original_corpus_dir,
            Experiment.PCA_COMPRESS_DIM,
            line_flag
        )
        # np.random.seed(Experiment.SEED_NUM)

    def process_features(self):
        self.joint.create_features()
        self.joint.pca_train_and_test_data()

    def fit_changing_sample_num(self, sample_num_list):
        data_num = self.joint.english_feature.get_train_data_num()
        for s in sample_num_list:
            sampled_indices = feat.BaseFeature.sample_indices(data_num, s)
            print sampled_indices
            self.joint.gcca_fit(s, 0.1, sampled_indices)
            self.joint.cca_fit(s, 0.1, sampled_indices)

    def calc_accuracy(self, start_dim=1, end_dim=100, dim_step=1):
        res_cca_list = []
        res_gcca_list = []

        print "|dim|CCA|GCCA|"
        for i in xrange(start_dim, end_dim, dim_step):
            res_cca = self.joint.cca_calc_search_precision(i)
            res_gcca = self.joint.gcca_calc_search_precision(i)
            print "|%d|%f|%f|" % (i, res_cca, res_gcca)
            res_cca_list.append(res_cca)
            res_gcca_list.append(res_gcca)

        return res_cca_list, res_gcca_list

    def plot_result(self, sample_num=500, reg_param=0.1):
        self.joint.gcca_transform(sample_num, reg_param=reg_param)
        self.joint.cca_transform(sample_num, reg_param=reg_param)
        self.joint.cca_plot()
        self.joint.gcca_plot()

    def calc_accuracy_changing_sample_num(self, sample_num_list, reg_param=0.1):

        res_cca_data = []
        res_gcca_data = []

        for num in sample_num_list:
            self.joint.gcca_transform(sample_num=num, reg_param=reg_param)
            self.joint.cca_transform(sample_num=num, reg_param=reg_param)

            res_cca_list, res_gcca_list = self.calc_accuracy(10, 310, 10)
            res_cca_data.append(res_cca_list)
            res_gcca_data.append(res_gcca_list)

        res_cca_arr = np.array(res_cca_data)
        res_gcca_arr = np.array(res_gcca_data)
        np.save('output/results/res_cca_arr.npy', res_cca_arr)
        np.save('output/results/res_gcca_arr.npy', res_gcca_arr)

        # joint.gcca_transform(mode='PART', line_flag=True, step=5)
        # res_cca_arr = np.load('output/results/res_cca_arr.npy')
        # res_gcca_arr = np.load('output/results/res_gcca_arr.npy')
        self.joint.plot_results(res_cca_arr, res_gcca_arr, sample_num_list, col_num=4)

    def fit_chenging_regparam(self, reg_params, sample_num=500):
        for r in reg_params:
            self.joint.gcca_fit(sample_num=sample_num, reg_param=r)
            self.joint.cca_fit(sample_num=sample_num, reg_param=r)

    def calc_accuracy_changing_reg_params(self, sample_num, reg_list):

        res_cca_data = []
        res_gcca_data = []

        for reg in reg_list:
            self.joint.gcca_transform(sample_num, reg_param=reg)
            self.joint.cca_transform(sample_num, reg_param=reg)

            res_cca_list, res_gcca_list = self.calc_accuracy(10, 310, 10)
            res_cca_data.append(res_cca_list)
            res_gcca_data.append(res_gcca_list)

        res_cca_arr = np.array(res_cca_data)
        res_gcca_arr = np.array(res_gcca_data)
        np.save('output/results/res_cca_reg_arr.npy', res_cca_arr)
        np.save('output/results/res_gcca_reg_arr.npy', res_gcca_arr)

        # joint.gcca_transform(line_flag=True, step=5)
        # res_cca_arr = np.load('output/results/res_cca_reg_arr.npy')
        # res_gcca_arr = np.load('output/results/res_gcca_reg_arr.npy')
        self.joint.plot_results(res_cca_arr, res_gcca_arr, reg_list, col_num=4)

    def plot_original_data(self):
        self.joint.plot_original_data()