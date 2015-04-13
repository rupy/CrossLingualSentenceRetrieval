__author__ = 'rupy'

from joint import Joint
import logging
import numpy as np

class Experiment:

    PCA_COMPRESS_DIM = 100

    def __init__(self):

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
            Experiment.PCA_COMPRESS_DIM
        )

    def fit_changing_step(self, line_flag=False, start_step=1, end_step=25, every_nth=1):

        for s in range(start_step, end_step, every_nth):
            self.joint.gcca_fit(line_flag=line_flag, step=s)
            self.joint.cca_fit(line_flag=line_flag, step=s)

    def calc_accuracy(self, start_dim=1, end_dim=100, dim_step=1):
        res_cca_list = []
        res_gcca_list = []

        for i in xrange(start_dim, end_dim, dim_step):
            res_cca = self.joint.cca_calc_search_precision(i)
            res_gcca = self.joint.gcca_calc_search_precision(i)
            print "|%d|%f|%f|" % (i, res_cca, res_gcca)
            res_cca_list.append(res_cca)
            res_gcca_list.append(res_gcca)

        return res_cca_list, res_gcca_list

    def calc_accuracy_changing_step(self, line_flag=False, reg_param=0.1, start_step=1, end_step=25, every_nth=1):

        res_cca_data = []
        res_gcca_data = []
        step_list = np.arange(start_step, end_step, every_nth)

        for step in step_list:
            self.joint.gcca_transform(line_flag=line_flag, step=step, reg_param=reg_param)
            self.joint.cca_transform(line_flag=line_flag, step=step, reg_param=reg_param)

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
        self.joint.plot_results(res_cca_arr, res_gcca_arr, step_list, col_num=2)

    def fit_chenging_regparam(self, reg_params, step=5):
        for r in reg_params:
            self.joint.gcca_fit(line_flag=False, step=step, reg_param=r)
            self.joint.cca_fit(line_flag=False, step=step, reg_param=r)

    def calc_accuracy_changing_reg_params(self, reg_list, line_flag=False, step=5):

        res_cca_data = []
        res_gcca_data = []

        for reg in reg_list:
            self.joint.gcca_transform(line_flag=line_flag, step=step, reg_param=reg)
            self.joint.cca_transform(line_flag=line_flag, step=step, reg_param=reg)

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
        self.joint.plot_results(res_cca_arr, res_gcca_arr, reg_list, col_num=2)
