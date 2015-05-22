__author__ = 'rupy'

from experiment import Experiment
from bridged_joint import BridgedJoint
import numpy as np
import base_feature as feat
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors

class BridgedExperiment(Experiment):

    def __init__(self, line_flag=False):

        Experiment.__init__(self, line_flag)

        self.joint = BridgedJoint(
            self.english_corpus_dir,
            self.img_features_npy,
            self.japanese_corpus_dir,
            self.img_original_dir,
            self.img_correspondence_path,
            self.japanese_original_corpus_dir,
            BridgedExperiment.PCA_COMPRESS_DIM,
            line_flag
        )

    def fit_changing_sample_num(self, sample_num_list):
        data_num = self.joint.english_feature.get_train_data_num()
        sampled_indices1 = feat.BaseFeature.all_indices1(data_num)
        sampled_indices2 = feat.BaseFeature.all_indices2(data_num)
        for s in sample_num_list:
            sampled_indices3 = feat.BaseFeature.sample_indices3(data_num, s)
            self.joint.bcca_fit(s, 0.01, sampled_indices1, sampled_indices2, sampled_indices3)
            if s != 0:
                self.joint.cca_fit(s, 0.01, sampled_indices3)

    def calc_accuracy(self, start_dim=1, end_dim=100, dim_step=1, cca_flag=True):
        res_cca_list = []
        res_bcca_list = []

        print "|dim|CCA|BCCA|"
        for i in xrange(start_dim, end_dim, dim_step):
            if cca_flag:
                res_cca = self.cca_calc_search_precision(i)
            else:
                res_cca = 0
            res_bcca = self.bcca_calc_search_precision(i)
            print "|%d|%f|%f|" % (i, res_cca, res_bcca)
            res_cca_list.append(res_cca)
            res_bcca_list.append(res_bcca)

        return res_cca_list, res_bcca_list

    def plot_result(self, sample_num=500, reg_param=0.1):
        self.joint.cca_transform(sample_num, reg_param)
        self.joint.bcca_transform(sample_num, reg_param)
        self.joint.cca_plot()
        self.joint.bcca_plot()

    def calc_accuracy_changing_sample_num(self, sample_num_list, reg_param=0.1, col_num=5):

        res_cca_data = []
        res_bcca_data = []

        for sample_num in sample_num_list:
            self.joint.bcca_transform(sample_num, reg_param)
            if sample_num != 0:
                self.joint.cca_transform(sample_num, reg_param)

            res_cca_list, res_bcca_list = self.calc_accuracy(BridgedExperiment.MIN_DIM, BridgedExperiment.MAX_DIM + 1, BridgedExperiment.DIM_STEP, sample_num != 0)
            res_cca_data.append(res_cca_list)
            res_bcca_data.append(res_bcca_list)

        res_cca_arr = np.array(res_cca_data)
        res_bcca_arr = np.array(res_bcca_data)
        # np.save('output/results/res_cca_arr.npy', res_cca_arr)
        # np.save('output/results/res_bcca_arr.npy', res_bcca_arr)

        # joint.bcca_transform(mode='PART', line_flag=True, step=5)
        # res_cca_arr = np.load('output/results/res_cca_arr.npy')
        # res_bcca_arr = np.load('output/results/res_bcca_arr.npy')
        self.plot_results(res_cca_arr, res_bcca_arr, sample_num_list, col_num)
        self.plot_max_results(res_cca_arr, res_bcca_arr, sample_num_list)

    def fit_chenging_regparam(self, reg_params, sample_num=500):
        data_num = self.joint.english_feature.get_train_data_num()
        for r in reg_params:
            sampled_indices = feat.BaseFeature.sample_indices(data_num, sample_num)
            self.joint.cca_fit(sample_num, r, sampled_indices)
            self.joint.bcca_fit(sample_num, r, sampled_indices)

    def calc_accuracy_changing_reg_params(self, sample_num, reg_list, col_num=5):

        res_cca_data = []
        res_bcca_data = []

        for reg in reg_list:
            self.joint.bcca_transform(sample_num,reg)
            self.joint.cca_transform(sample_num, reg)

            res_cca_list, res_bcca_list = self.calc_accuracy(BridgedExperiment.MIN_DIM, BridgedExperiment.MAX_DIM + 1 , BridgedExperiment.DIM_STEP)
            res_cca_data.append(res_cca_list)
            res_bcca_data.append(res_bcca_list)

        res_cca_arr = np.array(res_cca_data)
        res_bcca_arr = np.array(res_bcca_data)
        np.save('output/results/res_cca_reg_arr.npy', res_cca_arr)
        np.save('output/results/res_bcca_reg_arr.npy', res_bcca_arr)

        # joint.bcca_transform(line_flag=True, step=5)
        # res_cca_arr = np.load('output/results/res_cca_reg_arr.npy')
        # res_bcca_arr = np.load('output/results/res_bcca_reg_arr.npy')
        self.plot_results(res_cca_arr, res_bcca_arr, reg_list, col_num, 'REG')

    def plot_original_data(self):
        self.joint.plot_original_data()

    def bcca_calc_search_precision(self, min_dim, neighbor_num=1):

        en_mat, im_mat, jp_mat = self.joint.bcca.z_list[0][:, :min_dim], self.joint.bcca.z_list[1][:, :min_dim], self.joint.bcca.z_list[2][:, :min_dim]
        nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(en_mat)
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

    def plot_results(self, res_cca, res_bcca, title_list, col_num=2, mode='SAMPLE'):

        data_num = len(res_cca)
        row_num = data_num / col_num
        if row_num - float(data_num)/col_num != 0:
            print row_num
            row_num = row_num + 1

        fig = plt.figure()
        # plt.title('Accuracy')
        for i, (title, row_cca, row_bcca) in enumerate(zip(title_list, res_cca, res_bcca)):

            plt.subplot(row_num , col_num, i + 1)
            plt.plot(np.arange(len(row_cca)) * 10 + 10, row_cca, '-r')
            plt.plot(np.arange(len(row_bcca)) * 10 + 10, row_bcca, '-b')
            x_min, x_max = plt.gca().get_xlim()
            y_min, y_max = plt.gca().get_ylim()
            if mode == 'SAMPLE':
                plt.text(0.5 * (x_min + x_max), 0.5 * (y_min + y_max), 'sample:%d' % title, ha='center', va='center', color='gray')
            elif mode == 'REG':
                plt.text(0.5 * (x_min + x_max), 0.5 * (y_min + y_max), 'reg:%s' % title, ha='center', va='center', color='gray')
        plt.tight_layout()
        plt.show()

    def plot_max_results(self, res_cca_arr, res_bcca_arr, sample_num_list):
        plt.plot(sample_num_list, res_cca_arr.max(axis=1), '-g', label = "CCA")
        plt.plot(sample_num_list, res_bcca_arr.max(axis=1), '-b', label = "Bridged CCA")
        plt.legend()
        plt.ylabel("top 1 Retrieval Accuracy(%)")
        plt.xlabel("sampling num in <train3>")
        plt.show()
