#!/usr/bin/python
#-*- coding: utf-8 -*-

__author__ = 'rupy'

import numpy as np
import csv
import matplotlib.pyplot as plt
import math
from mpl_toolkits.axes_grid1 import ImageGrid
import logging
import sys
import os
import my_util as util

class ImageFeatures():

    def __init__(self, data_path, original_dir=None, correspondence_path=None):

        # log setting
        program = os.path.basename(sys.argv[0])
        self.logger = logging.getLogger(program)
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')

        self.data_path = data_path
        self.original_dir = original_dir
        self.correspondence_path = correspondence_path
        self.img_feature = None
        self.img_feature_copied = None

    def load_feature(self):
        self.img_feature = np.load(self.data_path)
        print self.img_feature.shape

    def load_feature_and_copy_line(self, line_count):
        img_feature = np.load(self.data_path)
        # copy rows
        self.logger.info("creating image feature matrix")
        img_features = []
        for idx, line_count in enumerate(line_count):
            self.logger.info("image id: %d", idx + 1)
            for i in xrange(line_count):
                img_features.append(img_feature[idx])
        self.img_feature = np.array(img_features)

    def get_train_data(self, step=2):

        # select every step-th row
        return self.img_feature[::step]

    def get_test_data(self, step=2):

        # select data except for every step-th data
        return util.skip_rows_by_step(self.img_feature, step)

    def plot_img_by_id(self, img_id):
        reader = csv.reader(open(self.correspondence_path, 'rb'))
        rows = np.array([row for row in reader])
        im = plt.imread(self.original_dir + rows[rows[:, 0]  == str(img_id), 1][0])
        plt.imshow(im)
        plt.show()

    def plot_img_by_ids(self, img_ids):

        reader = csv.reader(open(self.correspondence_path, 'rb'))
        rows = np.array([row for row in reader])

        data_num = len(img_ids)
        side_num = int(math.sqrt(data_num))
        if side_num - math.sqrt(data_num) != 0:
            side_num = side_num + 1

        fig = plt.figure()
        grid = ImageGrid(fig, 111, nrows_ncols = (side_num, side_num))
        all_axes = fig.get_axes()
        for ax in all_axes:
            ax.set_axis_off()
        for i, img_id in enumerate(img_ids):
            print img_id
            ax = grid[i]
            im = plt.imread(self.original_dir + rows[rows[:, 0] == str(img_id), 1][0])
            ax.imshow(im)
            ax.text(1, 1, str(i))
        plt.show()



# クロスリンガルセンテンスリトリーバル
# バラバラにする？
# 10サンプル以内に？トップ１？
# ccaで画像を抜きにして
# まずは半々
# 相関係数の比較
# GCCAを拡張して英語と画像、日本語と画像のペアしか無いサンプルから学習できるようにする
# 比較対象はない？理想的な状態との比較？