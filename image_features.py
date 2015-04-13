#!/usr/bin/python
#-*- coding: utf-8 -*-

__author__ = 'rupy'

import numpy as np
import csv
import matplotlib.pyplot as plt
import math
from mpl_toolkits.axes_grid1 import ImageGrid
from base_feature import BaseFeature

class ImageFeatures(BaseFeature):

    def __init__(self, data_dir, original_dir=None, correspondence_path=None):

        BaseFeature.__init__(self, data_dir, original_dir)

        self.correspondence_path = correspondence_path

    def load_feature_and_copy_line(self, line_count):
        img_feature = np.load(self.data_dir)
        # copy rows
        self.feature = img_feature.repeat(line_count, axis=0)
        print self.feature.shape

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

