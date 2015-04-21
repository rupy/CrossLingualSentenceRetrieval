#!/usr/bin/python
#-*- coding: utf-8 -*-

__author__ = 'rupy'

import logging
from joint import Joint

if __name__=="__main__":
    logging.root.setLevel(level=logging.INFO)


    english_corpus_dir = '../PascalSentenceDataset/english/'
    japanese_corpus_dir = '../PascalSentenceDataset/line_wakati/'
    japanese_original_corpus_dir = '../PascalSentenceDataset/japanese/'
    img_features_npy = 'pascal_features.npy'
    img_original_dir = '../PascalSentenceDataset/dataset/'
    img_correspondence_path = "../PascalSentenceDataset/correspondence.csv"
    joint = Joint(
        english_corpus_dir,
        img_features_npy,
        japanese_corpus_dir,
        img_original_dir,
        img_correspondence_path,
        japanese_original_corpus_dir
    )

    # retrieval
    joint.create_features()
    joint.pca_train_and_test_data()
    joint.cca_transform(line_flag=False, step=1, reg_param=0.1)
    joint.cca_plot()
    joint.gcca_transform(line_flag=False, step=1, reg_param=0.1)
    joint.gcca_plot()

    # joint.retrieval_j2e_by_cca(3)
    joint.retrieval_j2e_by_gcca(3)
    # joint.retrieval_j2i_by_gcca(3)
