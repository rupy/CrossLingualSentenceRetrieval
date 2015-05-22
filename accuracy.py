__author__ = 'rupy'

import numpy as np
from sklearn.metrics import precision_score

class Accuracy:

    def __init__(self):
        pass

    @staticmethod
    def precision(tp_num, fp_num):
        all_positive_num = tp_num + fp_num
        return 1.0 * tp_num / all_positive_num

    @staticmethod
    def recall(tp_num, fn_num):
        all_retrieved_num = tp_num + fn_num
        return tp_num / all_retrieved_num

    @staticmethod
    def average_precision(hit_list):

        hit = np.array(hit_list)
        search_num = len(hit_list)

        # calc precisions changing search_num
        precisions = np.array(
            [Accuracy.precision(
                len(hit[:(i+1)][hit[:(i+1)] == 1]), # true positive
                len(hit[:(i+1)][hit[:(i+1)] == 0])  # false positive
            ) for i in xrange(search_num)])

        # calc average precision
        hit_precisions = precisions[hit == 1]
        average_precision = hit_precisions.sum() / len(hit_precisions)
        return average_precision

    @staticmethod
    def mean_average_precision(ave_prec_list):
        ave_prec_arr = np.array(ave_prec_list)
        mean_average_precision = ave_prec_arr.mean()
        return mean_average_precision

    @staticmethod
    def calc_mean_average_precision(hit_list_list):
        ave_prec_list = []
        for hit_list in hit_list_list:
            ave_prec = Accuracy.average_precision(hit_list)
            ave_prec_list.append(ave_prec)
        mean_average_precision = Accuracy.mean_average_precision(ave_prec_list)
        return mean_average_precision

if __name__=="__main__":


    # hit1 = [1, 0, 1, 0, 1, 1, 0, 0, 0, 0]
    hit2 = [1, 0, 1, 0, 0, 1, 0, 0, 1, 1]
    hit3 = [0, 1, 0, 0, 1, 0, 1, 0, 0, 0]
    hit_list_list = [hit2, hit3]

    print Accuracy.calc_mean_average_precision(hit_list_list)

    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 1, 0, 0, 1]

    print precision_score(y_true, y_pred, average='micro')






