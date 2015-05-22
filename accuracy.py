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
        return 1.0 * tp_num / all_retrieved_num

    @staticmethod
    def average_precision(y_true, y_pred):

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        search_num = len(y_true)

        # calc precisions changing search_num
        precisions = np.array(
            [precision_score(y_true[:(i+1)], y_pred[:(i+1)], average='micro') for i in xrange(search_num)]
        )

        hit_precisions = precisions[y_true == y_pred]
        # calc average precision
        average_precision = hit_precisions.sum() / len(hit_precisions)
        return average_precision

    @staticmethod
    def mean_average_precision(ave_prec_list):
        ave_prec_arr = np.array(ave_prec_list)
        mean_average_precision = ave_prec_arr.mean()
        return mean_average_precision

    @staticmethod
    def calc_mean_average_precision(y_true_list, y_pred_list):
        ave_prec_list = []
        for y_true, y_pred in y_true_list, y_pred_list:
            ave_prec = Accuracy.average_precision(y_true, y_pred)
            ave_prec_list.append(ave_prec)
        mean_average_precision = Accuracy.mean_average_precision(ave_prec_list)
        return mean_average_precision

if __name__=="__main__":


    y_true = [1, 0, 1, 0, 1, 1, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    print Accuracy.average_precision(y_true, y_pred)

    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 1, 0, 0, 1]

    print precision_score(y_true, y_pred, average='micro')






