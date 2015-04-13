__author__ = 'rupy'

import logging
import experiment

if __name__=="__main__":
    logging.root.setLevel(level=logging.INFO)

    ex = experiment.Experiment()

    # !!!!!!!IMPORTANT!!!!!!!!!
    # this function must run first
    # if you ran this function run brefore, you don't have to run it again.
    # ex.process_features()

    ex.fit_changing_step(line_flag=True, start_step=25, end_step=250, every_nth=25)
