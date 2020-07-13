import numpy as np
import copy
import operator
from PyQt5.QtCore import QThread, pyqtSignal
from matplotlib import pyplot as plt
from imblearn.metrics import specificity_score
from sklearn.metrics import average_precision_score, roc_curve, precision_recall_fscore_support, auc
from sklearn.model_selection import StratifiedKFold
from scipy import interp

from functions import resampling_functions
from ui.widgets import Widgets


class Classifying(QThread):

    update_normal_classify_progress_bar = pyqtSignal(int)
    update_resample_classify_progress_bar = pyqtSignal(int)
    update_gui_after_classification = pyqtSignal()
    reraise_non_mt_exception_signal = pyqtSignal(Exception)

    def __init__(self, main_window, do_resampling):
        super(Classifying, self).__init__()
        self.main_window = main_window
        self.do_resampling = do_resampling
        self.__custom_pre_process()

    def run(self):
        try:
            classified_data = self.__classify()
            self.__store_classified_data(classified_data)
            self.__custom_post_process()
        except Exception as e:
            # print ('y')
            self.reraise_non_mt_exception_signal.emit(e)

    def __store_classified_data(self, classified_data):
        if self.do_resampling:
            self.main_window.state.classified_data_resampled_case = classified_data
        else:
            self.main_window.state.classified_data_normal_case = classified_data

    def __custom_post_process(self):
        if self.do_resampling:
            self.main_window.state.resample_classify_thread_finished = True
        else:
            self.main_window.state.normal_classify_thread_finished = True
        self.update_gui_after_classification.emit()

    def __custom_pre_process(self):
        self.main_window.widgets.get_progress_bar(Widgets.ProgressBars.ResampleClassifyProgressBar.value).setValue(
            0)
        self.main_window.widgets.get_progress_bar(Widgets.ProgressBars.NormalClassifyProgressBar.value).setValue(0)

    def __classify(self):
        classifying_data = {}
        classifying_data['main_tuples'] = []
        normal_dataset = self.main_window.state.dataset
        rand_state = np.random.RandomState(1)
        splits = self.main_window.state.number_of_folds
        cv = StratifiedKFold(n_splits=splits, random_state=rand_state)
        classifier = copy.deepcopy(self.main_window.state.classification_algorithm.value[1])
        X_normal = normal_dataset['x_values']
        y_normal = normal_dataset['y_values']
        tprs = []
        mean_fpr = np.linspace(0, 1, 100)
        roc_aucs = []
        i = 0
        g_means_1 = []
        g_means_2 = []
        pr_rec_f1s = []
        preds_list = []
        trues_list = []
        bal_accs = []
        average_precisions = []
        classifying_data['ClassAlg'] = self.main_window.state.classification_algorithm.value[0]
        classifying_data['SamplingAlg'] = self.main_window.state.sampling_algorithm_experiments_tab.value[
            0]
        for train, test in cv.split(X_normal, y_normal):
            if self.do_resampling:
                if self.main_window.state.sampling_algorithm_experiments_tab.value[0] == 'SMOTEBoost':
                    classifier_ = self.main_window.state.sampling_algorithm_experiments_tab.value[1].fit(
                        X_normal[train], y_normal[train].astype(int))
                else:
                    r_dataset = resampling_functions. \
                        do_resampling_without_writing_to_file(
                        self.main_window.state.sampling_algorithm_experiments_tab,
                        X_normal[train], y_normal[train])
                    classifier_ = classifier.fit(r_dataset['x_values'], r_dataset['y_values'])
            else:
                classifier_ = classifier.fit(X_normal[train], y_normal[train].astype(int))
            predicted_classes = classifier_.predict(X_normal[test])
            probas_ = classifier_.predict_proba(X_normal[test])
            preds_list.append(probas_)
            trues_list.append(y_normal[test])
            average_precision = average_precision_score(y_normal[test], probas_[:, 1])
            average_precisions.append(average_precision)
            fpr, tpr, thresholds = roc_curve(y_normal[test], probas_[:, 1])
            prf1 = precision_recall_fscore_support(y_normal[test], predicted_classes, average='binary')
            pr_rec_f1s.append(prf1)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            roc_aucs.append(roc_auc)
            specificity = specificity_score(y_normal[test], predicted_classes)
            g_means_1.append(np.sqrt(prf1[1] * specificity))
            g_means_2.append(np.sqrt(prf1[0] * prf1[1]))
            bal_accuracy = (prf1[1] + specificity) / 2
            bal_accs.append(bal_accuracy)
            classifying_data['main_tuples'].append(
                (fpr, tpr, roc_auc, i, y_normal[test], probas_[:, 1], average_precision))

            i += 1
            if self.do_resampling:
                self.update_resample_classify_progress_bar.emit(i)
            else:
                self.update_normal_classify_progress_bar.emit(i)
        classifying_data['pre_rec_f1_g_mean1_g_mean2_tuple'] = ((sum(map(operator.itemgetter(0), pr_rec_f1s)) / i),
                                                                (sum(map(operator.itemgetter(1), pr_rec_f1s)) / i),
                                                                (sum(map(operator.itemgetter(2), pr_rec_f1s)) / i),
                                                                sum(g_means_1) / i,
                                                                sum(g_means_2) / i)
        classifying_data['precision'] = list(map(operator.itemgetter(0), pr_rec_f1s))
        classifying_data['recall'] = list(map(operator.itemgetter(1), pr_rec_f1s))
        classifying_data['bal_acc'] = sum(bal_accs) / i
        classifying_data['preds_list'] = preds_list
        classifying_data['trues_list'] = trues_list
        classifying_data['avg_roc'] = sum(roc_aucs) / i
        classifying_data['average_precision'] = sum(average_precisions) / i


        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(roc_aucs)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        classifying_data['mean_values_tuple'] = (mean_fpr, mean_tpr, mean_auc, std_auc, tprs_lower, tprs_upper)
        return classifying_data