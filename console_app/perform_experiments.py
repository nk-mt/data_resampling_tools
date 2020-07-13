import argparse
import copy
import operator
import os
import sys
import warnings
from sklearn import svm
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from collections import Counter, defaultdict
from os.path import dirname, abspath, join

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.metrics import specificity_score
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, TomekLinks, NearMiss, \
    CondensedNearestNeighbour, OneSidedSelection, EditedNearestNeighbours, NeighbourhoodCleaningRule, \
    InstanceHardnessThreshold, RepeatedEditedNearestNeighbours, AllKNN
from sklearn.metrics import average_precision_score, roc_curve, precision_recall_fscore_support, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import tree

THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, '..', 'algs'))
sys.path.append(CODE_DIR)

import smote_boost


def over_sampling_algs():
    algs = list()
    algs.append(("No Rs Oversampling case", "No Re-sampling"))
    algs.append((RandomOverSampler(random_state=1), 'RO'))
    algs.append((SMOTE(random_state=1), 'SMOTE'))
    # algs.append((ADASYN(random_state=1), 'ADASYN'))
    # algs.append((SMOTETomek(random_state=1), 'SMOTE+TL'))
    # algs.append((SMOTEENN(random_state=1), 'SMOTE+ENN'))
    algs.append((smote_boost.SMOTEBoost(random_state=1), "SMOTEBoost"))
    return algs


def under_sampling_algs():
    algs = list()
    algs.append(("No Rs Undersampling case", "No Re-sampling"))
    algs.append((RandomUnderSampler(random_state=1), 'RU'))
    # algs.append((ClusterCentroids(random_state=1), 'CC'))
    algs.append((TomekLinks(), 'TL'))
    algs.append((NearMiss(version=1), 'NM1'))
    algs.append((NearMiss(version=2), 'NM2'))
    algs.append((NearMiss(version=3), 'NM3'))
    # algs.append((CondensedNearestNeighbour(random_state=1), 'CNN'))
    # algs.append((OneSidedSelection(random_state=1), 'OSS'))
    # algs.append((EditedNearestNeighbours(), 'ENN'))
    # algs.append((NeighbourhoodCleaningRule(), 'NCL'))
    # algs.append((InstanceHardnessThreshold(random_state=1), 'IHT'))
    # algs.append((RepeatedEditedNearestNeighbours(), 'RENN'))
    # algs.append((AllKNN(), 'AllKNN'))
    return algs


def is_over_sampling(sampling_alg, list_ovrsampling_algs):
    return type(sampling_alg).__name__ in [type(alg).__name__ for alg in list_ovrsampling_algs]


def gather_class_algs():
    algs = []
    algs.append(("CART", tree.DecisionTreeClassifier(criterion='entropy', random_state=np.random.RandomState(1))))
    algs.append(("SVM", svm.SVC(probability=True, random_state=np.random.RandomState(1))))
    return algs


def round(digit, digit_after_fp):
    if not isinstance(digit, str):
        return ("{:." + str(digit_after_fp) + "f}").format(digit)
    else:
        return digit


def has_header(dataframe):
    for el in dataframe:
        try:
            if str(el).count(".") == 2:
                continue
            float(el)
        except ValueError:
            return 1
    return None


def reorder_tuple_with_positive_class(tuples, positive_class):
    for idx, t in enumerate(tuples):
        if t[0] == positive_class:
            pos_idx = idx
    other_idx = 0 if pos_idx == 1 else 1
    return [tuples[other_idx], tuples[pos_idx]]


def do_experiments(dataset_files):
    rand_state = np.random.RandomState(1)
    cv = StratifiedKFold(n_splits=10, random_state=rand_state)
    latex_dict = {}
    latex_dict['Under-sampling'] = {}
    latex_dict['Over-sampling+Hybrid'] = {}
    class_algs = gather_class_algs()
    for alg_tuple in class_algs:
        alg_alias = alg_tuple[0]
        latex_dict['Under-sampling'][alg_alias] = {}
        latex_dict['Over-sampling+Hybrid'][alg_alias] = {}
    for d_fp in dataset_files:
            print ("Started re-sampling for {}".format(d_fp))
            first_row = pd.read_csv(d_fp, delimiter=',', nrows=1)
            header_row = has_header(first_row)
            tfr = pd.read_csv(d_fp, delimiter=',', iterator=True, header=header_row)
            ds_as_dataframe = pd.concat(tfr)
            columns_length = len(ds_as_dataframe.columns)
            x_values = ds_as_dataframe.iloc[:, :columns_length - 1].to_numpy()
            y_values = ds_as_dataframe.iloc[:, columns_length - 1:].to_numpy().flatten()
            t1_before, t2_before = Counter(y_values).most_common(2)

            ovr_s_algs = over_sampling_algs()
            undr_s_algs = under_sampling_algs()
            sampling_algs = undr_s_algs + ovr_s_algs

            d_fp = d_fp.split("/")[-1].split(".csv")[0]
            for alg_tuple in class_algs:
                alg_alias = alg_tuple[0]
                latex_dict['Under-sampling'][alg_alias][d_fp] = {}
                latex_dict['Over-sampling+Hybrid'][alg_alias][d_fp] = {}

                for sa in sampling_algs:

                    alg = copy.deepcopy(alg_tuple[1])

                    i = 0
                    pr_rec_f1s = []
                    g_means_1 = []
                    g_means_2 = []
                    roc_aucs = []
                    pr_aucs = []
                    bal_accs = []
                    sa_name = sa[1]
                    print("Current class. alg and sampling method: {} {}".format(alg_alias, sa_name))
                    percentage_pos_samples = []

                    for train, test in cv.split(x_values, y_values):
                        if sa_name == 'SMOTEBoost':
                            model = sa[0].fit(x_values[train], y_values[train].astype(int))
                            percentage_pos_samples.append(model.percentage_pos_examples)
                        elif sa_name == 'No Re-sampling':
                            model = alg.fit(x_values[train], y_values[train].astype(int))
                            percentage_pos_samples.append((t2_before[1] / len(y_values) * 100))
                        else:
                            x_resampled_values, y_resampled_values = sa[0].fit_resample(x_values[train], y_values[train].astype(int))
                            t1_after, t2_after = reorder_tuple_with_positive_class(Counter(y_resampled_values).most_common(2), t2_before[0])
                            percentage_pos_samples.append((t2_after[1] / len(y_resampled_values) * 100))
                            model = alg.fit(x_resampled_values, y_resampled_values)
                        predicted_classes = model.predict(x_values[test])
                        probas = model.predict_proba(x_values[test])
                        average_precision = average_precision_score(y_values[test], probas[:, 1])
                        pr_aucs.append(average_precision)
                        fpr, tpr, thresholds = roc_curve(y_values[test], probas[:, 1])
                        prf1 = precision_recall_fscore_support(y_values[test], predicted_classes, average='binary')
                        pr_rec_f1s.append(prf1)
                        specificity = specificity_score(y_values[test], predicted_classes, average='binary')
                        bal_accuracy = (prf1[1] + specificity) / 2
                        bal_accs.append(bal_accuracy)
                        g_mean_1 = np.sqrt(prf1[1] * specificity)
                        g_mean_2 = np.sqrt(prf1[0] * prf1[1])
                        g_means_1.append(g_mean_1)
                        g_means_2.append(g_mean_2)
                        roc_auc = auc(fpr, tpr)
                        roc_aucs.append(roc_auc)
                        i += 1

                    avg_percent_pos, avg_bal_acc, avg_pre, avg_rec, avg_f1, avg_g_mean, avg_g_mean_2, avg_roc_auc, avg_pr_auc = (sum(percentage_pos_samples) / i, sum(bal_accs) / i,
                    (sum(map(operator.itemgetter(0), pr_rec_f1s)) / i),
                    (sum(map(operator.itemgetter(1), pr_rec_f1s)) / i),
                    (sum(map(operator.itemgetter(2), pr_rec_f1s)) / i),
                    sum(g_means_1) / i,
                    sum(g_means_2) / i, sum(roc_aucs) / i,  sum(pr_aucs) / i)


                    if sa in ovr_s_algs:
                        latex_dict['Over-sampling+Hybrid'][alg_alias][d_fp][sa_name] = [round(avg_percent_pos, 1), round(avg_bal_acc, 3), round(avg_pre, 3), round(avg_rec, 3), round(avg_f1, 3), round(avg_g_mean, 3),
                        round(avg_g_mean_2, 3), round(avg_roc_auc, 3), round(avg_pr_auc, 3)]
                    else:
                        latex_dict['Under-sampling'][alg_alias][d_fp][sa_name] = [round(avg_percent_pos, 1), round(avg_bal_acc, 3), round(avg_pre, 3), round(avg_rec, 3), round(avg_f1, 3), round(avg_g_mean, 3),
                        round(avg_g_mean_2, 3), round(avg_roc_auc, 3), round(avg_pr_auc, 3)]

    generate_latex_output(latex_dict)
    print ("Process finished successfully. Check the latex-gen folder for the result tables.")


def append_multicolumn(value):
    return " & \multicolumn{1}{c|}{" + str(value) + "}"


def pre_table_content():
    content = r"\documentclass[12pt,oneside]{report}" + "\n" \
              r"\usepackage[a4paper, left=2cm, right=2cm, top=2.5cm, bottom=2.5cm]{geometry}" + "\n" \
              r"\usepackage[table]{xcolor}\usepackage{fancyhdr}\pagestyle{fancy}" + "\n" \
              r"\usepackage[T2A]{fontenc}\usepackage[english]{babel}" + "\n" \
              r"\usepackage[utf8]{inputenc}" + "\n" \
              r"\usepackage{longtable}" + "\n" \
              r"\usepackage{amssymb}" + "\n" \
              r"\usepackage{amsmath}" + "\n" \
              r"\usepackage{color}" + "\n" \
              r"\usepackage{caption}\captionsetup[table]{name=Таблица}\definecolor{lightgray}{gray}{0.9}" + "\n" \
              r"\fancyhead{}" + "\n" \
              r"\fancyhead[RO,LE]{Методи за работа с дебалансирани множества от данни в машинно самообучение}" + "\n" \
              r"\fancyfoot{}" + "\n" \
              r"\fancyfoot[C]{\thepage}" + "\n" \
              r"\begin{document}" + "\n"
    return content


def constant_factory(value):
    return lambda: value


def init_nested_dicts(parent_dict):
    for i in range(1, 9):
        if i not in parent_dict.keys():
            parent_dict[i] = defaultdict(dict)
            parent_dict[i]['NormalCase'] = defaultdict(dict)
            parent_dict[i]['ResampledCase'] = defaultdict(dict)


def transform_metric_idx(idx):
    if idx == 1: return "BA"
    elif idx == 2: return "PR"
    elif idx == 3:
        return "RE"
    elif idx == 4:
        return "F_{1}"
    elif idx == 5:
        return "G_{1}"
    elif idx == 6:
        return "G_{2}"
    elif idx == 7:
        return "AUC_{ROC}"
    elif idx == 8:
        return "AUC_{PR}"


def transform_idx(case, idx):
    idx += 1
    if case == 'Under-sampling':
        return under_sampling_algs()[idx][1]
    else:
        return over_sampling_algs()[idx][1]


def generate_latex_output(dict_with_data):
    best_results_dict = defaultdict(dict)
    pre_table_inited = False
    chapter_inited = False
    for sampling_version in dict_with_data.keys():
        result_tables_file = "./latex-gen/" + sampling_version + "-latex-tables.tex"
        os.makedirs(os.path.dirname(result_tables_file), exist_ok=True)
        # plus_sign = ""
        # if os.path.exists(result_tables_file):
        #     plus_sign = "+"
        with open(result_tables_file, "a+", encoding="utf8") as latex_file:
            if not pre_table_inited:
                latex_file.write(pre_table_content())
                pre_table_inited = True

            for class_alg in dict_with_data[sampling_version].keys():
                for d_name in dict_with_data[sampling_version][class_alg].keys():
                    d_fp_for_table = d_name.replace("_", "\\_")
                    init_nested_dicts(best_results_dict[d_fp_for_table])
            for class_alg in dict_with_data[sampling_version].keys():
                first_d_name = list(dict_with_data[sampling_version][class_alg].keys())[0]
                if not chapter_inited:
                    latex_file.write(
                    "\\chapter*{" + sampling_version + " results }" + "\n")
                    chapter_inited = True
                latex_file.write("\\begin{longtable}{|m{1.5cm}|m{1cm}|m{1cm}|m{1cm}|m{1cm}|m{1cm}|m{1cm}|m{1cm}|m{1cm}}\n")
                latex_file.write("\t\hline\n")
                latex_file.write("\t\multicolumn{9}{|c|}{" + sampling_version + " results " + class_alg + "} \\\\ \n")
                latex_file.write("\t\hline\n")
                latex_file.write("\t\multicolumn{1}{|c|}{DS \& SM (\\%Pos. ex.)}& \multicolumn{8}{c|}{Metrics} \\\\ \n")
                latex_file.write("\t\hline\n")
                latex_file.write("& \multicolumn{1}{c|}{$BA$} & \multicolumn{1}{c|}{$PR$} & \multicolumn{1}{c|}{$RE$} & \multicolumn{1}{c|}{$F_1$} & \multicolumn{1}{c|}{$G_1$} & \multicolumn{1}{c|}{$G_2$} & \multicolumn{1}{c|}{$AUC_{ROC}$} & \multicolumn{1}{c|}{$AUC_{PR}$} \\\\ \n")
                latex_file.write("\t\hline\n")
                for d_name in dict_with_data[sampling_version][class_alg].keys():
                    d_fp_for_table = d_name.replace("_", "\\_")
                    latex_file.write("\multicolumn{1}{|l|}{\\textit{" + d_fp_for_table +"}} & \multicolumn{8}{r|}{ } \\\\ \n")
                    latex_file.write("\t\hline\n")
                    latex_file.write("\t\hline\n")
                    latex_file.write("\t\hline\n")
                    sm_stats = [['not used'], [], [], [], [], [], [], [], []]
                    sm_stats_no_resampling = [['not used'], [], [], [], [], [], [], [], []]
                    sm_stats_sampling = [['not used'], [], [], [], [], [], [], [], []]
                    for sm, sm_values in dict_with_data[sampling_version][class_alg][d_name].items():
                        for i in range (1, 9):
                            if sm == 'No Re-sampling':
                                sm_stats_no_resampling[i].append(float(sm_values[i]))
                            else:
                                sm_stats_sampling[i].append(float(sm_values[i]))
                            sm_stats[i].append(float(sm_values[i]))
                    for i in range (1, 9):
                        if len(sm_stats_no_resampling[i]) > 0:
                            max_elem_no_resampling = max(sm_stats_no_resampling[i])
                            max_elem_indexes_no_rs = [index for index, value in enumerate(sm_stats_no_resampling[i]) if
                                                      value == max_elem_no_resampling]
                            for idx in max_elem_indexes_no_rs:
                                max_elem_rounded = round(max_elem_no_resampling, 3)
                                if max_elem_rounded not in best_results_dict[d_fp_for_table][i]['NormalCase']:
                                    best_results_dict[d_fp_for_table][i]['NormalCase'][max_elem_rounded][
                                        "ClassAlgs"] = list()
                                best_results_dict[d_fp_for_table][i]['NormalCase'][max_elem_rounded][
                                    "ClassAlgs"].append(class_alg)

                        if len(sm_stats_sampling[i]) > 0:
                            max_elem_sampling = max(sm_stats_sampling[i])
                            max_elem_indexes_sampling = [index for index, value in enumerate(sm_stats_sampling[i]) if
                                                      value == max_elem_sampling]
                            for idx in max_elem_indexes_sampling:
                                max_elem_rounded = round(max_elem_sampling, 3)
                                if max_elem_rounded not in best_results_dict[d_fp_for_table][i]['ResampledCase']:
                                    best_results_dict[d_fp_for_table][i]['ResampledCase'][max_elem_rounded][
                                        "ClassAlgs"] = list()
                                    best_results_dict[d_fp_for_table][i]['ResampledCase'][max_elem_rounded][
                                        "SamplingMethods"] = list()
                                best_results_dict[d_fp_for_table][i]['ResampledCase'][max_elem_rounded][
                                    "ClassAlgs"].append(class_alg)
                                # if idx == 0:
                                #     idx += 1
                                best_results_dict[d_fp_for_table][i]['ResampledCase'][max_elem_rounded][
                                    "SamplingMethods"].append(transform_idx(sampling_version, idx))

                        if len(sm_stats[i]) > 0:
                            max_elem = max(sm_stats[i])
                            max_elem_indexes = [index for index, value in enumerate(sm_stats[i]) if value == max_elem]
                            for idx in max_elem_indexes:
                                max_elem_rounded = round(max_elem, 3)
                                sm_stats[i][idx] = '\\textbf{' + max_elem_rounded + '}'

                    i = 0
                    for sm, sm_values in dict_with_data[sampling_version][class_alg][d_name].items():
                        latex_file.write("\multicolumn{1}{|r|}{\\textit{" + sm + " (" + str(sm_values[0]) + ")}}" + append_multicolumn(round(sm_stats[1][i], 3)) + append_multicolumn(round(sm_stats[2][i], 3)) + append_multicolumn(round(sm_stats[3][i], 3)) + append_multicolumn(round(sm_stats[4][i], 3)) + append_multicolumn(round(sm_stats[5][i], 3)) + append_multicolumn(round(sm_stats[6][i], 3)) + append_multicolumn(round(sm_stats[7][i], 3)) + append_multicolumn(round(sm_stats[8][i], 3)) +"\\\\ \n")
                        i += 1
                    latex_file.write("\t\hline\n")

                latex_file.write("\t\caption{}\n")
                latex_file.write("\end{longtable}\n")
            latex_file.write("\end{document}")
            chapter_inited = False
        pre_table_inited = False
    with open("./latex-gen/" + "BestResults-latex-tables.tex", "w+", encoding="utf8") as br_latex_file:
        br_latex_file.write(pre_table_content())
        br_latex_file.write("\\chapter*{Best results" + "}\n")
        br_latex_file.write("\\begin{longtable}{|m{1.5cm}|m{1cm}|m{1cm}|m{1cm}|m{0.5cm}|m{1.7cm}|}\n")
        br_latex_file.write("\t\\hline\n")
        br_latex_file.write("\t\multicolumn{6}{|c|}{Best results across all tests} \\\\")
        br_latex_file.write("\t\\hline\n")
        br_latex_file.write("\t\\multicolumn{1}{|c|}{Dataset \& metric} & \multicolumn{1}{c|}{BV w/o SM} & \multicolumn{1}{c|}{Alg}&\multicolumn{1}{c|}{BV w/ SM} & \multicolumn{1}{c|}{Alg} & \multicolumn{1}{c|}{SM} \\\\ \n")
        br_latex_file.write("\t\hline\n")
        for dataset, metric_dict in best_results_dict.items():
            br_latex_file.write("\t\multicolumn{1}{|l|}{\\textit{" + dataset + "}} & \multicolumn{5}{r|}{ } \\\\ \n")
            br_latex_file.write("\t\hline\n")
            br_latex_file.write("\t\hline\n")
            br_latex_file.write("\t\hline\n")
            for metric_index in metric_dict.keys():
                max_ele_nc = round(max(map(float, metric_dict[metric_index]['NormalCase'].keys())), 3)
                max_ele_rc = round(max(map(float, metric_dict[metric_index]['ResampledCase'].keys())), 3)
                max_ele_bold_ver_nc = "\\textbf{" + max_ele_nc + "}" if max_ele_nc >= max_ele_rc else max_ele_nc
                max_ele_bold_ver_rc = "\\textbf{" + max_ele_rc + "}" if max_ele_rc >= max_ele_nc else max_ele_rc

                nc_algs = extract_algs_sm_in_shortstacks(metric_dict[metric_index]['NormalCase'][max_ele_nc]['ClassAlgs'])
                rc_algs = extract_algs_sm_in_shortstacks(metric_dict[metric_index]['ResampledCase'][max_ele_rc]['ClassAlgs'])
                rc_sm = extract_algs_sm_in_shortstacks(metric_dict[metric_index]['ResampledCase'][max_ele_rc]['SamplingMethods'])

                br_latex_file.write("\t\multicolumn{1}{|r|}{$"+ transform_metric_idx(metric_index) +"$}  & \multicolumn{1}{c|}{" + max_ele_bold_ver_nc + "} & \multicolumn{1}{c|}{ " + nc_algs + "} & \multicolumn{1}{c|}{" + max_ele_bold_ver_rc + "} &\multicolumn{1}{c|}{" + rc_algs + "}  & \multicolumn{1}{c|}{ " + rc_sm + "} \\\\ \n")
                br_latex_file.write("\t\cline{5-6} \n")
            br_latex_file.write("\t\hline\n")
        br_latex_file.write("\t\caption{} \\\\ \n")
        br_latex_file.write("\end{longtable}")
        br_latex_file.write("\end{document}")


def extract_algs_sm_in_shortstacks(provided_list):
    provided_list = list(set(provided_list))
    result = "\shortstack[l]{"
    list_length = len(provided_list)
    for idx, el in enumerate(provided_list):
        result += el
        if idx < (list_length - 1):
            result += '/'
        if idx % 2 != 0 and idx > 0:
            if idx < (list_length - 1):
                result += '\\\\'
    result += "}"
    return result


def try_to_take_file_names(path):
    is_dir = os.path.isdir(path)
    if is_dir:
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        csv_files_presense = False
        for file in files:
            if file.lower().endswith(".csv"):
                csv_files_presense = True
                break
        if not csv_files_presense:
            print ("The folder you've selected is either empty or does not contain csv files!")
        else:
            return files
    else:
        print ("The path you've added, as -f param, is not pointing to a folder! Please point to a real folder.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="file path to dataset folder")
    args = parser.parse_args()
    if args.f:
        file_names = try_to_take_file_names(args.f)
        file_paths = [args.f + "/" + str(fn) for fn in file_names]
        do_experiments(file_paths)
    else:
        print ('ERROR: You have\'t added a path to the dataset folder. Please do that by using the -f param, '
           'and after that specify the path to the folder.')