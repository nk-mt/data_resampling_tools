import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QTableWidgetItem, QVBoxLayout, QHBoxLayout, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from sklearn.metrics import precision_recall_curve

from ui.widgets import Widgets


def update_widgets_after_classification(main_window):
    if main_window.state.resample_classify_thread_finished and main_window.state.normal_classify_thread_finished:
        main_window.state.number_of_runs += 1
        vboxLayout = main_window.findChild(QVBoxLayout, "resultsVerticalLayout")
        hLayout = QHBoxLayout()
        runsLabel = create_runs_label_widget(main_window)
        hLayout.addWidget(runsLabel)
        canvas = create_metrics_table(main_window)
        hLayout.addWidget(canvas)
        classified_data = [main_window.state.classified_data_normal_case, main_window.state.classified_data_resampled_case]
        hLayout.addWidget(create_roc_graph(classified_data))
        hLayout.addWidget(create_pr_graph(classified_data))

        vboxLayout.addLayout(hLayout)


def create_metrics_table(main_window):
    metrics_and_methods = np.array(
        ['Classifier', 'Sampling Method', 'Balanced Accuracy', 'Precision', 'Recall', 'F1', 'G1', 'G2', 'AUC_roc',
         'AUC_pr'])
    standard_case = [main_window.state.classified_data_normal_case['ClassAlg']] + ["---"] + [
        __round(main_window.state.classified_data_normal_case['bal_acc'], 3)] + list(
        map(lambda x: __round(x, 3),
            main_window.state.classified_data_normal_case['pre_rec_f1_g_mean1_g_mean2_tuple'])) + [
                        __round(main_window.state.classified_data_normal_case['avg_roc'], 3)] + [
                        __round(main_window.state.classified_data_normal_case['average_precision'], 3)]
    resampled_case = [main_window.state.classified_data_resampled_case['ClassAlg']] + [
        main_window.state.classified_data_resampled_case['SamplingAlg']] + [
                         __round(main_window.state.classified_data_resampled_case['bal_acc'], 3)] + list(
        map(lambda x: __round(x, 3),
            main_window.state.classified_data_resampled_case[
                'pre_rec_f1_g_mean1_g_mean2_tuple'])) + [
                         __round(main_window.state.classified_data_resampled_case['avg_roc'], 3)] + [
                         __round(main_window.state.classified_data_resampled_case['average_precision'], 3)]
    table_data = np.vstack((metrics_and_methods, standard_case, resampled_case)).T
    fig, ax = plt.subplots(1, 1)
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    df = pd.DataFrame(table_data, columns=[' ', 'Standard Case', 'Re-sampled Case'])
    ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    fig.tight_layout()
    canvas = FigureCanvasQTAgg(fig)
    canvas.setMinimumHeight(350)
    canvas.setMaximumHeight(350)
    return canvas


def create_runs_label_widget(main_window):
    runsLabel = QLabel("Run #{}".format(main_window.state.number_of_runs))
    runsLabel.setMinimumHeight(350)
    runsLabel.setMaximumHeight(350)
    return runsLabel


def clear_tables_and_graphs(main_window):
    main_window.state.number_of_runs = 0
    main_v_layout = main_window.findChild(QVBoxLayout, "resultsVerticalLayout")
    for i in reversed(range(main_v_layout.count())):
        __clear_layout(main_v_layout.itemAt(i))


def update_widgets_after_resampling(main_window, is_successful):
    main_window.widgets.get_progress_bar(Widgets.ProgressBars.ResampleProgressBar.value).setMaximum(100)
    if is_successful:
        main_window.widgets.get_progress_bar(Widgets.ProgressBars.ResampleProgressBar.value).setValue(100)
        resampled_dataset = main_window.state.resampled_dataset
        main_window.widgets.get_label(Widgets.Labels.TotalNumberOfExamplesResampledLabel.value).setText(
            str(resampled_dataset['number_of_examples']))
        main_window.widgets.get_label(Widgets.Labels.NumberOfPositiveExamplesResampledLabel.value).setText(
            str(resampled_dataset['number_of_positive_examples']))
        main_window.widgets.get_label(Widgets.Labels.TargetClassPercentageResampledLabel.value).setText(
            str(resampled_dataset['positive_examples_percentage']))
        main_window.widgets.get_label(Widgets.Labels.ImbalancedRatioResampledLabel.value).setText(
            str(resampled_dataset['imbalanced_ratio']))
        main_window.widgets.enable_disable_widgets(main_window.widgets.resampling_related_widgets(), True)


def update_widgets_after_datasetload(main_window, path, is_successful):
    main_window.setEnabled(True)
    main_window.widgets.get_label(Widgets.Labels.FilePathLabel.value).setText("FilePath:")
    main_window.widgets.get_label(Widgets.Labels.DatasetPickedLabel.value).setText(path)
    main_window.widgets.get_progress_bar(Widgets.ProgressBars.NormalClassifyProgressBar.value).setValue(0)
    main_window.widgets.get_progress_bar(Widgets.ProgressBars.ResampleClassifyProgressBar.value).setValue(0)
    main_window.widgets.get_progress_bar(Widgets.ProgressBars.ResampleProgressBar.value).setValue(0)
    if is_successful:
        main_window.widgets.get_label(Widgets.Labels.DatasetLoadingResultLabel.value).setText(
            "Status:  Successful load!")
        main_window.widgets.enable_disable_widgets(main_window.widgets.dataset_loading_related_widgets(True), True)
        dataset = main_window.state.dataset
        main_window.widgets.get_progress_bar(Widgets.ProgressBars.DatasetProgressBar.value).setMaximum(100)
        main_window.widgets.get_progress_bar(Widgets.ProgressBars.DatasetProgressBar.value).setValue(100)
        main_window.widgets.get_label(Widgets.Labels.TotalNumberOfExamplesLabel.value).setText(
            str(dataset['number_of_examples']))
        main_window.widgets.get_label(Widgets.Labels.NumberOfPositiveExamplesLabel.value).setText(
            str(dataset['number_of_positive_examples']))
        main_window.widgets.get_label(Widgets.Labels.TargetClassPercentageLabel.value).setText(
            str(dataset['positive_examples_percentage']))
        main_window.widgets.get_label(Widgets.Labels.ImbalancedRatioLabel.value).setText(
            str(dataset['imbalanced_ratio']))
        main_window.widgets.get_label(Widgets.Labels.SelectedDatasetExperimentsTabLabel.value).setText(
            str(dataset['name']))
        load_table(main_window)
        main_window.widgets.get_table(Widgets.Tables.DataTable.value).setEnabled(True);
    else:
        main_window.widgets.get_label(Widgets.Labels.DatasetLoadingResultLabel.value).setText(
            "Status:  Unsuccessful load!")
        main_window.widgets.enable_disable_widgets(main_window.widgets.dataset_loading_related_widgets(False), False)


def load_table(main_window):
    dataset = main_window.state.dataset
    rows_count = len(dataset['x_values'])
    rows_count = rows_count if rows_count < 10000 else 10000
    col_count = len(dataset['x_values'][0]) + 1
    data_table = main_window.widgets.get_table(Widgets.Tables.DataTable.value)
    data_table.setRowCount(rows_count)
    data_table.setColumnCount(col_count)
    for idx, el in enumerate(dataset['header_row']):
        data_table.setHorizontalHeaderItem(idx, QTableWidgetItem(el))
    for row in range(0, rows_count):
        for col in range(col_count):
            if col == col_count - 1:
                data_table.setItem(row, col, QTableWidgetItem(str(dataset['y_values'][row])))
            else:
                data_table.setItem(row, col, QTableWidgetItem(str(dataset['x_values'][row][col])))


def create_roc_graph(classified_data):
    f, ax = plt.subplots()
    mean_fpr, mean_tpr, mean_auc, std_auc, tprs_lower, tprs_upper = classified_data[0]['mean_values_tuple']
    re_mean_fpr, re_mean_tpr, re_mean_auc, re_std_auc, re_tprs_lower, re_tprs_upper = classified_data[1]['mean_values_tuple']
    ax.get_figure().set_size_inches(5, 5)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Random guess', alpha=.8)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (Standard case) (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)
    ax.plot(re_mean_fpr, re_mean_tpr, color='g',
            label=r'Mean ROC (Resampled case) (AUC = %0.2f $\pm$ %0.2f)' % (re_mean_auc, re_std_auc),
            lw=2, alpha=.8)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right", prop={'size': 7})
    ax.set_title('ROC chart')
    ax.xaxis.labelpad = -0.5
    canvas = FigureCanvasQTAgg(f)
    canvas.setMinimumHeight(350)
    canvas.setMaximumHeight(350)
    canvas.setMinimumSize(canvas.size())
    return canvas


def create_pr_graph(classified_data):
    f, ax = plt.subplots()
    nml_y_true = np.concatenate(classified_data[0]['trues_list'])
    nml_probas = np.concatenate(classified_data[0]['preds_list'])
    resampled_y_true = np.concatenate(classified_data[1]['trues_list'])
    resampled_probas = np.concatenate(classified_data[1]['preds_list'])
    pr, re,  _ = precision_recall_curve(nml_y_true, nml_probas[:, 1])
    resam_pr, resam_re, _ = precision_recall_curve(resampled_y_true, resampled_probas[:, 1])
    ax.get_figure().set_size_inches(5, 5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('PR chart')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.plot(pr, re, color='b', label="Standard case (AUC = {:.2f})".format(classified_data[0]['average_precision']))
    ax.plot(resam_re, resam_pr, color='g', label="Re-sampled case (AUC = {:.2f})".format(classified_data[1]['average_precision']))
    ax.legend(loc="upper right", prop={'size': 7})
    ax.xaxis.labelpad = -0.5
    canvas = FigureCanvasQTAgg(f)
    canvas.setMinimumHeight(350)
    canvas.setMaximumHeight(350)
    return canvas


def __clear_layout(layout):
    for i in reversed(range(layout.count())):
        layout.itemAt(i).widget().setParent(None)


def __round(digit, digit_after_fp):
    if not isinstance(digit, str):
        return ("{:." + str(digit_after_fp) + "f}").format(digit)
    else:
        return digit