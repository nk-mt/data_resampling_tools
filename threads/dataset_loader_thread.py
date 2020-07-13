import numpy as np
import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal

from functions.general_functions import compute_some_statistics_for_the_dataset
from ui.widgets import Widgets


class DatasetLoader(QThread):

    update_dataset_load_progress_bar_signal = pyqtSignal(object)
    update_gui_after_dataset_load_signal = pyqtSignal(str, bool)
    reraise_non_mt_exception_signal = pyqtSignal(Exception)

    def __init__(self, main_window, path):
        super(DatasetLoader, self).__init__()
        self.path = path
        self.main_window = main_window
        self.__custom_pre_process()

    def run(self):
        is_successful = True
        try:
            self.__load_dataset(self.path)
            compute_some_statistics_for_the_dataset(self.main_window.state.dataset, self.main_window.state, False)
        except Exception as e:
            self.reraise_non_mt_exception_signal.emit(e)
            self.main_window.setEnabled(True)
            is_successful = False
        self.__custom_post_process(is_successful)

    def __custom_post_process(self, is_successful):
        self.update_gui_after_dataset_load_signal.emit(self.path, is_successful)

    def __custom_pre_process(self):
        self.main_window.setEnabled(False)
        self.main_window.widgets.get_progress_bar(Widgets.ProgressBars.DatasetProgressBar.value).setVisible(True)
        self.main_window.widgets.get_progress_bar(Widgets.ProgressBars.DatasetProgressBar.value).setMaximum(0)
        self.main_window.widgets.get_progress_bar(Widgets.ProgressBars.DatasetProgressBar.value).setMinimum(0)
        self.main_window.widgets.get_label(Widgets.Labels.DatasetLoadingResultLabel.value).setText("")
        self.main_window.widgets.get_label(Widgets.Labels.FilePathLabel.value).setText("")
        self.main_window.widgets.get_label(Widgets.Labels.DatasetPickedLabel.value).setText("")
        self.main_window.widgets.get_label(Widgets.Labels.DatasetLoadingTextLabel.value).setText("Loading...")
        self.main_window.widgets.enable_disable_widgets(self.main_window.widgets.dataset_loading_related_widgets(False), False)

        self.main_window.widgets.get_label(Widgets.Labels.TotalNumberOfExamplesLabel.value).setText("")
        self.main_window.widgets.get_label(Widgets.Labels.NumberOfPositiveExamplesLabel.value).setText("")
        self.main_window.widgets.get_label(Widgets.Labels.TargetClassPercentageLabel.value).setText("")
        self.main_window.widgets.get_label(Widgets.Labels.ImbalancedRatioLabel.value).setText("")
        self.main_window.widgets.get_label(Widgets.Labels.SelectedDatasetExperimentsTabLabel.value).setText("")

        self.main_window.widgets.get_label(Widgets.Labels.TotalNumberOfExamplesResampledLabel.value).setText("")
        self.main_window.widgets.get_label(Widgets.Labels.NumberOfPositiveExamplesResampledLabel.value).setText("")
        self.main_window.widgets.get_label(Widgets.Labels.TargetClassPercentageResampledLabel.value).setText("")
        self.main_window.widgets.get_label(Widgets.Labels.ImbalancedRatioResampledLabel.value).setText("")
        self.main_window.widgets.get_label(Widgets.Labels.OutputDirectoryPickedLabel.value).setText("")

    def __load_dataset(self, path):
        first_row = pd.read_csv(path, delimiter=',', header=0, nrows=1)
        header_row = self.__has_header(first_row)
        tfr = pd.read_csv(path, delimiter=',', iterator=True, header=header_row)
        ds_as_dataframe = pd.concat(tfr)
        dataset = dict()
        columns_length = len(ds_as_dataframe.columns)
        if header_row:
            dataset['header_row'] = first_row.columns.to_numpy().flatten()
        else:
            dataset['header_row'] = np.array(['X_{}'.format(i) for i in range(columns_length - 1)] + ['Y'])
        dataset['dataset_as_dataframe'] = ds_as_dataframe
        dataset['x_values'] = ds_as_dataframe.iloc[:, :columns_length - 1].to_numpy()
        dataset['y_values'] = ds_as_dataframe.iloc[:, columns_length - 1:].to_numpy().flatten()
        dataset['name'] = path.split("/")[-1].split(".csv")[0]
        self.main_window.state.dataset = dataset

    def __has_header(self, df_firstrow):
        for el in df_firstrow:
            try:
                if str(el).count(".") == 2:
                    continue
                float(el)
            except ValueError:
                return 1
        return None