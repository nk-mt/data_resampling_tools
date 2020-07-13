from PyQt5.QtCore import QThread, pyqtSignal

from functions.general_functions import compute_some_statistics_for_the_dataset
from functions.resampling_functions import do_resampling
from ui.widgets import Widgets


class Resampling(QThread):

    update_gui_after_resampling_signal = pyqtSignal(bool)
    reraise_non_mt_exception_signal = pyqtSignal(Exception)

    def __init__(self, main_window):
        super(Resampling, self).__init__()
        self.main_window = main_window
        self.__custom_pre_process()

    def run(self):
        is_successful = True
        try:
            resampled_dataset = do_resampling(self.main_window.state)
            compute_some_statistics_for_the_dataset(resampled_dataset, self.main_window.state, True)
            self.main_window.state.resampled_dataset = resampled_dataset
        except Exception as e:
            self.reraise_non_mt_exception_signal.emit(e)
            is_successful = False
        self.__custom_post_process(is_successful)

    def __custom_pre_process(self):
        self.main_window.widgets.get_progress_bar(Widgets.ProgressBars.ResampleProgressBar.value).setValue(0)
        self.main_window.widgets.get_progress_bar(Widgets.ProgressBars.ResampleProgressBar.value).setMinimum(0)
        self.main_window.widgets.get_progress_bar(Widgets.ProgressBars.ResampleProgressBar.value).setMaximum(0)
        self.main_window.widgets.enable_disable_widgets(self.main_window.widgets.resampling_related_widgets(), False)

    def __custom_post_process(self, is_successful):
        self.update_gui_after_resampling_signal.emit(is_successful)

