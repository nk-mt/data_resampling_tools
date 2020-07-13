import sys
import traceback

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QApplication, QDialog

from algs.classification_algorithms import ClassificationAlgorithms
from algs.resampling_methods import ResamplingAlgorithms
from functions.ui_helping_functions import update_widgets_after_datasetload, update_widgets_after_resampling, \
    update_widgets_after_classification
from functions.ui_qt_slots import choose_dataset, \
    choose_outputdir, perform_resampling, \
    choose_sampling_algorithm, classify_datasets, choose_classification_algorithm, show_pca_graph, show_standard_graph, \
    store_selected_k, show_pair_plot_graph, \
    show_pie_chart, clear_experiments
from state import BasicState
from ui.error_dialog import Ui_Dialog
from ui.generated_pyqt_ui import Ui_DataResamplingTools
from ui.widgets import Widgets

QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)


class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_DataResamplingTools()
        self.ui.setupUi(self)
        self.state = BasicState()
        self.widgets = Widgets(self)
        # default algorithms
        self.state.sampling_algorithm_data_tab = ResamplingAlgorithms.RO
        self.state.sampling_algorithm_experiments_tab = ResamplingAlgorithms.RO
        self.state.classification_algorithm = ClassificationAlgorithms.CART

    # Signal callbacks here...

    def update_dataset_load_progress_bar(self, value):
        self.widgets.get_progress_bar(Widgets.ProgressBars.DatasetProgressBar.value).setValue(value)

    def update_normal_classify_progress_bar(self, value):
        self.widgets.get_progress_bar(Widgets.ProgressBars.NormalClassifyProgressBar.value).setValue(value)

    def update_resample_classify_progress_bar(self, value):
        self.widgets.get_progress_bar(Widgets.ProgressBars.ResampleClassifyProgressBar.value).setValue(value)

    def update_gui_after_classification(self):
        update_widgets_after_classification(self)

    def update_gui_after_dataset_load(self, value, is_successful):
        update_widgets_after_datasetload(self, value, is_successful)

    def update_gui_after_resampling(self, is_successful):
        update_widgets_after_resampling(self, is_successful)

    def do_setup(self):
        self.__register_callbacks()
        self.__fill_combo_boxes()
        self.__redefine_exceptions_hook()

    # Widget callbacks (user interactions)

    def __register_callbacks(self):
        self.widgets.get_button(Widgets.Buttons.DatasetButton.value).clicked.connect(lambda: choose_dataset(self))
        self.widgets.get_button(Widgets.Buttons.OutputDirectoryButton.value).clicked.connect(
            lambda: choose_outputdir(mw))
        self.widgets.get_button(Widgets.Buttons.ResampleButton.value).clicked.connect(lambda: perform_resampling(self))
        self.widgets.get_button(Widgets.Buttons.ClassifyButton.value).clicked.connect(lambda: classify_datasets(self))
        self.widgets.get_combo_box(Widgets.ComboBoxes.ResamplingAlgorithms.value).activated.connect(
            lambda: choose_sampling_algorithm(self, True))
        self.widgets.get_combo_box(Widgets.ComboBoxes.ResamplingAlgorithmsExperimentsCase.value).activated.connect(
            lambda: choose_sampling_algorithm(self, False))
        self.widgets.get_combo_box(Widgets.ComboBoxes.ClassificationAlgorithms.value).activated.connect(
            lambda: choose_classification_algorithm(self))
        self.widgets.get_combo_box(Widgets.ComboBoxes.NumberOfFoldsCV.value).activated.connect(
            lambda: store_selected_k(self))
        self.widgets.get_button(Widgets.Buttons.StandardGraphNormalDatasetButton.value).clicked.connect(
            lambda: show_standard_graph(self, False))
        self.widgets.get_button(Widgets.Buttons.PairPlotNormalDatasetButton.value).clicked.connect(
            lambda: show_pair_plot_graph(self, False))
        self.widgets.get_button(Widgets.Buttons.PairPlotResampledDatasetButton.value).clicked.connect(
            lambda: show_pair_plot_graph(self, True))
        self.widgets.get_button(Widgets.Buttons.StandardGraphResampledDatasetButton.value).clicked.connect(
            lambda: show_standard_graph(self, True))
        self.widgets.get_button(Widgets.Buttons.PcaGraphNormalDatasetButton.value).clicked.connect(
            lambda: show_pca_graph(self, False))
        self.widgets.get_button(Widgets.Buttons.PcaGraphResampledDatasetButton.value).clicked.connect(
            lambda: show_pca_graph(self, True))
        self.widgets.get_button(Widgets.Buttons.PieChartNormalDatasetButton.value).clicked.connect(
            lambda: show_pie_chart(self, False))
        self.widgets.get_button(Widgets.Buttons.PieChartResampledDatasetButton.value).clicked.connect(
            lambda: show_pie_chart(self, True))
        self.widgets.get_button(Widgets.Buttons.ClearButton.value).clicked.connect(lambda: clear_experiments(self))

    def __fill_combo_boxes(self):
        self.widgets.get_combo_box(Widgets.ComboBoxes.ResamplingAlgorithms.value).addItems(
            [ra.value[0] for ra in ResamplingAlgorithms if ra is not ResamplingAlgorithms.SMOTE_BOOST])
        self.widgets.get_combo_box(Widgets.ComboBoxes.ResamplingAlgorithmsExperimentsCase.value).addItems(
            [ra.value[0] for ra in ResamplingAlgorithms])
        self.widgets.get_combo_box(Widgets.ComboBoxes.ClassificationAlgorithms.value).addItems(
            [ca.value[0] for ca in ClassificationAlgorithms])
        # Folds for CV from 2 to 10. Default 10
        self.widgets.get_combo_box(Widgets.ComboBoxes.NumberOfFoldsCV.value).addItems(map(str, range(2,11)))
        self.widgets.get_combo_box(Widgets.ComboBoxes.NumberOfFoldsCV.value).setCurrentIndex(8)

    def __create_error_dialog(self, error_traceback, error_msg):
        dialog = QDialog(self)
        dialog.ui = Ui_Dialog()
        dialog.ui.setupUi(dialog)
        dialog.ui.errorTextPlaceholder.setText("Traceback (most recent call last):\n" +
                                               ''.join(traceback.format_tb(error_traceback)) + error_msg)

        dialog.setFocus(True)
        dialog.show()

    def __redefine_exceptions_hook(self):
        original_hook = sys.excepthook

        def call_original_exception_hook(exception_type, exception_message, error_traceback):
            self.__create_error_dialog(error_traceback, exception_type.__name__ + " " + str(exception_message))
            original_hook(exception_type, exception_message, error_traceback)
        sys.excepthook = call_original_exception_hook

    def reraise_non_mt_exception(self, exception):
        raise exception

if __name__ == '__main__':
    app = QApplication([])
    mw = MainWindow()
    mw.setWindowIcon(QIcon('logo.png'))
    mw.do_setup()
    mw.showMaximized()
    mw.show()
    app.exec_()

