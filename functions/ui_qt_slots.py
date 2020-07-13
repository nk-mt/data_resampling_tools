from PyQt5.QtWidgets import QFileDialog

from algs.classification_algorithms import ClassificationAlgorithms
from algs.resampling_methods import ResamplingAlgorithms
from functions.drawing_functions import draw_pca, \
    draw_popup_for_standard_graph, draw_pie_chart, draw_pair_plot_graph
from functions.ui_helping_functions import clear_tables_and_graphs
from threads.classifier_thread import Classifying
from threads.dataset_loader_thread import DatasetLoader
from threads.resampling_thread import Resampling
from ui.widgets import Widgets


def choose_dataset(main_window):
    ds_dialog = QFileDialog(main_window)
    ds_dialog.show()
    if ds_dialog.exec_():
        file_paths = ds_dialog.selectedFiles()
        main_window.dloader = DatasetLoader(main_window, file_paths[0])
        main_window.dloader.update_dataset_load_progress_bar_signal.connect(main_window.update_dataset_load_progress_bar)
        main_window.dloader.update_gui_after_dataset_load_signal.connect(main_window.update_gui_after_dataset_load)
        main_window.dloader.reraise_non_mt_exception_signal.connect(main_window.reraise_non_mt_exception)
        main_window.dloader.start()


def choose_outputdir(main_window):
    od = QFileDialog(main_window).getExistingDirectory()
    if od != "":
        main_window.widgets.get_label(Widgets.Labels.OutputDirectoryPickedLabel.value).setText(od)
        main_window.state.output_dir = od
        main_window.widgets.get_button(Widgets.Buttons.ResampleButton.value).setEnabled(True)


def choose_sampling_algorithm(main_window, data_tab):
    if data_tab:
        chosen_algorithm_name = main_window.widgets.\
            get_combo_box(Widgets.ComboBoxes.ResamplingAlgorithms.value).currentText()
        main_window.state.sampling_algorithm_data_tab = ResamplingAlgorithms.get_algorithm_by_name(chosen_algorithm_name)
    else:
        chosen_algorithm_name = main_window.widgets. \
            get_combo_box(Widgets.ComboBoxes.ResamplingAlgorithmsExperimentsCase.value).currentText()
        main_window.state.sampling_algorithm_experiments_tab = ResamplingAlgorithms.get_algorithm_by_name(chosen_algorithm_name)


def store_selected_k(main_window):
    number_of_folds = int (main_window.widgets.get_combo_box(Widgets.ComboBoxes.NumberOfFoldsCV.value).currentText())
    main_window.state.number_of_folds = number_of_folds
    main_window.widgets.get_progress_bar(Widgets.ProgressBars.NormalClassifyProgressBar.value).setMaximum(
        number_of_folds)
    main_window.widgets.get_progress_bar(Widgets.ProgressBars.ResampleClassifyProgressBar.value).setMaximum(
        number_of_folds)
    if main_window.widgets.get_progress_bar(Widgets.ProgressBars.NormalClassifyProgressBar.value).value() is not 0:
        main_window.widgets.get_progress_bar(Widgets.ProgressBars.NormalClassifyProgressBar.value).setValue(
            number_of_folds)
        main_window.widgets.get_progress_bar(Widgets.ProgressBars.ResampleClassifyProgressBar.value).setValue(
            number_of_folds)


def choose_classification_algorithm(main_window):
    chosen_algorithm_name = main_window.widgets. \
        get_combo_box(Widgets.ComboBoxes.ClassificationAlgorithms.value).currentText()
    main_window.state.classification_algorithm = ClassificationAlgorithms.get_algorithm_by_name(chosen_algorithm_name)


def perform_resampling(main_window):
    main_window.resampler = Resampling(main_window)
    main_window.resampler.reraise_non_mt_exception_signal.connect(main_window.reraise_non_mt_exception)
    main_window.resampler.update_gui_after_resampling_signal.connect(main_window.update_gui_after_resampling)
    main_window.resampler.start()


def classify_datasets(main_window):
    main_window.widgets.get_progress_bar(Widgets.ProgressBars.NormalClassifyProgressBar.value).setValue(0)
    main_window.classifier = Classifying(main_window, False)
    main_window.state.normal_classify_thread_finished = False
    main_window.classifier.update_gui_after_classification.connect(main_window.update_gui_after_classification)
    main_window.classifier.update_normal_classify_progress_bar.connect(main_window.update_normal_classify_progress_bar)
    main_window.classifier.start()

    main_window.widgets.get_progress_bar(Widgets.ProgressBars.ResampleClassifyProgressBar.value).setValue(0)
    main_window.classifier_rd = Classifying(main_window, True)
    main_window.state.resample_classify_thread_finished = False
    main_window.classifier_rd.update_gui_after_classification.connect(main_window.update_gui_after_classification)
    main_window.classifier_rd.reraise_non_mt_exception_signal.connect(main_window.reraise_non_mt_exception)
    main_window.classifier_rd.update_resample_classify_progress_bar.connect(main_window.update_resample_classify_progress_bar)
    main_window.classifier_rd.start()


def show_pair_plot_graph(main_window, is_resampled_case):
    draw_pair_plot_graph(main_window, is_resampled_case)


def show_standard_graph(main_window, is_resampled_case):
    draw_popup_for_standard_graph(main_window, is_resampled_case)


def show_pca_graph(main_window, is_resampled_case):
    draw_pca(main_window, is_resampled_case)


def show_pie_chart(main_window, is_resampled_case):
    draw_pie_chart(main_window, is_resampled_case)


def clear_experiments(main_window):
    clear_tables_and_graphs(main_window)




