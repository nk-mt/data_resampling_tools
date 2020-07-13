import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from pandas import np
from sklearn.metrics import average_precision_score, precision_recall_curve

matplotlib.get_backend()
matplotlib.use("QT5Agg")
import matplotlib.pyplot as plt
from enum import Enum
from PyQt5.QtWidgets import QPushButton, QComboBox, QLabel, QProgressBar, QScrollArea, QTableWidget


class Widgets:
    def __init__(self, main_window):
        self.buttons = {button.value: main_window.findChild(QPushButton, button.value) for button in Widgets.Buttons}
        self.combo_boxes = {combo_box.value: main_window.findChild(QComboBox, combo_box.value)
                            for combo_box in Widgets.ComboBoxes}
        self.labels = {label.value: main_window.findChild(QLabel, label.value) for label in Widgets.Labels}
        self.progress_bars = {progress_bar.value: main_window.findChild(QProgressBar, progress_bar.value)
                              for progress_bar in Widgets.ProgressBars}
        self.scroll_areas = {scroll_area.value: main_window.findChild(QScrollArea, scroll_area.value)
                             for scroll_area in Widgets.ScrollAreas}
        self.tables = {table.value: main_window.findChild(QTableWidget, table.value) for table in Widgets.Tables}

    class Buttons(Enum):
        DatasetButton = "datasetButton"
        OutputDirectoryButton = "outputDirectoryButton"
        ResampleButton = "resampleButton"
        ClassifyButton = "classifyButton"
        StandardGraphNormalDatasetButton = "standardGraphNormalDataset"
        PairPlotNormalDatasetButton = "pairPlotNormalDataset"
        PairPlotResampledDatasetButton = "pairPlotResampledDataset"
        StandardGraphResampledDatasetButton = "standardGraphResampledDataset"
        PcaGraphNormalDatasetButton = "pcaGraphNormalDataset"
        PcaGraphResampledDatasetButton = "pcaGraphResampledDataset"
        PieChartNormalDatasetButton = "pieChartNormalDataset"
        PieChartResampledDatasetButton = "pieChartResampledDataset"
        ClearButton = "clearGraphs"


    class ComboBoxes(Enum):
        ResamplingAlgorithms = "resamplingAlgorithms"
        ResamplingAlgorithmsExperimentsCase = "resamplingAlgorithmsExperimentsCase"
        ClassificationAlgorithms = "classAlgorithms"
        NumberOfFoldsCV = "numberOfFoldsCV"

    class Labels(Enum):
        DatasetPickedLabel = "datasetPickedLabel"
        DatasetStatisticsLabel = "datasetStatisticsLabel"
        ClassifyingStatusLabel = "classifyingStatusLabel"
        ResampledDatasetStatistics = "resampledDatasetStatistics"
        OutputDirectoryPickedLabel = "outputDirectoryPickedLabel"
        AfterClassificationStatistics = "afterClassificationStatistics"
        ClassificationAlgorithmLabel = "classAlgorithmLabel"
        FilePathLabel = "filePathLabel"
        DatasetLoadingResultLabel = "datasetLoadingResultLabel"
        DatasetLoadingTextLabel = "datasetLoadingTextLabel"
        TotalNumberOfExamplesLabel = "totalNumberOfExamplesLabel"
        NumberOfPositiveExamplesLabel = "numberOfPositivexamplesLabel"
        TargetClassPercentageLabel = "targetClassPercentageLabel"
        ImbalancedRatioLabel = "imbalancedRatioLabel"
        TotalNumberOfExamplesResampledLabel = "totalNumberOfExamplesResampledLabel"
        NumberOfPositiveExamplesResampledLabel = "numberOfPositiveExamplesResampledLabel"
        TargetClassPercentageResampledLabel = "targetClassPercentageResampledLabel"
        ImbalancedRatioResampledLabel = "imbalancedRatioResampledLabel"
        SelectedDatasetExperimentsTabLabel = "selectedDatasetExperimentsTab"

    class ProgressBars(Enum):
        DatasetProgressBar = "datasetProgressBar"
        ResampleProgressBar = "resampleProgressBar"
        NormalClassifyProgressBar = "normalClassifyProgressBar"
        ResampleClassifyProgressBar = "resampleClassifyProgressBar"

    class ScrollAreas(Enum):
        AfterClassificationStatisticsArea = "afterClassificationStatisticsArea"

    class Tables(Enum):
        DataTable = "dataTableWidget"

    def get_button(self, widget_id):
        return self.buttons[widget_id]

    def get_combo_box(self, widget_id):
        return self.combo_boxes[widget_id]

    def get_label(self, widget_id):
        return self.labels[widget_id]

    def get_progress_bar(self, widget_id):
        return self.progress_bars[widget_id]

    def get_scroll_area(self, widget_id):
        return self.scroll_areas[widget_id]

    def get_table(self, widget_id):
        return self.tables[widget_id]

    def enable_disable_widgets(self, widgets, enable):
        for w in widgets:
            if enable:
                w.setEnabled(True)
            else:
                w.setEnabled(False)

    def resampling_related_widgets(self):
        widgets = list()
        widgets.append(self.get_button(Widgets.Buttons.PcaGraphResampledDatasetButton.value))
        widgets.append(self.get_button(Widgets.Buttons.PieChartResampledDatasetButton.value))
        widgets.append(self.get_button(Widgets.Buttons.StandardGraphResampledDatasetButton.value))
        widgets.append(self.get_button(Widgets.Buttons.PairPlotResampledDatasetButton.value))
        return widgets

    def dataset_loading_related_widgets(self, enable):
        widgets = list()
        widgets.append(self.get_button(Widgets.Buttons.PieChartNormalDatasetButton.value))
        widgets.append(self.get_button(Widgets.Buttons.StandardGraphNormalDatasetButton.value))
        widgets.append(self.get_button(Widgets.Buttons.OutputDirectoryButton.value))
        widgets.append(self.get_button(Widgets.Buttons.PairPlotNormalDatasetButton.value))
        widgets.append(self.get_button(Widgets.Buttons.PcaGraphNormalDatasetButton.value))
        widgets.append(self.get_button(Widgets.Buttons.ClassifyButton.value))
        widgets.append(self.get_button(Widgets.Buttons.ClearButton.value))

        if not enable:
            widgets.append(self.get_button(Widgets.Buttons.PieChartResampledDatasetButton.value))
            widgets.append(self.get_button(Widgets.Buttons.StandardGraphResampledDatasetButton.value))
            widgets.append(self.get_button(Widgets.Buttons.PairPlotResampledDatasetButton.value))
            widgets.append(self.get_button(Widgets.Buttons.PcaGraphResampledDatasetButton.value))
            widgets.append(self.get_button(Widgets.Buttons.ResampleButton.value))

        widgets.append(self.get_combo_box(Widgets.ComboBoxes.ResamplingAlgorithms.value))
        widgets.append(self.get_combo_box(Widgets.ComboBoxes.ClassificationAlgorithms.value))
        widgets.append(self.get_combo_box(Widgets.ComboBoxes.ResamplingAlgorithmsExperimentsCase.value))
        widgets.append(self.get_combo_box(Widgets.ComboBoxes.NumberOfFoldsCV.value))
        return widgets