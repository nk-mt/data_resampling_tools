import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from PyQt5.QtWidgets import QDialog, QComboBox, QPushButton

from ui.window_standard_graph import Ui_Dialog

matplotlib.get_backend()
matplotlib.use("QT5Agg")
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


def draw_standard_xy_graph(dataset_tc_set, dataset, header_row, x_combo_box, y_combo_box):
    negative_tc, positive_tc = dataset_tc_set
    df = dataset['dataset_as_dataframe']
    df.columns = header_row
    df.loc[df['Y'] == negative_tc, ['Y']] = 'Negative class'
    df.loc[df['Y'] == positive_tc, ['Y']] = 'Positive class'
    ax = sns.scatterplot(x=x_combo_box.currentText(), y=y_combo_box.currentText(), hue="Y", palette={'Negative class' : 'blue', 'Positive class': 'red'}, label="", data=df)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.figure.show()


def draw_popup_for_standard_graph(main_window, is_resampled_case):
    w = QDialog(main_window)
    w.ui = Ui_Dialog()
    w.ui.setupUi(w)
    if is_resampled_case:
        dataset = main_window.state.resampled_dataset
    else:
        dataset = main_window.state.dataset

    dataset_tc_set = dataset['y_values_as_set']
    pd.DataFrame(np.random.randn(10, 2), columns=['a', 'b'])
    header_row = main_window.state.dataset['header_row']

    x_combo_box = w.findChild(QComboBox, "xComboBox")
    y_combo_box = w.findChild(QComboBox, "yComboBox")
    x_combo_box.addItems(main_window.state.dataset['header_row'])
    y_combo_box.addItems(main_window.state.dataset['header_row'])
    w.findChild(QPushButton, "showGraphButton").clicked.connect(lambda: draw_standard_xy_graph(dataset_tc_set, dataset, header_row, x_combo_box, y_combo_box))
    w.setFocus(True)
    w.show()


def draw_pie_chart(main_window, is_resampled_case):
    if is_resampled_case:
        dataset = main_window.state.resampled_dataset
    else:
        dataset = main_window.state.dataset
    negative_tc, positive_tc = main_window.state.dataset['y_values_as_set']
    df = dataset['dataset_as_dataframe']
    df.columns = main_window.state.dataset['header_row']
    df.loc[df['Y'] == negative_tc, ['Y']] = 'Negative class'
    df.loc[df['Y'] == positive_tc, ['Y']] = 'Positive class'
    plot = df.pivot_table(columns='Y', aggfunc='size').plot(kind='pie', autopct='%.1f', colors=['blue', 'red'],
                                                            label="", legend=True, subplots=False,
                                                            textprops={'color': "w"})
    plot.legend(loc="upper left")
    # __add_canvas_for_img(plot.figure)
    plot.figure.show()


def draw_pca(main_window, is_resampled_dataset):
    if is_resampled_dataset:
        dataset = main_window.state.resampled_dataset
    else:
        dataset = main_window.state.dataset
    pca = PCA(n_components=2)
    x_visible = pca.fit_transform(dataset['x_values'])
    y_values = dataset['y_values']
    negative_tc, positive_tc = main_window.state.dataset['y_values_as_set']
    f, ax = plt.subplots()
    f.set_size_inches(6, 6)
    ax.scatter(x_visible[y_values == negative_tc, 0], x_visible[y_values == negative_tc, 1], label="Negative class",
                     alpha=0.5, color='b')
    ax.scatter(x_visible[y_values == positive_tc, 0], x_visible[y_values == positive_tc, 1], label="Positive class",
               alpha=0.5, color='r')
    if is_resampled_dataset:
        ax.set_title("PCA Re-sampled data with {}".format( main_window.state.sampling_algorithm_data_tab.value[0]))
    else:
        ax.set_title("PCA Original data")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("PCA_X")
    ax.set_ylabel("PCA_Y")
    ax.legend(loc="upper left", prop={'size': 7})
    f.show()


def draw_pair_plot_graph(main_window, is_resampled_case):
    if is_resampled_case:
        dataset = main_window.state.resampled_dataset
    else:
        dataset = main_window.state.dataset
    df = dataset['dataset_as_dataframe']
    df.columns = main_window.state.dataset['header_row']
    negative_tc, positive_tc = main_window.state.dataset['y_values_as_set']
    df.loc[df['Y'] == negative_tc, ['Y']] = 'Negative class'
    df.loc[df['Y'] == positive_tc, ['Y']] = 'Positive class'
    pp = sns.pairplot(df, hue="Y", diag_kind="kde", palette={'Negative class': 'blue', 'Positive class': 'red'}, size=1);
    handles = pp._legend_data.values()
    labels = pp._legend_data.keys()
    del pp.fig.legends[0]
    pp.fig.legend(handles=handles, labels=labels, loc='lower center', ncol=2)
    pp.fig.subplots_adjust(top=0.92, bottom=0.085)
    pp.fig.show()


def __add_canvas_for_img(fig):
    dummy = plt.figure()
    cvs_manager = dummy.canvas.manager
    cvs_manager.canvas.figure = fig
    if dummy.legends is not None and len(dummy.legends) > 0:
        del dummy.legends[0]
    fig.set_canvas(cvs_manager.canvas)