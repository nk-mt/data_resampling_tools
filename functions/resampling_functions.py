import csv
import datetime

import numpy as np
import pandas as pd


def do_resampling(state):
    sampling_algorithm = state.sampling_algorithm_data_tab
    output_directory = state.output_dir
    dataset = state.dataset
    y_values = dataset['y_values']
    x_values = dataset['x_values']
    x_resampled_values, y_resampled_values = sampling_algorithm.value[1].fit_sample(x_values, y_values)
    resampled_dataset = dict()
    resampled_dataset['x_values'] = x_resampled_values
    resampled_dataset['y_values'] = y_resampled_values
    rd_as_df = pd.DataFrame(data=np.c_[x_resampled_values, np.vstack(y_resampled_values.T)], columns=dataset['header_row'])
    resampled_dataset['dataset_as_dataframe'] = rd_as_df
    __write_dataset_to_csv\
        (__create_resampled_file_name(output_directory, dataset['name'], sampling_algorithm.value[0]),
            x_values_param=x_resampled_values, y_values_param=y_resampled_values)
    return resampled_dataset


def do_resampling_without_writing_to_file(sampling_algorithm, x_values, y_values):
    x_resampled_values, y_resampled_values = sampling_algorithm.value[1].fit_resample(x_values, y_values.astype(int))
    resampled_dataset = dict()
    resampled_dataset['x_values'] = x_resampled_values
    resampled_dataset['y_values'] = y_resampled_values
    return resampled_dataset


def __create_resampled_file_name(output_dir, dataset_name, algorithm_name):
    time = str(datetime.datetime.now()).replace(" ", "_").replace(".", "_").replace(":", "_")
    return output_dir + "/" + dataset_name + "_" + algorithm_name.replace(" ", "_") + "_" + time + ".csv"


def __write_dataset_to_csv(path, dataset=None, x_values_param=None, y_values_param=None):
    if dataset is not None:
        x_values_param = dataset['data']
        y_values_param = dataset['target']
    with open(path, "w", newline="\n") as csv_output_file:
        dataset_writer = csv.writer(csv_output_file, delimiter=",")
        for row_idx, dv in enumerate(x_values_param):
            dataset_writer.writerow(np.append(dv, y_values_param[row_idx]))
